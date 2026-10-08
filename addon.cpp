#define ImTextureID ImU64
#include <imgui.h>
#include <reshade.hpp>
#include <com_ptr.hpp>
#include <crc32_hash.hpp>
#include <d3d11.h>
#include <nvsdk_ngx.h>
#include <nvsdk_ngx_params.h>
#include <nvsdk_ngx_helpers.h>
#include <set>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <cstdio>
#include <cstring>
#include <string>
#include <cmath>

#include "intermediate/PrepareMotionVectors.h"
#include "intermediate/0x0D1CD1AA.h"
#include "frame_generation_bridge.hpp"

using namespace reshade::api;

// With the Frame Generation bridge, ReShade also reports DirectX 12 objects (the bridge's
// own device and swap chain). Everything in this add-on except the bridge is DirectX 11 only.
static bool IsD3D11(device* d) {
    return d && d->get_api() == device_api::d3d11;
}
static bool IsD3D11(command_list* cmd_list) {
    return cmd_list && IsD3D11(cmd_list->get_device());
}

NVSDK_NGX_Parameter* capabilityParameters = nullptr;
NVSDK_NGX_Parameter* parameters = nullptr;
NVSDK_NGX_Handle* dlssHandle = nullptr;

com_ptr<ID3D11Texture2D> motionVectorTexture;
com_ptr<ID3D11UnorderedAccessView> motionVectorUAV;
com_ptr<ID3D11ComputeShader> prepareMotionVectorShader;

com_ptr<ID3D11Buffer> cbSharpenModify;

std::set<ID3D11PixelShader*> taaShaders;
std::set<ID3D11PixelShader*> sharpenShaders;

bool dlssAvailable = false;
uint32_t shaderHash;
uint32_t currentWidth = 0;
uint32_t currentHeight = 0;
void* mappedConstantBuffer = nullptr;
float jitter[] = { 0.0f, 0.0f };
bool invokedThisFrame = false;
bool needReset = false;
bool needReinitialize = false;
NVSDK_NGX_DLSS_Hint_Render_Preset preset = NVSDK_NGX_DLSS_Hint_Render_Preset_K;
bool autoExposure = false;
float sharpenMultiplier = 1.0f;

// ---------------------------------------------------------------------------
// Diagnostic frame recorder (v2).
// Records one frame of the game's render passes into ReShade.log so we can
// map the pipeline before adding DLSS Super Resolution and Frame Generation.
// Every texture gets a short id (T1, T2, ...) for the duration of the capture,
// so the data flow between passes can be followed (who writes, who reads).
// ---------------------------------------------------------------------------
struct DiagPass {
    ID3D11RenderTargetView* rtv = nullptr; // identity only, never dereferenced
    std::string line;
    uint32_t draws = 0;
    bool open = false;
};

std::mutex diagMutex;
std::unordered_map<uint64_t, uint32_t> pixelShaderHashes; // guarded by diagMutex
std::unordered_map<ID3D11DeviceContext*, DiagPass> diagPasses; // guarded by diagMutex
std::unordered_map<ID3D11Resource*, uint32_t> diagIds; // guarded by diagMutex
std::atomic<bool> diagRequested = false;
std::atomic<bool> diagCapturing = false;
uint32_t diagPassIndex = 0;
uint64_t diagBackBuffer = 0;

HMODULE addonModule = nullptr;
FILE* diagFile = nullptr;
bool diagFileStartedThisSession = false;

// Diagnostic lines go to their own file (FFXV-DLAA-diag.log, next to the add-on),
// not to ReShade.log. The file is restarted once per game session; every recorded
// frame of the same session is appended to it.
static std::wstring DiagFilePath() {
    wchar_t modulePath[MAX_PATH] = {};
    GetModuleFileNameW(addonModule, modulePath, MAX_PATH);
    std::wstring path = modulePath;
    size_t slash = path.find_last_of(L"\\/");
    path = (slash == std::wstring::npos) ? L"" : path.substr(0, slash + 1);
    return path + L"FFXV-DLAA-diag.log";
}

static void DiagOpenFile() {
    if (diagFile) {
        return;
    }
    if (_wfopen_s(&diagFile, DiagFilePath().c_str(), diagFileStartedThisSession ? L"a" : L"w") != 0) {
        diagFile = nullptr;
    }
    diagFileStartedThisSession = true;
}

static void DiagCloseFile() {
    if (diagFile) {
        fclose(diagFile);
        diagFile = nullptr;
    }
}

static void DiagLog(const std::string& text) {
    if (diagFile) {
        fputs(text.c_str(), diagFile);
        fputc('\n', diagFile);
    }
    else {
        reshade::log::message(reshade::log::level::info, text.c_str());
    }
}

// Caller must hold diagMutex.
static std::string DiagTex(ID3D11Resource* resource) {
    if (!resource) {
        return "-";
    }
    uint32_t id;
    auto it = diagIds.find(resource);
    if (it == diagIds.end()) {
        id = (uint32_t)diagIds.size() + 1;
        diagIds[resource] = id;
    }
    else {
        id = it->second;
    }

    char buffer[128];
    com_ptr<ID3D11Texture2D> texture;
    com_ptr<ID3D11Buffer> buf;
    if (SUCCEEDED(resource->QueryInterface(&texture)) && texture) {
        D3D11_TEXTURE2D_DESC desc;
        texture->GetDesc(&desc);
        snprintf(buffer, sizeof(buffer), "T%u(%ux%u f%u%s%s)", id, desc.Width, desc.Height, (unsigned)desc.Format,
            desc.ArraySize > 1 ? " array" : "",
            (uint64_t)resource == diagBackBuffer ? " BACKBUFFER" : "");
    }
    else if (SUCCEEDED(resource->QueryInterface(&buf)) && buf) {
        D3D11_BUFFER_DESC desc;
        buf->GetDesc(&desc);
        snprintf(buffer, sizeof(buffer), "B%u(buffer %u bytes)", id, desc.ByteWidth);
    }
    else {
        snprintf(buffer, sizeof(buffer), "R%u(other)", id);
    }
    return buffer;
}

// Caller must hold diagMutex.
static std::string DiagView(ID3D11View* view) {
    if (!view) {
        return "-";
    }
    com_ptr<ID3D11Resource> resource;
    view->GetResource(&resource);
    return DiagTex(resource.get());
}

// Caller must hold diagMutex.
static std::string DiagListSRVs(ID3D11ShaderResourceView** srvs, UINT count) {
    std::string text;
    for (UINT i = 0; i < count; ++i) {
        if (srvs[i]) {
            text += " t" + std::to_string(i) + "=" + DiagView(srvs[i]);
            srvs[i]->Release();
            srvs[i] = nullptr;
        }
    }
    return text.empty() ? " (none)" : text;
}

static void DiagFlushPass(DiagPass& pass) {
    if (pass.open) {
        DiagLog(pass.line + " draws=" + std::to_string(pass.draws));
        pass.open = false;
    }
}

static void DiagOnGpuWork(ID3D11DeviceContext* ctx, bool isDispatch) {
    if (!diagCapturing) {
        return;
    }
    std::lock_guard<std::mutex> lock(diagMutex);
    if (!diagCapturing) {
        return;
    }

    const char* contextTag = ctx->GetType() == D3D11_DEVICE_CONTEXT_DEFERRED ? " [deferred]" : "";
    DiagPass& pass = diagPasses[ctx];
    char header[64];

    if (isDispatch) {
        DiagFlushPass(pass);
        pass.rtv = nullptr;

        ID3D11UnorderedAccessView* uavs[2] = {};
        ctx->CSGetUnorderedAccessViews(0, 2, uavs);
        std::string uavText;
        for (int i = 0; i < 2; ++i) {
            if (uavs[i]) {
                uavText += " u" + std::to_string(i) + "=" + DiagView(uavs[i]);
                uavs[i]->Release();
            }
        }
        ID3D11ShaderResourceView* srvs[4] = {};
        ctx->CSGetShaderResources(0, 4, srvs);
        std::string srvText = DiagListSRVs(srvs, 4);

        snprintf(header, sizeof(header), "[DLAA-DIAG] #%03u COMPUTE", diagPassIndex++);
        DiagLog(std::string(header) + " out:" + (uavText.empty() ? " (none)" : uavText) + " in:" + srvText + contextTag);
        return;
    }

    ID3D11RenderTargetView* rtvs[8] = {};
    com_ptr<ID3D11DepthStencilView> dsv;
    ctx->OMGetRenderTargets(8, rtvs, &dsv);
    ID3D11RenderTargetView* rtv0 = rtvs[0];

    if (pass.open && rtv0 == pass.rtv) {
        pass.draws++;
        for (auto* view : rtvs) {
            if (view) {
                view->Release();
            }
        }
        return;
    }
    DiagFlushPass(pass);

    std::string rtText;
    for (int i = 0; i < 8; ++i) {
        if (rtvs[i]) {
            rtText += " rt" + std::to_string(i) + "=" + DiagView(rtvs[i]);
        }
    }
    if (rtText.empty()) {
        rtText = " (no color target)";
    }
    std::string dsvText = dsv ? DiagView(dsv.get()) : "-";

    com_ptr<ID3D11PixelShader> ps;
    ctx->PSGetShader(&ps, nullptr, nullptr);
    char psText[48];
    auto hashIt = pixelShaderHashes.find((uint64_t)ps.get());
    if (!ps) {
        snprintf(psText, sizeof(psText), "none");
    }
    else if (hashIt != pixelShaderHashes.end()) {
        snprintf(psText, sizeof(psText), "0x%08X", hashIt->second);
    }
    else {
        snprintf(psText, sizeof(psText), "?");
    }
    const char* tag = "";
    if (taaShaders.count(ps.get())) {
        tag = " <<TAA>>";
    }
    else if (sharpenShaders.count(ps.get())) {
        tag = " <<SHARPEN>>";
    }

    com_ptr<ID3D11BlendState> blendState;
    FLOAT blendFactor[4];
    UINT sampleMask;
    ctx->OMGetBlendState(&blendState, blendFactor, &sampleMask);
    bool blendOn = false;
    if (blendState) {
        D3D11_BLEND_DESC blendDesc;
        blendState->GetDesc(&blendDesc);
        blendOn = blendDesc.RenderTarget[0].BlendEnable != FALSE;
    }

    UINT viewportCount = 1;
    D3D11_VIEWPORT viewport = {};
    ctx->RSGetViewports(&viewportCount, &viewport);

    ID3D11ShaderResourceView* srvs[8] = {};
    ctx->PSGetShaderResources(0, 8, srvs);
    std::string srvText = DiagListSRVs(srvs, 8);

    char details[160];
    snprintf(details, sizeof(details), " viewport=%.0fx%.0f+%.0f,%.0f blend=%s ps=%s",
        viewport.Width, viewport.Height, viewport.TopLeftX, viewport.TopLeftY,
        blendOn ? "on" : "off", psText);
    snprintf(header, sizeof(header), "[DLAA-DIAG] #%03u DRAW", diagPassIndex++);

    pass.line = std::string(header) + " out:" + rtText + " depth=" + dsvText + details + tag + " in:" + srvText + contextTag;
    pass.rtv = rtv0;
    pass.draws = 1;
    pass.open = true;

    for (auto* view : rtvs) {
        if (view) {
            view->Release();
        }
    }
}

// Called from the TAA branch of OnDraw.
static void DiagOnTAA(uint32_t rtWidth, uint32_t rtHeight) {
    if (!diagCapturing) {
        return;
    }
    char buffer[200];
    snprintf(buffer, sizeof(buffer),
        "[DLAA-DIAG]      TAA: render size=%ux%u jitter=%.6f,%.6f (pixels %.3f,%.3f)",
        rtWidth, rtHeight, jitter[0], jitter[1], jitter[0] * rtWidth, jitter[1] * rtHeight);
    DiagLog(buffer);
}

static void DiagOnPresent(swapchain* swapchain) {
    std::lock_guard<std::mutex> lock(diagMutex);
    if (diagCapturing) {
        for (auto& entry : diagPasses) {
            DiagFlushPass(entry.second);
        }
        DiagLog("[DLAA-DIAG] ===== END OF FRAME =====");
        DiagCloseFile();
        reshade::log::message(reshade::log::level::info, "Diagnostic frame recorded to FFXV-DLAA-diag.log");
        diagCapturing = false;
        diagPasses.clear();
        diagIds.clear();
    }
    if (diagRequested) {
        diagRequested = false;
        diagPassIndex = 0;
        diagPasses.clear();
        diagIds.clear();
        diagBackBuffer = swapchain->get_current_back_buffer().handle;
        DiagOpenFile();
        std::string backBufferText = DiagTex((ID3D11Resource*)diagBackBuffer);
        char buffer[160];
        snprintf(buffer, sizeof(buffer), " dlss=%s", dlssAvailable ? "available" : "NOT available");
        DiagLog("[DLAA-DIAG] ===== START OF FRAME (v2) ===== backbuffer=" + backBufferText + buffer);
        diagCapturing = true;
    }
}

// ---------------------------------------------------------------------------
// DLSS Super Resolution (used when the game's Resolution Scaling is below 100%).
//
// FFXV renders AND post-processes (DOF, bloom, tonemapping) at the reduced
// resolution, and only then stretches the finished image to the output size
// with a simple pixel shader (0x1B6C8C68), right before drawing the HUD.
// In Super Resolution mode we:
//   1. skip the game's TAA and pass the raw, jittered image through,
//   2. let the game run its post-processing at the reduced resolution,
//   3. replace the final stretch with DLSS (low resolution -> output resolution).
// The HUD is then drawn by the game at full resolution on top of the DLSS output.
// ---------------------------------------------------------------------------
constexpr uint32_t kUpscaleShaderHash = 0x1B6C8C68;
std::set<ID3D11PixelShader*> upscaleShaders;

NVSDK_NGX_Parameter* srParameters = nullptr;
NVSDK_NGX_Handle* srHandle = nullptr;
uint32_t srInWidth = 0;
uint32_t srInHeight = 0;
uint32_t srOutWidth = 0;
uint32_t srOutHeight = 0;
DXGI_FORMAT srOutFormat = DXGI_FORMAT_UNKNOWN;
com_ptr<ID3D11Texture2D> srOutputTexture;

bool superResolutionEnabled = true;
std::atomic<uint32_t> outputWidth = 0;  // back buffer size, updated every present
std::atomic<uint32_t> outputHeight = 0;

// State handed from the TAA pass to the stretch pass within one frame.
bool srPending = false;
bool srEvaluatedThisFrame = false;
bool srNeedReset = true;
uint32_t srRenderWidth = 0;
uint32_t srRenderHeight = 0;
float srJitter[2] = { 0.0f, 0.0f };
com_ptr<ID3D11Resource> srDepth;
uint32_t srRetryCooldown = 0; // frames to wait before trying Super Resolution again after a failure
std::atomic<const char*> srPauseReason = "";

uint32_t mvWidth = 0;
uint32_t mvHeight = 0;

std::atomic<const char*> dlssStatus = "waiting for the first frame";
std::atomic<uint32_t> statusInWidth = 0;
std::atomic<uint32_t> statusInHeight = 0;
std::atomic<uint32_t> statusOutWidth = 0;
std::atomic<uint32_t> statusOutHeight = 0;

static void SetStatus(const char* status, uint32_t inW, uint32_t inH, uint32_t outW, uint32_t outH) {
    dlssStatus = status;
    statusInWidth = inW;
    statusInHeight = inH;
    statusOutWidth = outW;
    statusOutHeight = outH;
}

static void DiagNote(const char* text) {
    if (diagCapturing) {
        DiagLog(std::string("[DLAA-DIAG]      ") + text);
    }
}

void ReleaseDLSS() {
    if (parameters) {
        NVSDK_NGX_D3D11_DestroyParameters(parameters);
        parameters = nullptr;
    }
    if (dlssHandle) {
        NVSDK_NGX_D3D11_ReleaseFeature(dlssHandle);
        dlssHandle = nullptr;
    }
}

void ReleaseSuperResolution() {
    if (srParameters) {
        NVSDK_NGX_D3D11_DestroyParameters(srParameters);
        srParameters = nullptr;
    }
    if (srHandle) {
        NVSDK_NGX_D3D11_ReleaseFeature(srHandle);
        srHandle = nullptr;
    }
    srOutputTexture.reset();
    srInWidth = 0;
    srInHeight = 0;
    srOutWidth = 0;
    srOutHeight = 0;
    srOutFormat = DXGI_FORMAT_UNKNOWN;
}

static void ReleaseMotionVectors() {
    motionVectorUAV.reset();
    motionVectorTexture.reset();
    mvWidth = 0;
    mvHeight = 0;
}

static bool EnsureMotionVectorTexture(ID3D11Device* device, uint32_t width, uint32_t height) {
    if (motionVectorTexture && motionVectorUAV && mvWidth == width && mvHeight == height) {
        return true;
    }
    ReleaseMotionVectors();

    D3D11_TEXTURE2D_DESC motionVectorDesc = {};
    motionVectorDesc.Width = width;
    motionVectorDesc.Height = height;
    motionVectorDesc.Usage = D3D11_USAGE_DEFAULT;
    motionVectorDesc.ArraySize = 1;
    motionVectorDesc.Format = DXGI_FORMAT_R16G16_FLOAT;
    motionVectorDesc.SampleDesc.Count = 1;
    motionVectorDesc.SampleDesc.Quality = 0;
    motionVectorDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    motionVectorDesc.CPUAccessFlags = 0;
    motionVectorDesc.MiscFlags = 0;
    motionVectorDesc.MipLevels = 1;
    if (FAILED(device->CreateTexture2D(&motionVectorDesc, nullptr, &motionVectorTexture))) {
        ReleaseMotionVectors();
        return false;
    }

    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_R16G16_FLOAT;
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    uavDesc.Texture2D.MipSlice = 0;
    if (FAILED(device->CreateUnorderedAccessView(motionVectorTexture.get(), &uavDesc, &motionVectorUAV))) {
        ReleaseMotionVectors();
        return false;
    }

    mvWidth = width;
    mvHeight = height;
    return true;
}

// Picks the DLSS quality mode that matches the game's resolution scaling.
// Only used as a hint: the actual input size is always the game's render size.
static NVSDK_NGX_PerfQuality_Value QualityForScale(float scale) {
    if (scale >= 0.99f) return NVSDK_NGX_PerfQuality_Value_DLAA;
    if (scale >= 0.64f) return NVSDK_NGX_PerfQuality_Value_MaxQuality;
    if (scale >= 0.55f) return NVSDK_NGX_PerfQuality_Value_Balanced;
    if (scale >= 0.45f) return NVSDK_NGX_PerfQuality_Value_MaxPerf;
    return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
}

static bool EnsureSuperResolution(ID3D11DeviceContext* context, uint32_t inW, uint32_t inH, uint32_t outW, uint32_t outH, DXGI_FORMAT outFormat) {
    if (srHandle && srOutputTexture &&
        srInWidth == inW && srInHeight == inH &&
        srOutWidth == outW && srOutHeight == outH &&
        srOutFormat == outFormat) {
        return true;
    }
    ReleaseSuperResolution();

    NVSDK_NGX_D3D11_AllocateParameters(&srParameters);
    if (!srParameters) {
        return false;
    }
    NVSDK_NGX_Parameter_SetUI(srParameters, NVSDK_NGX_Parameter_Width, inW);
    NVSDK_NGX_Parameter_SetUI(srParameters, NVSDK_NGX_Parameter_Height, inH);
    NVSDK_NGX_Parameter_SetUI(srParameters, NVSDK_NGX_Parameter_OutWidth, outW);
    NVSDK_NGX_Parameter_SetUI(srParameters, NVSDK_NGX_Parameter_OutHeight, outH);
    NVSDK_NGX_Parameter_SetI(srParameters, NVSDK_NGX_Parameter_PerfQualityValue, QualityForScale((float)inW / (float)outW));
    // The input is already tonemapped (LDR), and motion vectors are at the render resolution.
    NVSDK_NGX_Parameter_SetI(srParameters, NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, NVSDK_NGX_DLSS_Feature_Flags_MVLowRes);
    NVSDK_NGX_Parameter_SetI(srParameters, NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects, 0);
    NVSDK_NGX_Parameter_SetUI(srParameters, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, preset);
    NVSDK_NGX_Parameter_SetUI(srParameters, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality, preset);
    NVSDK_NGX_Parameter_SetUI(srParameters, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, preset);
    NVSDK_NGX_Parameter_SetUI(srParameters, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance, preset);
    NVSDK_NGX_Parameter_SetUI(srParameters, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance, preset);

    NVSDK_NGX_Result result = NVSDK_NGX_D3D11_CreateFeature(context, NVSDK_NGX_Feature_SuperSampling, srParameters, &srHandle);
    if (NVSDK_NGX_FAILED(result) || !srHandle) {
        srHandle = nullptr;
        ReleaseSuperResolution();
        return false;
    }

    com_ptr<ID3D11Device> device;
    context->GetDevice(&device);
    D3D11_TEXTURE2D_DESC outputDesc = {};
    outputDesc.Width = outW;
    outputDesc.Height = outH;
    outputDesc.MipLevels = 1;
    outputDesc.ArraySize = 1;
    outputDesc.Format = outFormat;
    outputDesc.SampleDesc.Count = 1;
    outputDesc.Usage = D3D11_USAGE_DEFAULT;
    outputDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    if (FAILED(device->CreateTexture2D(&outputDesc, nullptr, &srOutputTexture))) {
        ReleaseSuperResolution();
        return false;
    }

    srInWidth = inW;
    srInHeight = inH;
    srOutWidth = outW;
    srOutHeight = outH;
    srOutFormat = outFormat;
    return true;
}

// ---------------------------------------------------------------------------
// Texture detail in Super Resolution (mip-map bias).
//
// At a reduced render resolution the GPU picks smaller texture mip levels, so
// textures look like the reduced resolution even after DLSS. The DLSS
// Programming Guide (section 3.5) requires a negative mip bias when DLSS is on:
//     bias = log2(render width / output width) - 1.0 + epsilon
// e.g. -2.0 at 50%, -1.415 at 75%. The game creates its samplers with its own
// bias, so while the 3D scene is drawn (from the start of the frame until the
// TAA pass) every pixel-shader sampler the game binds is swapped for a copy with
// the extra bias. Comparison samplers (shadows) are never touched.
// ---------------------------------------------------------------------------
bool mipBiasEnabled = true;
float mipBiasEpsilon = 0.0f;       // user adjustment: + = softer/stabler, - = sharper
std::atomic<float> activeMipBias = 0.0f; // shown in the menu
bool sceneMipBiasActive = false;   // true while the 3D scene of a Super Resolution frame is drawn
float cachedMipBias = 0.0f;        // bias the cached samplers were made with
std::unordered_map<std::string, com_ptr<ID3D11SamplerState>> biasedSamplers; // key: original desc bytes
std::set<ID3D11SamplerState*> ownBiasedSamplers; // identity only
thread_local bool settingOwnSamplers = false;

static void ReleaseBiasedSamplers() {
    biasedSamplers.clear();
    ownBiasedSamplers.clear();
}

static ID3D11SamplerState* GetBiasedSampler(ID3D11SamplerState* original) {
    if (!original || ownBiasedSamplers.count(original)) {
        return nullptr;
    }
    D3D11_SAMPLER_DESC desc;
    original->GetDesc(&desc);
    // Comparison (shadow) and min/max reduction samplers keep their original bias,
    // and samplers that never use mip levels are left alone.
    if ((unsigned)desc.Filter >= 0x80 || desc.MaxLOD <= 0.0f) {
        return nullptr;
    }

    std::string key((const char*)&desc, sizeof(desc));
    auto it = biasedSamplers.find(key);
    if (it != biasedSamplers.end()) {
        return it->second.get();
    }

    D3D11_SAMPLER_DESC biasedDesc = desc;
    float bias = desc.MipLODBias + cachedMipBias;
    biasedDesc.MipLODBias = bias < -16.0f ? -16.0f : (bias > 15.99f ? 15.99f : bias);

    com_ptr<ID3D11Device> device;
    original->GetDevice(&device);
    com_ptr<ID3D11SamplerState> biased;
    if (!device || FAILED(device->CreateSamplerState(&biasedDesc, &biased)) || !biased) {
        return nullptr;
    }
    ownBiasedSamplers.insert(biased.get());
    biasedSamplers[key] = biased;
    return biased.get();
}

// Called after the game binds pixel-shader samplers (ReShade has already applied the game's call).
void OnPushDescriptors(command_list* cmd_list,
    shader_stage stages,
    pipeline_layout layout,
    uint32_t layout_param,
    const descriptor_table_update& update) {
    if (!sceneMipBiasActive || settingOwnSamplers || !IsD3D11(cmd_list) ||
        update.type != descriptor_type::sampler || stages != shader_stage::pixel ||
        update.count == 0 || update.count > D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT ||
        update.binding + update.count > D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT) {
        return;
    }

    ID3D11SamplerState* samplers[D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT];
    const sampler* descriptors = (const sampler*)update.descriptors;
    bool changed = false;
    for (uint32_t i = 0; i < update.count; ++i) {
        samplers[i] = (ID3D11SamplerState*)descriptors[i].handle;
        if (ID3D11SamplerState* biased = GetBiasedSampler(samplers[i])) {
            samplers[i] = biased;
            changed = true;
        }
    }
    if (changed) {
        settingOwnSamplers = true;
        ((ID3D11DeviceContext*)cmd_list->get_native())->PSSetSamplers(update.binding, update.count, samplers);
        settingOwnSamplers = false;
    }
}

// Gives up on Super Resolution for a while (DLAA is used meanwhile).
static void PauseSuperResolution(const char* reason) {
    srRetryCooldown = 600;
    srPending = false;
    srPauseReason = reason;
    SetStatus(reason, 0, 0, 0, 0);
    DiagNote(reason);
}

void Cleanup() {
    ReleaseDLSS();
    ReleaseSuperResolution();
    ReleaseMotionVectors();
    ReleaseBiasedSamplers();
    sceneMipBiasActive = false;
    srDepth.reset();

    prepareMotionVectorShader.reset();

    if (capabilityParameters) {
        NVSDK_NGX_D3D11_DestroyParameters(capabilityParameters);
        capabilityParameters = nullptr;
    }
}

static void drawSettings(reshade::api::effect_runtime*)
{
    if (ImGui::DragFloat("Sharpen Amount", &sharpenMultiplier, 0.1f, 0.0f, 2.0f)) {
        reshade::set_config_value(nullptr, "DLAA", "SharpenAmount", sharpenMultiplier);
    }

    static const NVSDK_NGX_DLSS_Hint_Render_Preset presets[] = {
        NVSDK_NGX_DLSS_Hint_Render_Preset_F,
        NVSDK_NGX_DLSS_Hint_Render_Preset_J,
        NVSDK_NGX_DLSS_Hint_Render_Preset_K,
        NVSDK_NGX_DLSS_Hint_Render_Preset_L,
        NVSDK_NGX_DLSS_Hint_Render_Preset_M
    };
    static const char* presetNames[] = {
        "Preset F(DLSS 3.0)",
        "Preset J(DLSS 4.0 old)",
        "Preset K(DLSS 4.0)",
        "Preset L(DLSS 4.5 - very slow)",
        "Preset M(DLSS 4.5)"
    };
    const char* combo_preview_value;
    switch (preset)
    {
    case NVSDK_NGX_DLSS_Hint_Render_Preset_F:
        combo_preview_value = presetNames[0];
        break;
    case NVSDK_NGX_DLSS_Hint_Render_Preset_J:
        combo_preview_value = presetNames[1];
        break;
    case NVSDK_NGX_DLSS_Hint_Render_Preset_K:
        combo_preview_value = presetNames[2];
        break;
    case NVSDK_NGX_DLSS_Hint_Render_Preset_L:
        combo_preview_value = presetNames[3];
        break;
    case NVSDK_NGX_DLSS_Hint_Render_Preset_M:
        combo_preview_value = presetNames[4];
        break;
    default:
        preset = NVSDK_NGX_DLSS_Hint_Render_Preset_K;
        combo_preview_value = presetNames[2];
        break;
    }
    if (ImGui::BeginCombo("Preset", combo_preview_value, 0))
    {
        for (int i = 0; i < 5; i++) {
            bool isSelected = preset == presets[i];
            if (ImGui::Selectable(presetNames[i], isSelected)) {
                preset = presets[i];
                needReinitialize = true;
                reshade::set_config_value(nullptr, "DLAA", "Preset", preset);
            }
            if (isSelected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    if (ImGui::Checkbox("Enable Auto-exposure", &autoExposure))
    {
        reshade::set_config_value(nullptr, "DLAA", "AutoExposure", autoExposure);
        needReinitialize = true;
    }
    if (ImGui::Checkbox("DLSS Super Resolution when Resolution Scaling < 100%", &superResolutionEnabled))
    {
        reshade::set_config_value(nullptr, "DLAA", "SuperResolution", superResolutionEnabled);
        srRetryCooldown = 0;
    }
    if (ImGui::Checkbox("Full texture detail in Super Resolution (mip bias)", &mipBiasEnabled))
    {
        reshade::set_config_value(nullptr, "DLAA", "TextureDetail", mipBiasEnabled);
    }
    if (mipBiasEnabled) {
        if (ImGui::SliderFloat("Texture detail adjust (+ softer, - sharper)", &mipBiasEpsilon, -1.0f, 1.0f, "%.2f")) {
            reshade::set_config_value(nullptr, "DLAA", "TextureDetailAdjust", mipBiasEpsilon);
        }
        if (activeMipBias.load() != 0.0f) {
            ImGui::Text("Texture mip bias in use: %.2f", activeMipBias.load());
        }
    }
    if (statusInWidth > 0) {
        ImGui::Text("Status: %s (%ux%u -> %ux%u)", dlssStatus.load(),
            statusInWidth.load(), statusInHeight.load(), statusOutWidth.load(), statusOutHeight.load());
    }
    else {
        ImGui::Text("Status: %s", dlssStatus.load());
    }
    if (srRetryCooldown > 0) {
        ImGui::Text("Last Super Resolution problem: %s", srPauseReason.load());
    }

    ImGui::Separator();
    ImGui::TextUnformatted("Frame Generation (in development)");
    bool bridgeSetting = fg::bridgeRequested;
    reshade::get_config_value(nullptr, "DLAA", "FrameGenerationBridge", bridgeSetting);
    if (ImGui::Checkbox("Present through DirectX 12 (needed for Frame Generation, restart the game)", &bridgeSetting)) {
        reshade::set_config_value(nullptr, "DLAA", "FrameGenerationBridge", bridgeSetting);
    }
    ImGui::Text("DirectX 12 bridge: %s", fg::bridgeStatus.load());
    if (bridgeSetting != fg::bridgeRequested) {
        ImGui::TextUnformatted("Restart the game to apply this change.");
    }

    ImGui::Separator();
    ImGui::TextUnformatted("Diagnostic (for development)");
    if (ImGui::Button("Record 1 frame to ReShade.log")) {
        diagRequested = true;
    }
    if (diagRequested || diagCapturing) {
        ImGui::SameLine();
        ImGui::TextUnformatted("recording...");
    }
}

void OnInitDevice(reshade::api::device* device) {
    if (!IsD3D11(device)) {
        return;
    }
    reshade::get_config_value(nullptr, "DLAA", "SharpenAmount", sharpenMultiplier);
    if (sharpenMultiplier < 0.0f || sharpenMultiplier > 2.0f) {
        sharpenMultiplier = 1.0f;
    }
    int presetInt;
    reshade::get_config_value(nullptr, "DLAA", "Preset", presetInt);
    preset = (NVSDK_NGX_DLSS_Hint_Render_Preset)presetInt;
    if (preset != NVSDK_NGX_DLSS_Hint_Render_Preset_F &&
        preset != NVSDK_NGX_DLSS_Hint_Render_Preset_J &&
        preset != NVSDK_NGX_DLSS_Hint_Render_Preset_K &&
        preset != NVSDK_NGX_DLSS_Hint_Render_Preset_L &&
        preset != NVSDK_NGX_DLSS_Hint_Render_Preset_M) {
        preset = NVSDK_NGX_DLSS_Hint_Render_Preset_K;
    }
    reshade::get_config_value(nullptr, "DLAA", "AutoExposure", autoExposure);
    reshade::get_config_value(nullptr, "DLAA", "SuperResolution", superResolutionEnabled);
    reshade::get_config_value(nullptr, "DLAA", "FrameGenerationBridge", fg::bridgeRequested);
    fg::InstallHook((ID3D11Device*)device->get_native());
    reshade::get_config_value(nullptr, "DLAA", "TextureDetail", mipBiasEnabled);
    reshade::get_config_value(nullptr, "DLAA", "TextureDetailAdjust", mipBiasEpsilon);
    if (!(mipBiasEpsilon >= -1.0f && mipBiasEpsilon <= 1.0f)) {
        mipBiasEpsilon = 0.0f;
    }

    NVSDK_NGX_Result result = NVSDK_NGX_D3D11_Init(1,
        L"",
        (ID3D11Device*)device->get_native());

    if (NVSDK_NGX_FAILED(result)) {
        SetStatus("DLSS could not start (NGX init failed)", 0, 0, 0, 0);
        return;
    }

    result = NVSDK_NGX_D3D11_GetCapabilityParameters(&capabilityParameters);
    if (NVSDK_NGX_FAILED(result) || !capabilityParameters) {
        capabilityParameters = nullptr;
        SetStatus("DLSS could not start (no capability parameters)", 0, 0, 0, 0);
        return;
    }

    int needsUpdatedDriver = 0;
    int dlssSupported = 0;
    int featureInitResult = 0;
    if(NVSDK_NGX_FAILED(capabilityParameters->Get(NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsUpdatedDriver)) ||
        needsUpdatedDriver ||
        NVSDK_NGX_FAILED(capabilityParameters->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &dlssSupported)) ||
        !dlssSupported ||
        NVSDK_NGX_FAILED(capabilityParameters->Get(NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &featureInitResult)) ||
        !featureInitResult) {
        NVSDK_NGX_D3D11_DestroyParameters(capabilityParameters);
        capabilityParameters = nullptr;
        return;
    }
    dlssAvailable = true;

    ID3D11Device* d3d11Device = (ID3D11Device*)device->get_native();

    d3d11Device->CreateComputeShader(__PrepareMotionVectors.data(),
        __PrepareMotionVectors.size_bytes(),
        nullptr,
        &prepareMotionVectorShader);

    {
        D3D11_BUFFER_DESC bufferDesc;
        bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        bufferDesc.ByteWidth = 16;
        bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        bufferDesc.MiscFlags = 0;
        bufferDesc.StructureByteStride = 0;
        bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
        HRESULT hr = d3d11Device->CreateBuffer(&bufferDesc, nullptr, &cbSharpenModify);
    }
}

void OnDestroyDevice(reshade::api::device* device) {
    if (!IsD3D11(device)) {
        return;
    }
    Cleanup();
}

bool OnCreatePipeline(
    reshade::api::device* device,
    reshade::api::pipeline_layout layout,
    uint32_t subobjectCount,
    const reshade::api::pipeline_subobject* subobjects) {
    bool replacedShader = false;
    if (!IsD3D11(device)) {
        return false;
    }

    for (uint32_t i = 0; i < subobjectCount; ++i) {
        if (subobjects[i].type != reshade::api::pipeline_subobject_type::pixel_shader) {
            continue;
        }

        shader_desc* desc = (shader_desc*)subobjects[i].data;
        shaderHash = compute_crc32((const uint8_t*)desc->code, desc->code_size);

        if (shaderHash == 0x0D1CD1AA) {
            desc->code = __0x0D1CD1AA.data();
            desc->code_size = __0x0D1CD1AA.size_bytes();
            replacedShader = true;
        }
    }

    return replacedShader;
}

void OnInitPipeline(device* device,
    pipeline_layout layout,
    uint32_t subobjectCount,
    const pipeline_subobject* subobjects,
    pipeline pipeline) {
    if (!IsD3D11(device)) {
        return;
    }
    for (uint32_t i = 0; i < subobjectCount; ++i) {
        if (subobjects[i].type != reshade::api::pipeline_subobject_type::pixel_shader) {
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(diagMutex);
            pixelShaderHashes[pipeline.handle] = shaderHash;
        }
        if (shaderHash == 0x0DF0A97D) {
            taaShaders.insert((ID3D11PixelShader*)pipeline.handle);
        }
        if (shaderHash == 0x0D1CD1AA) {
            sharpenShaders.insert((ID3D11PixelShader*)pipeline.handle);
        }
        if (shaderHash == kUpscaleShaderHash) {
            upscaleShaders.insert((ID3D11PixelShader*)pipeline.handle);
        }
    }
}

void OnDestroyPipeline(reshade::api::device* device, reshade::api::pipeline pipeline) {
    if (!IsD3D11(device)) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(diagMutex);
        pixelShaderHashes.erase(pipeline.handle);
    }
    if (taaShaders.find((ID3D11PixelShader*)pipeline.handle) != taaShaders.end()) {
        taaShaders.erase((ID3D11PixelShader*)pipeline.handle);
    }
    if (sharpenShaders.find((ID3D11PixelShader*)pipeline.handle) != sharpenShaders.end()) {
        sharpenShaders.erase((ID3D11PixelShader*)pipeline.handle);
    }
    upscaleShaders.erase((ID3D11PixelShader*)pipeline.handle);
}

// Replaces the game's final stretch (render resolution -> output resolution) with DLSS.
// Returns true when the game's draw must be skipped.
static bool HandleUpscalePass(ID3D11DeviceContext* deviceContext) {
    com_ptr<ID3D11ShaderResourceView> sourceSRV;
    deviceContext->PSGetShaderResources(0, 1, &sourceSRV);
    com_ptr<ID3D11RenderTargetView> targetRTV;
    deviceContext->OMGetRenderTargets(1, &targetRTV, nullptr);
    if (!sourceSRV || !targetRTV) {
        return false;
    }

    com_ptr<ID3D11Resource> sourceResource;
    sourceSRV->GetResource(&sourceResource);
    com_ptr<ID3D11Resource> targetResource;
    targetRTV->GetResource(&targetResource);
    com_ptr<ID3D11Texture2D> sourceTexture;
    com_ptr<ID3D11Texture2D> targetTexture;
    if (!sourceResource || !targetResource ||
        FAILED(sourceResource->QueryInterface(&sourceTexture)) || !sourceTexture ||
        FAILED(targetResource->QueryInterface(&targetTexture)) || !targetTexture) {
        return false;
    }
    D3D11_TEXTURE2D_DESC sourceDesc;
    sourceTexture->GetDesc(&sourceDesc);
    D3D11_TEXTURE2D_DESC targetDesc;
    targetTexture->GetDesc(&targetDesc);

    // Only the stretch of the finished low-resolution image into a bigger target.
    if (sourceDesc.Width != srRenderWidth || sourceDesc.Height != srRenderHeight ||
        targetDesc.Width <= srRenderWidth || targetDesc.Height <= srRenderHeight ||
        targetDesc.SampleDesc.Count != 1) {
        return false;
    }

    // The stretch can be issued as more than one draw; DLSS already covered the whole image.
    if (srEvaluatedThisFrame) {
        return true;
    }

    if (!EnsureSuperResolution(deviceContext, srRenderWidth, srRenderHeight, targetDesc.Width, targetDesc.Height, targetDesc.Format)) {
        PauseSuperResolution("Super Resolution could not start, using DLAA");
        return false;
    }

    NVSDK_NGX_D3D11_DLSS_Eval_Params evalParams = {};
    evalParams.Feature.pInColor = sourceResource.get();
    evalParams.Feature.pInOutput = srOutputTexture.get();
    evalParams.pInDepth = srDepth.get();
    evalParams.pInMotionVectors = motionVectorTexture.get();
    evalParams.InJitterOffsetX = srJitter[0] * srRenderWidth;
    evalParams.InJitterOffsetY = srJitter[1] * srRenderHeight;
    evalParams.InReset = srNeedReset ? 1 : 0;
    evalParams.InMVScaleX = 1.0f;
    evalParams.InMVScaleY = 1.0f;
    evalParams.InRenderSubrectDimensions.Width = srRenderWidth;
    evalParams.InRenderSubrectDimensions.Height = srRenderHeight;
    NVSDK_NGX_Result result = NGX_D3D11_EVALUATE_DLSS_EXT(deviceContext, srHandle, srParameters, &evalParams);
    if (NVSDK_NGX_FAILED(result)) {
        PauseSuperResolution("Super Resolution failed to run, using DLAA");
        return false;
    }

    deviceContext->CopySubresourceRegion(targetResource.get(), 0, 0, 0, 0, srOutputTexture.get(), 0, nullptr);

    srEvaluatedThisFrame = true;
    SetStatus("DLSS Super Resolution", srRenderWidth, srRenderHeight, targetDesc.Width, targetDesc.Height);
    DiagNote("Super Resolution: DLSS replaced the game's stretch pass");
    return true;
}

bool OnDraw(reshade::api::command_list* cmd_list,
    uint32_t vertex_count,
    uint32_t instance_count,
    uint32_t first_vertex,
    uint32_t first_instance) {
    if (!IsD3D11(cmd_list)) {
        return false;
    }
    ID3D11DeviceContext* deviceContext = (ID3D11DeviceContext*)(cmd_list->get_native());
    DiagOnGpuWork(deviceContext, false);
    com_ptr<ID3D11PixelShader> shader;
    deviceContext->PSGetShader(&shader, nullptr, nullptr);

    if (dlssAvailable && taaShaders.find(shader.get()) != taaShaders.end()) {
        // The 3D scene is finished: stop adding texture mip bias for the rest of the frame.
        if (diagCapturing) {
            char note[160];
            snprintf(note, sizeof(note), "Texture mip bias during the 3D scene: %s, bias=%.2f, biased sampler copies=%zu",
                sceneMipBiasActive ? "on" : "off", cachedMipBias, biasedSamplers.size());
            DiagNote(note);
        }
        sceneMipBiasActive = false;

        com_ptr<ID3D11RenderTargetView> renderTargetView;
        deviceContext->OMGetRenderTargets(1, &renderTargetView, nullptr);

        if (!renderTargetView) {
            return false;
        }

        com_ptr<ID3D11Resource> renderTargetResource;
        renderTargetView->GetResource(&renderTargetResource);

        com_ptr<ID3D11Texture2D> renderTargetTexture;
        renderTargetResource->QueryInterface(&renderTargetTexture);
        if (!renderTargetTexture) {
            return false;
        }

        D3D11_TEXTURE2D_DESC renderTargetDesc;
        renderTargetTexture->GetDesc(&renderTargetDesc);

        uint32_t width = renderTargetDesc.Width;
        uint32_t height = renderTargetDesc.Height;

        if (needReinitialize) {
            ReleaseDLSS();
            ReleaseSuperResolution();
            needReinitialize = false;
        }

        com_ptr<ID3D11Device> device;
        deviceContext->GetDevice(&device);
        if (!EnsureMotionVectorTexture(device.get(), width, height)) {
            return false;
        }

        com_ptr<ID3D11Buffer> cbTemporalAA;
        deviceContext->PSGetConstantBuffers(0, 1, &cbTemporalAA);
        DiagOnTAA(width, height);

        com_ptr<ID3D11ShaderResourceView> inColorSRV;
        deviceContext->PSGetShaderResources(0, 1, &inColorSRV);

        com_ptr<ID3D11ShaderResourceView> inDepthSRV;
        deviceContext->PSGetShaderResources(3, 1, &inDepthSRV);

        com_ptr<ID3D11ShaderResourceView> inVelocitySRV;
        deviceContext->PSGetShaderResources(6, 1, &inVelocitySRV);

        if (!cbTemporalAA ||
            !inColorSRV ||
            !inDepthSRV ||
            !inVelocitySRV) {
            return false;
        }

        com_ptr<ID3D11Resource> inColor;
        inColorSRV->GetResource(&inColor);
        com_ptr<ID3D11Resource> inDepth;
        inDepthSRV->GetResource(&inDepth);

        {
            ID3D11ShaderResourceView* srvs[] = { inVelocitySRV.get() , inDepthSRV.get() };
            ID3D11UnorderedAccessView* uavs[] = { motionVectorUAV.get() };
            ID3D11Buffer* cbs[] = { cbTemporalAA.get() };
            deviceContext->CSSetShader(prepareMotionVectorShader.get(), 0, 0);
            deviceContext->CSSetShaderResources(0, 2, srvs);
            deviceContext->CSSetConstantBuffers(0, 1, cbs);
            deviceContext->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
            deviceContext->Dispatch((width + 7) / 8, (height + 7) / 8, 1);

            srvs[0] = nullptr;
            srvs[1] = nullptr;
            deviceContext->CSSetShaderResources(0, 2, srvs);
            uavs[0] = nullptr;
            deviceContext->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
        }

        // Super Resolution: the game renders below the output size, so DLSS runs later,
        // in place of the game's final stretch. Here we only pass the raw image through.
        uint32_t outW = outputWidth;
        uint32_t outH = outputHeight;
        if (superResolutionEnabled && srRetryCooldown == 0 && outW > width && outH > height) {
            com_ptr<ID3D11Texture2D> inColorTexture;
            inColor->QueryInterface(&inColorTexture);
            D3D11_TEXTURE2D_DESC inColorDesc = {};
            if (inColorTexture) {
                inColorTexture->GetDesc(&inColorDesc);
            }
            if (inColorTexture &&
                inColorDesc.Width == width && inColorDesc.Height == height &&
                inColorDesc.Format == renderTargetDesc.Format &&
                inColorDesc.SampleDesc.Count == 1 && renderTargetDesc.SampleDesc.Count == 1) {
                deviceContext->CopySubresourceRegion(renderTargetResource.get(), 0, 0, 0, 0, inColor.get(), 0, nullptr);
                srDepth = inDepth;
                srRenderWidth = width;
                srRenderHeight = height;
                srJitter[0] = jitter[0];
                srJitter[1] = jitter[1];
                srPending = true;
                DiagNote("Super Resolution: TAA skipped, raw image passed through");
                return true;
            }
            PauseSuperResolution("Super Resolution not possible here (TAA formats differ), using DLAA");
        }

        // DLAA: replaces the game's TAA at the same resolution (original behaviour of the mod).
        if (currentWidth != width ||
            currentHeight != height) {
            ReleaseDLSS();
        }

        if (!dlssHandle) {
            NVSDK_NGX_D3D11_AllocateParameters(&parameters);
            NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_Width, width);
            NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_Height, height);
            NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_OutWidth, width);
            NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_OutHeight, height);
            NVSDK_NGX_Parameter_SetI(parameters, NVSDK_NGX_Parameter_PerfQualityValue, NVSDK_NGX_PerfQuality_Value_DLAA);
            NVSDK_NGX_Parameter_SetI(parameters, NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, NVSDK_NGX_DLSS_Feature_Flags_IsHDR | (autoExposure ? NVSDK_NGX_DLSS_Feature_Flags_AutoExposure : 0));
            NVSDK_NGX_Parameter_SetI(parameters, NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects, 0);
            NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, preset);
            NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality, preset);
            NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, preset);
            NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance, preset);
            NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance, preset);
            NVSDK_NGX_Result result = NVSDK_NGX_D3D11_CreateFeature(deviceContext, NVSDK_NGX_Feature_SuperSampling, parameters, &dlssHandle);
            if (NVSDK_NGX_FAILED(result)) {
                dlssHandle = nullptr;
                dlssAvailable = false;
                SetStatus("DLSS could not start (create feature failed)", 0, 0, 0, 0);
                return false;
            }

            currentWidth = width;
            currentHeight = height;
        }

        {
            NVSDK_NGX_D3D11_DLSS_Eval_Params evalParams = {};
            evalParams.Feature.pInColor = inColor.get();
            evalParams.Feature.pInOutput = renderTargetResource.get();
            evalParams.pInDepth = inDepth.get();
            evalParams.pInMotionVectors = motionVectorTexture.get();
            evalParams.InJitterOffsetX = jitter[0] * width;
            evalParams.InJitterOffsetY = jitter[1] * height;
            evalParams.InReset = needReset ? 1 : 0;
            evalParams.InMVScaleX = 1.0f;
            evalParams.InMVScaleY = 1.0f;
            evalParams.InRenderSubrectDimensions.Width = width;
            evalParams.InRenderSubrectDimensions.Height = height;
            NVSDK_NGX_Result res = NGX_D3D11_EVALUATE_DLSS_EXT(deviceContext, dlssHandle, parameters, &evalParams);
        }

        invokedThisFrame = true;
        SetStatus("DLAA", width, height, width, height);
        return true;
    }
    else if (srPending && upscaleShaders.find(shader.get()) != upscaleShaders.end()) {
        return HandleUpscalePass(deviceContext);
    }
    else if (sharpenShaders.find(shader.get()) != sharpenShaders.end()) {
        {
            D3D11_MAPPED_SUBRESOURCE mappedResource;
            if (SUCCEEDED(deviceContext->Map(cbSharpenModify.get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource))) {
                // In Super Resolution mode the image is not anti-aliased yet at this point:
                // sharpening it would only amplify aliasing, so the game's sharpen is turned off.
                ((float*)mappedResource.pData)[0] = srPending ? 0.0f : sharpenMultiplier;
                deviceContext->Unmap(cbSharpenModify.get(), 0);
            }
        }

        ID3D11Buffer* cbs[] = { cbSharpenModify.get() };
        deviceContext->PSSetConstantBuffers(1, 1, cbs);
    }
    return false;
}

void OnMapBufferRegion(
    device* device,
    resource resource,
    uint64_t offset,
    uint64_t size,
    map_access access,
    void** mapped_data) {
    if (access != map_access::write_discard || !IsD3D11(device)) {
        return;
    }
    D3D11_BUFFER_DESC bd;
    ((ID3D11Buffer*)resource.handle)->GetDesc(&bd);
    if (bd.ByteWidth == 256) {
        mappedConstantBuffer = *mapped_data;
    }
}

void OnUnmapBufferRegion(
    device* device,
    resource resource) {
    if (!IsD3D11(device)) {
        return;
    }
    if (mappedConstantBuffer) {
        jitter[0] = ((float*)mappedConstantBuffer)[8];
        jitter[1] = ((float*)mappedConstantBuffer)[9];
        mappedConstantBuffer = nullptr;
    }
}

void OnPresent(command_queue* queue,
    swapchain* swapchain,
    const rect* source_rect,
    const rect* dest_rect,
    uint32_t dirty_rect_count,
    const rect* dirty_rects) {
    // Only once per frame rendered by the game (later, Frame Generation adds extra presents).
    if (swapchain->get_device()->get_api() == device_api::d3d12 && !fg::inGameFramePresent) {
        return;
    }
    needReset = !invokedThisFrame;
    invokedThisFrame = false;

    // TAA was skipped for Super Resolution but the game's stretch pass never came:
    // this scene does not use it, so fall back to DLAA for a while.
    if (srPending && !srEvaluatedThisFrame) {
        PauseSuperResolution("Super Resolution: stretch pass not found in this scene, using DLAA");
    }
    srNeedReset = !srEvaluatedThisFrame;

    // Texture detail for the next frame's 3D scene (only while Super Resolution is running).
    if (mipBiasEnabled && srEvaluatedThisFrame && srRenderWidth > 0 && srOutWidth > 0) {
        float bias = log2f((float)srRenderWidth / (float)srOutWidth) - 1.0f + mipBiasEpsilon;
        if (fabsf(bias - cachedMipBias) > 0.001f) {
            ReleaseBiasedSamplers();
            cachedMipBias = bias;
        }
        activeMipBias = bias;
        sceneMipBiasActive = true;
    }
    else {
        activeMipBias = 0.0f;
        sceneMipBiasActive = false;
    }

    srPending = false;
    srEvaluatedThisFrame = false;
    srDepth.reset();
    if (srRetryCooldown > 0) {
        srRetryCooldown--;
    }

    // Output size = back buffer size.
    resource backBuffer = swapchain->get_current_back_buffer();
    if (backBuffer.handle != 0) {
        resource_desc backBufferDesc = swapchain->get_device()->get_resource_desc(backBuffer);
        outputWidth = backBufferDesc.texture.width;
        outputHeight = backBufferDesc.texture.height;
    }

    DiagOnPresent(swapchain);
}

bool OnDrawIndexed(reshade::api::command_list* cmd_list,
    uint32_t index_count,
    uint32_t instance_count,
    uint32_t first_index,
    int32_t vertex_offset,
    uint32_t first_instance) {
    if (!IsD3D11(cmd_list)) {
        return false;
    }
    ID3D11DeviceContext* deviceContext = (ID3D11DeviceContext*)(cmd_list->get_native());
    DiagOnGpuWork(deviceContext, false);
    if (srPending) {
        com_ptr<ID3D11PixelShader> shader;
        deviceContext->PSGetShader(&shader, nullptr, nullptr);
        if (upscaleShaders.find(shader.get()) != upscaleShaders.end()) {
            return HandleUpscalePass(deviceContext);
        }
    }
    return false;
}

bool OnDispatch(reshade::api::command_list* cmd_list,
    uint32_t group_count_x,
    uint32_t group_count_y,
    uint32_t group_count_z) {
    if (!IsD3D11(cmd_list)) {
        return false;
    }
    DiagOnGpuWork((ID3D11DeviceContext*)(cmd_list->get_native()), true);
    return false;
}

extern "C" __declspec(dllexport) const char *NAME = "FFXV DLAA";
extern "C" __declspec(dllexport) const char *DESCRIPTION = "Replaces TAA with DLAA";

BOOL APIENTRY DllMain(HMODULE hModule, DWORD fdwReason, LPVOID)
{
	switch (fdwReason)
	{
	case DLL_PROCESS_ATTACH:
        addonModule = hModule;
		if (!reshade::register_addon(hModule))
			return FALSE;
        reshade::register_overlay(nullptr, drawSettings);
        reshade::register_event<reshade::addon_event::init_device>(OnInitDevice);
        reshade::register_event<reshade::addon_event::destroy_device>(OnDestroyDevice);
        reshade::register_event < reshade::addon_event::create_pipeline>(OnCreatePipeline);
        reshade::register_event<reshade::addon_event::init_pipeline>(OnInitPipeline);
        reshade::register_event<reshade::addon_event::destroy_pipeline>(OnDestroyPipeline);
        reshade::register_event<reshade::addon_event::draw>(OnDraw);
        reshade::register_event<reshade::addon_event::map_buffer_region>(OnMapBufferRegion);
        reshade::register_event<reshade::addon_event::unmap_buffer_region>(OnUnmapBufferRegion);
        reshade::register_event<reshade::addon_event::present>(OnPresent);
        reshade::register_event<reshade::addon_event::draw_indexed>(OnDrawIndexed);
        reshade::register_event<reshade::addon_event::dispatch>(OnDispatch);
        reshade::register_event<reshade::addon_event::push_descriptors>(OnPushDescriptors);
		break;
	case DLL_PROCESS_DETACH:
		fg::UninstallHook();
		reshade::unregister_addon(hModule);
		break;
	}

	return TRUE;
}
