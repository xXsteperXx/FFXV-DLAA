#pragma once
// ---------------------------------------------------------------------------
// DX11 -> DX12 presentation bridge + DLSS Frame Generation.
//
// DLSS Frame Generation only runs on DirectX 12, but FFXV is a DirectX 11 game.
// When the bridge is enabled, the game's swap chain creation is intercepted:
//   - the real swap chain is created on a DirectX 12 queue (through ReShade,
//     so ReShade still draws its menu and effects on it),
//   - the game receives a stand-in swap chain whose back buffer is a texture
//     shared between DirectX 11 and DirectX 12,
//   - on every Present, the game's finished frame is handed to DirectX 12.
//
// Without Frame Generation the frame is simply copied and presented.
// With Frame Generation, the add-on (DirectX 11 side) also shares the depth,
// the motion vectors, the image without HUD and the camera of each frame;
// DLSS-FG then creates the in-between frame, and a separate thread presents the
// generated frame followed by the real frame half a frame later (pacing).
//
// Synchronization uses two shared fences:
//   fence11to12: signaled by DirectX 11 when a frame is finished, waited by DX12
//   fence12:     signaled by DirectX 12 when it is done reading/copying,
//                waited by DirectX 11 (before it overwrites shared textures) and the CPU
//
// Must be included after <reshade.hpp>, <com_ptr.hpp>, <d3d11.h> and the NGX headers.
// ---------------------------------------------------------------------------
#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <nvsdk_ngx_helpers_dlssg.h>
#include "reflex.hpp"
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace fg {

// ---------------------------------------------------------------------------
// Settings and status (shown in the add-on menu)
// ---------------------------------------------------------------------------
inline bool bridgeRequested = false;           // config [DLAA] FrameGenerationBridge, read at startup
inline std::atomic<bool> bridgeActive = false; // a bridged swap chain currently exists
inline std::atomic<const char*> bridgeStatus = "off";

inline std::atomic<bool> frameGenerationEnabled = false; // config [DLAA] FrameGeneration
inline std::atomic<const char*> frameGenerationStatus = "off";
inline std::atomic<bool> frameGenerationRunning = false;
inline std::atomic<bool> frameGenerationRetry = false;  // set by the menu when Frame Generation is switched on
inline std::atomic<const char*> frameGenerationProblem = ""; // last problem preparing the inputs (kept until fixed)

// Latency settings (menu)
inline std::atomic<bool> reflexEnabled = true;   // NVIDIA Reflex low latency mode on the game's device
inline std::atomic<bool> reflexBoost = false;    // Reflex boost (GPU clocks kept high)
inline std::atomic<int> framesAhead = 1;         // frames the game may prepare ahead of the GPU (0..2)

// Statistics (menu)
inline std::atomic<uint64_t> statGenerated = 0;     // frames with a generated frame
inline std::atomic<uint64_t> statNotGenerated = 0;  // frames presented without one while Frame Generation is on
inline std::atomic<uint64_t> statCameraReused = 0;  // frames that reused the previous camera
inline std::atomic<float> statLongestFrameMs = 0.0f; // longest time spent in Present (last ~2 s)
inline std::atomic<const char*> statLastSkipReason = "-";
inline std::atomic<uint32_t> multiFrameCountMax = 0;    // reported by the driver (1 = only 2x)
inline std::atomic<bool> transposeMatrices = true;      // debug: matrix layout handed to DLSS-FG
inline std::atomic<int> depthInvertedMode = -1;         // debug: -1 automatic, 0 no, 1 yes
inline std::atomic<bool> lastDepthInverted = false;
inline std::atomic<bool> lastCameraFromGame = false;
inline std::atomic<bool> depthRangeFromGame = false;

// Called on the game's render thread at the end of every frame the game rendered
// (replaces ReShade's present event while the bridge is active).
inline void (*onGameFrameEnd)(uint64_t backBuffer, uint32_t width, uint32_t height) = nullptr;
inline std::atomic<ID3D11Texture2D*> backBuffer11 = nullptr; // identity of the game's back buffer

// ---------------------------------------------------------------------------
// Per-frame inputs for Frame Generation, filled by the add-on on the D3D11 side
// (render thread) and consumed by Present on the same thread.
// ---------------------------------------------------------------------------
struct FrameInputs {
    bool valid = false;        // depth, motion vectors and camera are from this frame
    bool hudlessValid = false; // the image without HUD was captured this frame
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    float clipToPrevClip[4][4] = {};
    float prevClipToClip[4][4] = {};
    float viewToClip[4][4] = {};
    float clipToView[4][4] = {};
    float nearPlane = 0.1f;
    float farPlane = 10000.0f;
    float fov = 1.0f;
    float aspect = 1.777f;
    bool depthInverted = false;
};
inline FrameInputs frameInputs;

// ---------------------------------------------------------------------------
// Devices, queue, fences
// ---------------------------------------------------------------------------
inline com_ptr<ID3D11Device> device11;         // native D3D11 device of the game
inline com_ptr<ID3D11Device5> device11_5;
inline com_ptr<ID3D11DeviceContext4> context11;
inline com_ptr<ID3D12Device> device12;         // created through ReShade (proxy)
inline com_ptr<ID3D12CommandQueue> queue12;
inline com_ptr<ID3D12Fence> fence11to12;
inline com_ptr<ID3D11Fence> fence11to12_11;
inline uint64_t value11to12 = 0;               // render thread only
inline com_ptr<ID3D12Fence> fence12;
inline com_ptr<ID3D11Fence> fence12_11;
inline uint64_t value12 = 0;                   // guarded by queueMutex
inline std::mutex queueMutex;                  // all DX12 queue submissions and presents
inline bool tearingSupported = false;
inline bool dx12InitTried = false;

typedef HRESULT(STDMETHODCALLTYPE* CreateSwapChainFn)(IDXGIFactory*, IUnknown*, DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain**);
inline CreateSwapChainFn previousCreateSwapChain = nullptr;
inline void** hookedVtableSlot = nullptr;

inline void Log(reshade::log::level level, const char* format, ...) {
    char buffer[600];
    int offset = snprintf(buffer, sizeof(buffer), "[FG bridge] ");
    va_list args;
    va_start(args, format);
    vsnprintf(buffer + offset, sizeof(buffer) - offset, format, args);
    va_end(args);
    reshade::log::message(level, buffer);
}

// CPU wait until DirectX 12 reached 'value' on fence12.
inline bool WaitForFence12(uint64_t value, DWORD timeoutMs = 2000) {
    if (!fence12 || fence12->GetCompletedValue() >= value) {
        return true;
    }
    thread_local HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (FAILED(fence12->SetEventOnCompletion(value, event))) {
        return false;
    }
    if (WaitForSingleObject(event, timeoutMs) != WAIT_OBJECT_0) {
        Log(reshade::log::level::warning, "GPU wait timed out (fence %llu).", (unsigned long long)value);
        return false;
    }
    return true;
}

// Must be called with queueMutex held. Returns the value signaled.
inline uint64_t SignalQueueLocked() {
    uint64_t value = ++value12;
    queue12->Signal(fence12.get(), value);
    return value;
}

// DirectX 11 marks "everything submitted so far is finished"; returns the value.
// Render thread only.
inline uint64_t SignalFromD3D11() {
    uint64_t value = ++value11to12;
    context11->Signal(fence11to12_11.get(), value);
    context11->Flush();
    return value;
}

// Waits until both DirectX 11 and DirectX 12 finished all submitted work. Render thread only.
inline void FlushGpu() {
    if (!context11 || !queue12 || !fence11to12_11 || !fence12) {
        return;
    }
    uint64_t d3d11Done = SignalFromD3D11();
    uint64_t allDone;
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        queue12->Wait(fence11to12.get(), d3d11Done);
        allDone = SignalQueueLocked();
    }
    WaitForFence12(allDone, 5000);
}

inline bool CreateSharedFence(com_ptr<ID3D12Fence>& fence, com_ptr<ID3D11Fence>& fence11) {
    HANDLE sharedFence = nullptr;
    if (FAILED(device12->CreateFence(0, D3D12_FENCE_FLAG_SHARED, __uuidof(ID3D12Fence), (void**)&fence)) ||
        FAILED(device12->CreateSharedHandle(fence.get(), nullptr, GENERIC_ALL, nullptr, &sharedFence)) ||
        FAILED(device11_5->OpenSharedFence(sharedFence, __uuidof(ID3D11Fence), (void**)&fence11)) || !fence11) {
        if (sharedFence) {
            CloseHandle(sharedFence);
        }
        fence11.reset();
        fence.reset();
        return false;
    }
    CloseHandle(sharedFence);
    return true;
}

// Creates the D3D12 device/queue on the same GPU as the game's D3D11 device.
// Called lazily from the swap chain hook (outside ReShade's own device creation).
inline bool InitDX12() {
    if (device12) {
        return true;
    }
    if (dx12InitTried) {
        return false;
    }
    dx12InitTried = true;

    if (!device11) {
        bridgeStatus = "failed: D3D11 device not known yet";
        return false;
    }
    if (FAILED(device11->QueryInterface(&device11_5)) || !device11_5) {
        bridgeStatus = "failed: needs Direct3D 11.4 (Windows 10)";
        return false;
    }
    {
        com_ptr<ID3D11DeviceContext> immediate;
        device11->GetImmediateContext(&immediate);
        if (!immediate || FAILED(immediate->QueryInterface(&context11)) || !context11) {
            bridgeStatus = "failed: needs Direct3D 11.4 context";
            return false;
        }
    }

    // Find the DXGI adapter the game uses.
    LUID luid = {};
    {
        com_ptr<IDXGIDevice> dxgiDevice;
        com_ptr<IDXGIAdapter> adapter;
        DXGI_ADAPTER_DESC adapterDesc = {};
        if (FAILED(device11->QueryInterface(&dxgiDevice)) || FAILED(dxgiDevice->GetAdapter(&adapter)) || FAILED(adapter->GetDesc(&adapterDesc))) {
            bridgeStatus = "failed: could not identify the GPU";
            return false;
        }
        luid = adapterDesc.AdapterLuid;
    }

    typedef HRESULT(WINAPI* CreateFactory1Fn)(REFIID, void**);
    HMODULE dxgiModule = GetModuleHandleW(L"dxgi.dll");
    CreateFactory1Fn createFactory = dxgiModule ? (CreateFactory1Fn)GetProcAddress(dxgiModule, "CreateDXGIFactory1") : nullptr;
    com_ptr<IDXGIFactory4> factory;
    if (!createFactory || FAILED(createFactory(__uuidof(IDXGIFactory4), (void**)&factory)) || !factory) {
        bridgeStatus = "failed: no DXGI 1.4 factory";
        return false;
    }
    com_ptr<IDXGIAdapter1> adapter;
    if (FAILED(factory->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter1), (void**)&adapter)) || !adapter) {
        bridgeStatus = "failed: GPU not found by DXGI";
        return false;
    }
    {
        com_ptr<IDXGIFactory5> factory5;
        BOOL allowTearing = FALSE;
        if (SUCCEEDED(factory->QueryInterface(&factory5)) && factory5 &&
            SUCCEEDED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing, sizeof(allowTearing)))) {
            tearingSupported = allowTearing != FALSE;
        }
    }

    HMODULE d3d12Module = LoadLibraryW(L"d3d12.dll");
    PFN_D3D12_CREATE_DEVICE createDevice = d3d12Module ? (PFN_D3D12_CREATE_DEVICE)GetProcAddress(d3d12Module, "D3D12CreateDevice") : nullptr;
    if (!createDevice || FAILED(createDevice(adapter.get(), D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), (void**)&device12)) || !device12) {
        device12.reset();
        bridgeStatus = "failed: could not create the DirectX 12 device";
        return false;
    }

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(device12->CreateCommandQueue(&queueDesc, __uuidof(ID3D12CommandQueue), (void**)&queue12)) || !queue12) {
        device12.reset();
        bridgeStatus = "failed: could not create the DirectX 12 queue";
        return false;
    }

    if (!CreateSharedFence(fence11to12, fence11to12_11) || !CreateSharedFence(fence12, fence12_11)) {
        fence11to12.reset();
        fence11to12_11.reset();
        fence12.reset();
        fence12_11.reset();
        queue12.reset();
        device12.reset();
        bridgeStatus = "failed: could not share fences between DX11 and DX12";
        return false;
    }

    Log(reshade::log::level::info, "DirectX 12 device created (tearing %s).", tearingSupported ? "supported" : "not supported");
    return true;
}

// Flip-model swap chains do not accept sRGB formats; the shared texture keeps the game's format.
inline DXGI_FORMAT FlipCompatibleFormat(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
        return format;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

inline bool CreateTexture12(uint32_t width, uint32_t height, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags,
    D3D12_HEAP_FLAGS heapFlags, D3D12_RESOURCE_STATES state, com_ptr<ID3D12Resource>& out) {
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = flags;
    return SUCCEEDED(device12->CreateCommittedResource(&heap, heapFlags, &desc, state, nullptr, __uuidof(ID3D12Resource), (void**)&out)) && out;
}

// A texture that DirectX 11 writes and DirectX 12 reads.
struct SharedTexture {
    com_ptr<ID3D12Resource> resource12;
    com_ptr<ID3D11Texture2D> texture11;
    uint32_t width = 0;
    uint32_t height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;

    void Reset() {
        resource12.reset();
        texture11.reset();
        width = 0;
        height = 0;
        format = DXGI_FORMAT_UNKNOWN;
    }
    bool Matches(uint32_t w, uint32_t h, DXGI_FORMAT f) const {
        return texture11 && resource12 && width == w && height == h && format == f;
    }
    bool Create(const char* name, uint32_t w, uint32_t h, DXGI_FORMAT f, D3D12_RESOURCE_FLAGS flags) {
        // First the same way as the shared back buffer (known to work), then without
        // simultaneous access as a fallback.
        if (TryCreate(name, w, h, f, flags | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS)) {
            return true;
        }
        return TryCreate(name, w, h, f, flags);
    }

private:
    bool TryCreate(const char* name, uint32_t w, uint32_t h, DXGI_FORMAT f, D3D12_RESOURCE_FLAGS flags) {
        Reset();
        D3D12_HEAP_PROPERTIES heap = {};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = w;
        desc.Height = h;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = f;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        desc.Flags = flags;
        HRESULT hr = device12->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
            __uuidof(ID3D12Resource), (void**)&resource12);
        if (FAILED(hr) || !resource12) {
            Log(reshade::log::level::warning, "Shared texture '%s' (%ux%u format %u flags 0x%X): DirectX 12 creation failed (0x%08X).",
                name, w, h, (unsigned)f, (unsigned)flags, (unsigned)hr);
            Reset();
            return false;
        }
        HANDLE sharedHandle = nullptr;
        hr = device12->CreateSharedHandle(resource12.get(), nullptr, GENERIC_ALL, nullptr, &sharedHandle);
        if (FAILED(hr)) {
            Log(reshade::log::level::warning, "Shared texture '%s': CreateSharedHandle failed (0x%08X).", name, (unsigned)hr);
            Reset();
            return false;
        }
        hr = device11_5->OpenSharedResource1(sharedHandle, __uuidof(ID3D11Texture2D), (void**)&texture11);
        CloseHandle(sharedHandle);
        if (FAILED(hr) || !texture11) {
            Log(reshade::log::level::warning, "Shared texture '%s' (flags 0x%X): DirectX 11 could not open it (0x%08X).", name, (unsigned)flags, (unsigned)hr);
            Reset();
            return false;
        }
        width = w;
        height = h;
        format = f;
        Log(reshade::log::level::info, "Shared texture '%s' created (%ux%u format %u flags 0x%X).", name, w, h, (unsigned)f, (unsigned)flags);
        return true;
    }
};

// Frame Generation inputs shared with DirectX 12 (created on demand by the add-on).
inline SharedTexture depthShared;   // R32_FLOAT, render resolution, written by the motion vector shader
inline SharedTexture motionShared;  // R16G16_FLOAT, render resolution, copied from the DLSS motion vectors
inline SharedTexture hudlessShared; // back buffer format, output resolution
inline com_ptr<ID3D11UnorderedAccessView> depthUAV11;
inline com_ptr<ID3D11RenderTargetView> hudlessRTV11;

// True when the add-on should prepare Frame Generation inputs this frame.
inline bool FrameGenerationWanted() {
    return bridgeActive && frameGenerationEnabled && device12;
}

// Depth and motion vector textures at the render resolution. Render thread only.
inline bool EnsureFrameInputs(uint32_t renderWidth, uint32_t renderHeight) {
    if (depthShared.Matches(renderWidth, renderHeight, DXGI_FORMAT_R32_FLOAT) &&
        motionShared.Matches(renderWidth, renderHeight, DXGI_FORMAT_R16G16_FLOAT) && depthUAV11) {
        return true;
    }
    // Do not retry every frame after a failure (it would flush the GPU each time).
    static uint32_t failedWidth = 0;
    static uint32_t failedHeight = 0;
    if (failedWidth == renderWidth && failedHeight == renderHeight) {
        return false;
    }
    FlushGpu(); // the previous textures may still be read by DirectX 12
    depthUAV11.reset();
    const D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (!depthShared.Create("depth", renderWidth, renderHeight, DXGI_FORMAT_R32_FLOAT, flags) ||
        !motionShared.Create("motion vectors", renderWidth, renderHeight, DXGI_FORMAT_R16G16_FLOAT, flags)) {
        depthShared.Reset();
        motionShared.Reset();
        failedWidth = renderWidth;
        failedHeight = renderHeight;
        frameGenerationProblem = "could not share the depth/motion vectors with DirectX 12 (see ReShade.log)";
        return false;
    }
    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_R32_FLOAT;
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    HRESULT hr = device11->CreateUnorderedAccessView(depthShared.texture11.get(), &uavDesc, &depthUAV11);
    if (FAILED(hr)) {
        Log(reshade::log::level::warning, "Could not create the DirectX 11 view of the shared depth (0x%08X).", (unsigned)hr);
        depthShared.Reset();
        motionShared.Reset();
        failedWidth = renderWidth;
        failedHeight = renderHeight;
        frameGenerationProblem = "could not write the depth for DirectX 12 (see ReShade.log)";
        return false;
    }
    failedWidth = 0;
    failedHeight = 0;
    frameGenerationProblem = "";
    return true;
}

// Image without HUD, in the back buffer format. Render thread only.
inline bool EnsureHudless(uint32_t width, uint32_t height, DXGI_FORMAT format) {
    if (hudlessShared.Matches(width, height, format) && hudlessRTV11) {
        return true;
    }
    static uint32_t failedWidth = 0;
    static uint32_t failedHeight = 0;
    if (failedWidth == width && failedHeight == height) {
        return false;
    }
    FlushGpu();
    hudlessRTV11.reset();
    if (!hudlessShared.Create("image without HUD", width, height, format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)) {
        failedWidth = width;
        failedHeight = height;
        return false;
    }
    HRESULT hr = device11->CreateRenderTargetView(hudlessShared.texture11.get(), nullptr, &hudlessRTV11);
    if (FAILED(hr)) {
        Log(reshade::log::level::warning, "Could not create the DirectX 11 view of the image without HUD (0x%08X).", (unsigned)hr);
        hudlessShared.Reset();
        failedWidth = width;
        failedHeight = height;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Camera data for DLSS-FG, from the game's TAA constants (cbTemporalAA).
//   c3..c6:  g_motionMatrix      (uv, depth) of this frame -> (uv, depth) of the previous frame
//   c7..c10: g_reconstructMatrix (uv, depth) -> camera view space (verified at runtime)
// Both are stored column by column (HLSL default), used as M * v.
// ---------------------------------------------------------------------------
inline void Multiply(const float a[4][4], const float b[4][4], float out[4][4]) {
    float r[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
        }
    }
    memcpy(out, r, sizeof(r));
}

inline bool Invert(const float m[4][4], float out[4][4]) {
    const float* a = &m[0][0];
    float inv[16];
    inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
    inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
    inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
    inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
    inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
    inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
    inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
    inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
    inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
    inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
    inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
    inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
    inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
    inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
    inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
    inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];
    float det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
    if (!std::isfinite(det) || std::fabs(det) < 1e-20f) {
        return false;
    }
    float invDet = 1.0f / det;
    float* o = &out[0][0];
    for (int i = 0; i < 16; ++i) {
        o[i] = inv[i] * invDet;
    }
    return true;
}

inline void TransformPoint(const float m[4][4], float x, float y, float z, float out[3], float* w = nullptr) {
    float r[4];
    for (int i = 0; i < 4; ++i) {
        r[i] = m[i][0] * x + m[i][1] * y + m[i][2] * z + m[i][3];
    }
    if (w) {
        *w = r[3];
    }
    float invW = std::fabs(r[3]) > 1e-12f ? 1.0f / r[3] : 0.0f;
    out[0] = r[0] * invW;
    out[1] = r[1] * invW;
    out[2] = r[2] * invW;
}

inline bool AllFinite(const float m[4][4]) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (!std::isfinite(m[i][j])) {
                return false;
            }
        }
    }
    return true;
}

// Fills the camera part of 'inputs' from the 64 floats of the TAA constant buffer.
// Returns false when the data does not look like a camera.
inline bool ComputeCamera(const float* c, uint32_t renderWidth, uint32_t renderHeight, FrameInputs& inputs) {
    float motion[4][4];
    float reconstruct[4][4];
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            motion[row][col] = c[12 + 4 * col + row];
            reconstruct[row][col] = c[28 + 4 * col + row];
        }
    }
    // Clip (NDC) <-> (uv, depth): u = 0.5x + 0.5, v = -0.5y + 0.5
    const float ndcToUv[4][4] = { { 0.5f, 0, 0, 0.5f }, { 0, -0.5f, 0, 0.5f }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };
    const float uvToNdc[4][4] = { { 2.0f, 0, 0, -1.0f }, { 0, -2.0f, 0, 1.0f }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };

    float temp[4][4];
    Multiply(motion, ndcToUv, temp);
    Multiply(uvToNdc, temp, inputs.clipToPrevClip);
    if (!AllFinite(inputs.clipToPrevClip) || !Invert(inputs.clipToPrevClip, inputs.prevClipToClip)) {
        return false;
    }

    // Projection: accept the reconstruct matrix only if the screen center maps onto the view axis.
    bool projectionFromGame = false;
    Multiply(reconstruct, ndcToUv, inputs.clipToView);
    if (AllFinite(inputs.clipToView) && Invert(inputs.clipToView, inputs.viewToClip)) {
        float atZero[3];
        float atOne[3];
        float wOne = 0.0f;
        TransformPoint(inputs.clipToView, 0.0f, 0.0f, 0.0f, atZero);
        TransformPoint(inputs.clipToView, 0.0f, 0.0f, 1.0f, atOne, &wOne);
        float zZero = std::fabs(atZero[2]);
        float zOne = std::fabs(wOne) > 1e-9f ? std::fabs(atOne[2]) : 1.0e6f;
        bool onAxis = std::fabs(atZero[0]) <= 1e-3f * (zZero + 1e-3f) && std::fabs(atZero[1]) <= 1e-3f * (zZero + 1e-3f);
        float p00 = inputs.viewToClip[0][0];
        float p11 = inputs.viewToClip[1][1];
        if (onAxis && zZero > 0.0f && std::isfinite(zOne) && std::fabs(p11) > 1e-6f && std::fabs(p00) > 1e-6f) {
            inputs.depthInverted = zZero > zOne;
            inputs.nearPlane = zZero < zOne ? zZero : zOne;
            inputs.farPlane = zZero < zOne ? zOne : zZero;
            if (inputs.farPlane > 1.0e6f) {
                inputs.farPlane = 1.0e6f;
            }
            inputs.fov = 2.0f * std::atan(1.0f / std::fabs(p11));
            inputs.aspect = std::fabs(p11 / p00);
            projectionFromGame = true;
        }
    }
    if (!projectionFromGame) {
        // Depth range from g_unprojectParams (c11): 1 / viewZ = c11.x * depth + c11.y
        // (in FFXV: near 0.2, far 12500, depth not inverted). Field of view is estimated.
        float n = 0.1f;
        float f = 10000.0f;
        bool inverted = false;
        const float a = c[44];
        const float b = c[45];
        const float zAtZero = b > 0.0f ? 1.0f / b : -1.0f;
        const float zAtOne = (a + b) > 0.0f ? 1.0f / (a + b) : 1.0e6f;
        depthRangeFromGame = std::isfinite(zAtZero) && zAtZero > 0.0f && std::isfinite(zAtOne) && std::fabs(zAtOne - zAtZero) > 1e-3f;
        if (depthRangeFromGame) {
            inverted = zAtZero > zAtOne;
            n = inverted ? zAtOne : zAtZero;
            f = inverted ? zAtZero : zAtOne;
            if (f > 1.0e6f) {
                f = 1.0e6f;
            }
        }
        const float fov = 1.0f; // about 57 degrees vertical
        const float aspect = renderHeight ? (float)renderWidth / (float)renderHeight : 1.777f;
        const float yScale = 1.0f / std::tan(fov * 0.5f);
        // Depth = A + B / viewZ: 0 at near and 1 at far (or the opposite when inverted).
        const float depthA = inverted ? -n / (f - n) : f / (f - n);
        const float depthB = inverted ? n * f / (f - n) : -n * f / (f - n);
        float p[4][4] = { { yScale / aspect, 0, 0, 0 }, { 0, yScale, 0, 0 }, { 0, 0, depthA, depthB }, { 0, 0, 1, 0 } };
        memcpy(inputs.viewToClip, p, sizeof(p));
        Invert(inputs.viewToClip, inputs.clipToView);
        inputs.nearPlane = n;
        inputs.farPlane = f;
        inputs.fov = fov;
        inputs.aspect = aspect;
        inputs.depthInverted = inverted;
    }
    int mode = depthInvertedMode;
    if (mode == 0 || mode == 1) {
        inputs.depthInverted = mode == 1;
    }
    lastDepthInverted = inputs.depthInverted;
    lastCameraFromGame = projectionFromGame;
    inputs.renderWidth = renderWidth;
    inputs.renderHeight = renderHeight;
    return true;
}

inline void CopyMatrixForDLSS(const float in[4][4], float out[4][4]) {
    // DLSS-FG expects row-major matrices for row vectors (v * M), i.e. the transpose
    // of the column-vector matrices computed above. Toggle kept for testing.
    if (transposeMatrices) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                out[i][j] = in[j][i];
            }
        }
    }
    else {
        memcpy(out, in, sizeof(float) * 16);
    }
}

// ---------------------------------------------------------------------------
// NGX (DLSS Frame Generation) on the DirectX 12 device
// ---------------------------------------------------------------------------
inline bool ngx12InitTried = false;
inline bool ngx12Available = false;
inline NVSDK_NGX_Parameter* fgParameters = nullptr;
inline NVSDK_NGX_Handle* fgHandle = nullptr;
inline uint32_t fgWidth = 0;
inline uint32_t fgHeight = 0;
inline uint32_t fgRenderWidth = 0;
inline uint32_t fgRenderHeight = 0;
inline DXGI_FORMAT fgFormat = DXGI_FORMAT_UNKNOWN;

inline bool InitNGX12() {
    if (ngx12InitTried) {
        return ngx12Available;
    }
    ngx12InitTried = true;
    if (NVSDK_NGX_FAILED(NVSDK_NGX_D3D12_Init(1, L"", device12.get()))) {
        frameGenerationStatus = "unavailable: NGX could not start on DirectX 12";
        return false;
    }
    NVSDK_NGX_Parameter* caps = nullptr;
    if (NVSDK_NGX_FAILED(NVSDK_NGX_D3D12_GetCapabilityParameters(&caps)) || !caps) {
        frameGenerationStatus = "unavailable: no NGX capabilities";
        return false;
    }
    int available = 0;
    int needsUpdatedDriver = 0;
    unsigned int mfgMax = 0;
    caps->Get(NVSDK_NGX_Parameter_FrameGeneration_Available, &available);
    caps->Get(NVSDK_NGX_Parameter_FrameInterpolation_NeedsUpdatedDriver, &needsUpdatedDriver);
    caps->Get(NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax, &mfgMax);
    NVSDK_NGX_D3D12_DestroyParameters(caps);
    multiFrameCountMax = mfgMax;
    Log(reshade::log::level::info, "DLSS Frame Generation available=%d, needs newer driver=%d, MultiFrameCountMax=%u.", available, needsUpdatedDriver, mfgMax);
    if (!available) {
        frameGenerationStatus = needsUpdatedDriver ? "unavailable: update the NVIDIA driver"
                                                   : "unavailable: check nvngx_dlssg.dll in the game folder, RTX 40+ GPU, Hardware-accelerated GPU scheduling";
        return false;
    }
    if (NVSDK_NGX_FAILED(NVSDK_NGX_D3D12_AllocateParameters(&fgParameters)) || !fgParameters) {
        frameGenerationStatus = "unavailable: NGX parameters";
        return false;
    }
    ngx12Available = true;
    return true;
}

// Render thread only, outside queueMutex.
inline void ReleaseFrameGenerationFeature() {
    if (fgHandle) {
        FlushGpu();
        NVSDK_NGX_D3D12_ReleaseFeature(fgHandle);
        fgHandle = nullptr;
    }
    fgWidth = fgHeight = fgRenderWidth = fgRenderHeight = 0;
    fgFormat = DXGI_FORMAT_UNKNOWN;
}

// ---------------------------------------------------------------------------
// The swap chain the game sees.
// ---------------------------------------------------------------------------
class BridgeSwapChain final : public IDXGISwapChain4 {
public:
    BridgeSwapChain(IDXGISwapChain3* real, IUnknown* gameDevice, const DXGI_SWAP_CHAIN_DESC& gameDesc, UINT realBufferCount, UINT realFlags)
        : _real(real), _gameDevice(gameDevice), _gameDesc(gameDesc), _realBufferCount(realBufferCount), _realFlags(realFlags) {
        _real->QueryInterface(&_real4);
        QueryPerformanceFrequency(&_qpcFrequency);
    }

    bool Initialize() {
        if (!CreateCommandRing(_renderRing) || !CreateCommandRing(_presentRing)) {
            return false;
        }
        if (!CreateBuffers()) {
            return false;
        }
        _presenter = std::thread([this]() { PresenterLoop(); });
        return true;
    }

    // IUnknown
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** object) override {
        if (!object) {
            return E_POINTER;
        }
        if (riid == __uuidof(IUnknown) || riid == __uuidof(IDXGIObject) || riid == __uuidof(IDXGIDeviceSubObject) ||
            riid == __uuidof(IDXGISwapChain) || riid == __uuidof(IDXGISwapChain1) || riid == __uuidof(IDXGISwapChain2) ||
            riid == __uuidof(IDXGISwapChain3) || (riid == __uuidof(IDXGISwapChain4) && _real4)) {
            AddRef();
            *object = static_cast<IDXGISwapChain4*>(this);
            return S_OK;
        }
        *object = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override {
        return (ULONG)InterlockedIncrement(&_ref);
    }
    ULONG STDMETHODCALLTYPE Release() override {
        ULONG ref = (ULONG)InterlockedDecrement(&_ref);
        if (ref == 0) {
            delete this;
        }
        return ref;
    }

    // IDXGIObject
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID name, UINT size, const void* data) override { return _real->SetPrivateData(name, size, data); }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID name, const IUnknown* unknown) override { return _real->SetPrivateDataInterface(name, unknown); }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID name, UINT* size, void* data) override { return _real->GetPrivateData(name, size, data); }
    HRESULT STDMETHODCALLTYPE GetParent(REFIID riid, void** parent) override { return _real->GetParent(riid, parent); }

    // IDXGIDeviceSubObject: the game's own D3D11 device
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID riid, void** device) override { return _gameDevice->QueryInterface(riid, device); }

    // IDXGISwapChain
    HRESULT STDMETHODCALLTYPE Present(UINT syncInterval, UINT flags) override {
        return PresentFrame(syncInterval, flags);
    }
    HRESULT STDMETHODCALLTYPE GetBuffer(UINT buffer, REFIID riid, void** surface) override {
        if (!_shared11) {
            return DXGI_ERROR_INVALID_CALL;
        }
        // The game created a DISCARD swap chain, so it only ever draws into one buffer.
        return _shared11->QueryInterface(riid, surface);
    }
    HRESULT STDMETHODCALLTYPE SetFullscreenState(BOOL fullscreen, IDXGIOutput* target) override {
        DrainPresenter();
        std::lock_guard<std::mutex> lock(queueMutex);
        return _real->SetFullscreenState(fullscreen, target);
    }
    HRESULT STDMETHODCALLTYPE GetFullscreenState(BOOL* fullscreen, IDXGIOutput** target) override {
        return _real->GetFullscreenState(fullscreen, target);
    }
    HRESULT STDMETHODCALLTYPE GetDesc(DXGI_SWAP_CHAIN_DESC* desc) override {
        if (!desc) {
            return E_INVALIDARG;
        }
        DXGI_SWAP_CHAIN_DESC realDesc = {};
        HRESULT hr = _real->GetDesc(&realDesc);
        if (FAILED(hr)) {
            return hr;
        }
        *desc = _gameDesc;
        desc->OutputWindow = realDesc.OutputWindow;
        desc->Windowed = realDesc.Windowed;
        desc->BufferDesc.RefreshRate = realDesc.BufferDesc.RefreshRate;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE ResizeBuffers(UINT bufferCount, UINT width, UINT height, DXGI_FORMAT format, UINT flags) override {
        return Resize(bufferCount, width, height, format, flags);
    }
    HRESULT STDMETHODCALLTYPE ResizeTarget(const DXGI_MODE_DESC* params) override { return _real->ResizeTarget(params); }
    HRESULT STDMETHODCALLTYPE GetContainingOutput(IDXGIOutput** output) override { return _real->GetContainingOutput(output); }
    HRESULT STDMETHODCALLTYPE GetFrameStatistics(DXGI_FRAME_STATISTICS* stats) override { return _real->GetFrameStatistics(stats); }
    HRESULT STDMETHODCALLTYPE GetLastPresentCount(UINT* count) override { return _real->GetLastPresentCount(count); }

    // IDXGISwapChain1
    HRESULT STDMETHODCALLTYPE GetDesc1(DXGI_SWAP_CHAIN_DESC1* desc) override {
        HRESULT hr = _real->GetDesc1(desc);
        if (SUCCEEDED(hr) && desc) {
            desc->Format = _gameDesc.BufferDesc.Format;
            desc->BufferCount = _gameDesc.BufferCount;
            desc->SwapEffect = _gameDesc.SwapEffect;
            desc->Flags = _gameDesc.Flags;
            desc->BufferUsage = _gameDesc.BufferUsage;
        }
        return hr;
    }
    HRESULT STDMETHODCALLTYPE GetFullscreenDesc(DXGI_SWAP_CHAIN_FULLSCREEN_DESC* desc) override { return _real->GetFullscreenDesc(desc); }
    HRESULT STDMETHODCALLTYPE GetHwnd(HWND* hwnd) override { return _real->GetHwnd(hwnd); }
    HRESULT STDMETHODCALLTYPE GetCoreWindow(REFIID riid, void** unknown) override { return _real->GetCoreWindow(riid, unknown); }
    HRESULT STDMETHODCALLTYPE Present1(UINT syncInterval, UINT flags, const DXGI_PRESENT_PARAMETERS*) override {
        return PresentFrame(syncInterval, flags);
    }
    BOOL STDMETHODCALLTYPE IsTemporaryMonoSupported() override { return _real->IsTemporaryMonoSupported(); }
    HRESULT STDMETHODCALLTYPE GetRestrictToOutput(IDXGIOutput** output) override { return _real->GetRestrictToOutput(output); }
    HRESULT STDMETHODCALLTYPE SetBackgroundColor(const DXGI_RGBA* color) override { return _real->SetBackgroundColor(color); }
    HRESULT STDMETHODCALLTYPE GetBackgroundColor(DXGI_RGBA* color) override { return _real->GetBackgroundColor(color); }
    HRESULT STDMETHODCALLTYPE SetRotation(DXGI_MODE_ROTATION rotation) override { return _real->SetRotation(rotation); }
    HRESULT STDMETHODCALLTYPE GetRotation(DXGI_MODE_ROTATION* rotation) override { return _real->GetRotation(rotation); }

    // IDXGISwapChain2
    HRESULT STDMETHODCALLTYPE SetSourceSize(UINT width, UINT height) override { return _real->SetSourceSize(width, height); }
    HRESULT STDMETHODCALLTYPE GetSourceSize(UINT* width, UINT* height) override { return _real->GetSourceSize(width, height); }
    HRESULT STDMETHODCALLTYPE SetMaximumFrameLatency(UINT latency) override { return _real->SetMaximumFrameLatency(latency); }
    HRESULT STDMETHODCALLTYPE GetMaximumFrameLatency(UINT* latency) override { return _real->GetMaximumFrameLatency(latency); }
    HANDLE STDMETHODCALLTYPE GetFrameLatencyWaitableObject() override { return _real->GetFrameLatencyWaitableObject(); }
    HRESULT STDMETHODCALLTYPE SetMatrixTransform(const DXGI_MATRIX_3X2_F* matrix) override { return _real->SetMatrixTransform(matrix); }
    HRESULT STDMETHODCALLTYPE GetMatrixTransform(DXGI_MATRIX_3X2_F* matrix) override { return _real->GetMatrixTransform(matrix); }

    // IDXGISwapChain3
    UINT STDMETHODCALLTYPE GetCurrentBackBufferIndex() override { return 0; }
    HRESULT STDMETHODCALLTYPE CheckColorSpaceSupport(DXGI_COLOR_SPACE_TYPE colorSpace, UINT* support) override { return _real->CheckColorSpaceSupport(colorSpace, support); }
    HRESULT STDMETHODCALLTYPE SetColorSpace1(DXGI_COLOR_SPACE_TYPE colorSpace) override { return _real->SetColorSpace1(colorSpace); }
    HRESULT STDMETHODCALLTYPE ResizeBuffers1(UINT bufferCount, UINT width, UINT height, DXGI_FORMAT format, UINT flags, const UINT*, IUnknown* const*) override {
        return Resize(bufferCount, width, height, format, flags);
    }

    // IDXGISwapChain4
    HRESULT STDMETHODCALLTYPE SetHDRMetaData(DXGI_HDR_METADATA_TYPE type, UINT size, void* metaData) override {
        return _real4 ? _real4->SetHDRMetaData(type, size, metaData) : DXGI_ERROR_UNSUPPORTED;
    }

private:
    static constexpr int kRingSize = 3;
    static constexpr int kSlots = 2;

    // Command allocators + list used by one thread.
    struct CommandRing {
        com_ptr<ID3D12CommandAllocator> allocators[kRingSize];
        uint64_t allocatorFence[kRingSize] = {};
        com_ptr<ID3D12GraphicsCommandList> list;
        uint64_t counter = 0;
    };

    // One frame handed to the presenter thread: generated frame + copy of the real frame.
    struct Slot {
        com_ptr<ID3D12Resource> generated; // UNORDERED_ACCESS state
        com_ptr<ID3D12Resource> real;      // UNORDERED_ACCESS state
        uint64_t freeFence = 0;            // GPU finished presenting this slot
        bool queued = false;               // guarded by _slotMutex
    };

    struct Job {
        int slot;
        bool hasGenerated;
        UINT syncInterval;
        UINT flags;
        double frameInterval; // seconds between frames rendered by the game
    };

    ~BridgeSwapChain() {
        {
            std::lock_guard<std::mutex> lock(_jobMutex);
            _quit = true;
        }
        _jobCv.notify_all();
        if (_presenter.joinable()) {
            _presenter.join();
        }
        FlushGpu();
        ReleaseSlots();
        ReleaseBuffers();
        bridgeActive = false;
        frameGenerationRunning = false;
        backBuffer11 = nullptr;
        bridgeStatus = "off (swap chain released)";
        Log(reshade::log::level::info, "Bridged swap chain released.");
    }

    bool CreateCommandRing(CommandRing& ring) {
        for (int i = 0; i < kRingSize; ++i) {
            if (FAILED(device12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), (void**)&ring.allocators[i]))) {
                return false;
            }
        }
        if (FAILED(device12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, ring.allocators[0].get(), nullptr, __uuidof(ID3D12GraphicsCommandList), (void**)&ring.list))) {
            return false;
        }
        ring.list->Close();
        return true;
    }

    // Opens the next command list of a ring. Returns the slot index used.
    int BeginCommands(CommandRing& ring) {
        int index = (int)(ring.counter++ % kRingSize);
        WaitForFence12(ring.allocatorFence[index]);
        ring.allocators[index]->Reset();
        ring.list->Reset(ring.allocators[index].get(), nullptr);
        return index;
    }

    static void Transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = resource;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = before;
        barrier.Transition.StateAfter = after;
        list->ResourceBarrier(1, &barrier);
    }

    UINT PresentFlags(UINT syncInterval, UINT flags) {
        UINT presentFlags = flags & (DXGI_PRESENT_DO_NOT_WAIT | DXGI_PRESENT_RESTART);
        if (syncInterval == 0 && tearingSupported) {
            BOOL fullscreen = FALSE;
            _real->GetFullscreenState(&fullscreen, nullptr);
            if (!fullscreen) {
                presentFlags |= DXGI_PRESENT_ALLOW_TEARING;
            }
        }
        return presentFlags;
    }

    void ReleaseBuffers() {
        _backBuffers12.clear();
        _shared11.reset();
        _shared12.reset();
        backBuffer11 = nullptr;
    }

    void ReleaseSlots() {
        for (Slot& slot : _slots) {
            slot.generated.reset();
            slot.real.reset();
            slot.freeFence = 0;
            slot.queued = false;
        }
    }

    bool CreateBuffers() {
        DXGI_SWAP_CHAIN_DESC realDesc = {};
        if (FAILED(_real->GetDesc(&realDesc))) {
            return false;
        }
        UINT width = realDesc.BufferDesc.Width;
        UINT height = realDesc.BufferDesc.Height;

        if (!CreateTexture12(width, height, _gameDesc.BufferDesc.Format,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS,
                D3D12_HEAP_FLAG_SHARED, D3D12_RESOURCE_STATE_COMMON, _shared12)) {
            Log(reshade::log::level::error, "Could not create the shared back buffer (%ux%u format %u).", width, height, (unsigned)_gameDesc.BufferDesc.Format);
            return false;
        }
        HANDLE sharedHandle = nullptr;
        if (FAILED(device12->CreateSharedHandle(_shared12.get(), nullptr, GENERIC_ALL, nullptr, &sharedHandle))) {
            ReleaseBuffers();
            return false;
        }
        HRESULT hr = device11_5->OpenSharedResource1(sharedHandle, __uuidof(ID3D11Texture2D), (void**)&_shared11);
        CloseHandle(sharedHandle);
        if (FAILED(hr) || !_shared11) {
            Log(reshade::log::level::error, "DirectX 11 could not open the shared back buffer.");
            ReleaseBuffers();
            return false;
        }

        _backBuffers12.resize(_realBufferCount);
        for (UINT i = 0; i < _realBufferCount; ++i) {
            if (FAILED(_real->GetBuffer(i, __uuidof(ID3D12Resource), (void**)&_backBuffers12[i]))) {
                ReleaseBuffers();
                return false;
            }
        }
        _gameDesc.BufferDesc.Width = width;
        _gameDesc.BufferDesc.Height = height;
        backBuffer11 = _shared11.get();
        return true;
    }

    // Output textures of Frame Generation (same size/format as the back buffer).
    bool EnsureSlots() {
        if (_slots[0].generated && _slots[0].real) {
            return true;
        }
        for (Slot& slot : _slots) {
            if (!CreateTexture12(_gameDesc.BufferDesc.Width, _gameDesc.BufferDesc.Height, _gameDesc.BufferDesc.Format,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, slot.generated) ||
                !CreateTexture12(_gameDesc.BufferDesc.Width, _gameDesc.BufferDesc.Height, _gameDesc.BufferDesc.Format,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, slot.real)) {
                ReleaseSlots();
                return false;
            }
        }
        return true;
    }

    HRESULT Resize(UINT bufferCount, UINT width, UINT height, DXGI_FORMAT format, UINT flags) {
        DrainPresenter();
        ReleaseFrameGenerationFeature();
        FlushGpu();
        ReleaseSlots();
        ReleaseBuffers();

        DXGI_FORMAT gameFormat = format != DXGI_FORMAT_UNKNOWN ? format : _gameDesc.BufferDesc.Format;
        DXGI_FORMAT realFormat = FlipCompatibleFormat(gameFormat);
        if (realFormat == DXGI_FORMAT_UNKNOWN) {
            Log(reshade::log::level::error, "ResizeBuffers with unsupported format %u.", (unsigned)gameFormat);
            return DXGI_ERROR_INVALID_CALL;
        }
        _realFlags = flags | (tearingSupported ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0);
        HRESULT hr;
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            hr = _real->ResizeBuffers(_realBufferCount, width, height, realFormat, _realFlags);
        }
        if (FAILED(hr)) {
            Log(reshade::log::level::error, "ResizeBuffers failed (0x%08X).", (unsigned)hr);
            return hr;
        }
        if (bufferCount != 0) {
            _gameDesc.BufferCount = bufferCount;
        }
        _gameDesc.BufferDesc.Format = gameFormat;
        _gameDesc.Flags = flags;
        if (!CreateBuffers()) {
            return E_FAIL;
        }
        _frameGenerationWasRunning = false;
        return S_OK;
    }

    void MeasureFrameInterval() {
        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        if (_lastFrameTime.QuadPart != 0) {
            double interval = (double)(now.QuadPart - _lastFrameTime.QuadPart) / (double)_qpcFrequency.QuadPart;
            if (interval > 0.002 && interval < 0.2) {
                _frameInterval = _frameInterval <= 0.0 ? interval : _frameInterval * 0.9 + interval * 0.1;
            }
        }
        _lastFrameTime = now;
    }

    static double QpcSeconds(const LARGE_INTEGER& from, const LARGE_INTEGER& to, const LARGE_INTEGER& frequency) {
        return (double)(to.QuadPart - from.QuadPart) / (double)frequency.QuadPart;
    }

    // NVIDIA Reflex on the game's DirectX 11 device: settings follow the menu.
    void ApplyReflexSettings() {
        bool want = reflexEnabled;
        bool boost = reflexBoost;
        if (!_reflexApplied || want != _reflexOn || boost != _reflexBoostOn) {
            if (reflex::SetMode(device11.get(), want, want && boost)) {
                _reflexOn = want;
                _reflexBoostOn = boost;
                _reflexApplied = true;
            }
            else if (!_reflexApplied) {
                _reflexApplied = true; // do not retry every frame
                _reflexOn = false;
            }
        }
    }

    void ReflexMarker(reflex::NV_LATENCY_MARKER_TYPE type) {
        if (_reflexOn) {
            reflex::Marker(device11.get(), _reflexFrameId, type);
        }
    }

    // At most 'framesAhead' frames may be waiting for the GPU when the game starts its next frame.
    // 'submitted' is the fence value of the DirectX 12 work of the frame just handed over.
    void LimitFramesAhead(uint64_t submitted) {
        if (submitted == 0) {
            return;
        }
        int ahead = framesAhead;
        if (ahead < 0) {
            ahead = 0;
        }
        if (ahead > 2) {
            ahead = 2;
        }
        uint64_t waitFor = 0;
        if (ahead == 0) {
            waitFor = submitted;
        }
        else if (_submittedCount >= (uint64_t)ahead) {
            waitFor = _submittedHistory[(_submittedCount - ahead) % kHistory];
        }
        _submittedHistory[_submittedCount % kHistory] = submitted;
        _submittedCount++;
        if (waitFor != 0) {
            WaitForFence12(waitFor, 200);
        }
    }

    void RecordFrameTime(double seconds) {
        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        float ms = (float)(seconds * 1000.0);
        if (ms > _windowLongestMs) {
            _windowLongestMs = ms;
        }
        if (_windowStart.QuadPart == 0) {
            _windowStart = now;
        }
        if (QpcSeconds(_windowStart, now, _qpcFrequency) >= 2.0) {
            statLongestFrameMs = _windowLongestMs;
            _windowLongestMs = 0.0f;
            _windowStart = now;
        }
        if (ms > 50.0f && QpcSeconds(_lastSlowLog, now, _qpcFrequency) > 1.0) {
            _lastSlowLog = now;
            Log(reshade::log::level::warning, "Slow frame: Present took %.1f ms (waiting for the presenter %.1f ms, frame limit %.1f ms, Reflex %.1f ms, generated=%d, reason: %s).",
                ms, _lastSlotWaitMs, _lastLimitWaitMs, _lastReflexMs, _lastGenerated ? 1 : 0, statLastSkipReason.load());
        }
    }

    HRESULT PresentFrame(UINT syncInterval, UINT flags) {
        if (flags & DXGI_PRESENT_TEST) {
            return _real->Present(syncInterval, DXGI_PRESENT_TEST);
        }
        if (!_shared12 || _backBuffers12.empty()) {
            return DXGI_ERROR_INVALID_CALL;
        }
        LARGE_INTEGER start;
        QueryPerformanceCounter(&start);
        _lastSlotWaitMs = 0.0;
        _lastLimitWaitMs = 0.0;
        _lastReflexMs = 0.0;
        _lastGenerated = false;

        ReflexMarker(reflex::RENDERSUBMIT_END);
        ReflexMarker(reflex::PRESENT_START);

        // End of a frame rendered by the game: let the add-on do its per-frame work.
        if (onGameFrameEnd) {
            onGameFrameEnd((uint64_t)_shared11.get(), _gameDesc.BufferDesc.Width, _gameDesc.BufferDesc.Height);
        }
        MeasureFrameInterval();

        FrameInputs inputs = frameInputs;
        frameInputs.valid = false;
        frameInputs.hudlessValid = false;
        if (frameGenerationRetry.exchange(false)) {
            _frameGenerationFailed = false;
        }

        HRESULT hr = S_OK;
        uint64_t submitted = 0;
        bool throughPresenter = false;
        if (frameGenerationEnabled && !_frameGenerationFailed) {
            // Frames without Frame Generation data (menus, loading, videos) also go through the
            // presenter, so its rhythm is not broken by switching back and forth.
            throughPresenter = PresentThroughPresenter(syncInterval, flags, inputs, inputs.valid, submitted);
        }
        else if (!frameGenerationEnabled) {
            frameGenerationStatus = "off";
        }
        if (!throughPresenter) {
            frameGenerationRunning = false;
            _frameGenerationWasRunning = false;
            DrainPresenter();
            hr = PresentDirect(syncInterval, flags, submitted);
        }

        ReflexMarker(reflex::PRESENT_END);

        LARGE_INTEGER beforeLimit;
        QueryPerformanceCounter(&beforeLimit);
        LimitFramesAhead(submitted);
        LARGE_INTEGER afterLimit;
        QueryPerformanceCounter(&afterLimit);
        _lastLimitWaitMs = QpcSeconds(beforeLimit, afterLimit, _qpcFrequency) * 1000.0;

        // Start of the next frame (Reflex sleeps here when it is useful).
        ApplyReflexSettings();
        if (_reflexOn) {
            reflex::Sleep(device11.get());
            _reflexFrameId++;
            ReflexMarker(reflex::SIMULATION_START);
            ReflexMarker(reflex::RENDERSUBMIT_START);
        }
        LARGE_INTEGER end;
        QueryPerformanceCounter(&end);
        _lastReflexMs = QpcSeconds(afterLimit, end, _qpcFrequency) * 1000.0;
        RecordFrameTime(QpcSeconds(start, end, _qpcFrequency));
        return hr;
    }

    // Frame Generation off: copy the game's frame and present it right away.
    HRESULT PresentDirect(UINT syncInterval, UINT flags, uint64_t& copyDone) {
        uint64_t gameFrameDone = SignalFromD3D11();
        HRESULT hr;
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            queue12->Wait(fence11to12.get(), gameFrameDone);

            int ringIndex = BeginCommands(_renderRing);
            ID3D12GraphicsCommandList* list = _renderRing.list.get();
            UINT index = _real->GetCurrentBackBufferIndex();
            ID3D12Resource* backBuffer = _backBuffers12[index < _backBuffers12.size() ? index : 0].get();
            Transition(list, backBuffer, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
            list->CopyResource(backBuffer, _shared12.get());
            Transition(list, backBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
            list->Close();
            ID3D12CommandList* lists[] = { list };
            queue12->ExecuteCommandLists(1, lists);

            hr = _real->Present(syncInterval, PresentFlags(syncInterval, flags));

            copyDone = SignalQueueLocked();
            _renderRing.allocatorFence[ringIndex] = copyDone;
        }
        // The game may only draw into the shared back buffer again after the copy finished.
        context11->Wait(fence12_11.get(), copyDone);

        if (FAILED(hr) && hr != DXGI_ERROR_WAS_STILL_DRAWING) {
            Log(reshade::log::level::warning, "Present failed (0x%08X).", (unsigned)hr);
        }
        return hr;
    }

    // Frame Generation on: generate the in-between frame (when 'generate' and possible) and hand
    // the frame(s) to the presenter thread. Returns false if this frame must be presented directly.
    bool PresentThroughPresenter(UINT syncInterval, UINT flags, const FrameInputs& inputs, bool generate, uint64_t& submitted) {
        if (!InitNGX12()) {
            _frameGenerationFailed = true;
            return false;
        }
        if (!EnsureSlots()) {
            frameGenerationStatus = "failed: could not create the output textures";
            _frameGenerationFailed = true;
            return false;
        }
        if (generate && (!depthShared.resource12 || !motionShared.resource12 ||
            depthShared.width != inputs.renderWidth || depthShared.height != inputs.renderHeight)) {
            generate = false;
        }
        if (!generate) {
            statLastSkipReason = "no frame data from the game this frame";
            frameGenerationStatus = "waiting for frame data (normal in menus, loading screens and videos)";
        }

        const uint32_t width = _gameDesc.BufferDesc.Width;
        const uint32_t height = _gameDesc.BufferDesc.Height;
        const DXGI_FORMAT format = _gameDesc.BufferDesc.Format;
        if (generate && fgHandle && (fgWidth != width || fgHeight != height || fgFormat != format ||
            fgRenderWidth != inputs.renderWidth || fgRenderHeight != inputs.renderHeight)) {
            DrainPresenter();
            ReleaseFrameGenerationFeature();
            _frameGenerationWasRunning = false;
        }

        // Wait for a free slot (the presenter may still be showing the slot from two frames ago).
        int slotIndex = _nextSlot;
        _nextSlot = (_nextSlot + 1) % kSlots;
        Slot& slot = _slots[slotIndex];
        {
            LARGE_INTEGER before;
            QueryPerformanceCounter(&before);
            std::unique_lock<std::mutex> lock(_slotMutex);
            bool free = _slotCv.wait_for(lock, std::chrono::milliseconds(500), [&]() { return !slot.queued; });
            LARGE_INTEGER after;
            QueryPerformanceCounter(&after);
            _lastSlotWaitMs = QpcSeconds(before, after, _qpcFrequency) * 1000.0;
            if (!free) {
                Log(reshade::log::level::warning, "Presenter thread is stuck; presenting without Frame Generation.");
                return false;
            }
        }
        WaitForFence12(slot.freeFence);

        uint64_t gameFrameDone = SignalFromD3D11();
        bool generated = false;
        bool createdThisFrame = false;
        uint64_t workDone;
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            queue12->Wait(fence11to12.get(), gameFrameDone);

            int ringIndex = BeginCommands(_renderRing);
            ID3D12GraphicsCommandList* list = _renderRing.list.get();

            if (generate && !fgHandle) {
                createdThisFrame = true;
                NVSDK_NGX_DLSSG_Create_Params createParams = {};
                createParams.Width = width;
                createParams.Height = height;
                createParams.NativeBackbufferFormat = (unsigned int)format;
                createParams.RenderWidth = inputs.renderWidth;
                createParams.RenderHeight = inputs.renderHeight;
                createParams.DynamicResolutionScaling = false;
                if (NVSDK_NGX_FAILED(NGX_D3D12_CREATE_DLSSG(list, 1, 1, &fgHandle, fgParameters, &createParams)) || !fgHandle) {
                    fgHandle = nullptr;
                    frameGenerationStatus = "failed: DLSS Frame Generation could not be created";
                    Log(reshade::log::level::error, "DLSS Frame Generation feature creation failed (%ux%u, render %ux%u).",
                        width, height, inputs.renderWidth, inputs.renderHeight);
                    _frameGenerationFailed = true;
                }
                else {
                    fgWidth = width;
                    fgHeight = height;
                    fgFormat = format;
                    fgRenderWidth = inputs.renderWidth;
                    fgRenderHeight = inputs.renderHeight;
                    Log(reshade::log::level::info, "DLSS Frame Generation created (%ux%u, render %ux%u).", width, height, inputs.renderWidth, inputs.renderHeight);
                }
            }

            ID3D12Resource* readInputs[] = { _shared12.get(), depthShared.resource12.get(), motionShared.resource12.get(),
                inputs.hudlessValid ? hudlessShared.resource12.get() : nullptr };
            // On the frame the feature is created, only the real frame is shown.
            bool evaluate = generate && fgHandle && !createdThisFrame;
            if (evaluate) {
                for (ID3D12Resource* resource : readInputs) {
                    if (resource) {
                        Transition(list, resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    }
                }

                NVSDK_NGX_D3D12_DLSSG_Eval_Params evalParams = {};
                evalParams.pBackbuffer = _shared12.get();
                evalParams.pDepth = depthShared.resource12.get();
                evalParams.pMVecs = motionShared.resource12.get();
                evalParams.pHudless = inputs.hudlessValid ? hudlessShared.resource12.get() : nullptr;
                evalParams.pOutputInterpFrame = slot.generated.get();
                evalParams.pOutputRealFrame = slot.real.get();

                NVSDK_NGX_DLSSG_Opt_Eval_Params optParams = {};
                optParams.multiFrameCount = 1;
                optParams.multiFrameIndex = 1;
                CopyMatrixForDLSS(inputs.viewToClip, optParams.cameraViewToClip);
                CopyMatrixForDLSS(inputs.clipToView, optParams.clipToCameraView);
                CopyMatrixForDLSS(inputs.clipToPrevClip, optParams.clipToPrevClip);
                CopyMatrixForDLSS(inputs.prevClipToClip, optParams.prevClipToClip);
                for (int i = 0; i < 4; ++i) {
                    optParams.clipToLensClip[i][i] = 1.0f;
                }
                optParams.mvecScale[0] = 1.0f;
                optParams.mvecScale[1] = 1.0f;
                optParams.cameraUp[1] = 1.0f;
                optParams.cameraRight[0] = 1.0f;
                optParams.cameraFwd[2] = 1.0f;
                optParams.cameraNear = inputs.nearPlane;
                optParams.cameraFar = inputs.farPlane;
                optParams.cameraFOV = inputs.fov;
                optParams.cameraAspectRatio = inputs.aspect;
                optParams.colorBuffersHDR = false;
                optParams.depthInverted = inputs.depthInverted;
                optParams.cameraMotionIncluded = true;
                optParams.reset = !_frameGenerationWasRunning;
                optParams.menuDetectionEnabled = true;

                generated = NVSDK_NGX_SUCCEED(NGX_D3D12_EVALUATE_DLSSG(list, fgHandle, fgParameters, &evalParams, &optParams));

                for (ID3D12Resource* resource : readInputs) {
                    if (resource) {
                        Transition(list, resource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
                    }
                }
            }
            if (!generated) {
                // Keep a copy of the real frame so the presenter can still show it.
                Transition(list, slot.real.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
                list->CopyResource(slot.real.get(), _shared12.get());
                Transition(list, slot.real.get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }

            list->Close();
            ID3D12CommandList* lists[] = { list };
            queue12->ExecuteCommandLists(1, lists);
            workDone = SignalQueueLocked();
            _renderRing.allocatorFence[ringIndex] = workDone;
        }
        // DirectX 11 may overwrite the back buffer, depth and motion vectors once DX12 read them.
        context11->Wait(fence12_11.get(), workDone);
        submitted = workDone;

        if (generated) {
            frameGenerationRunning = true;
            frameGenerationStatus = "running (2x)";
            statGenerated++;
        }
        else {
            statNotGenerated++;
            if (generate && createdThisFrame) {
                statLastSkipReason = "Frame Generation was (re)created this frame";
            }
            else if (generate && fgHandle) {
                statLastSkipReason = "evaluation failed";
                frameGenerationStatus = "failed: DLSS Frame Generation evaluation failed";
                Log(reshade::log::level::error, "DLSS Frame Generation evaluation failed; turning it off for this session.");
                _frameGenerationFailed = true;
            }
        }
        _frameGenerationWasRunning = generated;
        _lastGenerated = generated;

        {
            std::lock_guard<std::mutex> lock(_slotMutex);
            slot.queued = true;
        }
        {
            std::lock_guard<std::mutex> lock(_jobMutex);
            _jobs.push_back(Job{ slotIndex, generated, syncInterval, flags, _frameInterval });
        }
        _jobCv.notify_all();
        return true;
    }

    // Copies a texture (UNORDERED_ACCESS state) into the current back buffer and presents it.
    uint64_t PresentTexture(ID3D12Resource* source, UINT syncInterval, UINT flags) {
        std::lock_guard<std::mutex> lock(queueMutex);
        int ringIndex = BeginCommands(_presentRing);
        ID3D12GraphicsCommandList* list = _presentRing.list.get();
        UINT index = _real->GetCurrentBackBufferIndex();
        ID3D12Resource* backBuffer = _backBuffers12[index < _backBuffers12.size() ? index : 0].get();
        Transition(list, backBuffer, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
        Transition(list, source, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyResource(backBuffer, source);
        Transition(list, source, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Transition(list, backBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
        list->Close();
        ID3D12CommandList* lists[] = { list };
        queue12->ExecuteCommandLists(1, lists);

        HRESULT hr = _real->Present(syncInterval, PresentFlags(syncInterval, flags));
        if (FAILED(hr) && hr != DXGI_ERROR_WAS_STILL_DRAWING) {
            Log(reshade::log::level::warning, "Present failed (0x%08X).", (unsigned)hr);
        }
        uint64_t done = SignalQueueLocked();
        _presentRing.allocatorFence[ringIndex] = done;
        return done;
    }

    bool JobWaiting() {
        std::lock_guard<std::mutex> lock(_jobMutex);
        return _quit || !_jobs.empty();
    }

    // Waits until 'target' (QPC ticks) with ~1 ms precision, or until the game hands over a new frame.
    void WaitUntil(const LARGE_INTEGER& target) {
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
        thread_local HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        for (;;) {
            if (JobWaiting()) {
                return;
            }
            LARGE_INTEGER now;
            QueryPerformanceCounter(&now);
            double remaining = QpcSeconds(now, target, _qpcFrequency);
            if (remaining <= 0.0) {
                return;
            }
            if (remaining > 0.0015 && timer) {
                double step = remaining - 0.0005 < 0.001 ? remaining - 0.0005 : 0.001;
                LARGE_INTEGER due;
                due.QuadPart = -(LONGLONG)(step * 10000000.0); // relative, 100 ns units
                if (SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) {
                    WaitForSingleObject(timer, 10);
                    continue;
                }
            }
            YieldProcessor();
        }
    }

    // Presenter thread: shows the generated frame, then the real frame half a frame later.
    void PresenterLoop() {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(_jobMutex);
                _jobCv.wait(lock, [&]() { return _quit || !_jobs.empty(); });
                if (_jobs.empty()) {
                    break; // quitting
                }
                job = _jobs.front();
                _jobs.pop_front();
            }
            Slot& slot = _slots[job.slot];

            if (job.hasGenerated) {
                PresentTexture(slot.generated.get(), job.syncInterval, job.flags);
                if (job.syncInterval == 0 && job.frameInterval > 0.0) {
                    // Pace: the real frame goes half a frame after the generated one,
                    // unless the game already finished its next frame.
                    double wait = job.frameInterval * 0.5;
                    if (wait > 0.05) {
                        wait = 0.05;
                    }
                    LARGE_INTEGER target;
                    QueryPerformanceCounter(&target);
                    target.QuadPart += (LONGLONG)(wait * (double)_qpcFrequency.QuadPart);
                    WaitUntil(target);
                }
            }
            uint64_t done = PresentTexture(slot.real.get(), job.syncInterval, job.flags);

            {
                std::lock_guard<std::mutex> lock(_slotMutex);
                slot.freeFence = done;
                slot.queued = false;
            }
            _slotCv.notify_all();
        }
    }

    // Waits until the presenter thread has shown everything handed to it.
    void DrainPresenter() {
        std::unique_lock<std::mutex> lock(_slotMutex);
        _slotCv.wait_for(lock, std::chrono::milliseconds(1000), [&]() {
            for (const Slot& slot : _slots) {
                if (slot.queued) {
                    return false;
                }
            }
            return true;
        });
    }

    volatile LONG _ref = 1;
    com_ptr<IDXGISwapChain3> _real;
    com_ptr<IDXGISwapChain4> _real4;
    com_ptr<IUnknown> _gameDevice;
    DXGI_SWAP_CHAIN_DESC _gameDesc;
    UINT _realBufferCount;
    UINT _realFlags;

    com_ptr<ID3D12Resource> _shared12;
    com_ptr<ID3D11Texture2D> _shared11;
    std::vector<com_ptr<ID3D12Resource>> _backBuffers12;

    CommandRing _renderRing;  // used by the game's render thread
    CommandRing _presentRing; // used by the presenter thread

    Slot _slots[kSlots];
    int _nextSlot = 0;
    std::mutex _slotMutex;
    std::condition_variable _slotCv;

    std::thread _presenter;
    std::mutex _jobMutex;
    std::condition_variable _jobCv;
    std::deque<Job> _jobs;
    bool _quit = false;

    LARGE_INTEGER _qpcFrequency = {};
    LARGE_INTEGER _lastFrameTime = {};
    double _frameInterval = 0.0;
    bool _frameGenerationWasRunning = false;
    bool _frameGenerationFailed = false;

    // Latency
    static constexpr int kHistory = 4;
    uint64_t _submittedHistory[kHistory] = {};
    uint64_t _submittedCount = 0;
    bool _reflexApplied = false;
    bool _reflexOn = false;
    bool _reflexBoostOn = false;
    uint64_t _reflexFrameId = 1;

    // Statistics
    LARGE_INTEGER _windowStart = {};
    LARGE_INTEGER _lastSlowLog = {};
    float _windowLongestMs = 0.0f;
    double _lastSlotWaitMs = 0.0;
    double _lastLimitWaitMs = 0.0;
    double _lastReflexMs = 0.0;
    bool _lastGenerated = false;
};

// Replaces IDXGIFactory::CreateSwapChain (vtable slot 10). Runs before ReShade's own hook.
inline HRESULT STDMETHODCALLTYPE CreateSwapChainHook(IDXGIFactory* factory, IUnknown* device, DXGI_SWAP_CHAIN_DESC* desc, IDXGISwapChain** swapChain) {
    com_ptr<ID3D11Device> gameDevice;
    if (!bridgeRequested || !device || !desc || !swapChain ||
        FAILED(device->QueryInterface(&gameDevice)) || !gameDevice) {
        return previousCreateSwapChain(factory, device, desc, swapChain);
    }

    if (!InitDX12()) {
        Log(reshade::log::level::warning, "Not bridging: %s", bridgeStatus.load());
        return previousCreateSwapChain(factory, device, desc, swapChain);
    }

    DXGI_FORMAT realFormat = FlipCompatibleFormat(desc->BufferDesc.Format);
    if (realFormat == DXGI_FORMAT_UNKNOWN) {
        bridgeStatus = "failed: unsupported back buffer format";
        Log(reshade::log::level::warning, "Not bridging: back buffer format %u is not supported.", (unsigned)desc->BufferDesc.Format);
        return previousCreateSwapChain(factory, device, desc, swapChain);
    }

    DXGI_SWAP_CHAIN_DESC realDesc = *desc;
    realDesc.BufferDesc.Format = realFormat;
    realDesc.SampleDesc.Count = 1;
    realDesc.SampleDesc.Quality = 0;
    realDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    realDesc.BufferCount = 3;
    realDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    realDesc.Flags = desc->Flags | (tearingSupported ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0);

    IDXGISwapChain* realSwapChain = nullptr;
    HRESULT hr = previousCreateSwapChain(factory, queue12.get(), &realDesc, &realSwapChain);
    com_ptr<IDXGISwapChain3> realSwapChain3;
    if (SUCCEEDED(hr) && realSwapChain) {
        realSwapChain->QueryInterface(&realSwapChain3);
        realSwapChain->Release();
    }
    if (FAILED(hr) || !realSwapChain3) {
        bridgeStatus = "failed: DirectX 12 swap chain could not be created";
        Log(reshade::log::level::warning, "Not bridging: DirectX 12 swap chain creation failed (0x%08X).", (unsigned)hr);
        return previousCreateSwapChain(factory, device, desc, swapChain);
    }

    BridgeSwapChain* bridge = new BridgeSwapChain(realSwapChain3.get(), device, *desc, realDesc.BufferCount, realDesc.Flags);
    if (!bridge->Initialize()) {
        bridge->Release();
        bridgeStatus = "failed: shared back buffer could not be created";
        Log(reshade::log::level::warning, "Not bridging: shared back buffer could not be created.");
        return previousCreateSwapChain(factory, device, desc, swapChain);
    }

    *swapChain = bridge;
    bridgeActive = true;
    bridgeStatus = "active (game presents through DirectX 12)";
    Log(reshade::log::level::info, "Game swap chain bridged to DirectX 12 (%ux%u, format %u).",
        desc->BufferDesc.Width, desc->BufferDesc.Height, (unsigned)desc->BufferDesc.Format);
    return S_OK;
}

// Installs the swap chain hook. Called once the game's D3D11 device exists
// (the game creates its swap chain right after that).
inline void InstallHook(ID3D11Device* nativeDevice) {
    if (!bridgeRequested || hookedVtableSlot) {
        return;
    }
    device11 = nativeDevice;

    typedef HRESULT(WINAPI* CreateFactory1Fn)(REFIID, void**);
    HMODULE dxgiModule = GetModuleHandleW(L"dxgi.dll");
    CreateFactory1Fn createFactory = dxgiModule ? (CreateFactory1Fn)GetProcAddress(dxgiModule, "CreateDXGIFactory1") : nullptr;
    com_ptr<IDXGIFactory> factory;
    if (!createFactory || FAILED(createFactory(__uuidof(IDXGIFactory), (void**)&factory)) || !factory) {
        bridgeStatus = "failed: no DXGI factory";
        return;
    }

    void** vtable = *(void***)factory.get();
    void** slot = &vtable[10]; // IDXGIFactory::CreateSwapChain
    DWORD oldProtection = 0;
    if (!VirtualProtect(slot, sizeof(void*), PAGE_EXECUTE_READWRITE, &oldProtection)) {
        bridgeStatus = "failed: could not hook swap chain creation";
        return;
    }
    previousCreateSwapChain = (CreateSwapChainFn)*slot;
    *slot = (void*)&CreateSwapChainHook;
    VirtualProtect(slot, sizeof(void*), oldProtection, &oldProtection);
    hookedVtableSlot = slot;
    bridgeStatus = "waiting for the game's swap chain";
    Log(reshade::log::level::info, "Swap chain hook installed.");
}

inline void UninstallHook() {
    if (!hookedVtableSlot) {
        return;
    }
    DWORD oldProtection = 0;
    if (*hookedVtableSlot == (void*)&CreateSwapChainHook &&
        VirtualProtect(hookedVtableSlot, sizeof(void*), PAGE_EXECUTE_READWRITE, &oldProtection)) {
        *hookedVtableSlot = (void*)previousCreateSwapChain;
        VirtualProtect(hookedVtableSlot, sizeof(void*), oldProtection, &oldProtection);
    }
    hookedVtableSlot = nullptr;
}

} // namespace fg
