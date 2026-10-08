#pragma once
// ---------------------------------------------------------------------------
// DX11 -> DX12 presentation bridge (Frame Generation, step 1).
//
// DLSS Frame Generation only runs on DirectX 12, but FFXV is a DirectX 11 game.
// When enabled, this bridge intercepts the game's swap chain creation:
//   - the real swap chain is created on a DirectX 12 queue (through ReShade,
//     so ReShade still draws its menu and effects on it),
//   - the game receives a stand-in swap chain whose back buffer is a texture
//     shared between DirectX 11 and DirectX 12,
//   - on every Present, the game's finished frame is copied into the DX12 back
//     buffer and presented.
// The game keeps rendering in DirectX 11 exactly as before. Frame Generation
// (step 2) will run between the copy and the Present.
//
// Must be included after <reshade.hpp>, <com_ptr.hpp> and <d3d11.h>.
// ---------------------------------------------------------------------------
#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <atomic>
#include <cstdarg>
#include <vector>

namespace fg {

inline bool bridgeRequested = false;          // config [DLAA] FrameGenerationBridge, read at startup
inline std::atomic<bool> bridgeActive = false; // a bridged swap chain currently exists
inline std::atomic<const char*> bridgeStatus = "off";
inline bool inGameFramePresent = false;       // true while the bridge presents a frame the game rendered

inline com_ptr<ID3D11Device> device11;         // native D3D11 device of the game
inline com_ptr<ID3D11Device5> device11_5;
inline com_ptr<ID3D11DeviceContext4> context11;
inline com_ptr<ID3D12Device> device12;         // created through ReShade (proxy)
inline com_ptr<ID3D12CommandQueue> queue12;
inline com_ptr<ID3D12Fence> fence12;           // shared with D3D11 (fence11 is the same fence)
inline com_ptr<ID3D11Fence> fence11;
inline HANDLE fenceEvent = nullptr;
inline uint64_t fenceValue = 0;
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

inline bool WaitForFence(uint64_t value, DWORD timeoutMs = 2000) {
    if (!fence12 || fence12->GetCompletedValue() >= value) {
        return true;
    }
    if (FAILED(fence12->SetEventOnCompletion(value, fenceEvent))) {
        return false;
    }
    if (WaitForSingleObject(fenceEvent, timeoutMs) != WAIT_OBJECT_0) {
        Log(reshade::log::level::warning, "GPU wait timed out (fence %llu).", (unsigned long long)value);
        return false;
    }
    return true;
}

// Waits until both the D3D11 and the D3D12 side finished all submitted work.
inline void FlushGpu() {
    if (!context11 || !queue12 || !fence11) {
        return;
    }
    uint64_t d3d11Done = ++fenceValue;
    context11->Signal(fence11.get(), d3d11Done);
    context11->Flush();
    queue12->Wait(fence12.get(), d3d11Done);
    uint64_t allDone = ++fenceValue;
    queue12->Signal(fence12.get(), allDone);
    WaitForFence(allDone, 5000);
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

    HANDLE sharedFence = nullptr;
    if (FAILED(device12->CreateFence(0, D3D12_FENCE_FLAG_SHARED, __uuidof(ID3D12Fence), (void**)&fence12)) ||
        FAILED(device12->CreateSharedHandle(fence12.get(), nullptr, GENERIC_ALL, nullptr, &sharedFence)) ||
        FAILED(device11_5->OpenSharedFence(sharedFence, __uuidof(ID3D11Fence), (void**)&fence11)) || !fence11) {
        if (sharedFence) {
            CloseHandle(sharedFence);
        }
        fence11.reset();
        fence12.reset();
        queue12.reset();
        device12.reset();
        bridgeStatus = "failed: could not share a fence between DX11 and DX12";
        return false;
    }
    CloseHandle(sharedFence);
    fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);

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

// ---------------------------------------------------------------------------
// The swap chain the game sees.
// ---------------------------------------------------------------------------
class BridgeSwapChain final : public IDXGISwapChain4 {
public:
    BridgeSwapChain(IDXGISwapChain3* real, IUnknown* gameDevice, const DXGI_SWAP_CHAIN_DESC& gameDesc, UINT realBufferCount, UINT realFlags)
        : _real(real), _gameDevice(gameDevice), _gameDesc(gameDesc), _realBufferCount(realBufferCount), _realFlags(realFlags) {
        _real->QueryInterface(&_real4);
    }

    bool Initialize() {
        for (int i = 0; i < kFrames; ++i) {
            if (FAILED(device12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), (void**)&_allocators[i]))) {
                return false;
            }
            _allocatorFence[i] = 0;
        }
        if (FAILED(device12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, _allocators[0].get(), nullptr, __uuidof(ID3D12GraphicsCommandList), (void**)&_commandList))) {
            return false;
        }
        _commandList->Close();
        return CreateBuffers();
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
    static constexpr int kFrames = 3;

    ~BridgeSwapChain() {
        FlushGpu();
        ReleaseBuffers();
        bridgeActive = false;
        bridgeStatus = "off (swap chain released)";
        Log(reshade::log::level::info, "Bridged swap chain released.");
    }

    void ReleaseBuffers() {
        _backBuffers12.clear();
        _shared11.reset();
        _shared12.reset();
    }

    bool CreateBuffers() {
        DXGI_SWAP_CHAIN_DESC realDesc = {};
        if (FAILED(_real->GetDesc(&realDesc))) {
            return false;
        }
        UINT width = realDesc.BufferDesc.Width;
        UINT height = realDesc.BufferDesc.Height;

        D3D12_HEAP_PROPERTIES heap = {};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = width;
        desc.Height = height;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = _gameDesc.BufferDesc.Format;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
        if (FAILED(device12->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                __uuidof(ID3D12Resource), (void**)&_shared12))) {
            Log(reshade::log::level::error, "Could not create the shared back buffer (%ux%u format %u).", width, height, (unsigned)desc.Format);
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
        return true;
    }

    HRESULT Resize(UINT bufferCount, UINT width, UINT height, DXGI_FORMAT format, UINT flags) {
        FlushGpu();
        ReleaseBuffers();

        DXGI_FORMAT gameFormat = format != DXGI_FORMAT_UNKNOWN ? format : _gameDesc.BufferDesc.Format;
        DXGI_FORMAT realFormat = FlipCompatibleFormat(gameFormat);
        if (realFormat == DXGI_FORMAT_UNKNOWN) {
            Log(reshade::log::level::error, "ResizeBuffers with unsupported format %u.", (unsigned)gameFormat);
            return DXGI_ERROR_INVALID_CALL;
        }
        _realFlags = flags | (tearingSupported ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0);
        HRESULT hr = _real->ResizeBuffers(_realBufferCount, width, height, realFormat, _realFlags);
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
        return S_OK;
    }

    HRESULT PresentFrame(UINT syncInterval, UINT flags) {
        if (flags & DXGI_PRESENT_TEST) {
            return _real->Present(syncInterval, DXGI_PRESENT_TEST);
        }
        if (!_shared12 || _backBuffers12.empty()) {
            return DXGI_ERROR_INVALID_CALL;
        }

        // 1. The game finished drawing into the shared back buffer (DX11).
        uint64_t gameFrameDone = ++fenceValue;
        context11->Signal(fence11.get(), gameFrameDone);
        context11->Flush();

        // 2. DX12 waits for it, then copies it into the current DX12 back buffer.
        queue12->Wait(fence12.get(), gameFrameDone);

        const int slot = (int)(_frameCounter++ % kFrames);
        WaitForFence(_allocatorFence[slot]);
        _allocators[slot]->Reset();
        _commandList->Reset(_allocators[slot].get(), nullptr);

        UINT index = _real->GetCurrentBackBufferIndex();
        ID3D12Resource* backBuffer = _backBuffers12[index < _backBuffers12.size() ? index : 0].get();

        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = backBuffer;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        _commandList->ResourceBarrier(1, &barrier);
        _commandList->CopyResource(backBuffer, _shared12.get());
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
        _commandList->ResourceBarrier(1, &barrier);
        _commandList->Close();
        ID3D12CommandList* lists[] = { _commandList.get() };
        queue12->ExecuteCommandLists(1, lists);

        // 3. Present (ReShade draws its menu/effects here).
        UINT presentFlags = flags & (DXGI_PRESENT_DO_NOT_WAIT | DXGI_PRESENT_RESTART);
        if (syncInterval == 0 && tearingSupported) {
            BOOL fullscreen = FALSE;
            _real->GetFullscreenState(&fullscreen, nullptr);
            if (!fullscreen) {
                presentFlags |= DXGI_PRESENT_ALLOW_TEARING;
            }
        }
        inGameFramePresent = true;
        HRESULT hr = _real->Present(syncInterval, presentFlags);
        inGameFramePresent = false;

        // 4. The game may only draw into the shared back buffer again after the copy finished.
        uint64_t copyDone = ++fenceValue;
        queue12->Signal(fence12.get(), copyDone);
        _allocatorFence[slot] = copyDone;
        context11->Wait(fence11.get(), copyDone);

        if (FAILED(hr) && hr != DXGI_ERROR_WAS_STILL_DRAWING) {
            Log(reshade::log::level::warning, "Present failed (0x%08X).", (unsigned)hr);
        }
        return hr;
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
    com_ptr<ID3D12CommandAllocator> _allocators[kFrames];
    uint64_t _allocatorFence[kFrames] = {};
    com_ptr<ID3D12GraphicsCommandList> _commandList;
    uint64_t _frameCounter = 0;
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
