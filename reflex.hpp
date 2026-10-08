#pragma once
// ---------------------------------------------------------------------------
// NVIDIA Reflex (low latency mode + boost) through NVAPI, loaded at runtime.
//
// The structures and function IDs below are copied from NVIDIA's official NVAPI
// headers (github.com/NVIDIA/nvapi: nvapi.h and nvapi_interface.h), so no extra
// library is needed: nvapi64.dll is part of every NVIDIA driver.
//
//   NvAPI_Initialize            0x0150E828
//   NvAPI_D3D_SetSleepMode      0xAC1CA9E0
//   NvAPI_D3D_Sleep             0x852CD1D2
//   NvAPI_D3D_SetLatencyMarker  0xD9984C05
// ---------------------------------------------------------------------------
#include <windows.h>
#include <atomic>
#include <cstdint>

namespace reflex {

typedef uint8_t NvBool;
typedef int NvAPI_Status; // 0 = NVAPI_OK

typedef enum {
    SIMULATION_START = 0,
    SIMULATION_END = 1,
    RENDERSUBMIT_START = 2,
    RENDERSUBMIT_END = 3,
    PRESENT_START = 4,
    PRESENT_END = 5,
    INPUT_SAMPLE = 6,
} NV_LATENCY_MARKER_TYPE;

struct NV_SET_SLEEP_MODE_PARAMS_V1 {
    uint32_t version;
    NvBool bLowLatencyMode;
    NvBool bLowLatencyBoost;
    uint32_t minimumIntervalUs;
    NvBool bUseMarkersToOptimize;
    NvBool bUseMinQueueTime;
    uint8_t rsvd[30];
};

struct NV_LATENCY_MARKER_PARAMS_V1 {
    uint32_t version;
    uint64_t frameID;
    NV_LATENCY_MARKER_TYPE markerType;
    uint64_t rsvd0;
    uint8_t rsvd[56];
};

// MAKE_NVAPI_VERSION(type, 1) = sizeof(type) | (1 << 16)
constexpr uint32_t kSleepModeParamsVersion = (uint32_t)(sizeof(NV_SET_SLEEP_MODE_PARAMS_V1) | (1u << 16));
constexpr uint32_t kLatencyMarkerParamsVersion = (uint32_t)(sizeof(NV_LATENCY_MARKER_PARAMS_V1) | (1u << 16));

typedef void* (__cdecl* QueryInterfaceFn)(uint32_t id);
typedef NvAPI_Status(__cdecl* InitializeFn)();
typedef NvAPI_Status(__cdecl* SetSleepModeFn)(IUnknown* device, NV_SET_SLEEP_MODE_PARAMS_V1* params);
typedef NvAPI_Status(__cdecl* SleepFn)(IUnknown* device);
typedef NvAPI_Status(__cdecl* SetLatencyMarkerFn)(IUnknown* device, NV_LATENCY_MARKER_PARAMS_V1* params);

inline bool loadTried = false;
inline bool loaded = false;
inline SetSleepModeFn setSleepMode = nullptr;
inline SleepFn sleep = nullptr;
inline SetLatencyMarkerFn setLatencyMarker = nullptr;
inline std::atomic<const char*> status = "off";

inline bool Load() {
    if (loadTried) {
        return loaded;
    }
    loadTried = true;
    HMODULE module = LoadLibraryW(L"nvapi64.dll");
    QueryInterfaceFn query = module ? (QueryInterfaceFn)(void*)GetProcAddress(module, "nvapi_QueryInterface") : nullptr;
    if (!query) {
        status = "unavailable: nvapi64.dll not found";
        return false;
    }
    InitializeFn initialize = (InitializeFn)query(0x0150E828);
    setSleepMode = (SetSleepModeFn)query(0xAC1CA9E0);
    sleep = (SleepFn)query(0x852CD1D2);
    setLatencyMarker = (SetLatencyMarkerFn)query(0xD9984C05);
    if (!initialize || !setSleepMode || !sleep || initialize() != 0) {
        status = "unavailable: NVAPI could not start";
        return false;
    }
    loaded = true;
    return true;
}

// Applies the settings to the device (the settings persist, so only call on change).
inline bool SetMode(IUnknown* device, bool lowLatency, bool boost) {
    if (!device || !Load()) {
        return false;
    }
    NV_SET_SLEEP_MODE_PARAMS_V1 params = {};
    params.version = kSleepModeParamsVersion;
    params.bLowLatencyMode = lowLatency ? 1 : 0;
    params.bLowLatencyBoost = boost ? 1 : 0;
    params.minimumIntervalUs = 0;
    params.bUseMarkersToOptimize = 0;
    NvAPI_Status result = setSleepMode(device, &params);
    if (result != 0) {
        status = "error: NvAPI_D3D_SetSleepMode failed";
        return false;
    }
    status = !lowLatency ? "off" : (boost ? "on + boost" : "on");
    return true;
}

// Must be called exactly once per frame, at the start of the frame.
inline void Sleep(IUnknown* device) {
    if (loaded && sleep && device) {
        sleep(device);
    }
}

inline void Marker(IUnknown* device, uint64_t frameId, NV_LATENCY_MARKER_TYPE type) {
    if (loaded && setLatencyMarker && device) {
        NV_LATENCY_MARKER_PARAMS_V1 params = {};
        params.version = kLatencyMarkerParamsVersion;
        params.frameID = frameId;
        params.markerType = type;
        setLatencyMarker(device, &params);
    }
}

} // namespace reflex
