#pragma once
// ---------------------------------------------------------------------------
// Multi Frame Generation (3x / 4x) on RTX 40 (Ada).
//
// NVIDIA's nvngx_dlssg.dll only offers more than one generated frame on RTX 50.
// Two patches, applied to the DLL in memory only (the file on disk is never changed,
// because NGX checks its signature when it loads it):
//
// 1. Architecture gates: every "cmp <reg>, 0x1b0" (0x1b0 = RTX 50 architecture id) is
//    changed to compare against 0x190 (RTX 40). One of them decides the reported
//    MultiFrameCountMax, another one the path used while generating.
//    "mov <reg>, 0x1b0" (a lookup table) is left alone.
//
// 2. Temporal fix: the interpolation kernel blends the two real frames with a fixed
//    0.5, so 3x/4x would show copies of the same half-way frame. Its PTX is rewritten
//    to use the kernel's own time parameter, and the precompiled sm_89 binary is
//    dropped so the driver compiles the edited PTX.
//
// Credits: this is a port of the MIT-licensed work of
//   dashdogy        - RTX40MFG-Unlock (diagnosis, gates, PTX rewrite and truncation method)
//   ImDreamt        - MFGAdaUnlock-RenoDx (ReShade add-on adaptation)
//   mavismmg        - MFGAdaUnlock-RenoDx fork (provider profiles, safety checks)
// https://github.com/mavismmg/MFGAdaUnlock-RenoDx (src/addons/mfgunlock/addon.cpp, midpoint.hpp)
//
// Only providers whose kernel matches a known profile get the temporal fix. The
// FFXV-DLAA build ships nvngx_dlssg.dll 310.5.0, which matches the first profile.
// ---------------------------------------------------------------------------
#include <windows.h>
#include <tlhelp32.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <string>
#include <vector>

namespace mfg {

// Status for the menu
inline std::atomic<const char*> status = "not checked yet";
inline std::atomic<int> gatesPatched = 0;
inline std::atomic<bool> temporalFixApplied = false;
inline char temporalDetail[200] = "";

namespace internal {

constexpr uint32_t kFatbinMagic = 0xBA55ED50u;
constexpr size_t kOuterHeader = 16;
constexpr uint32_t kPtxKind = 1;
constexpr uint32_t kAdaArch = 89;
constexpr uint64_t kUncompressedFlags = 0x41;

struct TemporalProfile {
    size_t ptxBytes;
    const char* entryName;
    const char* descriptorName;
    ptrdiff_t entryNameOffset;
    ptrdiff_t descriptorNameOffset;
};

constexpr TemporalProfile kTemporalProfiles[] = {
    { 99362, "main_kernel", "dlfg_kernel", 0x10, 0x28 },                              // up to 310.8
    { 99626, "Kernel_EstimateIntermMvecsScatter", "EstimateIntermMvecsScatter", 0x10, -0x08 }, // 310.9
};

constexpr size_t kExpectedMidpoints = 104;
constexpr char kJoinLabel[] = "$L__BB0_3:";
constexpr char kMidpointBits[] = "0f3F000000";
constexpr char kMulPrefix[] = "mul.ftz.f32 ";
constexpr char kCurrToPrev[] = "%f136";
constexpr char kPrevToCurr[] = "%f134";

inline uint16_t ReadU16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, sizeof(v)); return v; }
inline uint32_t ReadU32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, sizeof(v)); return v; }
inline uint64_t ReadU64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, sizeof(v)); return v; }

// Plain LZ4 block format.
inline bool Lz4BlockDecompress(const uint8_t* src, size_t srcSize, uint8_t* dst, size_t dstSize) {
    size_t in = 0;
    size_t out = 0;
    while (in < srcSize) {
        const uint8_t token = src[in++];
        size_t literals = token >> 4;
        if (literals == 15) {
            uint8_t ext = 0;
            do {
                if (in >= srcSize) return false;
                ext = src[in++];
                literals += ext;
            } while (ext == 0xFF);
        }
        if (literals > srcSize - in || literals > dstSize - out) return false;
        std::memcpy(dst + out, src + in, literals);
        in += literals;
        out += literals;
        if (in == srcSize) break;
        if (srcSize - in < 2) return false;
        const size_t back = (size_t)src[in] | ((size_t)src[in + 1] << 8);
        in += 2;
        if (back == 0 || back > out) return false;
        size_t match = 4 + (token & 0x0F);
        if ((token & 0x0F) == 15) {
            uint8_t ext = 0;
            do {
                if (in >= srcSize) return false;
                ext = src[in++];
                match += ext;
            } while (ext == 0xFF);
        }
        if (match > dstSize - out) return false;
        for (size_t i = 0; i < match; ++i) dst[out + i] = dst[out + i - back];
        out += match;
    }
    return in == srcSize && out == dstSize;
}

// Locates the sm_89 PTX entry inside a fatbin by walking its entry list.
inline bool FindAdaPtxEntry(const uint8_t* fat, size_t fatSize, size_t& entryOffset) {
    if (fatSize < kOuterHeader || ReadU32(fat) != kFatbinMagic) return false;
    if (ReadU16(fat + 6) != kOuterHeader) return false;
    const uint64_t declared = ReadU64(fat + 8);
    if (declared + kOuterHeader != fatSize) return false;
    size_t p = kOuterHeader;
    while (p + 64 <= fatSize) {
        const uint32_t kind = ReadU16(fat + p);
        const uint32_t hdr = ReadU32(fat + p + 4);
        const uint64_t payload = ReadU64(fat + p + 8);
        if (hdr < 64 || payload == 0) return false;
        if (p + hdr + payload > fatSize) return false;
        if (kind == kPtxKind && ReadU32(fat + p + 28) == kAdaArch) {
            entryOffset = p;
            return true;
        }
        p += hdr + payload;
    }
    return false;
}

// Decompresses the Ada PTX, rewrites the blend weights, and re-emits a truncated
// fatbin that ends after it.
inline bool BuildTemporalFatbin(const uint8_t* fat, size_t fatSize, const TemporalProfile& profile,
    std::vector<uint8_t>& out, std::string& why) {
    size_t entry = 0;
    if (!FindAdaPtxEntry(fat, fatSize, entry)) { why = "no sm_89 PTX entry"; return false; }
    const uint32_t hdr = ReadU32(fat + entry + 4);
    const uint32_t compressed = ReadU32(fat + entry + 16);
    const uint64_t raw = ReadU64(fat + entry + 56);
    if (compressed == 0 || raw == 0 || raw > (8u << 20)) { why = "PTX entry is not compressed as expected"; return false; }
    if (raw != profile.ptxBytes) { why = "PTX size does not match the known kernel"; return false; }

    std::vector<uint8_t> ptx((size_t)raw);
    if (!Lz4BlockDecompress(fat + entry + hdr, compressed, ptx.data(), ptx.size())) { why = "LZ4 decompression failed"; return false; }

    const std::string entrySignature = std::string(".entry ") + profile.entryName + "(";
    const std::string parameterName = std::string(profile.entryName) + "_param_0";
    const std::string parameterSignature = std::string(".param .align 8 .b8 ") + parameterName + "[144]";
    const std::string ptxText((const char*)ptx.data(), ptx.size());
    if (ptxText.find(entrySignature) == std::string::npos || ptxText.find(parameterSignature) == std::string::npos ||
        ptxText.find(".reg .f32 %f<1362>;") == std::string::npos) {
        why = "temporal kernel signature changed";
        return false;
    }

    const char* begin = (const char*)ptx.data();
    const size_t n = ptx.size();
    const size_t labelLength = sizeof(kJoinLabel) - 1;
    size_t label = SIZE_MAX;
    for (size_t i = 0; i + labelLength <= n; ++i) {
        if (std::memcmp(begin + i, kJoinLabel, labelLength) != 0) continue;
        if (label != SIZE_MAX) { why = "join label is not unique"; return false; }
        label = i;
    }
    if (label == SIZE_MAX) { why = "join label not found"; return false; }
    size_t insertion = label + labelLength;
    while (insertion < n && begin[insertion] != '\n') ++insertion;
    if (insertion >= n) { why = "join label has no line end"; return false; }
    ++insertion;

    const size_t midLength = sizeof(kMidpointBits) - 1;
    const size_t mulLength = sizeof(kMulPrefix) - 1;
    std::vector<size_t> marks;
    for (size_t i = 0; i + midLength < n; ++i) {
        if (std::memcmp(begin + i, kMidpointBits, midLength) != 0) continue;
        if (begin[i + midLength] != ';') continue;
        size_t line = i;
        while (line > 0 && begin[line - 1] != '\n') --line;
        if (i - line < mulLength) continue;
        if (std::memcmp(begin + line, kMulPrefix, mulLength) != 0) continue;
        marks.push_back(i);
    }
    if (marks.size() != kExpectedMidpoints) { why = "unexpected number of midpoint multiplies"; return false; }
    if (marks.front() <= insertion) { why = "first midpoint precedes the injection point"; return false; }

    const std::string temporalInput =
        "ld.param.f32 %f134, [" + parameterName + "+32];\r\n"
        "mov.f32 %f135, 0f3F800000;\r\n"
        "sub.ftz.f32 %f136, %f135, %f134;\r\n";

    std::vector<uint8_t> patched;
    patched.reserve(n + temporalInput.size());
    auto append = [&patched](const void* p, size_t bytes) {
        const uint8_t* b = (const uint8_t*)p;
        patched.insert(patched.end(), b, b + bytes);
    };
    append(ptx.data(), insertion);
    append(temporalInput.data(), temporalInput.size());
    size_t source = insertion;
    const size_t half = kExpectedMidpoints / 2;
    for (size_t i = 0; i < marks.size(); ++i) {
        append(ptx.data() + source, marks[i] - source);
        append(i < half ? kCurrToPrev : kPrevToCurr, 5);
        source = marks[i] + midLength;
    }
    append(ptx.data() + source, n - source);

    const size_t padded = (patched.size() + 7) & ~(size_t)7;
    const size_t finalSize = entry + hdr + padded;
    out.assign(fat, fat + entry + hdr);
    out.resize(finalSize, 0);
    std::memcpy(out.data() + entry + hdr, patched.data(), patched.size());
    const uint64_t payload64 = padded;
    const uint32_t zero32 = 0;
    const uint64_t zero64 = 0;
    std::memcpy(out.data() + entry + 8, &payload64, sizeof(payload64));
    std::memcpy(out.data() + entry + 16, &zero32, sizeof(zero32));
    std::memcpy(out.data() + entry + 40, &kUncompressedFlags, sizeof(kUncompressedFlags));
    std::memcpy(out.data() + entry + 56, &zero64, sizeof(zero64));
    const uint64_t outer = finalSize - kOuterHeader;
    std::memcpy(out.data() + 8, &outer, sizeof(outer));
    return true;
}

inline const TemporalProfile* FindTemporalProfile(const uint8_t* fat, size_t fatSize) {
    size_t entry = 0;
    if (!FindAdaPtxEntry(fat, fatSize, entry)) return nullptr;
    const uint64_t raw = ReadU64(fat + entry + 56);
    for (const TemporalProfile& profile : kTemporalProfiles) {
        if (raw == profile.ptxBytes) return &profile;
    }
    return nullptr;
}

inline bool PointsToCString(const uint8_t* base, size_t imageSize, uint64_t value, const char* expected) {
    const uintptr_t start = (uintptr_t)base;
    if (value < start || value >= start + imageSize) return false;
    const size_t length = std::strlen(expected);
    if (value + length + 1 > start + imageSize) return false;
    return std::memcmp((const char*)value, expected, length + 1) == 0;
}

inline bool ReadRelativePointer(const uint8_t* base, size_t imageSize, const uint8_t* slot, ptrdiff_t displacement, uint64_t& value) {
    const uintptr_t start = (uintptr_t)base;
    const uintptr_t end = start + imageSize;
    const uintptr_t field = (uintptr_t)slot + displacement;
    if (field < start || field > end || sizeof(value) > end - field) return false;
    std::memcpy(&value, (const void*)field, sizeof(value));
    return true;
}

inline const IMAGE_NT_HEADERS64* NtHeaders(HMODULE module) {
    const uint8_t* base = (const uint8_t*)module;
    const IMAGE_DOS_HEADER* dos = (const IMAGE_DOS_HEADER*)base;
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) return nullptr;
    const IMAGE_NT_HEADERS64* nt = (const IMAGE_NT_HEADERS64*)(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE || nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) return nullptr;
    return nt;
}

inline bool ModuleContains(HMODULE module, const char* needle, size_t needleLength) {
    const IMAGE_NT_HEADERS64* nt = NtHeaders(module);
    if (!nt) return false;
    const uint8_t* base = (const uint8_t*)module;
    const IMAGE_SECTION_HEADER* section = IMAGE_FIRST_SECTION(nt);
    for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++section) {
        if ((section->Characteristics & IMAGE_SCN_MEM_READ) == 0) continue;
        const uint8_t* start = base + section->VirtualAddress;
        const size_t size = section->Misc.VirtualSize;
        for (size_t offset = 0; offset + needleLength <= size; ++offset) {
            if (std::memcmp(start + offset, needle, needleLength) == 0) return true;
        }
    }
    return false;
}

// The DLSS Frame Generation provider: by file name / driver store path, or by its NGX
// export plus the kernel name (renamed copies).
inline bool IsDlssgProvider(HMODULE module, const wchar_t* path) {
    if (module == nullptr) return false;
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCWSTR)&IsDlssgProvider, &self);
    if (module == self) return false; // this add-on contains the same strings
    std::wstring lower(path ? path : L"");
    for (wchar_t& c : lower) {
        if (c >= L'A' && c <= L'Z') c = (wchar_t)(c - L'A' + L'a');
    }
    if (lower.find(L"nvngx_dlssg") != std::wstring::npos || lower.find(L"\\models\\dlssg\\") != std::wstring::npos) {
        return true;
    }
    if (GetProcAddress(module, "NVSDK_NGX_D3D12_PopulateDeviceParameters_Impl") == nullptr) return false;
    static const char marker[] = "dlfg_kernel";
    return ModuleContains(module, marker, sizeof(marker) - 1);
}

struct PatchedModule {
    HMODULE module;
    std::vector<uint8_t*> gateSites;
    bool temporal;
};
inline std::vector<PatchedModule> patchedModules;
inline std::vector<HMODULE> rejectedModules;

inline int PatchArchGates(HMODULE module, std::vector<uint8_t*>& sites) {
    const IMAGE_NT_HEADERS64* nt = NtHeaders(module);
    if (!nt) return -1;
    uint8_t* base = (uint8_t*)module;
    std::vector<uint8_t*> found;
    const IMAGE_SECTION_HEADER* section = IMAGE_FIRST_SECTION(nt);
    for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++section) {
        if ((section->Characteristics & IMAGE_SCN_MEM_EXECUTE) == 0) continue;
        uint8_t* start = base + section->VirtualAddress;
        const size_t size = section->Misc.VirtualSize;
        for (size_t o = 0; o + 6 <= size; ++o) {
            // 3D imm32: cmp eax, 0x1b0
            if (start[o] == 0x3D && start[o + 1] == 0xB0 && start[o + 2] == 0x01 && start[o + 3] == 0x00 && start[o + 4] == 0x00) {
                found.push_back(start + o + 1);
                continue;
            }
            // 81 /7 imm32: cmp r32, 0x1b0
            if (start[o] == 0x81 && start[o + 1] >= 0xF8 && start[o + 2] == 0xB0 && start[o + 3] == 0x01 &&
                start[o + 4] == 0x00 && start[o + 5] == 0x00) {
                found.push_back(start + o + 2);
            }
        }
    }
    if (found.empty() || found.size() > 4) {
        return -(int)found.size() - 1; // unexpected layout: leave it alone
    }
    for (uint8_t* site : found) {
        DWORD oldProtection = 0;
        if (!VirtualProtect(site, 1, PAGE_EXECUTE_READWRITE, &oldProtection)) continue;
        *site = 0x90; // 0x1b0 (RTX 50) -> 0x190 (RTX 40)
        DWORD ignored = 0;
        VirtualProtect(site, 1, oldProtection, &ignored);
        FlushInstructionCache(GetCurrentProcess(), site, 1);
        sites.push_back(site);
    }
    return (int)sites.size();
}

// Redirects every descriptor of the temporal kernel to a rebuilt fatbin.
inline bool ApplyTemporalFix(HMODULE module, std::string& detail) {
    const IMAGE_NT_HEADERS64* nt = NtHeaders(module);
    if (!nt) { detail = "not a 64-bit module"; return false; }
    uint8_t* base = (uint8_t*)module;
    const size_t imageSize = nt->OptionalHeader.SizeOfImage;
    const uintptr_t start = (uintptr_t)base;

    std::vector<uint64_t*> slots;
    const uint8_t* fat = nullptr;
    size_t fatSize = 0;
    const TemporalProfile* selected = nullptr;

    const IMAGE_SECTION_HEADER* section = IMAGE_FIRST_SECTION(nt);
    for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++section) {
        if ((section->Characteristics & IMAGE_SCN_MEM_READ) == 0) continue;
        if ((section->Characteristics & IMAGE_SCN_MEM_EXECUTE) != 0) continue;
        uint8_t* data = base + section->VirtualAddress;
        const size_t size = section->Misc.VirtualSize;
        for (size_t offset = 0; offset + sizeof(uint64_t) <= size; offset += sizeof(uint64_t)) {
            uint64_t value = 0;
            std::memcpy(&value, data + offset, sizeof(value));
            if (value < start || imageSize < kOuterHeader || value > start + imageSize - kOuterHeader) continue;
            const uint8_t* candidate = (const uint8_t*)value;
            if (ReadU32(candidate) != kFatbinMagic) continue;

            const TemporalProfile* nameProfile = nullptr;
            for (const TemporalProfile& profile : kTemporalProfiles) {
                uint64_t entryName = 0;
                uint64_t descriptorName = 0;
                if (!ReadRelativePointer(base, imageSize, data + offset, profile.entryNameOffset, entryName) ||
                    !ReadRelativePointer(base, imageSize, data + offset, profile.descriptorNameOffset, descriptorName)) {
                    continue;
                }
                if (PointsToCString(base, imageSize, entryName, profile.entryName) &&
                    PointsToCString(base, imageSize, descriptorName, profile.descriptorName)) {
                    nameProfile = &profile;
                    break;
                }
            }
            if (!nameProfile) continue;

            const uint64_t declared = ReadU64(candidate + 8);
            const size_t total = (size_t)declared + kOuterHeader;
            if (total < 1024 || total > (16u << 20)) continue;
            if (total > start + imageSize - value) continue;
            const TemporalProfile* fatProfile = FindTemporalProfile(candidate, total);
            if (!fatProfile || fatProfile != nameProfile) continue;
            if (!fat) {
                fat = candidate;
                fatSize = total;
                selected = fatProfile;
            }
            else if (candidate != fat) {
                continue;
            }
            slots.push_back((uint64_t*)(data + offset));
        }
    }
    if (!fat || slots.empty() || !selected) { detail = "no known temporal kernel (other DLSS-G version)"; return false; }

    std::vector<uint8_t> rebuilt;
    std::string why;
    if (!BuildTemporalFatbin(fat, fatSize, *selected, rebuilt, why)) { detail = why; return false; }

    // Kept for the lifetime of the process: descriptors keep pointing to it.
    void* memory = VirtualAlloc(nullptr, rebuilt.size(), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!memory) { detail = "allocation failed"; return false; }
    std::memcpy(memory, rebuilt.data(), rebuilt.size());
    int written = 0;
    for (uint64_t* slot : slots) {
        DWORD oldProtection = 0;
        if (!VirtualProtect(slot, sizeof(uint64_t), PAGE_READWRITE, &oldProtection)) continue;
        *slot = (uint64_t)memory;
        DWORD ignored = 0;
        VirtualProtect(slot, sizeof(uint64_t), oldProtection, &ignored);
        written++;
    }
    if (written == 0) {
        VirtualFree(memory, 0, MEM_RELEASE);
        detail = "descriptor not writable";
        return false;
    }
    char text[160];
    snprintf(text, sizeof(text), "%d kernel descriptor(s) redirected (%s)", written, selected->descriptorName);
    detail = text;
    return true;
}

} // namespace internal

// Finds every loaded DLSS Frame Generation provider and patches the ones not patched yet.
// 'log' receives one line per action. Returns the number of modules newly patched.
template <typename LogFn>
inline int PatchLoadedProviders(LogFn&& log) {
    int newlyPatched = 0;
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
    if (snapshot == INVALID_HANDLE_VALUE) {
        return 0;
    }
    MODULEENTRY32W entry = {};
    entry.dwSize = sizeof(entry);
    std::vector<std::pair<HMODULE, std::wstring>> candidates;
    if (Module32FirstW(snapshot, &entry)) {
        do {
            candidates.push_back({ entry.hModule, entry.szExePath });
        } while (Module32NextW(snapshot, &entry));
    }
    CloseHandle(snapshot);

    // A module that was unloaded and loaded again (same address, fresh unpatched copy)
    // must be patched again: only count it as done if its gates are still changed.
    for (size_t i = 0; i < internal::patchedModules.size();) {
        const internal::PatchedModule& patched = internal::patchedModules[i];
        bool loaded = false;
        for (const auto& candidate : candidates) {
            if (candidate.first == patched.module) loaded = true;
        }
        if (!loaded || patched.gateSites.empty() || *patched.gateSites[0] != 0x90) {
            internal::patchedModules.erase(internal::patchedModules.begin() + i);
        }
        else {
            ++i;
        }
    }

    for (const auto& candidate : candidates) {
        HMODULE module = candidate.first;
        bool known = false;
        for (const internal::PatchedModule& patched : internal::patchedModules) {
            if (patched.module == module) known = true;
        }
        for (HMODULE rejected : internal::rejectedModules) {
            if (rejected == module) known = true;
        }
        if (known || !internal::IsDlssgProvider(module, candidate.second.c_str())) {
            continue;
        }
        char path[MAX_PATH] = {};
        WideCharToMultiByte(CP_UTF8, 0, candidate.second.c_str(), -1, path, MAX_PATH, nullptr, nullptr);

        internal::PatchedModule patched = { module, {}, false };
        int gates = internal::PatchArchGates(module, patched.gateSites);
        if (gates <= 0) {
            log("MFG unlock: %s: %s (%d gate(s) found); left alone.", path,
                gates == 0 ? "code could not be made writable" : "unexpected code layout", gates < 0 ? -gates - 1 : 0);
            internal::rejectedModules.push_back(module);
            status = "not possible: unknown nvngx_dlssg.dll version (see log)";
            continue;
        }
        std::string detail;
        patched.temporal = internal::ApplyTemporalFix(module, detail);
        snprintf(temporalDetail, sizeof(temporalDetail), "%s", detail.c_str());
        log("MFG unlock: %s: %d architecture gate(s) changed; temporal fix %s (%s).", path, gates,
            patched.temporal ? "applied" : "NOT applied", detail.c_str());
        gatesPatched += gates;
        if (patched.temporal) {
            temporalFixApplied = true;
        }
        status = patched.temporal ? "unlocked" : "unlocked, but 3x/4x frames may repeat (temporal fix not applied)";
        internal::patchedModules.push_back(std::move(patched));
        newlyPatched++;
    }
    return newlyPatched;
}

// True when a loaded provider is patched right now (checked in memory).
inline bool AnyProviderPatched() {
    for (const internal::PatchedModule& patched : internal::patchedModules) {
        if (!patched.gateSites.empty()) {
            HMODULE stillLoaded = nullptr;
            if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                    (LPCWSTR)patched.gateSites[0], &stillLoaded) && stillLoaded == patched.module &&
                *patched.gateSites[0] == 0x90) {
                return true;
            }
        }
    }
    return false;
}

} // namespace mfg
