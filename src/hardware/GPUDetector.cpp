#include "hardware/GPUDetector.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cwchar>

#if defined(_WIN32)
    #include <windows.h>
    #include <dxgi1_4.h>
    #pragma comment(lib, "dxgi.lib")
#else
    #include <cstdlib>
    #include <dirent.h>
    #include <fstream>
    #include <sstream>
    #include <string>
#endif

namespace agr {

namespace {

GPUVendor vendorFromId(uint32_t vendorId) {
    switch (vendorId) {
        case 0x8086: return GPUVendor::Intel;
        case 0x1002:
        case 0x1022: return GPUVendor::AMD;
        case 0x10DE: return GPUVendor::NVIDIA;
        default: return GPUVendor::Unknown;
    }
}

} // namespace

#if defined(_WIN32)

std::vector<GPUInfo> GPUDetector::detect() {
    std::vector<GPUInfo> gpus;

    IDXGIFactory1* factory = nullptr;
    HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory));
    if (FAILED(hr) || factory == nullptr) {
        return gpus; // No DXGI available; return empty rather than fabricating data.
    }

    IDXGIAdapter1* adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        if (SUCCEEDED(adapter->GetDesc1(&desc))) {
            // Skip the software rasterizer ("Microsoft Basic Render Driver").
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                GPUInfo info;

                // Wide -> UTF-8 via the Win32 API rather than the
                // deprecated <codecvt>/std::wstring_convert (removed in
                // C++26, deprecated since C++17). Behavior is identical.
                std::wstring wname(desc.Description);
                int utf8Len = WideCharToMultiByte(CP_UTF8, 0, wname.c_str(),
                                                   static_cast<int>(wname.size()),
                                                   nullptr, 0, nullptr, nullptr);
                std::string utf8Name(utf8Len, '\0');
                if (utf8Len > 0) {
                    WideCharToMultiByte(CP_UTF8, 0, wname.c_str(),
                                         static_cast<int>(wname.size()),
                                         utf8Name.data(), utf8Len, nullptr, nullptr);
                }
                info.name = utf8Name;

                info.vendor = vendorFromId(desc.VendorId);
                info.dedicated_vram_mb = static_cast<uint64_t>(desc.DedicatedVideoMemory / (1024ULL * 1024ULL));
                info.shared_system_memory_mb = static_cast<uint64_t>(desc.SharedSystemMemory / (1024ULL * 1024ULL));
                info.total_accessible_memory_mb = info.dedicated_vram_mb + info.shared_system_memory_mb;
                info.detected_via_native_api = true;

                // Heuristic (DXGI does not expose an explicit integrated/
                // dedicated flag): integrated GPUs typically report a small
                // dedicated aperture and rely mostly on shared system RAM.
                if (info.dedicated_vram_mb < 512 && info.shared_system_memory_mb > 0) {
                    info.kind = GPUKind::Integrated;
                } else {
                    info.kind = GPUKind::Dedicated;
                }

                gpus.push_back(info);
            }
        }
        adapter->Release();
        adapter = nullptr;
    }

    factory->Release();
    return gpus;
}

#else // Non-Windows dev-time fallback (NOT the Phase-1 target platform)

namespace {

std::string readFileTrimmed(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    std::string line;
    std::getline(f, line);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' ')) {
        line.pop_back();
    }
    return line;
}

} // namespace

std::vector<GPUInfo> GPUDetector::detect() {
    std::vector<GPUInfo> gpus;

    DIR* dir = opendir("/sys/class/drm");
    if (dir == nullptr) {
        return gpus; // No DRM subsystem visible (e.g. headless container) -> honestly empty.
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name(entry->d_name);
        // Only look at primary card nodes: "card0", "card1", ... (skip
        // "cardN-<connector>" render/output subnodes).
        if (name.rfind("card", 0) != 0) continue;
        if (name.find('-') != std::string::npos) continue;

        std::string base = "/sys/class/drm/" + name + "/device/";
        std::string vendorHex = readFileTrimmed(base + "vendor");
        if (vendorHex.empty()) continue;

        GPUInfo info;
        info.detected_via_native_api = true;
        try {
            uint32_t vendorId = static_cast<uint32_t>(std::stoul(vendorHex, nullptr, 16));
            info.vendor = vendorFromId(vendorId);
        } catch (...) {
            info.vendor = GPUVendor::Unknown;
        }

        info.name = std::string("GPU (") + name + ")";
        // Kind cannot be reliably determined from sysfs alone without
        // vendor-specific tooling; leave Unknown rather than guessing.
        info.kind = GPUKind::Unknown;

        gpus.push_back(info);
    }
    closedir(dir);

    return gpus;
}

#endif

} // namespace agr
