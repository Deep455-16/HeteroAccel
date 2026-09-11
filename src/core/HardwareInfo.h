#pragma once
//
// HardwareInfo.h
//
// Clean, API-agnostic representation of everything Phase 1 detects.
// Nothing in this file depends on Windows headers, Vulkan headers, or
// CUDA headers -- those live only inside the detector .cpp files.
//

#include <cstdint>
#include <string>
#include <vector>

namespace agr {

// ---------------------------------------------------------------------
// CPU
// ---------------------------------------------------------------------
struct CPUInfo {
    std::string vendor;          // e.g. "GenuineIntel", "AuthenticAMD"
    std::string model_name;      // e.g. "Intel(R) Core(TM) i5-1235U"
    uint32_t logical_processors = 0;
    uint32_t physical_cores = 0; // 0 if it could not be determined
    std::string architecture;    // "x86_64", "arm64", "unknown"
};

// ---------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------
struct MemoryInfo {
    uint64_t total_physical_mb = 0;
    uint64_t available_physical_mb = 0;
};

// ---------------------------------------------------------------------
// GPU enumeration (vendor-neutral)
// ---------------------------------------------------------------------
enum class GPUVendor { Unknown, Intel, AMD, NVIDIA };
enum class GPUKind { Unknown, Integrated, Dedicated };

struct GPUInfo {
    std::string name;
    GPUVendor vendor = GPUVendor::Unknown;
    GPUKind kind = GPUKind::Unknown;
    uint64_t dedicated_vram_mb = 0;          // 0 if none / not applicable
    uint64_t shared_system_memory_mb = 0;    // 0 if none / not applicable
    uint64_t total_accessible_memory_mb = 0; // best-effort sum
    // True only when this record came from a real enumeration API
    // (DXGI on Windows). Never fabricated.
    bool detected_via_native_api = false;
};

// ---------------------------------------------------------------------
// Vulkan
// ---------------------------------------------------------------------
struct VulkanQueueFamilyInfo {
    uint32_t index = 0;
    uint32_t queue_count = 0;
    bool supports_graphics = false;
    bool supports_compute = false;
    bool supports_transfer = false;
};

struct VulkanDeviceInfo {
    std::string name;
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    std::string device_type; // "Discrete GPU","Integrated GPU","CPU","Virtual GPU","Other"
    uint32_t api_version_major = 0;
    uint32_t api_version_minor = 0;
    uint32_t api_version_patch = 0;
    uint32_t driver_version_raw = 0;
    uint64_t device_local_memory_mb = 0;
    uint64_t host_visible_memory_mb = 0;
    std::vector<VulkanQueueFamilyInfo> queue_families;
    bool has_compute_capable_queue = false;
};

struct VulkanInfo {
    bool available = false;
    std::string unavailable_reason; // populated when available == false
    uint32_t instance_api_version_major = 0;
    uint32_t instance_api_version_minor = 0;
    uint32_t instance_api_version_patch = 0;
    std::vector<VulkanDeviceInfo> devices;
};

// ---------------------------------------------------------------------
// CUDA (optional)
// ---------------------------------------------------------------------
struct CUDADeviceInfo {
    int index = 0;
    std::string name;
};

struct CUDAInfo {
    bool available = false;
    std::string unavailable_reason;
    int driver_version = 0; // e.g. 12040 == 12.4, 0 if unknown
    std::vector<CUDADeviceInfo> devices;
};

// ---------------------------------------------------------------------
// Aggregate
// ---------------------------------------------------------------------
struct HardwareInfo {
    CPUInfo cpu;
    MemoryInfo memory;
    std::vector<GPUInfo> gpus;
    VulkanInfo vulkan;
    CUDAInfo cuda;
};

inline const char* toString(GPUVendor v) {
    switch (v) {
        case GPUVendor::Intel: return "Intel";
        case GPUVendor::AMD: return "AMD";
        case GPUVendor::NVIDIA: return "NVIDIA";
        default: return "Unknown";
    }
}

inline const char* toString(GPUKind k) {
    switch (k) {
        case GPUKind::Integrated: return "Integrated";
        case GPUKind::Dedicated: return "Dedicated";
        default: return "Unknown";
    }
}

} // namespace agr
