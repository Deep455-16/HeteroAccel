// src/orchestration/HardwareDetector.h
// Phase 21: Hardware Detection & Capability Registry

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <iostream>

namespace agr {
namespace orchestration {

struct DeviceInfo {
    enum class Vendor { Unknown, Intel, NVIDIA, AMD, ARM, Apple };
    enum class APISupport : uint32_t {
        None   = 0,
        CPU    = 1 << 0,
        Vulkan = 1 << 1,
        CUDA   = 1 << 2,
        OpenCL = 1 << 3
    };

    std::string name;
    Vendor vendor = Vendor::Unknown;
    uint32_t api_support = 0;   // bitmask of APISupport
    size_t vram_bytes = 0;      // 0 for CPU
    size_t system_ram_bytes = 0;
    bool vulkan_ok = false;
    bool cuda_ok = false;
    int vulkan_queue_count = 0;
    std::string vulkan_device_name;
    std::string driver_version;
    bool is_software_emulated = false;  // CPU-based Vulkan (llvmpipe etc.); not a physical GPU

    // CUDA details (Phase 26); populated only for devices CUDA can enumerate.
    int cuda_device_index = -1;      // index in the CUDA driver's device list
    int cuda_compute_major = 0;
    int cuda_compute_minor = 0;
    size_t cuda_total_bytes = 0;
};

inline const char* vendorToString(DeviceInfo::Vendor v) {
    switch (v) {
        case DeviceInfo::Vendor::Intel:  return "Intel";
        case DeviceInfo::Vendor::NVIDIA: return "NVIDIA";
        case DeviceInfo::Vendor::AMD:    return "AMD";
        case DeviceInfo::Vendor::ARM:    return "ARM";
        case DeviceInfo::Vendor::Apple:  return "Apple";
        default:                         return "Unknown";
    }
}

class HardwareDetector {
public:
    static HardwareDetector& instance();

    void detect();   // call once at startup; idempotent
    const std::vector<DeviceInfo>& devices() const;
    size_t available_system_ram() const;
    bool has_vulkan() const;
    bool has_physical_vulkan_gpu() const { return has_physical_vulkan_gpu_; }
    bool has_cuda() const;
    const DeviceInfo* best_gpu() const;  // returns nullptr if CPU-only
    const DeviceInfo* best_cuda_gpu() const;  // CUDA-usable device with most VRAM; nullptr if none

    /// True when this binary was built with HETEROACCEL_ENABLE_CUDA (llama.cpp GGML_CUDA).
    static bool cuda_inference_compiled();
    /// Why CUDA is not usable (empty when has_cuda()).
    const std::string& cuda_unavailable_reason() const { return cuda_reason_; }
    void print_summary(std::ostream& out) const;

    // Test support / simulation helpers
    void reset();
    void set_mock_environment(const std::vector<DeviceInfo>& mock_devices,
                              size_t mock_avail_ram,
                              bool mock_has_vulkan,
                              bool mock_has_cuda);

private:
    HardwareDetector() = default;

    std::vector<DeviceInfo> devices_;
    size_t total_system_ram_ = 0;
    mutable size_t mock_avail_ram_ = 0;
    bool has_vulkan_ = false;
    bool has_physical_vulkan_gpu_ = false;
    bool has_cuda_ = false;
    std::string cuda_reason_;
    bool detected_ = false;
    bool is_mocked_ = false;
};

} // namespace orchestration
} // namespace agr
