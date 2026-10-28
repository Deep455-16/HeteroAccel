// src/orchestration/HardwareDetector.cpp
// Phase 21: Hardware Detection & Capability Registry

#include "orchestration/HardwareDetector.h"
#include "hardware/CPUDetector.h"
#include "hardware/MemoryDetector.h"
#include "hardware/GPUDetector.h"
#include "hardware/VulkanDetector.h"
#include "hardware/CUDADetector.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

// Vulkan macros for version parsing
#ifndef VK_API_VERSION_MAJOR
#define VK_API_VERSION_MAJOR(v) ((uint32_t)(v) >> 22)
#define VK_API_VERSION_MINOR(v) (((uint32_t)(v) >> 12) & 0x3FFU)
#define VK_API_VERSION_PATCH(v) ((uint32_t)(v) & 0xFFFU)
#endif

#include <algorithm>
#include <iomanip>

namespace agr {
namespace orchestration {

static DeviceInfo::Vendor detectVendorFromIdOrName(uint32_t vendor_id, const std::string& name) {
    if (vendor_id == 0x8086) return DeviceInfo::Vendor::Intel;
    if (vendor_id == 0x10DE) return DeviceInfo::Vendor::NVIDIA;
    if (vendor_id == 0x1002) return DeviceInfo::Vendor::AMD;
    if (vendor_id == 0x13B5) return DeviceInfo::Vendor::ARM;
    if (vendor_id == 0x106B) return DeviceInfo::Vendor::Apple;

    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower.find("intel") != std::string::npos) return DeviceInfo::Vendor::Intel;
    if (lower.find("nvidia") != std::string::npos || lower.find("geforce") != std::string::npos) return DeviceInfo::Vendor::NVIDIA;
    if (lower.find("amd") != std::string::npos || lower.find("radeon") != std::string::npos) return DeviceInfo::Vendor::AMD;
    if (lower.find("arm") != std::string::npos) return DeviceInfo::Vendor::ARM;
    if (lower.find("apple") != std::string::npos) return DeviceInfo::Vendor::Apple;

    return DeviceInfo::Vendor::Unknown;
}

HardwareDetector& HardwareDetector::instance() {
    static HardwareDetector inst;
    return inst;
}

void HardwareDetector::detect() {
    if (is_mocked_) {
        return;
    }

    devices_.clear();
    has_vulkan_ = false;
    has_cuda_ = false;

    // 1. Detect System RAM
    auto mem_info = MemoryDetector::detect();
    total_system_ram_ = static_cast<size_t>(mem_info.total_physical_mb) * 1024ULL * 1024ULL;

    // 2. Detect CPU
    auto cpu_info = CPUDetector::detect();
    DeviceInfo cpu_dev;
    cpu_dev.name = cpu_info.model_name.empty() ? "Host CPU" : cpu_info.model_name;
    cpu_dev.vendor = detectVendorFromIdOrName(0, cpu_info.vendor + " " + cpu_info.model_name);
    cpu_dev.api_support = static_cast<uint32_t>(DeviceInfo::APISupport::CPU);
    cpu_dev.vram_bytes = 0;
    cpu_dev.system_ram_bytes = total_system_ram_;
    cpu_dev.vulkan_ok = false;
    cpu_dev.cuda_ok = false;
    cpu_dev.vulkan_queue_count = 0;
    cpu_dev.driver_version = "Native OS";
    devices_.push_back(cpu_dev);

    // 3. Detect Vulkan GPUs
    auto vk_info = VulkanDetector::detect();
    if (vk_info.available && !vk_info.devices.empty()) {
        for (const auto& dev : vk_info.devices) {
            DeviceInfo gpu_dev;
            gpu_dev.name = dev.name;
            gpu_dev.vulkan_device_name = dev.name;
            gpu_dev.vendor = detectVendorFromIdOrName(dev.vendor_id, dev.name);
            gpu_dev.api_support = static_cast<uint32_t>(DeviceInfo::APISupport::Vulkan);
            gpu_dev.vram_bytes = static_cast<size_t>(dev.device_local_memory_mb) * 1024ULL * 1024ULL;
            gpu_dev.system_ram_bytes = total_system_ram_;
            gpu_dev.vulkan_ok = dev.has_compute_capable_queue;
            gpu_dev.cuda_ok = false;

            int total_queues = 0;
            for (const auto& qf : dev.queue_families) {
                total_queues += qf.queue_count;
            }
            gpu_dev.vulkan_queue_count = total_queues;

            char drv_buf[64];
            snprintf(drv_buf, sizeof(drv_buf), "%u.%u.%u",
                     VK_API_VERSION_MAJOR(dev.driver_version_raw),
                     VK_API_VERSION_MINOR(dev.driver_version_raw),
                     VK_API_VERSION_PATCH(dev.driver_version_raw));
            gpu_dev.driver_version = drv_buf;

            if (gpu_dev.vulkan_ok) {
                has_vulkan_ = true;
            }
            devices_.push_back(gpu_dev);
        }
    } else {
        // Fall back to DXGI GPU detection if Vulkan didn't find GPUs
        auto dxgi_gpus = GPUDetector::detect();
        for (const auto& g : dxgi_gpus) {
            DeviceInfo gpu_dev;
            gpu_dev.name = g.name;
            gpu_dev.vendor = detectVendorFromIdOrName(0, g.name);
            gpu_dev.api_support = static_cast<uint32_t>(DeviceInfo::APISupport::None);
            gpu_dev.vram_bytes = static_cast<size_t>(g.dedicated_vram_mb) * 1024ULL * 1024ULL;
            gpu_dev.system_ram_bytes = total_system_ram_;
            gpu_dev.vulkan_ok = false;
            gpu_dev.cuda_ok = false;
            gpu_dev.vulkan_queue_count = 0;
            gpu_dev.driver_version = "DXGI";
            devices_.push_back(gpu_dev);
        }
    }

    // 4. Detect CUDA
#ifdef HETEROACCEL_CUDA_ENABLED
    auto cuda_info = CUDADetector::detect();
    if (cuda_info.available && !cuda_info.devices.empty()) {
        has_cuda_ = true;
        for (const auto& cdev : cuda_info.devices) {
            bool found_existing = false;
            for (auto& existing : devices_) {
                if (existing.name.find(cdev.name) != std::string::npos ||
                    cdev.name.find(existing.name) != std::string::npos) {
                    existing.cuda_ok = true;
                    existing.api_support |= static_cast<uint32_t>(DeviceInfo::APISupport::CUDA);
                    found_existing = true;
                    break;
                }
            }
            if (!found_existing) {
                DeviceInfo cuda_dev;
                cuda_dev.name = cdev.name;
                cuda_dev.vendor = DeviceInfo::Vendor::NVIDIA;
                cuda_dev.api_support = static_cast<uint32_t>(DeviceInfo::APISupport::CUDA);
                cuda_dev.vram_bytes = 0; // if unknown
                cuda_dev.system_ram_bytes = total_system_ram_;
                cuda_dev.vulkan_ok = false;
                cuda_dev.cuda_ok = true;
                cuda_dev.vulkan_queue_count = 0;
                cuda_dev.driver_version = std::to_string(cuda_info.driver_version);
                devices_.push_back(cuda_dev);
            }
        }
    }
#else
    has_cuda_ = false;
#endif

    detected_ = true;
}

const std::vector<DeviceInfo>& HardwareDetector::devices() const {
    return devices_;
}

size_t HardwareDetector::available_system_ram() const {
    if (is_mocked_) {
        return mock_avail_ram_;
    }

#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return static_cast<size_t>(status.ullAvailPhys);
    }
#endif
    auto mem_info = MemoryDetector::detect();
    return static_cast<size_t>(mem_info.available_physical_mb) * 1024ULL * 1024ULL;
}

bool HardwareDetector::has_vulkan() const {
    return has_vulkan_;
}

bool HardwareDetector::has_cuda() const {
    return has_cuda_;
}

const DeviceInfo* HardwareDetector::best_gpu() const {
    const DeviceInfo* best = nullptr;
    size_t max_vram = 0;

    for (const auto& dev : devices_) {
        // Must be a GPU (has Vulkan or CUDA support or vram > 0)
        bool is_gpu = (dev.api_support & static_cast<uint32_t>(DeviceInfo::APISupport::Vulkan)) ||
                      (dev.api_support & static_cast<uint32_t>(DeviceInfo::APISupport::CUDA)) ||
                      dev.vulkan_ok || dev.cuda_ok || (dev.vram_bytes > 0);

        // Don't treat pure CPU device as GPU
        if (dev.api_support == static_cast<uint32_t>(DeviceInfo::APISupport::CPU)) {
            continue;
        }

        if (is_gpu) {
            if (!best || dev.vram_bytes > max_vram) {
                best = &dev;
                max_vram = dev.vram_bytes;
            }
        }
    }

    return best;
}

void HardwareDetector::print_summary(std::ostream& out) const {
    out << "========================================================\n";
    out << " HeteroAccel Phase 21 Hardware Capability Registry\n";
    out << "========================================================\n";
    out << " Total System RAM:     " << (total_system_ram_ / (1024 * 1024)) << " MB\n";
    out << " Available System RAM: " << (available_system_ram() / (1024 * 1024)) << " MB\n";
    out << " Vulkan Compute OK:    " << (has_vulkan_ ? "YES" : "NO") << "\n";
    out << " CUDA Compute OK:      " << (has_cuda_ ? "YES" : "NO") << "\n";
    out << " Total Devices Found:  " << devices_.size() << "\n\n";

    for (size_t i = 0; i < devices_.size(); ++i) {
        const auto& d = devices_[i];
        out << " [" << i << "] Device: " << d.name << "\n";
        out << "     Vendor:       " << vendorToString(d.vendor) << "\n";
        out << "     APIs:        ";
        if (d.api_support & static_cast<uint32_t>(DeviceInfo::APISupport::CPU)) out << " CPU";
        if (d.api_support & static_cast<uint32_t>(DeviceInfo::APISupport::Vulkan)) out << " Vulkan";
        if (d.api_support & static_cast<uint32_t>(DeviceInfo::APISupport::CUDA)) out << " CUDA";
        if (d.api_support & static_cast<uint32_t>(DeviceInfo::APISupport::OpenCL)) out << " OpenCL";
        out << "\n";
        if (d.vram_bytes > 0) {
            out << "     VRAM:         " << (d.vram_bytes / (1024 * 1024)) << " MB\n";
        }
        if (!d.vulkan_device_name.empty()) {
            out << "     Vulkan Dev:   " << d.vulkan_device_name << "\n";
            out << "     Vulkan Queues:" << d.vulkan_queue_count << "\n";
            out << "     Driver Ver:   " << d.driver_version << "\n";
        }
        out << "\n";
    }

    const auto* best = best_gpu();
    if (best) {
        out << " Selected Best GPU: " << best->name << " (VRAM: "
            << (best->vram_bytes / (1024 * 1024)) << " MB)\n";
    } else {
        out << " Selected Best GPU: None (CPU Only mode)\n";
    }
    out << "========================================================\n";
}

void HardwareDetector::reset() {
    is_mocked_ = false;
    detected_ = false;
    devices_.clear();
    total_system_ram_ = 0;
    mock_avail_ram_ = 0;
    has_vulkan_ = false;
    has_cuda_ = false;
}

void HardwareDetector::set_mock_environment(const std::vector<DeviceInfo>& mock_devices,
                                            size_t mock_avail_ram,
                                            bool mock_has_vulkan,
                                            bool mock_has_cuda) {
    is_mocked_ = true;
    detected_ = true;
    devices_ = mock_devices;
    mock_avail_ram_ = mock_avail_ram;
    total_system_ram_ = mock_avail_ram;
    has_vulkan_ = mock_has_vulkan;
    has_cuda_ = mock_has_cuda;
}

} // namespace orchestration
} // namespace agr
