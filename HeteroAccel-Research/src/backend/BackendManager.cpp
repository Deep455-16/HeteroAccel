// src/backend/BackendManager.cpp
#include "backend/BackendManager.h"
#include "hardware/HardwareDetector.h"

#include <algorithm>

namespace agr {

// Base scores before workload adjustments
static constexpr double CPU_SCORE_BASE    = 100.0;
static constexpr double VULKAN_SCORE_BASE = 300.0;
static constexpr double CUDA_SCORE_BASE   = 500.0;

BackendManager::BackendManager(VulkanBackend& vulkanBackend)
    : vulkan_(vulkanBackend) {}

void BackendManager::discover() {
    devices_.clear();
    discoverCPU();
    discoverVulkan();
    discoverCUDA();

    // Sort: available first, then by score descending
    std::stable_sort(devices_.begin(), devices_.end(),
        [](const ComputeDevice& a, const ComputeDevice& b) {
            if (a.is_available != b.is_available)
                return static_cast<int>(a.is_available) > static_cast<int>(b.is_available);
            return a.compute_score > b.compute_score;
        });

    discovered_ = true;
}

void BackendManager::discoverCPU() {
    HardwareInfo info = HardwareDetector::detectAll();

    ComputeDevice dev;
    dev.backend          = ComputeBackend::CPU;
    dev.is_available     = true;
    dev.supports_compute = true;
    dev.name             = info.cpu.model_name.empty() ? "CPU" : info.cpu.model_name;
    dev.vendor           = info.cpu.vendor;
    dev.compute_units    = info.cpu.physical_cores;
    dev.memory_capacity  = static_cast<size_t>(info.memory.total_physical_mb) * 1024ULL * 1024ULL;
    dev.memory_available = static_cast<size_t>(info.memory.available_physical_mb) * 1024ULL * 1024ULL;
    // Score: base + bonus per core
    dev.compute_score    = CPU_SCORE_BASE
                         + static_cast<double>(info.cpu.physical_cores) * 2.0;
    devices_.push_back(std::move(dev));
}

void BackendManager::discoverVulkan() {
    ComputeDevice dev;
    dev.backend = ComputeBackend::VULKAN;

    if (!vulkan_.isAvailable() && !vulkan_.initialize()) {
        dev.is_available       = false;
        dev.name               = "Vulkan";
        dev.unavailable_reason = vulkan_.lastError();
        dev.compute_score      = 0.0;
        devices_.push_back(std::move(dev));
        return;
    }

    dev.is_available     = true;
    dev.supports_compute = true;
    dev.name             = vulkan_.deviceName();
    dev.vendor           = "Vulkan";
    dev.compute_score    = VULKAN_SCORE_BASE;

    // Query heap sizes via physicalDevice()
    VkPhysicalDevice pd = vulkan_.physicalDevice();
    if (pd != VK_NULL_HANDLE) {
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(pd, &mp);
        for (uint32_t i = 0; i < mp.memoryHeapCount; ++i) {
            if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                dev.memory_capacity += static_cast<size_t>(mp.memoryHeaps[i].size);
            }
        }
        dev.memory_available = dev.memory_capacity;

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(pd, &props);
        dev.api_version = std::to_string(VK_VERSION_MAJOR(props.apiVersion)) + "."
                        + std::to_string(VK_VERSION_MINOR(props.apiVersion));
        dev.driver_version = static_cast<int>(props.driverVersion);

        // Bonus per GB of device-local memory
        dev.compute_score += static_cast<double>(dev.memory_capacity)
                           / (1024.0 * 1024.0 * 1024.0) * 10.0;
    }

    devices_.push_back(std::move(dev));
}

void BackendManager::discoverCUDA() {
    HardwareInfo info = HardwareDetector::detectAll();

    ComputeDevice dev;
    dev.backend = ComputeBackend::CUDA;

    if (!info.cuda.available) {
        dev.is_available       = false;
        dev.name               = "CUDA";
        dev.unavailable_reason = info.cuda.unavailable_reason.empty()
                                     ? "No NVIDIA driver or GPU found"
                                     : info.cuda.unavailable_reason;
        dev.compute_score      = 0.0;
        devices_.push_back(std::move(dev));
        return;
    }

    dev.is_available     = true;
    dev.supports_compute = true;
    dev.vendor           = "NVIDIA";
    dev.driver_version   = info.cuda.driver_version;
    dev.compute_score    = CUDA_SCORE_BASE;

    if (!info.cuda.devices.empty()) {
        dev.name = info.cuda.devices[0].name;
    } else {
        dev.name = "CUDA GPU";
    }

    devices_.push_back(std::move(dev));
}

std::vector<ComputeDevice> BackendManager::availableDevices() const {
    std::vector<ComputeDevice> result;
    for (const auto& d : devices_) {
        if (d.is_available) result.push_back(d);
    }
    return result;
}

bool BackendManager::isBackendAvailable(ComputeBackend backend) const {
    const ComputeDevice* d = getDevice(backend);
    return d && d->is_available;
}

const ComputeDevice* BackendManager::getDevice(ComputeBackend backend) const {
    for (const auto& d : devices_) {
        if (d.backend == backend) return &d;
    }
    return nullptr;
}

} // namespace agr
