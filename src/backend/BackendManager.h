// src/backend/BackendManager.h
// Discovers and owns representations of all compute backends.
// This is the component that hides CUDA/Vulkan details from the caller.
#pragma once
#include "backend/ComputeDevice.h"
#include "gpu/VulkanBackend.h"
#include <vector>

namespace agr {

/// Discovers CPU, Vulkan, and CUDA backends at startup.
/// The higher-level code never needs to know HOW discovery works.
class BackendManager {
public:
    explicit BackendManager(VulkanBackend& vulkanBackend);

    /// Probe all backends. Safe to call multiple times.
    void discover();

    /// All discovered devices (available and unavailable), sorted best-first.
    const std::vector<ComputeDevice>& allDevices() const { return devices_; }

    /// Only available devices.
    std::vector<ComputeDevice> availableDevices() const;

    bool isBackendAvailable(ComputeBackend backend) const;

    /// Returns nullptr if backend not found.
    const ComputeDevice* getDevice(ComputeBackend backend) const;

    bool discovered() const { return discovered_; }

private:
    void discoverCPU();
    void discoverVulkan();
    void discoverCUDA();

    VulkanBackend&             vulkan_;
    std::vector<ComputeDevice> devices_;
    bool                       discovered_ = false;
};

} // namespace agr
