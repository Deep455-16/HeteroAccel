// src/backend/ComputeDevice.h
// Unified hardware-independent device model for HeteroAccel.
// CPU, Vulkan, and CUDA are all represented as ComputeDevice instances.
// The user never chooses a backend -- HeteroAccel decides.
#pragma once
#include <cstddef>
#include <string>

namespace agr {

enum class ComputeBackend { CPU, VULKAN, CUDA };

inline const char* toString(ComputeBackend b) {
    switch (b) {
        case ComputeBackend::CPU:    return "CPU";
        case ComputeBackend::VULKAN: return "Vulkan";
        case ComputeBackend::CUDA:   return "CUDA";
        default:                     return "unknown";
    }
}

/// Hardware-independent representation of a compute device.
/// Allows CPU, Vulkan, and CUDA to be compared uniformly.
struct ComputeDevice {
    ComputeBackend backend     = ComputeBackend::CPU;
    bool   is_available        = false;
    std::string name;              ///< "Intel Core i5-1235U", "Intel Iris Xe", "NVIDIA RTX 4090"
    std::string vendor;
    std::string unavailable_reason;

    // Memory
    size_t memory_capacity  = 0;   ///< Total device memory in bytes
    size_t memory_available = 0;   ///< Estimated free memory in bytes

    // Compute
    bool   supports_compute = false;
    int    compute_units    = 0;   ///< Physical cores (CPU) / CUs (Vulkan) / SMs (CUDA)
    double compute_score    = 0.0; ///< Capability score -- higher = preferred

    // Diagnostics
    std::string api_version;
    int    driver_version   = 0;

    bool isValid() const { return is_available; }
};

/// Workload descriptor -- hints to the selector about what the task needs.
struct WorkloadHint {
    size_t required_memory    = 0;     ///< Minimum memory in bytes (0 = don't check)
    bool   prefer_gpu         = true;  ///< Prefer GPU when capable
    bool   allow_cpu_fallback = true;
    bool   compute_intensive  = true;  ///< Dense compute prefers GPU
};

} // namespace agr
