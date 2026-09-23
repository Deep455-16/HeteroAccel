// src/mem/MemoryTypes.h
//
// Hardware-independent memory abstractions for Phase 4.
// No Vulkan/CUDA headers included here.
//
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

namespace agr {

enum class MemoryLocation { CPU, GPU, ACCELERATOR, DISK };
enum class MemoryState    { FREE, RESIDENT, LOADING, EVICTING };
enum class MemoryPriority { CRITICAL = 0, HIGH = 1, NORMAL = 2, LOW = 3 };
enum class Residency      { DISK, CPU, GPU, CPU_AND_GPU };
enum class PressureLevel  { NORMAL, WARNING, HIGH, CRITICAL };

inline const char* toString(MemoryLocation l) {
    switch (l) {
        case MemoryLocation::CPU:         return "CPU";
        case MemoryLocation::GPU:         return "GPU";
        case MemoryLocation::ACCELERATOR: return "ACCELERATOR";
        case MemoryLocation::DISK:        return "DISK";
        default:                          return "unknown";
    }
}
inline const char* toString(MemoryState s) {
    switch (s) {
        case MemoryState::FREE:     return "FREE";
        case MemoryState::RESIDENT: return "RESIDENT";
        case MemoryState::LOADING:  return "LOADING";
        case MemoryState::EVICTING: return "EVICTING";
        default:                    return "unknown";
    }
}
inline const char* toString(MemoryPriority p) {
    switch (p) {
        case MemoryPriority::CRITICAL: return "CRITICAL";
        case MemoryPriority::HIGH:     return "HIGH";
        case MemoryPriority::NORMAL:   return "NORMAL";
        case MemoryPriority::LOW:      return "LOW";
        default:                       return "unknown";
    }
}
inline const char* toString(PressureLevel p) {
    switch (p) {
        case PressureLevel::NORMAL:   return "NORMAL";
        case PressureLevel::WARNING:  return "WARNING";
        case PressureLevel::HIGH:     return "HIGH";
        case PressureLevel::CRITICAL: return "CRITICAL";
        default:                      return "unknown";
    }
}

// ---------------------------------------------------------------------------
// MemoryBlock — opaque, manager-owned descriptor for one allocation
// ---------------------------------------------------------------------------
struct MemoryBlock {
    uint64_t       id       = 0;
    size_t         size     = 0;
    MemoryLocation location = MemoryLocation::CPU;
    MemoryState    state    = MemoryState::FREE;
    MemoryPriority priority = MemoryPriority::NORMAL;
    Residency      residency = Residency::CPU;

    std::chrono::steady_clock::time_point last_accessed{};
    uint64_t access_count   = 0;
    uint64_t backend_handle = 0; // CPUAllocator: raw ptr; GPUAllocator: Buffer::id

    bool isValid() const { return id != 0; }
};

// ---------------------------------------------------------------------------
// TransferRecord — telemetry for one CPU<->GPU transfer
// ---------------------------------------------------------------------------
struct TransferRecord {
    uint64_t       id          = 0;
    MemoryLocation source      = MemoryLocation::CPU;
    MemoryLocation destination = MemoryLocation::GPU;
    size_t         bytes       = 0;

    std::chrono::steady_clock::time_point start_time{};
    std::chrono::steady_clock::time_point end_time{};

    double duration_ms    = 0.0;
    double bandwidth_gbps = 0.0;
    bool   success        = false;
};

// ---------------------------------------------------------------------------
// MemoryStats — aggregate runtime statistics
// ---------------------------------------------------------------------------
struct MemoryStats {
    size_t cpu_total_bytes        = 0;
    size_t cpu_used_bytes         = 0;
    size_t cpu_peak_bytes         = 0;

    size_t gpu_device_local_bytes = 0;
    size_t gpu_used_bytes         = 0;
    size_t gpu_peak_bytes         = 0;

    size_t   allocation_count     = 0;
    size_t   transfer_count       = 0;
    size_t   pool_reuse_count     = 0;
    size_t   eviction_count       = 0;

    uint64_t bytes_uploaded       = 0;
    uint64_t bytes_downloaded     = 0;

    double upload_bandwidth_gbps   = 0.0;
    double download_bandwidth_gbps = 0.0;

    PressureLevel cpu_pressure = PressureLevel::NORMAL;
    PressureLevel gpu_pressure = PressureLevel::NORMAL;
};

} // namespace agr
