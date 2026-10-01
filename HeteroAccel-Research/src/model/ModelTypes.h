// src/model/ModelTypes.h
#pragma once

#include "backend/ComputeDevice.h"
#include "mem/MemoryTypes.h"
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

namespace agr {

/// Explicit logical residency state for a model resource.
enum class ResourceResidency {
    DISK,     ///< Resource exists only on persistent storage.
    COLD,     ///< Known to the runtime but not resident in active memory.
    WARM,     ///< Resident in system RAM (host memory) ready for fast promotion.
    HOT,      ///< Resident on the currently usable execution device (GPU or CPU if fallback).
    LOADING,  ///< Currently being loaded from disk or transferred to device.
    EVICTING  ///< Currently being removed from a memory tier.
};

inline const char* toString(ResourceResidency r) {
    switch(r) {
        case ResourceResidency::DISK:     return "DISK";
        case ResourceResidency::COLD:     return "COLD";
        case ResourceResidency::WARM:     return "WARM";
        case ResourceResidency::HOT:      return "HOT";
        case ResourceResidency::LOADING:  return "LOADING";
        case ResourceResidency::EVICTING: return "EVICTING";
        default:                          return "UNKNOWN";
    }
}

/// Generic model-resource representation (format-independent).
struct ModelResource {
    uint64_t    id = 0;
    std::string name;
    size_t      size_bytes = 0;
    
    // Abstract source location (e.g., file path, offset)
    std::string source_path;
    size_t      source_offset = 0;
    
    // Tracking current physical allocations in Phase 4 MemoryManager
    MemoryBlock block; // Will be valid when WARM or HOT

    ResourceResidency current_residency = ResourceResidency::DISK;
    MemoryPriority    priority = MemoryPriority::NORMAL;
    
    // Telemetry
    uint64_t access_count = 0;
    std::chrono::steady_clock::time_point last_access_time{};
};

/// Generic representation for executable model components.
struct ModelLayer {
    uint64_t    id = 0;
    uint64_t    model_id = 0;
    std::string name;
    
    // Data dependencies (the resources required to execute this layer, e.g. weights)
    std::vector<uint64_t> input_resources;
    std::vector<uint64_t> output_resources; // Temporary output buffers
    
    // Graph dependencies (which layers must run before this one)
    std::vector<uint64_t> depends_on_layers;
    
    // Capability requirements
    bool        prefer_gpu = true;
    size_t      estimated_compute_ops = 0;
    
    // Runtime state
    uint64_t access_count = 0;
    std::chrono::steady_clock::time_point last_access_time{};
};

} // namespace agr
