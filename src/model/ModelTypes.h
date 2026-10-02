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

/// VLM/VLA Readiness: Type of resource
enum class ResourceType {
    GENERIC,
    TEXT_WEIGHTS,
    VISION_ENCODER,
    AUDIO_ENCODER,
    PROJECTION,
    DECODER,
    EMBEDDINGS,
    KV_CACHE
};

inline const char* toString(ResourceType t) {
    switch(t) {
        case ResourceType::GENERIC:        return "GENERIC";
        case ResourceType::TEXT_WEIGHTS:   return "TEXT_WEIGHTS";
        case ResourceType::VISION_ENCODER: return "VISION_ENCODER";
        case ResourceType::AUDIO_ENCODER:  return "AUDIO_ENCODER";
        case ResourceType::PROJECTION:     return "PROJECTION";
        case ResourceType::DECODER:        return "DECODER";
        case ResourceType::EMBEDDINGS:     return "EMBEDDINGS";
        case ResourceType::KV_CACHE:       return "KV_CACHE";
        default:                           return "UNKNOWN";
    }
}

/// Generic model-resource representation (format-independent).
struct ModelResource {
    uint64_t    id = 0;
    std::string name;
    size_t      size_bytes = 0;
    ResourceType type = ResourceType::GENERIC;
    
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

/// Workload Classification
enum class WorkloadClass {
    DEFAULT,
    LOW_LATENCY,
    THROUGHPUT,
    MEMORY_BOUND,
    COMPUTE_BOUND,
    STREAMING,
    BACKGROUND,
    INTERACTIVE,
    BATCH
};

inline const char* toString(WorkloadClass c) {
    switch(c) {
        case WorkloadClass::DEFAULT:       return "DEFAULT";
        case WorkloadClass::LOW_LATENCY:   return "LOW_LATENCY";
        case WorkloadClass::THROUGHPUT:    return "THROUGHPUT";
        case WorkloadClass::MEMORY_BOUND:  return "MEMORY_BOUND";
        case WorkloadClass::COMPUTE_BOUND: return "COMPUTE_BOUND";
        case WorkloadClass::STREAMING:     return "STREAMING";
        case WorkloadClass::BACKGROUND:    return "BACKGROUND";
        case WorkloadClass::INTERACTIVE:   return "INTERACTIVE";
        case WorkloadClass::BATCH:         return "BATCH";
        default:                           return "UNKNOWN";
    }
}

/// Execution Strategy for large models
enum class ExecutionStrategy {
    AUTO,
    FULL_RESIDENT,
    PARTIAL_RESIDENT,
    STREAMING,
    MEMORY_PRESSURE,
    CPU_FALLBACK,
    ACCELERATOR_OFFLOAD,
    HYBRID
};

inline const char* toString(ExecutionStrategy s) {
    switch(s) {
        case ExecutionStrategy::AUTO:                return "AUTO";
        case ExecutionStrategy::FULL_RESIDENT:       return "FULL_RESIDENT";
        case ExecutionStrategy::PARTIAL_RESIDENT:    return "PARTIAL_RESIDENT";
        case ExecutionStrategy::STREAMING:           return "STREAMING";
        case ExecutionStrategy::MEMORY_PRESSURE:     return "MEMORY_PRESSURE";
        case ExecutionStrategy::CPU_FALLBACK:        return "CPU_FALLBACK";
        case ExecutionStrategy::ACCELERATOR_OFFLOAD: return "ACCELERATOR_OFFLOAD";
        case ExecutionStrategy::HYBRID:              return "HYBRID";
        default:                                     return "UNKNOWN";
    }
}

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
