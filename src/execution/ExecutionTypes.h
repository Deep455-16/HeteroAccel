// src/execution/ExecutionTypes.h
// Phase 15: Heterogeneous Execution Graph Types

#pragma once

#include <cstdint>
#include <string>

namespace agr {

enum class ExecutionNodeType {
    COMPUTE,
    DATA_READ,
    DATA_WRITE,
    TRANSFER,
    RESIDENCY,
    PREFETCH,
    EVICTION,
    SYNCHRONIZATION,
    BARRIER,
    CUSTOM
};

inline const char* toString(ExecutionNodeType type) {
    switch (type) {
        case ExecutionNodeType::COMPUTE:         return "COMPUTE";
        case ExecutionNodeType::DATA_READ:       return "DATA_READ";
        case ExecutionNodeType::DATA_WRITE:      return "DATA_WRITE";
        case ExecutionNodeType::TRANSFER:        return "TRANSFER";
        case ExecutionNodeType::RESIDENCY:       return "RESIDENCY";
        case ExecutionNodeType::PREFETCH:        return "PREFETCH";
        case ExecutionNodeType::EVICTION:        return "EVICTION";
        case ExecutionNodeType::SYNCHRONIZATION: return "SYNCHRONIZATION";
        case ExecutionNodeType::BARRIER:         return "BARRIER";
        case ExecutionNodeType::CUSTOM:          return "CUSTOM";
        default:                                 return "UNKNOWN";
    }
}

enum class ExecutionResource {
    CPU,
    VULKAN,
    CUDA,
    NPU,
    RAM,
    SSD,
    NONE
};

inline const char* toString(ExecutionResource res) {
    switch (res) {
        case ExecutionResource::CPU:    return "CPU";
        case ExecutionResource::VULKAN: return "VULKAN";
        case ExecutionResource::CUDA:   return "CUDA";
        case ExecutionResource::NPU:    return "NPU";
        case ExecutionResource::RAM:    return "RAM";
        case ExecutionResource::SSD:    return "SSD";
        case ExecutionResource::NONE:   return "NONE";
        default:                        return "UNKNOWN";
    }
}

enum class GraphNodeState {
    CREATED,
    READY,
    RUNNING,
    COMPLETED,
    FAILED,
    CANCELLED,
    BLOCKED,
    SKIPPED
};

inline const char* toString(GraphNodeState state) {
    switch (state) {
        case GraphNodeState::CREATED:   return "CREATED";
        case GraphNodeState::READY:     return "READY";
        case GraphNodeState::RUNNING:   return "RUNNING";
        case GraphNodeState::COMPLETED: return "COMPLETED";
        case GraphNodeState::FAILED:    return "FAILED";
        case GraphNodeState::CANCELLED: return "CANCELLED";
        case GraphNodeState::BLOCKED:   return "BLOCKED";
        case GraphNodeState::SKIPPED:   return "SKIPPED";
        default:                        return "UNKNOWN";
    }
}

} // namespace agr
