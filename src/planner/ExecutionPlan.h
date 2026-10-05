// src/planner/ExecutionPlan.h
// Phase 13: Explainable description of how a workload should be executed.
//
// The plan is a decision, not an execution. Tensor streaming, per-tensor
// placement, and native kernels remain later phases. A plan may name a
// future strategy and mark it not currently executable.
#pragma once

#include "model/ModelTypes.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace agr {

/// Compute placement selected by the planner.
/// HYBRID means the workload is split across CPU and an accelerator
/// (partial residency). UNSUPPORTED means no viable placement exists.
enum class ExecutionMode {
    CPU,
    VULKAN,
    CUDA,
    HYBRID,
    UNSUPPORTED
};

inline const char* toString(ExecutionMode mode) {
    switch (mode) {
        case ExecutionMode::CPU:         return "CPU";
        case ExecutionMode::VULKAN:      return "Vulkan";
        case ExecutionMode::CUDA:        return "CUDA";
        case ExecutionMode::HYBRID:      return "HYBRID";
        case ExecutionMode::UNSUPPORTED: return "UNSUPPORTED";
        default:                         return "UNSUPPORTED";
    }
}

/// Machine-readable reason for a planning decision or rejection.
enum class PlanReason {
    VULKAN_FEASIBLE,
    CUDA_FEASIBLE,
    ACCELERATOR_MEMORY_SUFFICIENT,
    ACCELERATOR_MEMORY_INSUFFICIENT,
    ENGINE_UNAVAILABLE,
    ENGINE_REJECTED,
    CPU_FALLBACK,
    STREAMING_REQUIRED,
    UNSUPPORTED_MODEL,
    UNKNOWN_MEMORY,
    INSUFFICIENT_RAM,
    NO_REGISTERED_ENGINES,
    INVALID_CONFIGURATION,
    CONTRADICTORY_CAPABILITY,
    FUTURE_CAPABILITY,
    PARTIAL_RESIDENCY,
    QUANTIZATION_UNKNOWN
};

inline const char* toString(PlanReason reason) {
    switch (reason) {
        case PlanReason::VULKAN_FEASIBLE:                    return "PLAN_REASON_VULKAN_FEASIBLE";
        case PlanReason::CUDA_FEASIBLE:                      return "PLAN_REASON_CUDA_FEASIBLE";
        case PlanReason::ACCELERATOR_MEMORY_SUFFICIENT:      return "PLAN_REASON_ACCELERATOR_MEMORY_SUFFICIENT";
        case PlanReason::ACCELERATOR_MEMORY_INSUFFICIENT:    return "PLAN_REASON_ACCELERATOR_MEMORY_INSUFFICIENT";
        case PlanReason::ENGINE_UNAVAILABLE:                 return "PLAN_REASON_ENGINE_UNAVAILABLE";
        case PlanReason::ENGINE_REJECTED:                    return "PLAN_REASON_ENGINE_REJECTED";
        case PlanReason::CPU_FALLBACK:                       return "PLAN_REASON_CPU_FALLBACK";
        case PlanReason::STREAMING_REQUIRED:                 return "PLAN_REASON_STREAMING_REQUIRED";
        case PlanReason::UNSUPPORTED_MODEL:                  return "PLAN_REASON_UNSUPPORTED_MODEL";
        case PlanReason::UNKNOWN_MEMORY:                     return "PLAN_REASON_UNKNOWN_MEMORY";
        case PlanReason::INSUFFICIENT_RAM:                   return "PLAN_REASON_INSUFFICIENT_RAM";
        case PlanReason::NO_REGISTERED_ENGINES:              return "PLAN_REASON_NO_REGISTERED_ENGINES";
        case PlanReason::INVALID_CONFIGURATION:              return "PLAN_REASON_INVALID_CONFIGURATION";
        case PlanReason::CONTRADICTORY_CAPABILITY:           return "PLAN_REASON_CONTRADICTORY_CAPABILITY";
        case PlanReason::FUTURE_CAPABILITY:                  return "PLAN_REASON_FUTURE_CAPABILITY";
        case PlanReason::PARTIAL_RESIDENCY:                  return "PLAN_REASON_PARTIAL_RESIDENCY";
        case PlanReason::QUANTIZATION_UNKNOWN:               return "PLAN_REASON_QUANTIZATION_UNKNOWN";
        default:                                             return "PLAN_REASON_UNKNOWN";
    }
}

/// Phase 13 placement decision.
///
/// This is not the Phase 5 scheduler `ExecutionPlan`, which is the
/// cost-model result for one submitted workload. Callers own this plan
/// by value. Planning does not run inference.
struct ModelExecutionPlan {
    bool viable = false;
    /// True only when a currently shipped engine can carry the plan out.
    /// STREAMING is planned but not executable until Phase 14.
    bool executable = false;

    std::string engine_key;
    ExecutionMode mode = ExecutionMode::UNSUPPORTED;
    ExecutionStrategy residency = ExecutionStrategy::AUTO;
    ExecutionStrategy fallback = ExecutionStrategy::AUTO;

    bool streaming_required = false;
    uint64_t memory_budget_bytes = 0;
    uint64_t estimated_model_memory_bytes = 0;
    std::optional<uint32_t> gpu_layer_budget;
    std::optional<uint32_t> context_limit;

    /// 0..100. Identical inputs produce an identical confidence.
    int confidence = 0;

    std::vector<PlanReason> reasons;
    std::string explanation;
    std::vector<std::string> warnings;
    std::vector<std::string> rejected_engines;

    bool hasReason(PlanReason reason) const;
    std::string formatToString() const;
};

/// Knobs that do not rediscover hardware. Defaults preserve the
/// accelerator-first decision tree.
struct PlannerConfig {
    bool allow_accelerator = true;
    bool allow_partial_residency = true;
    bool allow_streaming_plan = true;
    bool allow_cpu_fallback = true;
    /// 0 means no extra ceiling beyond the model's reported context.
    uint32_t context_ceiling = 0;
};

} // namespace agr
