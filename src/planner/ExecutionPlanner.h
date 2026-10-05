// src/planner/ExecutionPlanner.h
// Phase 13: Execution Planner.
//
// Turns a Phase 12 CapabilityReport plus registered engine capabilities
// into a ModelExecutionPlan. The planner does not inspect GGUF files, probe
// Vulkan, or detect CUDA. It does not run inference.
#pragma once

#include "analysis/CapabilityReport.h"
#include "engine/ExecutionEngineRegistry.h"
#include "planner/ExecutionPlan.h"

namespace agr {

class ExecutionPlanner {
public:
    ExecutionPlanner() = default;

    /// Deterministic for identical requirements, report, registry
    /// contents, and config. Registry key order does not affect the result.
    ModelExecutionPlan plan(const ModelRequirements& requirements,
                       const CapabilityReport& report,
                       const ExecutionEngineRegistry& registry,
                       const PlannerConfig& config = {}) const;
};

} // namespace agr
