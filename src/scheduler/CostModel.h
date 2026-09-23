// src/scheduler/CostModel.h
// Evaluates compute candidates for a given workload.
#pragma once

#include "backend/BackendManager.h"
#include "scheduler/SchedulerTypes.h"
#include "scheduler/PerformanceHistory.h"

namespace agr {

class CostModel {
public:
    CostModel(const BackendManager& backendManager, const PerformanceHistory& history);

    /// Evaluates all backends and selects the one with the lowest total cost.
    ExecutionPlan evaluate(const Workload& workload) const;

    /// Evaluates a specific backend. Returns a plan with total_cost_ms. 
    /// If infeasible (e.g. OOM), returns a plan with total_cost_ms = infinity.
    ExecutionPlan evaluateCandidate(const Workload& workload, const ComputeDevice& device) const;

private:
    const BackendManager&     backendManager_;
    const PerformanceHistory& history_;
};

} // namespace agr
