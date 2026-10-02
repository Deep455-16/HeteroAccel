// src/scheduler/ExecutionPolicyEngine.h
// Phase 9: Selects the appropriate ExecutionStrategy for a given workload.
#pragma once
#include "backend/ComputeDevice.h"
#include "backend/BackendManager.h"
#include "mem/MemoryManager.h"
#include "model/ModelTypes.h"
#include "scheduler/PerformanceHistory.h"
#include "scheduler/WorkloadRegistry.h"
#include <string>

namespace agr {

/// Input descriptor for strategy selection
struct PolicyInput {
    size_t          model_size_bytes = 0;
    size_t          available_gpu_mem = 0;
    size_t          available_cpu_mem = 0;
    PressureLevel   memory_pressure = PressureLevel::NORMAL;
    bool            gpu_available = false;
    WorkloadClass   wclass = WorkloadClass::DEFAULT;
    WorkloadPriority priority = WorkloadPriority::NORMAL;
    ProfileKey      profile_key;
    int             active_workload_count = 0;
};

/// Output: the selected strategy with reasoning
struct PolicyDecision {
    ExecutionStrategy strategy = ExecutionStrategy::AUTO;
    int               n_gpu_layers = 99;
    int               n_threads = 4;
    int               prefetch_distance = 1;
    bool              streaming_enabled = true;
    std::string       reason;
};

/// Selects execution strategy based on hardware state, memory pressure, and history.
class ExecutionPolicyEngine {
public:
    ExecutionPolicyEngine(const BackendManager& backends,
                          MemoryManager& memMgr,
                          const PerformanceHistory& history);

    /// Select the best strategy for the given workload input.
    PolicyDecision selectStrategy(const PolicyInput& input) const;

    /// Compute n_gpu_layers based on model size and available GPU memory.
    int computeGpuLayers(size_t model_size_bytes, size_t gpu_mem_bytes, int total_layers) const;

    /// Compute adaptive prefetch distance based on memory pressure.
    int adaptivePrefetchDistance(PressureLevel pressure, int base_distance = 2) const;

private:
    const BackendManager&     backends_;
    MemoryManager&            memMgr_;
    const PerformanceHistory& history_;

    static constexpr double kFullResidentThreshold = 0.85;
    static constexpr double kPartialResidentMin    = 0.15;
};

} // namespace agr
