// src/collaboration/WorkloadPartitioner.h
// Query Understanding, Task Decomposition, and Dynamic Capability-Aware
// Workload Partitioning across HydroXL, llama.cpp, and Colibri.

#pragma once

#include "collaboration/CollaborativeResourceManager.h"
#include "collaboration/CollaborativeTypes.h"
#include "orchestration/HardwareDetector.h"

#include <string>
#include <vector>

namespace agr {
namespace collaboration {

struct PartitionPlan {
    QueryCategory query_category = QueryCategory::GENERAL_REASONING_AND_SYNTHESIS;
    std::vector<CollaborativeTask> tasks;
    std::vector<std::vector<std::string>> waves; ///< Task IDs per topological wave
    bool memory_constrained_mode = false;
    std::string plan_summary;

    // Phase 33: resolved generation budget and template warnings
    generation::TaskProfile resolved_profile = generation::TaskProfile::EXPLANATION;
    generation::GenerationBudget effective_budget;
    std::vector<std::string> budget_warnings;
};

class WorkloadPartitioner {
public:
    WorkloadPartitioner() = default;

    /// Classifies a user prompt into a high-level QueryCategory.
    static QueryCategory classifyQuery(const std::string& prompt);

    /// Selects the optimal EngineId for a given TaskType and ModelMemoryProfile.
    /// Separates engine selection from hardware backend selection (Section 10).
    static EngineId selectEngineForTask(TaskType task_type,
                                        const ModelMemoryProfile& model_profile,
                                        std::string& reason_out);

    /// Selects the optimal HardwareBackend for an already-selected EngineId.
    /// Respects each engine's real hardware capabilities, VRAM headroom, and
    /// cross-engine GPU contention within the same wave.
    static HardwareBackend selectHardwareBackendForEngine(
        EngineId engine,
        const ModelMemoryProfile& model_profile,
        const orchestration::HardwareDetector& hw,
        const std::string& preferred_gpu_backend,
        bool gpu_already_claimed_in_wave,
        std::string& orchestrator_backend_id_out,
        std::string& reason_out);

    /// Builds a validated, dependency-aware multi-engine PartitionPlan using
    /// HeteroAccel's ExecutionGraph wave decomposition.
    static PartitionPlan createPlan(
        const CollaborativeRequest& req,
        const ModelMemoryProfile& model_profile,
        const CollaborativeResourceManager& resource_mgr,
        const orchestration::HardwareDetector& hw);
};

} // namespace collaboration
} // namespace agr
