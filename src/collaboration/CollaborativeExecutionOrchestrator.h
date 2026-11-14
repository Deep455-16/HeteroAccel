// src/collaboration/CollaborativeExecutionOrchestrator.h
// Central Collaborative Multi-Engine Orchestrator for HeteroAccel
//
// Coordinates HydroXL (native engine), llama.cpp (compatibility engine), and
// Colibri (specialized structural/complexity/verification engine) through:
//   1. Query understanding and task decomposition
//   2. Capability- and memory-aware workload partitioning (DAG wave model)
//   3. Separate engine selection vs. hardware backend selection (CPU/Vulkan/CUDA)
//   4. Parallel wave execution when memory permits, or sequential model-swapping
//      execution under constrained RAM budgets (~16 GB RAM)
//   5. Standardized cross-engine intermediate result exchange
//   6. Result validation, deduplication, conflict resolution, and coherent fusion
//   7. Unified streaming output and participation telemetry

#pragma once

#include "collaboration/CollaborativeResourceManager.h"
#include "collaboration/CollaborativeTypes.h"
#include "collaboration/ResultFusionEngine.h"
#include "collaboration/WorkloadPartitioner.h"

#include <functional>
#include <map>
#include <mutex>
#include <string>

namespace agr {
namespace collaboration {

class CollaborativeExecutionOrchestrator {
public:
    /// Optional custom task executor signature for deterministic testing/override.
    using CustomTaskExecutor = std::function<IntermediateResult(
        const CollaborativeTask& task,
        const CollaborativeRequest& req)>;

    CollaborativeExecutionOrchestrator();
    ~CollaborativeExecutionOrchestrator();

    /// Ensures HydroXL ("hydroxl"), llama.cpp ("llama.cpp"), and Colibri ("colibri")
    /// are registered in ExecutionEngineRegistry and EngineOrchestrator.
    static void ensureEnginesRegistered();

    /// Executes a collaborative multi-engine request synchronously.
    CollaborativeResult execute(const CollaborativeRequest& req);

    /// Executes a collaborative multi-engine request and streams the unified
    /// fused response incrementally through `cb`.
    CollaborativeResult executeStreaming(const CollaborativeRequest& req,
                                         CollaborativeStreamCallback cb);

    /// Access the underlying shared resource manager (e.g., to inspect residency
    /// or evict cached models).
    CollaborativeResourceManager& resourceManager() { return resource_mgr_; }
    const CollaborativeResourceManager& resourceManager() const { return resource_mgr_; }

    /// Registers an optional custom task executor for a specific EngineId (used
    /// by unit tests to inject controlled engine outputs or simulated failures).
    void setCustomExecutor(EngineId engine, CustomTaskExecutor executor);
    void clearCustomExecutors();

private:
    IntermediateResult executeSingleTask(
        CollaborativeTask& task,
        const CollaborativeRequest& req,
        const ModelMemoryProfile& model_profile,
        bool memory_constrained_mode);

    IntermediateResult runEngineBackend(
        EngineId engine,
        HardwareBackend backend,
        const std::string& orchestrator_backend_id,
        const CollaborativeTask& task,
        const CollaborativeRequest& req,
        const ModelMemoryProfile& model_profile,
        bool memory_constrained_mode);

    CollaborativeResourceManager resource_mgr_;
    mutable std::mutex custom_exec_mu_;
    std::map<EngineId, CustomTaskExecutor> custom_executors_;
};

} // namespace collaboration
} // namespace agr
