// src/execution/ExecutionGraphScheduler.h
// Phase 15: Execution Graph Scheduler
//
// Orchestrates the execution of the execution graph. Handles ready-node detection,
// resource conflict avoidance, failure propagation, and telemetry.

#pragma once

#include "execution/ExecutionGraph.h"
#include "execution/ExecutionGraphContext.h"
#include <vector>
#include <map>
#include <mutex>
#include <future>
#include <string>

namespace agr {

struct ExecutionGraphResult {
    bool success = false;
    std::string error_message;
    std::vector<NodeTelemetry> telemetry;
};

class ExecutionGraphScheduler {
public:
    ExecutionGraphScheduler() = default;
    ~ExecutionGraphScheduler() = default;

    /// Execute the graph to completion or failure.
    ExecutionGraphResult execute(ExecutionGraph& graph, ExecutionGraphContext& context);

private:
    struct NodeStateTracker {
        GraphNodeState state = GraphNodeState::CREATED;
        int in_degree = 0;
        std::string fail_reason;
        double start_ms = 0.0;
        double end_ms = 0.0;
    };

    double getCurrentTimeMs() const;
    void failDependentNodes(ExecutionGraph& graph, uint64_t failed_id, std::map<uint64_t, NodeStateTracker>& tracker, const std::string& reason);
    bool checkResourceAvailable(ExecutionResource res, const std::map<ExecutionResource, int>& active_resources) const;
};

} // namespace agr
