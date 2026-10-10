// src/execution/ExecutionGraphBuilder.h
// Phase 15: Execution Graph Builder
// Converts a ModelExecutionPlan into a concrete ExecutionGraph.

#pragma once

#include "execution/ExecutionGraph.h"
#include "planner/ExecutionPlan.h"
#include "model/ModelRegion.h"
#include <string>
#include <vector>

namespace agr {

class ExecutionGraphBuilder {
public:
    ExecutionGraphBuilder() = default;

    /// Build a graph from an execution plan. 
    /// Adds residency nodes based on regions, followed by a synchronization barrier, 
    /// followed by compute node(s).
    /// If the plan is UNSUPPORTED, builds an empty graph or returns false.
    bool build(const ModelExecutionPlan& plan,
               const std::string& model_path,
               const std::vector<ModelRegion>& regions,
               const std::string& prompt,
               ExecutionGraph& out_graph,
               std::string& out_error);

private:
    uint64_t next_id_ = 1;
};

} // namespace agr
