// src/execution/ExecutionGraph.h
// Phase 15: Heterogeneous Execution Graph
//
// Represents a DAG of execution nodes.

#pragma once

#include "execution/IExecutionNode.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace agr {

class ExecutionGraph {
public:
    ExecutionGraph() = default;
    ~ExecutionGraph() = default;

    // Disallow copy
    ExecutionGraph(const ExecutionGraph&) = delete;
    ExecutionGraph& operator=(const ExecutionGraph&) = delete;

    // Allow move
    ExecutionGraph(ExecutionGraph&&) = default;
    ExecutionGraph& operator=(ExecutionGraph&&) = default;

    /// Add a node to the graph. The graph takes ownership.
    /// Returns the node ID (which is extracted from the node).
    uint64_t addNode(std::unique_ptr<IExecutionNode> node);

    /// Add a directed edge indicating `predecessor_id` must finish before `successor_id` starts.
    bool addDependency(uint64_t predecessor_id, uint64_t successor_id);

    /// Validate the graph (check for cycles and missing nodes).
    bool validate(std::string& out_error) const;

    /// Get all node IDs
    std::vector<uint64_t> getAllNodeIds() const;

    /// Get predecessors for a given node ID
    std::vector<uint64_t> getPredecessors(uint64_t node_id) const;

    /// Get successors for a given node ID
    std::vector<uint64_t> getSuccessors(uint64_t node_id) const;

    /// Get node pointer (read-only access)
    IExecutionNode* getNode(uint64_t node_id) const;

private:
    bool detectCycle() const;
    bool dfs(uint64_t node_id, std::unordered_map<uint64_t, int>& state) const;

    std::unordered_map<uint64_t, std::unique_ptr<IExecutionNode>> nodes_;
    
    // Adjacency list: node -> list of successors
    std::unordered_map<uint64_t, std::vector<uint64_t>> adj_;
    
    // Reverse Adjacency list: node -> list of predecessors
    std::unordered_map<uint64_t, std::vector<uint64_t>> rev_adj_;
};

} // namespace agr
