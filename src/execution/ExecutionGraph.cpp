// src/execution/ExecutionGraph.cpp

#include "execution/ExecutionGraph.h"
#include <algorithm>
#include <sstream>

namespace agr {

uint64_t ExecutionGraph::addNode(std::unique_ptr<IExecutionNode> node) {
    if (!node) return 0;
    uint64_t id = node->getId();
    nodes_[id] = std::move(node);
    
    // Ensure adjacency lists have an entry for the node
    if (adj_.find(id) == adj_.end()) {
        adj_[id] = std::vector<uint64_t>();
    }
    if (rev_adj_.find(id) == rev_adj_.end()) {
        rev_adj_[id] = std::vector<uint64_t>();
    }
    return id;
}

bool ExecutionGraph::addDependency(uint64_t predecessor_id, uint64_t successor_id) {
    if (nodes_.find(predecessor_id) == nodes_.end() || nodes_.find(successor_id) == nodes_.end()) {
        return false;
    }

    auto& succs = adj_[predecessor_id];
    if (std::find(succs.begin(), succs.end(), successor_id) == succs.end()) {
        succs.push_back(successor_id);
    }

    auto& preds = rev_adj_[successor_id];
    if (std::find(preds.begin(), preds.end(), predecessor_id) == preds.end()) {
        preds.push_back(predecessor_id);
    }
    return true;
}

bool ExecutionGraph::validate(std::string& out_error) const {
    if (nodes_.empty()) {
        out_error = "Graph is empty.";
        // Technically empty graph is valid, but usually we want to know if it's empty.
        // We'll let it be valid but some might consider it invalid.
        // Let's allow empty graphs as valid, but we will return early from validation.
        return true;
    }

    if (detectCycle()) {
        out_error = "Cycle detected in execution graph dependencies.";
        return false;
    }

    return true;
}

std::vector<uint64_t> ExecutionGraph::getAllNodeIds() const {
    std::vector<uint64_t> ids;
    ids.reserve(nodes_.size());
    for (const auto& pair : nodes_) {
        ids.push_back(pair.first);
    }
    // Sort to keep determinism when iterating
    std::sort(ids.begin(), ids.end());
    return ids;
}

std::vector<uint64_t> ExecutionGraph::getPredecessors(uint64_t node_id) const {
    auto it = rev_adj_.find(node_id);
    if (it != rev_adj_.end()) return it->second;
    return {};
}

std::vector<uint64_t> ExecutionGraph::getSuccessors(uint64_t node_id) const {
    auto it = adj_.find(node_id);
    if (it != adj_.end()) return it->second;
    return {};
}

IExecutionNode* ExecutionGraph::getNode(uint64_t node_id) const {
    auto it = nodes_.find(node_id);
    if (it != nodes_.end()) return it->second.get();
    return nullptr;
}

bool ExecutionGraph::detectCycle() const {
    // 0 = unvisited, 1 = visiting, 2 = visited
    std::unordered_map<uint64_t, int> state;
    for (const auto& pair : nodes_) {
        state[pair.first] = 0;
    }

    // Since it's a DAG, we can start from any node. 
    // To be deterministic, we could sort the keys, but it doesn't matter for correctness.
    for (const auto& pair : nodes_) {
        if (state[pair.first] == 0) {
            if (dfs(pair.first, state)) {
                return true; // Cycle found
            }
        }
    }
    return false;
}

bool ExecutionGraph::dfs(uint64_t node_id, std::unordered_map<uint64_t, int>& state) const {
    state[node_id] = 1; // visiting
    auto it = adj_.find(node_id);
    if (it != adj_.end()) {
        for (uint64_t next_id : it->second) {
            if (state[next_id] == 1) {
                return true; // back-edge = cycle
            }
            if (state[next_id] == 0) {
                if (dfs(next_id, state)) {
                    return true;
                }
            }
        }
    }
    state[node_id] = 2; // visited
    return false;
}

} // namespace agr
