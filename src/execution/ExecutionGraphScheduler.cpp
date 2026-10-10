// src/execution/ExecutionGraphScheduler.cpp

#include "execution/ExecutionGraphScheduler.h"
#include <chrono>
#include <thread>
#include <queue>
#include <iostream>

namespace agr {

double ExecutionGraphScheduler::getCurrentTimeMs() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();
}

bool ExecutionGraphScheduler::checkResourceAvailable(ExecutionResource res, const std::map<ExecutionResource, int>& active_resources) const {
    // If ANY resource can only be used by one node at a time, we enforce it here.
    // For simplicity, let's assume CPU can run concurrently (no limit), 
    // but accelerators (VULKAN, CUDA, NPU) and SSD/RAM are exclusive per operation to avoid 
    // destructive conflicts in this generic layer. 
    // Real engines would handle finer locking.
    if (res == ExecutionResource::NONE) return true;
    if (res == ExecutionResource::CPU) return true; // Multiple CPU threads fine

    auto it = active_resources.find(res);
    if (it != active_resources.end() && it->second > 0) {
        return false; // Exclusive resource is busy
    }
    return true;
}

void ExecutionGraphScheduler::failDependentNodes(ExecutionGraph& graph, uint64_t failed_id, std::map<uint64_t, NodeStateTracker>& tracker, const std::string& reason) {
    std::queue<uint64_t> q;
    for (uint64_t succ : graph.getSuccessors(failed_id)) {
        q.push(succ);
    }
    while (!q.empty()) {
        uint64_t curr = q.front();
        q.pop();
        if (tracker[curr].state == GraphNodeState::CREATED || tracker[curr].state == GraphNodeState::READY) {
            tracker[curr].state = GraphNodeState::BLOCKED;
            tracker[curr].fail_reason = "Blocked by failure of predecessor: " + reason;
            for (uint64_t succ : graph.getSuccessors(curr)) {
                q.push(succ);
            }
        }
    }
}

ExecutionGraphResult ExecutionGraphScheduler::execute(ExecutionGraph& graph, ExecutionGraphContext& context) {
    ExecutionGraphResult result;
    std::string err;
    if (!graph.validate(err)) {
        result.success = false;
        result.error_message = err;
        return result;
    }

    auto all_ids = graph.getAllNodeIds();
    std::map<uint64_t, NodeStateTracker> tracker;
    for (uint64_t id : all_ids) {
        tracker[id] = NodeStateTracker();
        tracker[id].in_degree = static_cast<int>(graph.getPredecessors(id).size());
        if (tracker[id].in_degree == 0) {
            tracker[id].state = GraphNodeState::READY;
        }
    }

    std::map<ExecutionResource, int> active_resources;
    
    // Futures mapping: node_id -> future
    std::map<uint64_t, std::future<GraphNodeResult>> active_tasks;
    
    bool overall_success = true;
    bool active = true;

    while (active) {
        // Check for cancellation
        if (context.isCancelled()) {
            overall_success = false;
            result.error_message = "Execution cancelled.";
            // Mark remaining as cancelled
            for (auto& pair : tracker) {
                if (pair.second.state == GraphNodeState::READY || pair.second.state == GraphNodeState::CREATED) {
                    pair.second.state = GraphNodeState::CANCELLED;
                }
            }
        }

        // Dispatch ready nodes if resources are available
        for (uint64_t id : all_ids) {
            if (tracker[id].state == GraphNodeState::READY && !context.isCancelled()) {
                IExecutionNode* node = graph.getNode(id);
                ExecutionResource res = node->getResource();
                if (checkResourceAvailable(res, active_resources)) {
                    // Mark as running
                    tracker[id].state = GraphNodeState::RUNNING;
                    tracker[id].start_ms = getCurrentTimeMs();
                    active_resources[res]++;
                    
                    // Dispatch
                    active_tasks[id] = std::async(std::launch::async, [node, &context]() {
                        return node->execute(context);
                    });
                }
            }
        }

        // Wait/poll for completions
        active = false;
        if (!active_tasks.empty()) {
            active = true;
            bool task_finished = false;
            while (!task_finished && !active_tasks.empty()) {
                for (auto it = active_tasks.begin(); it != active_tasks.end(); ) {
                    uint64_t id = it->first;
                    if (it->second.wait_for(std::chrono::milliseconds(5)) == std::future_status::ready) {
                        GraphNodeResult node_res = it->second.get();
                        task_finished = true;
                        
                        IExecutionNode* node = graph.getNode(id);
                        ExecutionResource res = node->getResource();
                        active_resources[res]--;
                        
                        tracker[id].end_ms = getCurrentTimeMs();
                        
                        if (node_res.success) {
                            tracker[id].state = GraphNodeState::COMPLETED;
                            // Decrement in-degree for successors
                            for (uint64_t succ : graph.getSuccessors(id)) {
                                tracker[succ].in_degree--;
                                if (tracker[succ].in_degree == 0 && tracker[succ].state == GraphNodeState::CREATED) {
                                    tracker[succ].state = GraphNodeState::READY;
                                }
                            }
                        } else {
                            tracker[id].state = GraphNodeState::FAILED;
                            tracker[id].fail_reason = node_res.error_message;
                            overall_success = false;
                            failDependentNodes(graph, id, tracker, node_res.error_message);
                        }
                        
                        it = active_tasks.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
        }
        
        // Also check if there's any READY tasks left but not dispatched (could be blocked by resources)
        for (const auto& pair : tracker) {
            if (pair.second.state == GraphNodeState::READY || pair.second.state == GraphNodeState::RUNNING) {
                active = true;
                break;
            }
        }
    }

    // Collect telemetry
    for (uint64_t id : all_ids) {
        IExecutionNode* node = graph.getNode(id);
        NodeTelemetry t;
        t.node_id = id;
        t.type = node->getType();
        t.resource = node->getResource();
        t.start_time_ms = tracker[id].start_ms;
        t.end_time_ms = tracker[id].end_ms;
        t.duration_ms = (tracker[id].end_ms >= tracker[id].start_ms && tracker[id].start_ms > 0) ? (tracker[id].end_ms - tracker[id].start_ms) : 0.0;
        t.final_state = tracker[id].state;
        t.success = (t.final_state == GraphNodeState::COMPLETED);
        t.failure_reason = tracker[id].fail_reason;
        
        result.telemetry.push_back(t);
    }
    
    // Check if any didn't finish
    for (const auto& pair : tracker) {
        if (pair.second.state != GraphNodeState::COMPLETED && pair.second.state != GraphNodeState::SKIPPED && pair.second.state != GraphNodeState::CANCELLED) {
            overall_success = false;
            if (result.error_message.empty()) {
                result.error_message = "Not all nodes completed successfully.";
            }
            break;
        }
    }

    result.success = overall_success;
    return result;
}

} // namespace agr
