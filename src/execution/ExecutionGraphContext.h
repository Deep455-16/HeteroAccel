// src/execution/ExecutionGraphContext.h
// Phase 15: Execution Graph Context
// Provides required services to nodes during execution.

#pragma once

#include "execution/ExecutionTypes.h"
#include <atomic>
#include <string>

namespace agr {

class ModelResidencyManager;
class IExecutionEngine;

struct NodeTelemetry {
    uint64_t node_id = 0;
    ExecutionNodeType type = ExecutionNodeType::CUSTOM;
    ExecutionResource resource = ExecutionResource::NONE;
    
    double start_time_ms = 0.0;
    double end_time_ms = 0.0;
    double duration_ms = 0.0;
    
    GraphNodeState final_state = GraphNodeState::CREATED;
    bool success = false;
    std::string failure_reason;
};

class ExecutionGraphContext {
public:
    ExecutionGraphContext(std::atomic<bool>* cancel_flag = nullptr)
        : cancel_flag_(cancel_flag) {}

    virtual ~ExecutionGraphContext() = default;

    /// Is cancellation requested?
    bool isCancelled() const {
        if (!cancel_flag_) return false;
        return cancel_flag_->load(std::memory_order_relaxed);
    }

    std::atomic<bool>* getCancelFlag() const { return cancel_flag_; }

    // Dependencies (Phase 14 integration and Engine integration)
    virtual ModelResidencyManager* getResidencyManager() const { return nullptr; }
    virtual IExecutionEngine* getEngine() const { return nullptr; }

private:
    std::atomic<bool>* cancel_flag_ = nullptr;
};

struct GraphNodeResult {
    bool success = false;
    std::string error_message;
};

} // namespace agr
