// src/execution/ExecutionNodes.h
// Phase 15: Standard Execution Nodes

#pragma once

#include "execution/IExecutionNode.h"
#include "model/ModelRegion.h"
#include "model/ModelResidencyManager.h"
#include "engine/IExecutionEngine.h"

namespace agr {

class ResidencyNode : public IExecutionNode {
public:
    ResidencyNode(uint64_t id, const std::string& name, ModelRegion region, ModelResidencyTarget target, ExecutionResource res)
        : id_(id), name_(name), region_(region), target_(target), resource_(res) {}

    uint64_t getId() const override { return id_; }
    std::string getName() const override { return name_; }
    ExecutionNodeType getType() const override { return ExecutionNodeType::RESIDENCY; }
    ExecutionResource getResource() const override { return resource_; }

    GraphNodeResult execute(ExecutionGraphContext& context) override {
        GraphNodeResult res;
        auto* mgr = context.getResidencyManager();
        if (!mgr) {
            res.success = false;
            res.error_message = "No ResidencyManager available in context";
            return res;
        }

        bool ok = mgr->ensureResident(region_, target_, context.getCancelFlag());
        if (ok) {
            res.success = true;
        } else {
            res.success = false;
            res.error_message = mgr->lastError();
        }
        return res;
    }

private:
    uint64_t id_;
    std::string name_;
    ModelRegion region_;
    ModelResidencyTarget target_;
    ExecutionResource resource_;
};

class ComputeNode : public IExecutionNode {
public:
    ComputeNode(uint64_t id, const std::string& name, const std::string& prompt, ExecutionResource res)
        : id_(id), name_(name), prompt_(prompt), resource_(res) {}

    uint64_t getId() const override { return id_; }
    std::string getName() const override { return name_; }
    ExecutionNodeType getType() const override { return ExecutionNodeType::COMPUTE; }
    ExecutionResource getResource() const override { return resource_; }

    GraphNodeResult execute(ExecutionGraphContext& context) override {
        GraphNodeResult res;
        auto* engine = context.getEngine();
        if (!engine) {
            res.success = false;
            res.error_message = "No ExecutionEngine available in context";
            return res;
        }

        // Real execution via engine abstraction boundary.
        // We do not fake operations here. We use the highest fidelity operation 
        // currently exposed by the engine, which is the execute() method.
        agr::ExecutionContext ectx;
        ectx.prompt = prompt_;
        ectx.cancel_flag = context.getCancelFlag();
        
        auto result = engine->execute(ectx);
        if (result.success) {
            res.success = true;
        } else {
            res.success = false;
            res.error_message = result.error;
        }
        return res;
    }

private:
    uint64_t id_;
    std::string name_;
    std::string prompt_;
    ExecutionResource resource_;
};

class SynchronizationNode : public IExecutionNode {
public:
    SynchronizationNode(uint64_t id, const std::string& name)
        : id_(id), name_(name) {}

    uint64_t getId() const override { return id_; }
    std::string getName() const override { return name_; }
    ExecutionNodeType getType() const override { return ExecutionNodeType::SYNCHRONIZATION; }
    ExecutionResource getResource() const override { return ExecutionResource::NONE; }

    GraphNodeResult execute(ExecutionGraphContext&) override {
        // Just acts as a barrier in the DAG. Execution completes immediately.
        GraphNodeResult res;
        res.success = true;
        return res;
    }

private:
    uint64_t id_;
    std::string name_;
};

} // namespace agr
