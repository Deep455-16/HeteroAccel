// src/execution/IExecutionNode.h
// Phase 15: Execution Graph Node Interface

#pragma once

#include "execution/ExecutionTypes.h"
#include "execution/ExecutionGraphContext.h"
#include <string>

namespace agr {

class IExecutionNode {
public:
    virtual ~IExecutionNode() = default;

    virtual uint64_t getId() const = 0;
    virtual std::string getName() const = 0;
    virtual ExecutionNodeType getType() const = 0;
    virtual ExecutionResource getResource() const = 0;

    /// Execute the node's operation using the provided context.
    /// This is a real execution (e.g. upload to Vulkan, compute on engine),
    /// not a sleep/simulation.
    virtual GraphNodeResult execute(ExecutionGraphContext& context) = 0;
};

} // namespace agr
