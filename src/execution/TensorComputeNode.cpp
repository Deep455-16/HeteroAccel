// src/execution/TensorComputeNode.cpp
// Phase 16: TensorComputeNode execute() implementation.

#include "execution/TensorComputeNode.h"
#include "execution/ExecutionGraphContext.h"

namespace agr {

GraphNodeResult TensorComputeNode::execute(ExecutionGraphContext& ctx) {
    // Check cancellation first
    if (ctx.isCancelled()) {
        return {false, "TensorComputeNode: execution cancelled"};
    }

    if (!exec_) {
        return {false, "TensorComputeNode: no IBackendExecutor provided"};
    }

    if (!exec_->isAvailable()) {
        return {false, std::string("TensorComputeNode: backend '") + exec_->name() + "' is not available"};
    }

    primitives::BackendOpResult res;

    switch (op_) {
    case TensorOpType::COPY:
        if (!dst_ || !a_) return {false, "TensorComputeNode(COPY): null tensor pointer"};
        res = exec_->copy(*dst_, *a_);
        break;

    case TensorOpType::FILL:
        if (!dst_) return {false, "TensorComputeNode(FILL): null dst tensor"};
        res = exec_->fill(*dst_, scalar_);
        break;

    case TensorOpType::ADD:
        if (!dst_ || !a_ || !b_) return {false, "TensorComputeNode(ADD): null tensor pointer"};
        res = exec_->add(*dst_, *a_, *b_);
        break;

    case TensorOpType::MULTIPLY:
        if (!dst_ || !a_ || !b_) return {false, "TensorComputeNode(MULTIPLY): null tensor pointer"};
        res = exec_->multiply(*dst_, *a_, *b_);
        break;

    case TensorOpType::SCALE:
        if (!dst_ || !a_) return {false, "TensorComputeNode(SCALE): null tensor pointer"};
        res = exec_->scale(*dst_, *a_, scalar_);
        break;

    default:
        return {false, std::string("TensorComputeNode: unsupported op type: ") + toString(op_)};
    }

    if (!res.success) {
        return {false, res.error};
    }
    return {true, ""};
}

} // namespace agr
