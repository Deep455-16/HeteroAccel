// src/execution/TensorComputeNode.h
// Phase 16: Execution graph node that dispatches a tensor operation via
//           IBackendExecutor (Phase 16 backend abstraction).
//
// This extends the Phase 15 ExecutionNodes.h set. It follows the same
// IExecutionNode contract. The node holds references to:
//   - an IBackendExecutor (supplied at construction)
//   - input Tensors (non-owning pointers into caller-managed storage)
//   - output Tensor (non-owning pointer)
//
// Ownership rules:
//   - TensorComputeNode does NOT own the tensors or the executor.
//   - Caller must ensure all referenced objects outlive the graph execution.
//
// Supported operation types (Phase 16):
//   COPY, FILL, ADD, MULTIPLY, SCALE, SUM, MAX
//
// On unsupported op or unavailable backend: returns GraphNodeResult::fail.
// On shape/type mismatch: returns GraphNodeResult::fail.

#pragma once

#include "execution/IExecutionNode.h"
#include "execution/ExecutionTypes.h"
#include "primitives/IBackendExecutor.h"
#include "primitives/Tensor.h"

#include <string>

namespace agr {

// Tensor operation types supported by TensorComputeNode
enum class TensorOpType {
    COPY,
    FILL,
    ADD,
    MULTIPLY,
    SCALE,
    SUM,
    MAX,
};

inline const char* toString(TensorOpType op) {
    switch (op) {
        case TensorOpType::COPY:     return "COPY";
        case TensorOpType::FILL:     return "FILL";
        case TensorOpType::ADD:      return "ADD";
        case TensorOpType::MULTIPLY: return "MULTIPLY";
        case TensorOpType::SCALE:    return "SCALE";
        case TensorOpType::SUM:      return "SUM";
        case TensorOpType::MAX:      return "MAX";
        default:                     return "UNKNOWN";
    }
}

class TensorComputeNode : public IExecutionNode {
public:
    // ----- Constructors for each operation -----

    /// COPY: dst = src
    static TensorComputeNode makeCopy(uint64_t id,
                                       primitives::IBackendExecutor* exec,
                                       primitives::Tensor* dst,
                                       primitives::Tensor* src) {
        TensorComputeNode n;
        n.id_ = id; n.name_ = "TensorCopy"; n.op_ = TensorOpType::COPY;
        n.exec_ = exec; n.dst_ = dst; n.a_ = src;
        return n;
    }

    /// FILL: dst = fill(value)
    static TensorComputeNode makeFill(uint64_t id,
                                       primitives::IBackendExecutor* exec,
                                       primitives::Tensor* dst,
                                       float value) {
        TensorComputeNode n;
        n.id_ = id; n.name_ = "TensorFill"; n.op_ = TensorOpType::FILL;
        n.exec_ = exec; n.dst_ = dst; n.scalar_ = value;
        return n;
    }

    /// ADD: dst = a + b
    static TensorComputeNode makeAdd(uint64_t id,
                                      primitives::IBackendExecutor* exec,
                                      primitives::Tensor* dst,
                                      primitives::Tensor* a,
                                      primitives::Tensor* b) {
        TensorComputeNode n;
        n.id_ = id; n.name_ = "TensorAdd"; n.op_ = TensorOpType::ADD;
        n.exec_ = exec; n.dst_ = dst; n.a_ = a; n.b_ = b;
        return n;
    }

    /// MULTIPLY: dst = a * b  (elementwise)
    static TensorComputeNode makeMultiply(uint64_t id,
                                           primitives::IBackendExecutor* exec,
                                           primitives::Tensor* dst,
                                           primitives::Tensor* a,
                                           primitives::Tensor* b) {
        TensorComputeNode n;
        n.id_ = id; n.name_ = "TensorMultiply"; n.op_ = TensorOpType::MULTIPLY;
        n.exec_ = exec; n.dst_ = dst; n.a_ = a; n.b_ = b;
        return n;
    }

    /// SCALE: dst = src * scalar
    static TensorComputeNode makeScale(uint64_t id,
                                        primitives::IBackendExecutor* exec,
                                        primitives::Tensor* dst,
                                        primitives::Tensor* src,
                                        float scalar) {
        TensorComputeNode n;
        n.id_ = id; n.name_ = "TensorScale"; n.op_ = TensorOpType::SCALE;
        n.exec_ = exec; n.dst_ = dst; n.a_ = src; n.scalar_ = scalar;
        return n;
    }

    // IExecutionNode interface
    uint64_t          getId()       const override { return id_; }
    std::string       getName()     const override { return name_; }
    ExecutionNodeType getType()     const override { return ExecutionNodeType::COMPUTE; }
    ExecutionResource getResource() const override { return ExecutionResource::CPU; }

    GraphNodeResult execute(ExecutionGraphContext& ctx) override;

private:
    TensorComputeNode() = default;

    uint64_t   id_     = 0;
    std::string name_;
    TensorOpType op_   = TensorOpType::COPY;

    primitives::IBackendExecutor* exec_ = nullptr;
    primitives::Tensor*           dst_  = nullptr;
    primitives::Tensor*           a_    = nullptr;
    primitives::Tensor*           b_    = nullptr;
    float                         scalar_ = 0.0f;
};

} // namespace agr
