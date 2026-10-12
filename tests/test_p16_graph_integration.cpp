// tests/test_p16_graph_integration.cpp
// Phase 16: TensorComputeNode integration with Phase 15 ExecutionGraph.
// Builds a small graph of tensor ops and executes it via ExecutionGraphScheduler.

#include "execution/ExecutionGraph.h"
#include "execution/ExecutionGraphScheduler.h"
#include "execution/ExecutionGraphContext.h"
#include "execution/TensorComputeNode.h"
#include "primitives/cpu/CpuBackendExecutor.h"
#include "mini_test.h"

#include <iostream>
#include <cmath>
#include <memory>

using namespace agr;
using namespace agr::primitives;

static TensorShape F32(std::initializer_list<size_t> dims) {
    return TensorShape(std::vector<size_t>(dims), DataType::FLOAT32);
}

static bool near(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

// Graph: Fill(a, 3.0) -> Fill(b, 4.0) -> Add(c = a+b) [depends on both fills]
void test_fill_then_add() {
    CpuBackendExecutor exec;

    Tensor a = Tensor::zeros(F32({8}));
    Tensor b = Tensor::zeros(F32({8}));
    Tensor c = Tensor::zeros(F32({8}));

    ExecutionGraph g;

    auto n1 = std::make_unique<TensorComputeNode>(TensorComputeNode::makeFill(1, &exec, &a, 3.0f));
    auto n2 = std::make_unique<TensorComputeNode>(TensorComputeNode::makeFill(2, &exec, &b, 4.0f));
    auto n3 = std::make_unique<TensorComputeNode>(TensorComputeNode::makeAdd(3, &exec, &c, &a, &b));

    g.addNode(std::move(n1));
    g.addNode(std::move(n2));
    g.addNode(std::move(n3));

    // c depends on a and b being filled first
    g.addDependency(1, 3);
    g.addDependency(2, 3);

    std::string err;
    AGR_CHECK(g.validate(err));

    ExecutionGraphScheduler sched;
    ExecutionGraphContext ctx;
    auto res = sched.execute(g, ctx);

    AGR_CHECK(res.success);

    // Every element should be 7.0f
    const float* p = c.dataAs<const float>();
    AGR_CHECK(p != nullptr);
    for (uint64_t i = 0; i < 8; ++i) AGR_CHECK(near(p[i], 7.0f));
}

// Graph: Scale by 2 then copy result
void test_scale_then_copy() {
    CpuBackendExecutor exec;

    Tensor src = Tensor::zeros(F32({4}));
    Tensor scaled = Tensor::zeros(F32({4}));
    Tensor dst = Tensor::zeros(F32({4}));

    {
        float* p = src.dataAs<float>();
        p[0]=1.0f; p[1]=2.0f; p[2]=3.0f; p[3]=4.0f;
    }

    ExecutionGraph g;
    g.addNode(std::make_unique<TensorComputeNode>(TensorComputeNode::makeScale(1, &exec, &scaled, &src, 2.0f)));
    g.addNode(std::make_unique<TensorComputeNode>(TensorComputeNode::makeCopy(2, &exec, &dst, &scaled)));
    g.addDependency(1, 2);

    std::string err;
    AGR_CHECK(g.validate(err));

    ExecutionGraphScheduler sched;
    ExecutionGraphContext ctx;
    auto res = sched.execute(g, ctx);
    AGR_CHECK(res.success);

    const float* p = dst.dataAs<const float>();
    AGR_CHECK(near(p[0], 2.0f));
    AGR_CHECK(near(p[1], 4.0f));
    AGR_CHECK(near(p[2], 6.0f));
    AGR_CHECK(near(p[3], 8.0f));
}

// Graph: Cancellation stops tensor ops
void test_cancellation() {
    CpuBackendExecutor exec;
    Tensor t = Tensor::zeros(F32({4}));

    ExecutionGraph g;
    g.addNode(std::make_unique<TensorComputeNode>(TensorComputeNode::makeFill(1, &exec, &t, 99.0f)));

    std::string err;
    AGR_CHECK(g.validate(err));

    std::atomic<bool> cancel{true}; // already cancelled
    ExecutionGraphContext ctx(&cancel);
    ExecutionGraphScheduler sched;
    auto res = sched.execute(g, ctx);

    AGR_CHECK(!res.success);
}

int main() {
    test_fill_then_add();
    test_scale_then_copy();
    test_cancellation();
    std::cout << "test_p16_graph_integration PASSED\n";
    AGR_TEST_MAIN_END();
}
