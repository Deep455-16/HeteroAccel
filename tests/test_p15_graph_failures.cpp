// tests/test_p15_graph_failures.cpp

#include "execution/ExecutionGraphScheduler.h"
#include <cassert>
#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>

using namespace agr;

class FailNode : public IExecutionNode {
public:
    FailNode(uint64_t id) : id_(id) {}
    uint64_t getId() const override { return id_; }
    std::string getName() const override { return "FailNode"; }
    ExecutionNodeType getType() const override { return ExecutionNodeType::CUSTOM; }
    ExecutionResource getResource() const override { return ExecutionResource::NONE; }

    GraphNodeResult execute(ExecutionGraphContext&) override {
        return {false, "Intended failure for test"};
    }
private:
    uint64_t id_;
};

class SuccessNode : public IExecutionNode {
public:
    SuccessNode(uint64_t id, bool* executed_flag = nullptr) : id_(id), executed_(executed_flag) {}
    uint64_t getId() const override { return id_; }
    std::string getName() const override { return "SuccessNode"; }
    ExecutionNodeType getType() const override { return ExecutionNodeType::CUSTOM; }
    ExecutionResource getResource() const override { return ExecutionResource::NONE; }

    GraphNodeResult execute(ExecutionGraphContext&) override {
        if (executed_) *executed_ = true;
        return {true, ""};
    }
private:
    uint64_t id_;
    bool* executed_;
};

void test_failure_propagation() {
    ExecutionGraph g;
    bool executed3 = false;
    
    // 1(Fail) -> 2(Success) -> 3(Success)
    // 2 and 3 should be BLOCKED and never execute
    g.addNode(std::make_unique<FailNode>(1));
    g.addNode(std::make_unique<SuccessNode>(2));
    g.addNode(std::make_unique<SuccessNode>(3, &executed3));

    g.addDependency(1, 2);
    g.addDependency(2, 3);

    ExecutionGraphScheduler sched;
    ExecutionGraphContext ctx;
    auto res = sched.execute(g, ctx);

    assert(!res.success);
    assert(!executed3);

    // Check telemetry states
    bool found_fail = false, found_blocked = false;
    for (const auto& t : res.telemetry) {
        if (t.node_id == 1) {
            assert(t.final_state == GraphNodeState::FAILED);
            assert(!t.success);
            found_fail = true;
        } else if (t.node_id == 3) {
            assert(t.final_state == GraphNodeState::BLOCKED);
            found_blocked = true;
        }
    }
    assert(found_fail && found_blocked);
}

void test_independent_branch_continuation() {
    ExecutionGraph g;
    bool executed3 = false;
    
    // 1(Fail) -> 2(Success)
    // 3(Success) [independent]
    g.addNode(std::make_unique<FailNode>(1));
    g.addNode(std::make_unique<SuccessNode>(2));
    g.addNode(std::make_unique<SuccessNode>(3, &executed3));

    g.addDependency(1, 2); // 2 depends on 1, should block

    ExecutionGraphScheduler sched;
    ExecutionGraphContext ctx;
    auto res = sched.execute(g, ctx);

    assert(!res.success); // Overall graph fails because node 1 failed
    assert(executed3); // But 3 must have executed since it's independent
}

class CancelNode : public IExecutionNode {
public:
    CancelNode(uint64_t id) : id_(id) {}
    uint64_t getId() const override { return id_; }
    std::string getName() const override { return "CancelNode"; }
    ExecutionNodeType getType() const override { return ExecutionNodeType::CUSTOM; }
    ExecutionResource getResource() const override { return ExecutionResource::NONE; }

    GraphNodeResult execute(ExecutionGraphContext& ctx) override {
        // Trigger cancellation midway
        if (ctx.getCancelFlag()) {
            *ctx.getCancelFlag() = true;
        }
        return {true, ""};
    }
private:
    uint64_t id_;
};

void test_cancellation() {
    ExecutionGraph g;
    bool executed3 = false;
    
    // 1(Cancel) -> 2(Success) -> 3(Success)
    g.addNode(std::make_unique<CancelNode>(1));
    g.addNode(std::make_unique<SuccessNode>(2));
    g.addNode(std::make_unique<SuccessNode>(3, &executed3));

    g.addDependency(1, 2);
    g.addDependency(2, 3);

    std::atomic<bool> cancel_flag{false};
    ExecutionGraphContext ctx(&cancel_flag);
    ExecutionGraphScheduler sched;
    
    auto res = sched.execute(g, ctx);

    assert(!res.success);
    assert(res.error_message == "Execution cancelled.");
    assert(!executed3); // Should not have executed due to cancellation

    // 2 and 3 should be CANCELLED state
    for (const auto& t : res.telemetry) {
        if (t.node_id == 2 || t.node_id == 3) {
            assert(t.final_state == GraphNodeState::CANCELLED);
        }
    }
}

int main() {
    test_failure_propagation();
    test_independent_branch_continuation();
    test_cancellation();
    std::cout << "test_p15_graph_failures PASSED\n";
    return 0;
}
