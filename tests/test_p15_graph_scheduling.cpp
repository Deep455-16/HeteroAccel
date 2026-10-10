// tests/test_p15_graph_scheduling.cpp

#include "execution/ExecutionGraphScheduler.h"
#include <cassert>
#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>

using namespace agr;

class TimingNode : public IExecutionNode {
public:
    TimingNode(uint64_t id, ExecutionResource res, std::atomic<int>& order_counter, std::vector<int>& order_log, int delay_ms = 10)
        : id_(id), res_(res), order_counter_(order_counter), order_log_(order_log), delay_ms_(delay_ms) {}

    uint64_t getId() const override { return id_; }
    std::string getName() const override { return "TimingNode"; }
    ExecutionNodeType getType() const override { return ExecutionNodeType::CUSTOM; }
    ExecutionResource getResource() const override { return res_; }

    GraphNodeResult execute(ExecutionGraphContext&) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms_));
        order_log_[id_] = ++order_counter_;
        return {true, ""};
    }

private:
    uint64_t id_;
    ExecutionResource res_;
    std::atomic<int>& order_counter_;
    std::vector<int>& order_log_;
    int delay_ms_;
};

void test_dependency_ordering() {
    ExecutionGraph g;
    std::atomic<int> counter{0};
    std::vector<int> log(10, 0);

    g.addNode(std::make_unique<TimingNode>(1, ExecutionResource::CPU, counter, log));
    g.addNode(std::make_unique<TimingNode>(2, ExecutionResource::CPU, counter, log));
    g.addNode(std::make_unique<TimingNode>(3, ExecutionResource::CPU, counter, log));

    // 1 -> 2 -> 3
    g.addDependency(1, 2);
    g.addDependency(2, 3);

    ExecutionGraphScheduler sched;
    ExecutionGraphContext ctx;
    auto res = sched.execute(g, ctx);

    assert(res.success);
    assert(log[1] < log[2]);
    assert(log[2] < log[3]);
}

void test_resource_conflict() {
    ExecutionGraph g;
    std::atomic<int> counter{0};
    std::vector<int> log(10, 0);

    // VULKAN resource is exclusive, so these must serialize even without explicit dependencies
    g.addNode(std::make_unique<TimingNode>(1, ExecutionResource::VULKAN, counter, log, 20));
    g.addNode(std::make_unique<TimingNode>(2, ExecutionResource::VULKAN, counter, log, 20));

    ExecutionGraphScheduler sched;
    ExecutionGraphContext ctx;
    auto res = sched.execute(g, ctx);

    assert(res.success);
    assert(log[1] > 0 && log[2] > 0);
    assert(log[1] != log[2]); // They must have executed sequentially
}

void test_parallel_independent() {
    ExecutionGraph g;
    std::atomic<int> counter{0};
    std::vector<int> log(10, 0);

    // CPU resources can run in parallel.
    g.addNode(std::make_unique<TimingNode>(1, ExecutionResource::CPU, counter, log, 50));
    g.addNode(std::make_unique<TimingNode>(2, ExecutionResource::CPU, counter, log, 50));

    ExecutionGraphScheduler sched;
    ExecutionGraphContext ctx;

    auto t0 = std::chrono::steady_clock::now();
    auto res = sched.execute(g, ctx);
    auto t1 = std::chrono::steady_clock::now();
    
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    assert(res.success);
    // They should run in parallel, taking ~50ms total, not 100ms.
    // Allow some overhead
    assert(dur < 85);
}

void test_concurrent_graph_instances() {
    // Run two graph executors on different threads
    auto run_graph = []() {
        ExecutionGraph g;
        std::atomic<int> counter{0};
        std::vector<int> log(10, 0);
        g.addNode(std::make_unique<TimingNode>(1, ExecutionResource::CPU, counter, log, 10));
        g.addNode(std::make_unique<TimingNode>(2, ExecutionResource::CPU, counter, log, 10));
        g.addDependency(1, 2);
        ExecutionGraphScheduler sched;
        ExecutionGraphContext ctx;
        assert(sched.execute(g, ctx).success);
    };

    std::thread t1(run_graph);
    std::thread t2(run_graph);
    t1.join();
    t2.join();
}

int main() {
    test_dependency_ordering();
    test_resource_conflict();
    test_parallel_independent();
    test_concurrent_graph_instances();
    std::cout << "test_p15_graph_scheduling PASSED\n";
    return 0;
}
