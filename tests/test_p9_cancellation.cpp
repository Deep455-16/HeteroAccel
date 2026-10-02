// tests/test_p9_cancellation.cpp
// Phase 9: Workload cancellation — signal propagates, resources not corrupted.
#include "mini_test.h"
#include "scheduler/WorkloadRegistry.h"
#include <thread>
#include <chrono>
#include <atomic>

using namespace agr;

int main() {
    std::cout << "== test_p9_cancellation ==\n";

    WorkloadRegistry reg;

    // 1. Register and immediately cancel a QUEUED workload
    uint64_t id1 = reg.registerWorkload("fast_cancel", "/model/x.gguf",
        WorkloadPriority::NORMAL, WorkloadClass::INTERACTIVE);
    AGR_CHECK(reg.getState(id1) == WorkloadState::QUEUED);
    reg.cancel(id1);
    AGR_CHECK(reg.getState(id1) == WorkloadState::CANCELLED);
    AGR_CHECK(reg.isCancelled(id1));
    std::cout << "  ok: queued workload cancelled immediately\n";

    // 2. Cancel a running workload - flag set, state not auto-completed
    uint64_t id2 = reg.registerWorkload("running_cancel", "/model/y.gguf",
        WorkloadPriority::HIGH, WorkloadClass::BATCH);
    reg.setRunning(id2);
    AGR_CHECK(reg.getState(id2) == WorkloadState::RUNNING);
    reg.cancel(id2);
    AGR_CHECK(reg.isCancelled(id2));
    // State stays RUNNING until the execution code observes cancel_flag and calls setCancelled()
    reg.setCancelled(id2);
    AGR_CHECK(reg.getState(id2) == WorkloadState::CANCELLED);
    std::cout << "  ok: running workload can be cancelled\n";

    // 3. Cancel flag propagation to inference code via pointer
    uint64_t id3 = reg.registerWorkload("llm_gen", "/model/llm.gguf");
    std::atomic<bool>* flag = reg.cancelFlag(id3);
    AGR_CHECK(flag != nullptr);
    AGR_CHECK(!flag->load());

    // Simulate a background generation observing the flag
    std::atomic<bool> work_stopped{false};
    std::thread worker([&]() {
        for (int i = 0; i < 200; ++i) {
            if (flag->load(std::memory_order_acquire)) {
                work_stopped.store(true, std::memory_order_release);
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    reg.cancel(id3);
    worker.join();

    AGR_CHECK(work_stopped.load());
    std::cout << "  ok: cancel flag stops background worker\n";

    // 4. Cancellation does not affect unrelated workloads
    uint64_t id4 = reg.registerWorkload("safe_workload", "/model/z.gguf",
        WorkloadPriority::NORMAL, WorkloadClass::DEFAULT);
    uint64_t id5 = reg.registerWorkload("victim", "/model/v.gguf",
        WorkloadPriority::BACKGROUND, WorkloadClass::BACKGROUND);

    reg.cancel(id5);
    AGR_CHECK(reg.isCancelled(id5));
    AGR_CHECK(!reg.isCancelled(id4));
    AGR_CHECK(reg.getState(id4) == WorkloadState::QUEUED);
    std::cout << "  ok: cancelling one workload does not affect another\n";

    // 5. Invalid cancel returns false
    bool res = reg.cancel(99999);
    AGR_CHECK(!res);
    std::cout << "  ok: invalid cancel returns false\n";

    // 6. Prune old completed workloads
    uint64_t id6 = reg.registerWorkload("done", "/model/d.gguf");
    reg.setCompleted(id6);
    size_t before = reg.totalCount();
    reg.pruneOld(0.0); // 0 seconds = prune all terminal
    size_t after = reg.totalCount();
    AGR_CHECK(after < before);
    std::cout << "  ok: pruneOld removes completed entries\n";

    AGR_TEST_MAIN_END();
}
