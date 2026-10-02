// tests/test_p9_workload_registry.cpp
// Phase 9: WorkloadRegistry - priority, isolation, cancellation
#include "mini_test.h"
#include "scheduler/WorkloadRegistry.h"
#include <thread>

int main() {
    std::cout << "== test_p9_workload_registry ==\n";
    agr::WorkloadRegistry reg;

    // 1. Register workloads
    uint64_t id1 = reg.registerWorkload("llm-a", "/model/a.gguf",
        agr::WorkloadPriority::NORMAL, agr::WorkloadClass::INTERACTIVE);
    uint64_t id2 = reg.registerWorkload("llm-b", "/model/b.gguf",
        agr::WorkloadPriority::HIGH, agr::WorkloadClass::BATCH);
    uint64_t id3 = reg.registerWorkload("bg-embed", "/model/c.gguf",
        agr::WorkloadPriority::BACKGROUND, agr::WorkloadClass::BACKGROUND);
    AGR_CHECK(id1 != id2 && id2 != id3);
    AGR_CHECK(reg.totalCount() == 3);
    AGR_CHECK(reg.activeCount() == 3);
    std::cout << "  ok: 3 workloads registered\n";

    // 2. Priority sorting: HIGH should come first
    auto sorted = reg.activeWorkloadsSorted();
    AGR_CHECK(sorted.size() == 3);
    AGR_CHECK(sorted[0]->priority == agr::WorkloadPriority::HIGH);
    AGR_CHECK(sorted[2]->priority == agr::WorkloadPriority::BACKGROUND);
    std::cout << "  ok: priority ordering correct\n";

    // 3. State transitions
    reg.setRunning(id1);
    AGR_CHECK(reg.getState(id1) == agr::WorkloadState::RUNNING);
    reg.setCompleted(id1);
    AGR_CHECK(reg.getState(id1) == agr::WorkloadState::COMPLETED);
    AGR_CHECK(reg.activeCount() == 2); // id1 done
    std::cout << "  ok: state transitions work\n";

    // 4. Cancellation
    AGR_CHECK(!reg.isCancelled(id2));
    bool cancelled = reg.cancel(id2);
    AGR_CHECK(cancelled);
    AGR_CHECK(reg.isCancelled(id2));
    std::cout << "  ok: cancellation works\n";

    // 5. cancelFlag pointer
    auto* flag = reg.cancelFlag(id3);
    AGR_CHECK(flag != nullptr);
    AGR_CHECK(!flag->load());
    reg.cancel(id3);
    AGR_CHECK(flag->load());
    std::cout << "  ok: cancel flag pointer works\n";

    // 6. Failure recording
    reg.setFailed(id3, "backend error");
    AGR_CHECK(reg.getState(id3) == agr::WorkloadState::FAILED);
    AGR_CHECK(reg.activeCount() == 0);
    std::cout << "  ok: failure recording works\n";

    // 7. Snapshot
    auto snap = reg.snapshot();
    AGR_CHECK(snap.size() == 3);
    std::cout << "  ok: snapshot returns all entries\n";

    // 8. Isolation: failing one doesn't affect another
    uint64_t id4 = reg.registerWorkload("llm-d", "/model/d.gguf");
    uint64_t id5 = reg.registerWorkload("llm-e", "/model/e.gguf");
    reg.setRunning(id4);
    reg.setFailed(id4, "OOM");
    AGR_CHECK(reg.getState(id4) == agr::WorkloadState::FAILED);
    AGR_CHECK(reg.getState(id5) == agr::WorkloadState::QUEUED);
    std::cout << "  ok: failure isolation works\n";

    AGR_TEST_MAIN_END();
}
