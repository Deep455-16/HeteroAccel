// tests/test_p31_collaborative_orchestrator.cpp
// Unified Collaborative Runtime — Multi-Engine Orchestration, DAG Waves,
// Intermediate Result Exchange, Conflict Resolution, and Fallback Tests.
//
// Validates (Section 15.2):
//   1. Query classification across code optimization, debugging, technical Q&A,
//      and general reasoning.
//   2. Three-engine collaboration (HydroXL + llama.cpp + Colibri) with DAG
//      topological wave decomposition (Wave 0 independent -> Wave 1 dependent).
//   3. Intermediate-result exchange: Wave 1 receives Wave 0 outputs in its
//      input_context.
//   4. Parallel wave execution: Wave 0 tasks run concurrently when memory allows.
//   5. Dynamic task reassignment when model architecture is unsupported by HydroXL.
//   6. Result fusion & deduplication: overlapping findings across engines are
//      deduplicated rather than blindly concatenated.
//   7. Conflict detection & resolution: opposing complexity/thread-safety claims
//      are detected, resolved, and preserved in disagreements.
//   8. Partial engine failure & fallback: when HydroXL or llama.cpp fails, the
//      orchestrator falls back to a surviving engine and completes the query.

#include "mini_test.h"
#include "collaboration/CollaborativeExecutionOrchestrator.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

using namespace agr::collaboration;

int main() {
    std::cout << "=== test_p31_collaborative_orchestrator ===\n";

    // 1. Query Classification tests
    {
        AGR_CHECK(WorkloadPartitioner::classifyQuery(
            "Optimize this C++ nested for(int i=0; i<n; ++i) loop") ==
            QueryCategory::CODE_ANALYSIS_AND_OPTIMIZATION);
        AGR_CHECK(WorkloadPartitioner::classifyQuery(
            "Find the deadlock and race condition in this mutex code") ==
            QueryCategory::DEBUGGING_AND_VERIFICATION);
        AGR_CHECK(WorkloadPartitioner::classifyQuery(
            "Explain how transformer grouped-query attention and quantization work") ==
            QueryCategory::COMPLEX_TECHNICAL_QUESTION);
        AGR_CHECK(WorkloadPartitioner::classifyQuery(
            "Summarize the key milestones of modern computing history") ==
            QueryCategory::GENERAL_REASONING_AND_SYNTHESIS);
        std::cout << "  [OK] Query understanding & category classification\n";
    }

    // 2. Three-Engine Collaboration, Parallel Wave 0, Sequential Wave 1, and
    //    Intermediate Result Exchange
    {
        CollaborativeExecutionOrchestrator orch;
        std::atomic<int> wave0_active{0};
        std::atomic<int> max_wave0_concurrent{0};
        std::string captured_colibri_upstream;

        orch.setCustomExecutor(EngineId::HYDROXL, [&](const CollaborativeTask& task, const CollaborativeRequest&) {
            int cur = ++wave0_active;
            int prev = max_wave0_concurrent.load();
            while (cur > prev && !max_wave0_concurrent.compare_exchange_weak(prev, cur)) {}
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            --wave0_active;

            IntermediateResult r;
            r.task_id = task.task_id;
            r.engine_id = "hydroxl";
            r.backend_used = "CPU";
            r.structured_result = "Tiled matrix multiplication improves cache locality by fitting blocks into L1 cache. "
                                  "Inner loop vectorization uses SIMD FMA instructions.";
            r.tokens_generated = 22;
            r.confidence_score = 0.88f;
            r.success = true;
            return r;
        });

        orch.setCustomExecutor(EngineId::LLAMA_CPP, [&](const CollaborativeTask& task, const CollaborativeRequest&) {
            int cur = ++wave0_active;
            int prev = max_wave0_concurrent.load();
            while (cur > prev && !max_wave0_concurrent.compare_exchange_weak(prev, cur)) {}
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            --wave0_active;

            IntermediateResult r;
            r.task_id = task.task_id;
            r.engine_id = "llama.cpp";
            r.backend_used = "CPU";
            // Intentionally include one overlapping finding with HydroXL to test deduplication
            r.structured_result = "Tiled matrix multiplication improves cache locality by fitting blocks into L1 cache. "
                                  "Handle non-multiple matrix dimensions with boundary tail loops.";
            r.tokens_generated = 24;
            r.confidence_score = 0.86f;
            r.success = true;
            return r;
        });

        // Leave Colibri on its REAL engine implementation to test real Colibri + upstream context exchange!
        // We also inspect the final task list to verify Colibri received HydroXL & llama.cpp outputs.
        CollaborativeRequest req;
        req.prompt = "Optimize a C++ matrix multiplication kernel with nested loops:\n"
                     "for (int i = 0; i < N; ++i) { for (int j = 0; j < N; ++j) { for (int k = 0; k < N; ++k) { C[i*N+j] += A[i*N+k]*B[k*N+j]; } } }";
        req.allow_parallel_waves = true;
        req.max_ram_budget_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL; // 16 GB budget

        CollaborativeResult res = orch.execute(req);
        AGR_CHECK(res.success);
        AGR_CHECK(res.tasks.size() == 3);
        AGR_CHECK(res.execution_waves.size() == 2);
        AGR_CHECK(res.execution_waves[0].size() == 2);
        AGR_CHECK(res.execution_waves[1].size() == 1);

        // Verify all 3 engines participated
        AGR_CHECK(res.engineParticipated("hydroxl"));
        AGR_CHECK(res.engineParticipated("llama.cpp"));
        AGR_CHECK(res.engineParticipated("colibri"));

        // Verify parallel execution occurred in Wave 0
        AGR_CHECK(res.resource_telemetry.concurrent_wave_executed);
        AGR_CHECK(max_wave0_concurrent.load() >= 2);

        // Verify intermediate-result exchange: Task 3 (Colibri in Wave 1) must have
        // received the upstream outputs from Task 1 (HydroXL) and Task 2 (llama.cpp)
        const auto& colibri_task = res.tasks[2];
        AGR_CHECK(colibri_task.assigned_engine == EngineId::COLIBRI);
        AGR_CHECK(colibri_task.wave_index == 1);
        AGR_CHECK(colibri_task.input_context.find("[Upstream hydroxl (task_1_hydroxl)]") != std::string::npos);
        AGR_CHECK(colibri_task.input_context.find("[Upstream llama.cpp (task_2_llamacpp)]") != std::string::npos);

        // Verify deduplication reduced the overlapping finding count
        AGR_CHECK(res.fusion.raw_findings_count > res.fusion.deduplicated_findings_count);
        AGR_CHECK(res.fusion.validation_passed);
        AGR_CHECK(!res.unified_response.empty());
        std::cout << "  [OK] 3-engine DAG, parallel Wave 0, intermediate exchange to Wave 1, and deduplication\n";
    }

    // 3. Conflict Detection & Resolution Test
    {
        CollaborativeExecutionOrchestrator orch;
        orch.setCustomExecutor(EngineId::HYDROXL, [](const CollaborativeTask& task, const CollaborativeRequest&) {
            IntermediateResult r;
            r.task_id = task.task_id;
            r.engine_id = "hydroxl";
            r.backend_used = "CPU";
            r.structured_result = "The algorithm runs in O(N) time and the shared queue is thread-safe.";
            r.confidence_score = 0.80f;
            r.success = true;
            return r;
        });

        orch.setCustomExecutor(EngineId::LLAMA_CPP, [](const CollaborativeTask& task, const CollaborativeRequest&) {
            IntermediateResult r;
            r.task_id = task.task_id;
            r.engine_id = "llama.cpp";
            r.backend_used = "CPU";
            r.structured_result = "Worst-case nested traversal is O(N^2) and unsynchronized access causes a race condition.";
            r.confidence_score = 0.87f;
            r.success = true;
            return r;
        });

        CollaborativeRequest req;
        req.prompt = "Verify complexity and thread safety of this nested queue processor.";
        CollaborativeResult res = orch.execute(req);
        AGR_CHECK(res.success);
        AGR_CHECK(res.fusion.conflicts_detected >= 2);
        AGR_CHECK(!res.fusion.resolved_conflicts.empty());
        AGR_CHECK(!res.fusion.preserved_disagreements.empty());
        AGR_CHECK(res.unified_response.find("Conflict Resolved:") != std::string::npos);
        std::cout << "  [OK] Cross-engine conflict detection (complexity & concurrency) and resolution\n";
    }

    // 4. Partial Engine Failure & Fallback Test
    {
        CollaborativeExecutionOrchestrator orch;
        // Inject failure on HydroXL; verify Task 1 falls back to llama.cpp and succeeds
        orch.setCustomExecutor(EngineId::HYDROXL, [](const CollaborativeTask& task, const CollaborativeRequest&) {
            IntermediateResult r;
            r.task_id = task.task_id;
            r.engine_id = "hydroxl";
            r.backend_used = "CPU";
            r.success = false;
            r.error = "Simulated HydroXL tensor allocation failure";
            return r;
        });

        orch.setCustomExecutor(EngineId::LLAMA_CPP, [](const CollaborativeTask& task, const CollaborativeRequest&) {
            IntermediateResult r;
            r.task_id = task.task_id;
            r.engine_id = "llama.cpp";
            r.backend_used = "CPU";
            r.structured_result = "Fallback execution completed structural and implementation analysis cleanly.";
            r.confidence_score = 0.85f;
            r.success = true;
            return r;
        });

        CollaborativeRequest req;
        req.prompt = "Analyze virtual memory paging and TLB shootdown overhead.";
        req.allow_engine_fallback = true;

        CollaborativeResult res = orch.execute(req);
        AGR_CHECK(res.success);
        AGR_CHECK(res.tasks[0].state == TaskState::COMPLETED);
        AGR_CHECK(res.tasks[0].fallback_occurred);
        AGR_CHECK(res.tasks[0].assigned_engine == EngineId::LLAMA_CPP);
        AGR_CHECK(res.tasks[0].fallback_reason.find("Simulated HydroXL tensor allocation failure") != std::string::npos);
        std::cout << "  [OK] Partial engine failure handled via automatic cross-engine fallback\n";
    }

    // 5. Dynamic Task Assignment when Architecture is Unsupported by HydroXL
    {
        ModelMemoryProfile unsupported_arch_prof;
        unsupported_arch_prof.arch = "falcon_mamba_experimental";
        unsupported_arch_prof.quant_type = 12;

        std::string reason;
        EngineId chosen = WorkloadPartitioner::selectEngineForTask(
            TaskType::STRUCTURAL_AND_REASONING, unsupported_arch_prof, reason);
        AGR_CHECK(chosen == EngineId::LLAMA_CPP);
        AGR_CHECK(reason.find("reassigned to llama.cpp") != std::string::npos);
        std::cout << "  [OK] Dynamic capability-aware engine reassignment for unsupported architecture\n";
    }

    AGR_TEST_MAIN_END();
}
