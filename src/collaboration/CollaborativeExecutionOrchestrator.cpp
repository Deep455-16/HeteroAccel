// src/collaboration/CollaborativeExecutionOrchestrator.cpp
// Central Collaborative Multi-Engine Orchestrator implementation.

#include "collaboration/CollaborativeExecutionOrchestrator.h"
#include "colibri/ColibriEngine.h"
#include "engine/ExecutionEngineRegistry.h"
#include "hydroxl/HydroXLEngine.h"
#include "orchestration/EngineOrchestrator.h"
#include "orchestration/HardwareDetector.h"
#include "orchestration/OrchestratorInit.h"

#include <chrono>
#include <future>
#include <iomanip>
#include <sstream>

namespace agr {
namespace collaboration {

std::string CollaborativeResult::formatTelemetryReport() const {
    std::ostringstream out;
    out << "=== HeteroAccel Collaborative Execution Telemetry ===\n";
    out << "Status          : " << (success ? "SUCCESS" : ("FAILED (" + error + ")")) << "\n";
    out << "Query Category  : " << queryCategoryToString(query_category) << "\n";
    out << "Timing (ms)     : Total=" << std::fixed << std::setprecision(2) << total_ms
        << " (Plan=" << planning_ms
        << ", Exec=" << execution_ms
        << ", Fusion=" << fusion_ms << ")\n";
    out << "Execution Waves : " << execution_waves.size()
        << " (ConcurrentWave=" << (resource_telemetry.concurrent_wave_executed ? "YES" : "NO")
        << ", MemoryConstrainedSwap=" << (resource_telemetry.memory_constrained_mode ? "YES" : "NO")
        << ")\n";
    out << "Memory Budget   : " << (resource_telemetry.effective_ram_budget_bytes / (1024 * 1024))
        << " MB (Available RAM=" << (resource_telemetry.available_ram_bytes / (1024 * 1024))
        << " MB, Peak Resident=" << (resource_telemetry.peak_resident_bytes / (1024 * 1024))
        << " MB, Loaded=" << resource_telemetry.models_loaded_count
        << ", Evicted=" << resource_telemetry.models_evicted_count << ")\n\n";

    out << "--- Task DAG Execution Breakdown ---\n";
    for (const auto& t : tasks) {
        out << "  [" << t.task_id << "] Wave " << t.wave_index
            << " | Engine=" << engineIdToString(t.assigned_engine)
            << " | Backend=" << hardwareBackendToString(t.assigned_backend)
            << " (" << t.orchestrator_backend_id << ")"
            << " | State=" << taskStateToString(t.state)
            << " | Time=" << std::fixed << std::setprecision(2) << t.execution_ms << " ms"
            << " | Conf=" << std::setprecision(2) << t.output.confidence_score;
        if (t.fallback_occurred) {
            out << " | Fallback: " << t.fallback_reason;
        }
        out << "\n";
    }

    out << "\n--- Engine Participation Summary ---\n";
    for (const auto& c : fusion.engine_contributions) {
        out << "  Engine: " << std::left << std::setw(10) << c.engine_id
            << " | Backend: " << std::setw(8) << c.primary_backend_used
            << " | Completed: " << c.tasks_completed << "/" << c.tasks_assigned
            << " | Tokens: " << c.total_tokens_generated
            << " | Time: " << std::fixed << std::setprecision(2) << c.total_execution_ms << " ms"
            << " | AvgConf: " << std::setprecision(2) << c.average_confidence << "\n";
    }

    out << "\n--- Validation & Fusion Metrics ---\n";
    out << "  Validation Passed : " << (fusion.validation_passed ? "YES" : "NO") << "\n";
    out << "  Overall Confidence: " << std::fixed << std::setprecision(2) << fusion.overall_confidence << "\n";
    out << "  Findings          : " << fusion.raw_findings_count << " raw -> "
        << fusion.deduplicated_findings_count << " deduplicated\n";
    out << "  Conflicts Detected: " << fusion.conflicts_detected << "\n";
    return out.str();
}

CollaborativeExecutionOrchestrator::CollaborativeExecutionOrchestrator() {
    ensureEnginesRegistered();
}

CollaborativeExecutionOrchestrator::~CollaborativeExecutionOrchestrator() {
    resource_mgr_.evictAll();
}

void CollaborativeExecutionOrchestrator::ensureEnginesRegistered() {
    // 1. Ensure all 3 engines are registered in ExecutionEngineRegistry
    auto& reg = ExecutionEngineRegistry::instance();
    if (!reg.isRegistered("hydroxl")) {
        HydroXLEngine::registerInRegistry();
    }
    if (!reg.isRegistered("colibri")) {
        ColibriEngine::registerInRegistry();
    }

    // 2. Ensure default backends are registered in EngineOrchestrator
    auto& hw = orchestration::HardwareDetector::instance();
    if (hw.devices().empty()) {
        hw.detect();
    }
    auto& orch = orchestration::EngineOrchestrator::instance();
    if (orch.backends().empty()) {
        orchestration::register_default_backends(orch, hw);
    }
}

void CollaborativeExecutionOrchestrator::setCustomExecutor(
    EngineId engine,
    CustomTaskExecutor executor) {
    std::lock_guard<std::mutex> lk(custom_exec_mu_);
    custom_executors_[engine] = std::move(executor);
}

void CollaborativeExecutionOrchestrator::clearCustomExecutors() {
    std::lock_guard<std::mutex> lk(custom_exec_mu_);
    custom_executors_.clear();
}

IntermediateResult CollaborativeExecutionOrchestrator::runEngineBackend(
    EngineId engine,
    HardwareBackend backend,
    const std::string& orchestrator_backend_id,
    const CollaborativeTask& task,
    const CollaborativeRequest& req,
    const ModelMemoryProfile& model_profile,
    bool memory_constrained_mode) {

    IntermediateResult out;
    out.task_id = task.task_id;
    out.engine_id = engineIdToString(engine);
    out.backend_used = hardwareBackendToString(backend);
    out.orchestrator_backend_id = orchestrator_backend_id;

    // 1. Check for custom test executor override first
    {
        CustomTaskExecutor custom_fn;
        {
            std::lock_guard<std::mutex> lk(custom_exec_mu_);
            auto it = custom_executors_.find(engine);
            if (it != custom_executors_.end()) {
                custom_fn = it->second;
            }
        }
        if (custom_fn) {
            auto t0 = std::chrono::steady_clock::now();
            out = custom_fn(task, req);
            auto t1 = std::chrono::steady_clock::now();
            if (out.execution_ms <= 0.0) {
                out.execution_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
            ResultFusionEngine::enrichIntermediateResult(task, out);
            return out;
        }
    }

    auto t0 = std::chrono::steady_clock::now();

    // 2. Specialized Colibri execution path (passes both original query and upstream
    //    intermediate context into ColibriEngine::analyze)
    if (engine == EngineId::COLIBRI) {
        std::string err;
        auto inf_eng = resource_mgr_.acquireEngine(
            "colibri_cpu", req.model_path, model_profile.colibri_estimated_bytes,
            memory_constrained_mode, err);
        if (!inf_eng) {
            out.success = false;
            out.error = "Colibri initialization failed: " + err;
            return out;
        }

        auto* colibri_adapter = dynamic_cast<orchestration::ColibriEngineAdapter*>(inf_eng.get());
        if (colibri_adapter) {
            ColibriAnalysisReport rep = colibri_adapter->engine().analyze(req.prompt, task.input_context);
            auto t1 = std::chrono::steady_clock::now();
            out.execution_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (out.execution_ms < 0.05) out.execution_ms = 0.05;
            out.structured_result = rep.formatted_summary;
            out.key_findings = rep.key_findings;
            for (const auto& r : rep.detected_risks) {
                out.key_findings.push_back("Risk: " + r);
            }
            for (const auto& a : rep.optimization_actions) {
                out.key_findings.push_back("Optimization: " + a);
            }
            out.supporting_evidence = rep.supporting_evidence;
            out.confidence_score = rep.confidence_score;
            out.validated = rep.validated;
            out.tokens_generated = static_cast<int>(rep.formatted_summary.size() / 4) + 1;
            out.tokens_per_second = static_cast<float>(out.tokens_generated / (out.execution_ms / 1000.0));
            out.peak_memory_bytes = colibri_adapter->engine().estimatedWorkspaceBytes();
            out.gpu_used = false;
            out.success = true;
            ResultFusionEngine::enrichIntermediateResult(task, out);
            return out;
        }
    }

    // 3. Neural Engine execution path (HydroXL or llama.cpp) via CollaborativeResourceManager
    if (req.model_path.empty()) {
        out.success = false;
        out.error = "Model path is empty for neural engine '" + out.engine_id + "'";
        return out;
    }

    size_t est_bytes = (engine == EngineId::HYDROXL)
                           ? model_profile.hydroxl_estimated_bytes
                           : model_profile.llamacpp_estimated_bytes;

    std::string init_err;
    auto inf_eng = resource_mgr_.acquireEngine(
        orchestrator_backend_id, req.model_path, est_bytes,
        memory_constrained_mode, init_err);
    if (!inf_eng) {
        out.success = false;
        out.error = init_err.empty() ? ("Failed to initialize " + orchestrator_backend_id) : init_err;
        return out;
    }

    orchestration::InferenceRequest ireq;
    ireq.prompt = task.input_context.empty() ? req.prompt : task.input_context;
    ireq.max_new_tokens = req.max_new_tokens_per_task > 0 ? req.max_new_tokens_per_task : 16;
    ireq.temperature = req.temperature;
    ireq.top_p = req.top_p;

    orchestration::InferenceResult ires;
    if (backend != HardwareBackend::CPU) {
        // Coordinate GPU execution lock when running on Vulkan or CUDA
        auto gpu_lock = resource_mgr_.acquireGpuExecutionLock();
        ires = inf_eng->run(ireq);
    } else {
        ires = inf_eng->run(ireq);
    }

    auto t1 = std::chrono::steady_clock::now();
    out.execution_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    out.structured_result = ires.output_text;
    out.prompt_tokens = ires.prompt_tokens;
    out.tokens_generated = ires.output_tokens;
    out.tokens_per_second = ires.tokens_per_second;
    out.peak_memory_bytes = ires.peak_memory_bytes > 0 ? ires.peak_memory_bytes : est_bytes;
    out.gpu_used = ires.gpu_used;
    out.success = ires.success;
    out.error = ires.error;

    ResultFusionEngine::enrichIntermediateResult(task, out);
    return out;
}

IntermediateResult CollaborativeExecutionOrchestrator::executeSingleTask(
    CollaborativeTask& task,
    const CollaborativeRequest& req,
    const ModelMemoryProfile& model_profile,
    bool memory_constrained_mode) {

    if (req.cancel_flag && req.cancel_flag->load()) {
        task.state = TaskState::CANCELLED;
        task.output.success = false;
        task.output.error = "Cancelled before task execution";
        return task.output;
    }

    task.state = TaskState::RUNNING;
    auto t0 = std::chrono::steady_clock::now();

    IntermediateResult res = runEngineBackend(
        task.assigned_engine,
        task.assigned_backend,
        task.orchestrator_backend_id,
        task,
        req,
        model_profile,
        memory_constrained_mode);

    // 1. Hardware backend fallback (GPU -> CPU within the same engine)
    if (!res.success && task.assigned_backend != HardwareBackend::CPU) {
        std::string primary_err = res.error;
        std::string cpu_backend_id = (task.assigned_engine == EngineId::HYDROXL)
                                         ? "hydroxl_cpu"
                                         : "llamacpp_cpu";
        CollaborativeTask fallback_task = task;
        fallback_task.assigned_backend = HardwareBackend::CPU;
        fallback_task.orchestrator_backend_id = cpu_backend_id;

        IntermediateResult cpu_res = runEngineBackend(
            fallback_task.assigned_engine,
            HardwareBackend::CPU,
            cpu_backend_id,
            fallback_task,
            req,
            model_profile,
            memory_constrained_mode);

        if (cpu_res.success) {
            task.assigned_backend = HardwareBackend::CPU;
            task.orchestrator_backend_id = cpu_backend_id;
            task.fallback_occurred = true;
            task.fallback_reason = "GPU backend failed (" + primary_err + "); fell back to " + cpu_backend_id;
            res = std::move(cpu_res);
        }
    }

    // 2. Cross-engine fallback (if the assigned engine itself failed and allow_engine_fallback is enabled)
    if (!res.success && req.allow_engine_fallback) {
        std::string primary_eng_err = res.error;
        std::vector<std::pair<EngineId, std::string>> backup_candidates;
        if (task.assigned_engine == EngineId::HYDROXL) {
            backup_candidates.push_back({EngineId::LLAMA_CPP, "llamacpp_cpu"});
            backup_candidates.push_back({EngineId::COLIBRI, "colibri_cpu"});
        } else if (task.assigned_engine == EngineId::LLAMA_CPP) {
            backup_candidates.push_back({EngineId::HYDROXL, "hydroxl_cpu"});
            backup_candidates.push_back({EngineId::COLIBRI, "colibri_cpu"});
        } else {
            backup_candidates.push_back({EngineId::HYDROXL, "hydroxl_cpu"});
            backup_candidates.push_back({EngineId::LLAMA_CPP, "llamacpp_cpu"});
        }

        for (const auto& cand : backup_candidates) {
            CollaborativeTask fb_task = task;
            fb_task.assigned_engine = cand.first;
            fb_task.assigned_backend = HardwareBackend::CPU;
            fb_task.orchestrator_backend_id = cand.second;

            IntermediateResult fb_res = runEngineBackend(
                cand.first,
                HardwareBackend::CPU,
                cand.second,
                fb_task,
                req,
                model_profile,
                memory_constrained_mode);

            if (fb_res.success) {
                task.assigned_engine = cand.first;
                task.assigned_backend = HardwareBackend::CPU;
                task.orchestrator_backend_id = cand.second;
                task.fallback_occurred = true;
                task.fallback_reason = "Engine fallback to " + std::string(engineIdToString(cand.first)) +
                                       " after primary failure: " + primary_eng_err;
                res = std::move(fb_res);
                break;
            }
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    task.execution_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    task.peak_memory_bytes = res.peak_memory_bytes;
    task.output = res;
    task.state = res.success ? TaskState::COMPLETED : TaskState::FAILED;
    return task.output;
}

CollaborativeResult CollaborativeExecutionOrchestrator::execute(const CollaborativeRequest& req) {
    return executeStreaming(req, nullptr);
}

CollaborativeResult CollaborativeExecutionOrchestrator::executeStreaming(
    const CollaborativeRequest& req,
    CollaborativeStreamCallback cb) {

    CollaborativeResult result;
    auto t_total0 = std::chrono::steady_clock::now();

    if (req.prompt.empty()) {
        result.success = false;
        result.error = "Prompt cannot be empty";
        return result;
    }

    if (req.cancel_flag && req.cancel_flag->load()) {
        result.success = false;
        result.error = "Cancelled before planning";
        return result;
    }

    ensureEnginesRegistered();
    resource_mgr_.configureBudget(req.max_ram_budget_bytes);

    // 1. Planning & Task Decomposition
    auto t_plan0 = std::chrono::steady_clock::now();
    ModelMemoryProfile model_profile = CollaborativeResourceManager::estimateModelMemory(
        req.model_path, static_cast<uint32_t>(req.n_ctx > 0 ? req.n_ctx : 512));

    const auto& hw = orchestration::HardwareDetector::instance();
    PartitionPlan plan = WorkloadPartitioner::createPlan(req, model_profile, resource_mgr_, hw);
    auto t_plan1 = std::chrono::steady_clock::now();

    result.planning_ms = std::chrono::duration<double, std::milli>(t_plan1 - t_plan0).count();
    result.query_category = plan.query_category;
    result.tasks = plan.tasks;
    result.execution_waves = plan.waves;

    // Map task_id -> index in result.tasks
    std::map<std::string, size_t> task_index;
    for (size_t i = 0; i < result.tasks.size(); ++i) {
        task_index[result.tasks[i].task_id] = i;
    }

    // 2. Wave-by-Wave DAG Execution
    auto t_exec0 = std::chrono::steady_clock::now();
    bool concurrent_wave_ran = false;

    for (size_t w = 0; w < result.execution_waves.size(); ++w) {
        if (req.cancel_flag && req.cancel_flag->load()) {
            for (auto& t : result.tasks) {
                if (t.state == TaskState::PENDING || t.state == TaskState::READY) {
                    t.state = TaskState::CANCELLED;
                }
            }
            result.success = false;
            result.error = "Cancelled during wave execution";
            return result;
        }

        const auto& wave_ids = result.execution_waves[w];
        std::vector<size_t> ready_indices;

        // Resolve upstream dependencies and exchange intermediate results
        for (const auto& tid : wave_ids) {
            auto idx_it = task_index.find(tid);
            if (idx_it == task_index.end()) continue;
            size_t idx = idx_it->second;
            auto& task = result.tasks[idx];

            std::ostringstream upstream_ctx;
            bool deps_satisfied = true;
            for (const auto& dep_id : task.dependencies) {
                auto dep_it = task_index.find(dep_id);
                if (dep_it == task_index.end()) continue;
                const auto& dep_task = result.tasks[dep_it->second];
                if (dep_task.state == TaskState::COMPLETED && dep_task.output.success) {
                    upstream_ctx << "\n[Upstream " << dep_task.output.engine_id
                                 << " (" << dep_task.task_id << ")]: "
                                 << dep_task.output.structured_result;
                } else if (!req.allow_engine_fallback) {
                    deps_satisfied = false;
                }
            }

            if (!deps_satisfied) {
                task.state = TaskState::FAILED;
                task.output.success = false;
                task.output.error = "Upstream dependency failed";
                continue;
            }

            std::string extra = upstream_ctx.str();
            if (!extra.empty()) {
                task.input_context += extra;
            }
            task.state = TaskState::READY;
            ready_indices.push_back(idx);
        }

        // Execute tasks in this wave:
        // - Parallel if allow_parallel_waves is true AND memory is NOT constrained
        // - Sequential with model residency swapping if memory_constrained_mode is true
        bool run_parallel = req.allow_parallel_waves &&
                            !plan.memory_constrained_mode &&
                            (ready_indices.size() > 1);

        if (run_parallel) {
            concurrent_wave_ran = true;
            std::vector<std::future<void>> futures;
            futures.reserve(ready_indices.size());
            for (size_t idx : ready_indices) {
                futures.push_back(std::async(std::launch::async, [this, idx, &result, &req, &model_profile, &plan]() {
                    executeSingleTask(result.tasks[idx], req, model_profile, plan.memory_constrained_mode);
                }));
            }
            for (auto& f : futures) {
                f.get();
            }
        } else {
            auto wave_t0 = std::chrono::steady_clock::now();
            for (size_t idx : ready_indices) {
                auto now = std::chrono::steady_clock::now();
                result.tasks[idx].queue_wait_ms =
                    std::chrono::duration<double, std::milli>(now - wave_t0).count();
                executeSingleTask(result.tasks[idx], req, model_profile, plan.memory_constrained_mode);
            }
        }
    }

    auto t_exec1 = std::chrono::steady_clock::now();
    result.execution_ms = std::chrono::duration<double, std::milli>(t_exec1 - t_exec0).count();

    // Check that at least one task succeeded
    bool any_completed = false;
    std::string first_err;
    for (const auto& t : result.tasks) {
        if (t.state == TaskState::COMPLETED && t.output.success) {
            any_completed = true;
        } else if (first_err.empty() && !t.output.error.empty()) {
            first_err = t.output.error;
        }
    }

    if (!any_completed) {
        result.success = false;
        result.error = first_err.empty() ? "All collaborative tasks failed" : first_err;
        result.resource_telemetry = resource_mgr_.telemetrySnapshot(
            model_profile, plan.memory_constrained_mode, concurrent_wave_ran);
        return result;
    }

    // 3. Result Validation and Fusion
    auto t_fuse0 = std::chrono::steady_clock::now();
    result.fusion = ResultFusionEngine::fuse(plan.query_category, req.prompt, result.tasks);
    result.unified_response = result.fusion.synthesized_response;
    auto t_fuse1 = std::chrono::steady_clock::now();
    result.fusion_ms = std::chrono::duration<double, std::milli>(t_fuse1 - t_fuse0).count();

    // 4. Unified Incremental Streaming (if callback provided)
    if (cb && !result.unified_response.empty()) {
        std::istringstream iss(result.unified_response);
        std::string line;
        int chunk_id = 0;
        while (std::getline(iss, line)) {
            if (req.cancel_flag && req.cancel_flag->load()) {
                result.success = false;
                result.error = "Cancelled during unified response streaming";
                return result;
            }
            if (!cb(line + "\n", chunk_id++)) {
                result.success = false;
                result.error = "Cancelled by streaming callback";
                return result;
            }
        }
    }

    auto t_total1 = std::chrono::steady_clock::now();
    result.total_ms = std::chrono::duration<double, std::milli>(t_total1 - t_total0).count();
    result.resource_telemetry = resource_mgr_.telemetrySnapshot(
        model_profile, plan.memory_constrained_mode, concurrent_wave_ran);
    result.success = true;
    return result;
}

} // namespace collaboration
} // namespace agr
