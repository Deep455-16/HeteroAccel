// src/inference/HeteroRuntime.h
// Phase 7: Top-level HeteroAccel public API.
//
// This is the single entry point for application code. Users call:
//
//   HeteroRuntime runtime;
//   runtime.initialize();
//   auto result = runtime.generate(path, prompt, opts);
//   runtime.shutdown();
//
// Internally it orchestrates:
//   Phase 4 MemoryManager + Phase 5 Scheduler + Phase 6 ModelManager
//   -> LlamaCppBackend -> llama.cpp -> CPU/Vulkan/CUDA
//
// Backend selection is AUTOMATIC. The caller does not choose CPU/GPU.
#pragma once

#include "inference/InferenceTypes.h"
#include "engine/IExecutionEngine.h"
#include "analysis/CapabilityReport.h"
#include "backend/BackendManager.h"
#include "backend/DeviceSelector.h"
#include "gpu/VulkanBackend.h"
#include "mem/MemoryManager.h"
#include "scheduler/AdaptiveScheduler.h"
#include "hardware/HardwareDetector.h"
#include "profiler/Profiler.h"
#include "scheduler/AutoTuner.h"
// Phase 9
#include "scheduler/WorkloadRegistry.h"
#include "scheduler/ExecutionPolicyEngine.h"
#include "scheduler/Phase9Telemetry.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace agr {

/// Top-level HeteroAccel runtime.
/// One instance per process is the recommended pattern.
class HeteroRuntime {
public:
    HeteroRuntime() = default;
    ~HeteroRuntime() { shutdown(); }

    // Non-copyable
    HeteroRuntime(const HeteroRuntime&) = delete;
    HeteroRuntime& operator=(const HeteroRuntime&) = delete;

    /// Initialize hardware detection, memory manager, and scheduler.
    /// Must be called before any model operations.
    bool initialize();

    /// Load a GGUF model. HeteroAccel selects backend automatically.
    /// Returns a handle (model path) to use with generate().
    bool loadModel(const std::string& model_path, const GenerationOptions& opts = {});

    /// Run inference on a previously loaded model. Automatically selects backend.
    InferenceResult generate(const std::string& model_path,
                             const std::string& prompt,
                             const GenerationOptions& opts = {});

    /// Run inference with per-token streaming callback.
    InferenceResult generateStreaming(const std::string& model_path,
                                      const std::string& prompt,
                                      TokenCallback cb,
                                      const GenerationOptions& opts = {});

    /// Unload a specific model and free its resources.
    void unloadModel(const std::string& model_path);

    /// Unload all models and free all resources.
    void shutdown();

    /// Get the last error message (e.g. if initialize() or loadModel() fails).
    const std::string& lastError() const { return last_error_; }

    /// Hardware/runtime diagnostics.
    std::string diagnosticsReport() const;
    
    // Phase 12: Model Inspection and Capability Analysis
    CapabilityReport analyzeModel(const std::string& model_path);

    /// Whether initialize() succeeded.
    bool isInitialized() const { return initialized_; }

    const HardwareInfo&    hardware() const { return hardware_; }
    const BackendManager&  backendManager() const { return *backendMgr_; }

    // Phase 9: Workload management
    /// Register a workload for tracking. Returns workload ID.
    uint64_t registerWorkload(const std::string& name,
                              const std::string& model_path,
                              WorkloadPriority priority = WorkloadPriority::NORMAL,
                              WorkloadClass wclass = WorkloadClass::DEFAULT);

    /// Cancel a registered workload.
    bool cancelWorkload(uint64_t workload_id);

    /// Get snapshot of all workloads for diagnostics.
    std::vector<WorkloadEntry> workloadSnapshot() const;

    /// Phase 9 workload-aware generate: registers workload automatically.
    InferenceResult generateWorkload(const std::string& model_path,
                                     const std::string& prompt,
                                     const GenerationOptions& opts = {},
                                     WorkloadPriority priority = WorkloadPriority::NORMAL,
                                     WorkloadClass wclass = WorkloadClass::DEFAULT);

private:
    /// Determine n_gpu_layers and cpu_only based on scheduler decision, model size, and pressure.
    /// Will reload the backend if configuration changes significantly (e.g. GPU -> CPU fallback).
    TuningConfig resolveAndApplyConfig(const std::string& model_path, int max_threads, WorkloadClass wclass);

    /// Execute inference using the loaded model.
    /// CONCURRENCY LIMITATIONS (INTENTIONAL):
    /// - Different backend/model instances may execute concurrently.
    /// - Requests sharing the same LlamaCppBackend instance are serialized by its backend mutex.
    /// - This is intentional for correctness, as the underlying llama_context is not thread-safe.
    InferenceResult doGenerate(const std::string& model_path,
                               const std::string& prompt,
                               const GenerationOptions& opts,
                               TokenCallback cb,
                               uint64_t workload_id = 0);

    std::string last_error_;
    bool initialized_ = false;
    mutable std::mutex mutex_;

    HardwareInfo hardware_;

    // Phase 4 + 5 subsystems
    std::unique_ptr<VulkanBackend>    vulkan_;
    std::unique_ptr<BackendManager>   backendMgr_;
    std::unique_ptr<MemoryManager>    memMgr_;
    std::unique_ptr<AdaptiveScheduler> scheduler_;
    std::unique_ptr<Profiler>         profiler_;
    std::unique_ptr<AutoTuner>        autoTuner_;

    // Phase 9 subsystems
    std::unique_ptr<WorkloadRegistry>      workloadRegistry_;
    std::unique_ptr<ExecutionPolicyEngine> policyEngine_;

    // Loaded model backends, keyed by model path
    std::unordered_map<std::string, std::shared_ptr<IExecutionEngine>> engines_;
    std::unordered_map<std::string, TuningConfig> loaded_configs_;
};

} // namespace agr
