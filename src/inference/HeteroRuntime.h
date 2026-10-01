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
#include "inference/IInferenceBackend.h"
#include "backend/BackendManager.h"
#include "backend/DeviceSelector.h"
#include "gpu/VulkanBackend.h"
#include "mem/MemoryManager.h"
#include "scheduler/AdaptiveScheduler.h"
#include "hardware/HardwareDetector.h"
#include "profiler/Profiler.h"
#include "scheduler/AutoTuner.h"

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

    /// Hardware/runtime diagnostics.
    std::string diagnosticsReport() const;

    /// Whether initialize() succeeded.
    bool isInitialized() const { return initialized_; }

    const HardwareInfo&    hardware() const { return hardware_; }
    const BackendManager&  backendManager() const { return *backendMgr_; }

private:
    /// Determine n_gpu_layers and cpu_only based on scheduler decision.
    void resolveBackendConfig(int& out_gpu_layers, bool& out_cpu_only) const;

    InferenceResult doGenerate(const std::string& model_path,
                               const std::string& prompt,
                               const GenerationOptions& opts,
                               TokenCallback cb);

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

    // Loaded model backends, keyed by model path
    std::unordered_map<std::string, std::unique_ptr<IInferenceBackend>> backends_;
};

} // namespace agr
