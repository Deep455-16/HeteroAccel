// src/collaboration/CollaborativeResourceManager.h
// Shared Resource & Memory-Aware Residency Coordinator for Multi-Engine Execution
//
// Coordinates model residency, system RAM budgets, VRAM budgets, and GPU device
// locks across HydroXL, llama.cpp, and Colibri.
//
// Key guarantee (Sections 9 & 11):
// On constrained hardware (e.g., ~16 GB RAM with integrated GPU or tight memory
// budget), this coordinator prevents loading multiple heavy neural models at the
// same time by switching the execution wave strategy to sequential model swapping
// while keeping all three engines in the collaborative workflow.

#pragma once

#include "collaboration/CollaborativeTypes.h"
#include "orchestration/IInferenceEngine.h"

#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace agr {
namespace collaboration {

struct ModelMemoryProfile {
    std::string model_path;
    std::string arch;
    int         quant_type = 12;
    size_t      file_size_bytes = 0;
    uint32_t    n_layers = 24;
    uint32_t    n_embd = 896;
    uint32_t    n_heads = 14;
    uint32_t    n_kv_heads = 2;
    uint32_t    n_ff = 4864;
    uint32_t    vocab_size = 151936;
    uint32_t    n_ctx = 512;

    size_t      hydroxl_estimated_bytes = 0;  ///< Dequantized F32 weights + KV + scratch
    size_t      llamacpp_estimated_bytes = 0; ///< Quantized weights + KV + compute buffer
    size_t      colibri_estimated_bytes = 0;  ///< Lightweight GGUF metadata + analyzer workspace
};

class CollaborativeResourceManager {
public:
    CollaborativeResourceManager();
    ~CollaborativeResourceManager();

    /// Inspects a GGUF file (or synthetic model metadata) and computes per-engine
    /// memory footprints using MemoryEstimator.
    static ModelMemoryProfile estimateModelMemory(const std::string& model_path,
                                                  uint32_t n_ctx = 512);

    /// Configures the active memory budget. If budget_bytes == 0, uses 80% of
    /// detected available system RAM.
    void configureBudget(size_t budget_bytes);

    size_t availableRamBytes() const;
    size_t effectiveBudgetBytes() const;

    /// Returns true if HydroXL and llama.cpp can both hold the model resident in
    /// RAM simultaneously within the effective budget.
    bool canKeepConcurrentNeuralEngines(const ModelMemoryProfile& profile) const;

    /// Acquires an initialized IInferenceEngine for (orchestrator_backend_id, model_path).
    /// When memory_constrained_mode is active, automatically evicts any other
    /// resident heavy neural engine before initializing the requested one so
    /// peak resident memory stays within budget.
    std::shared_ptr<orchestration::IInferenceEngine> acquireEngine(
        const std::string& orchestrator_backend_id,
        const std::string& model_path,
        size_t estimated_engine_bytes,
        bool memory_constrained_mode,
        std::string& error_out);

    /// Explicitly evicts a specific backend_id from residency.
    void releaseEngine(const std::string& orchestrator_backend_id);

    /// Evicts all resident engines and resets residency counters.
    void evictAll();

    /// GPU execution serialization lock for when multiple tasks target the same
    /// physical GPU (Vulkan or CUDA) in a wave.
    std::unique_lock<std::mutex> acquireGpuExecutionLock();

    ResourceCoordinationTelemetry telemetrySnapshot(const ModelMemoryProfile& profile,
                                                    bool memory_constrained_mode,
                                                    bool concurrent_wave_executed) const;

private:
    struct ResidentEntry {
        std::shared_ptr<orchestration::IInferenceEngine> engine;
        std::string model_path;
        size_t estimated_bytes = 0;
        bool is_heavy_neural = false;
    };

    mutable std::mutex mu_;
    mutable std::mutex gpu_exec_mu_;
    std::map<std::string, ResidentEntry> resident_;

    size_t configured_budget_bytes_ = 0;
    size_t current_resident_bytes_ = 0;
    size_t peak_resident_bytes_ = 0;
    int    models_loaded_count_ = 0;
    int    models_evicted_count_ = 0;
    mutable int gpu_lock_acquisitions_ = 0;
};

} // namespace collaboration
} // namespace agr
