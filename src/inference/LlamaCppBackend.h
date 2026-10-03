// src/inference/LlamaCppBackend.h
// Phase 7: LlamaCppBackend implements IInferenceBackend using the existing
// LlamaCppEngine from Phase 3 (src/llm/LlamaCppEngine).
//
// ARCHITECTURE BOUNDARY:
//   HeteroAccel (scheduling, memory, residency)
//       |
//       v
//   LlamaCppBackend  <-- this file
//       |
//       v
//   LlamaCppEngine  (src/llm/) -- wraps llama.cpp C API
//       |
//       v
//   llama.cpp (CPU / Vulkan / CUDA)
//
// HeteroAccel controls the execution configuration through the supported
// llama.cpp API: n_gpu_layers, n_threads, n_ctx, n_batch, temperature,
// seed, and cpu_only mode. HeteroAccel does NOT implement custom per-tensor
// device placement; GPU-layer allocation via n_gpu_layers is the mechanism
// used to distribute work between CPU and GPU.
// llama.cpp internally maps those layers to Vulkan/CUDA devices.
#pragma once

#include "inference/IInferenceBackend.h"
#include "llm/LlamaCppEngine.h"
#include "llm/LlmConfig.h"

#include <atomic>
#include <mutex>
#include <string>

namespace agr {

class LlamaCppBackend : public IInferenceBackend {
public:
    LlamaCppBackend() = default;
    ~LlamaCppBackend() override { unload(); }

    // Non-copyable
    LlamaCppBackend(const LlamaCppBackend&) = delete;
    LlamaCppBackend& operator=(const LlamaCppBackend&) = delete;

    bool loadModel(const std::string& path, int n_gpu_layers, bool cpu_only) override;
    bool createContext(const GenerationOptions& opts) override;

    ModelInfo  modelInfo() const override;
    ModelState state()     const override { return state_; }

    InferenceResult generate(const InferenceRequest& req) override;
    InferenceResult generateStreaming(const InferenceRequest& req,
                                      TokenCallback cb) override;

    void unload() override;

    std::string backendName() const override { return "LlamaCppBackend"; }
    std::string lastError()   const override { return lastError_; }

private:
    InferenceResult runInference(const InferenceRequest& req, TokenCallback cb);

    mutable std::mutex  mutex_;
    LlamaCppEngine      engine_;
    LlmConfig           config_;

    std::atomic<ModelState> state_{ModelState::DISCOVERED};
    std::string lastError_;

    // Cached model metadata (populated after loadModel succeeds)
    ModelInfo cachedInfo_;
};

} // namespace agr
