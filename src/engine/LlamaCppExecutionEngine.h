// src/engine/LlamaCppExecutionEngine.h
// Phase 11: IExecutionEngine adapter wrapping LlamaCppBackend.
//
// This class implements the universal IExecutionEngine interface and
// internally delegates to the existing LlamaCppBackend / LlamaCppEngine
// pipeline. All llama.cpp-specific types remain behind this boundary.
//
// ARCHITECTURE:
//
//   HeteroRuntime
//        │
//        ▼
//   IExecutionEngine (generic)
//        │
//        ▼
//   LlamaCppExecutionEngine   ← this file (Phase 11 adapter)
//        │
//        ▼
//   LlamaCppBackend           (Phase 7 backend)
//        │
//        ▼
//   LlamaCppEngine            (Phase 3 llama.cpp wrapper)
//        │
//        ▼
//   llama.cpp                 (CPU / Vulkan / CUDA)
//
// CONCURRENCY (intentional, Phase 11):
//   - execute() / executeStreaming() acquire backend_mutex_ before
//     calling into LlamaCppBackend. This serialises concurrent calls
//     on the SAME engine instance, consistent with llama_context not
//     being thread-safe.
//   - Distinct LlamaCppExecutionEngine instances may run concurrently;
//     the Phase 10 llama_backend_init reference counter handles that.
//
// PHASE 11 SCOPE:
//   - Does NOT implement automatic device planning (Phase 13).
//   - Does NOT implement tensor-level streaming (Phase 14).
//   - Does NOT perform per-tensor device placement.
//   - n_gpu_layers from ExecutionContext.n_gpu_layers is the only
//     GPU/CPU split mechanism, forwarded to LlamaCppBackend unchanged.
#pragma once

#include "engine/IExecutionEngine.h"
#include "inference/LlamaCppBackend.h"

#include <memory>
#include <mutex>
#include <string>

namespace agr {

/// IExecutionEngine adapter over the existing LlamaCppBackend.
///
/// External code should obtain instances via the registry:
///   ExecutionEngineRegistry::instance().create("llama.cpp")
class LlamaCppExecutionEngine final : public IExecutionEngine {
public:
    LlamaCppExecutionEngine();
    ~LlamaCppExecutionEngine() override;

    // Non-copyable
    LlamaCppExecutionEngine(const LlamaCppExecutionEngine&) = delete;
    LlamaCppExecutionEngine& operator=(const LlamaCppExecutionEngine&) = delete;

    // -------------------------------------------------------------------
    // Identity & capabilities
    // -------------------------------------------------------------------
    EngineIdentity identity() const override;
    EngineCapabilitySet capabilities() const override;

    // -------------------------------------------------------------------
    // Engine lifecycle
    // -------------------------------------------------------------------
    bool initialize() override;
    void shutdown() override;
    bool isInitialized() const override;

    // -------------------------------------------------------------------
    // Model lifecycle
    // -------------------------------------------------------------------
    bool loadModel(const ModelDescriptor& descriptor) override;
    void unloadModel() override;
    bool isModelLoaded() const override;
    ModelDescriptor currentModel() const override;

    // -------------------------------------------------------------------
    // Execution
    // -------------------------------------------------------------------
    ExecutionResult execute(const ExecutionContext& ctx) override;
    ExecutionResult executeStreaming(const ExecutionContext& ctx,
                                    TokenCallback cb) override;

    // -------------------------------------------------------------------
    // Error handling
    // -------------------------------------------------------------------
    EngineError lastEngineError() const override;

private:
    /// Map an ExecutionContext to the InferenceRequest format used
    /// by the existing LlamaCppBackend pipeline.
    InferenceRequest buildRequest(const ExecutionContext& ctx) const;

    /// Map an InferenceResult from LlamaCppBackend to ExecutionResult.
    ExecutionResult mapResult(const InferenceResult& r) const;

    mutable std::mutex    engine_mutex_;  ///< Serialises execute() calls on this instance
    bool                  initialized_ = false;
    ModelDescriptor       currentDescriptor_;
    bool                  modelLoaded_ = false;
    EngineError           lastError_;

    /// Underlying Phase 7 backend (holds llama_model + llama_context).
    std::shared_ptr<LlamaCppBackend> backend_;
};

} // namespace agr
