// src/hydroxl/HydroXLEngine.h
// Phase 17: HydroXL IExecutionEngine implementation.
//
// HydroXLEngine is a GENUINE NATIVE inference engine. It does NOT delegate
// to llama.cpp for model execution, tokenization, or sampling.
//
// It implements IExecutionEngine and registers itself under the key
// "hydroxl" in ExecutionEngineRegistry.
//
// Backend: CPU (FLOAT32 scalar arithmetic)
// Claimed: BACKEND_CPU only. GPU execution is NOT claimed.
//
// PARTIALLY IMPLEMENTED note:
//   Real-model inference requires the model file to be present at
//   ctx.model_path (passed via ModelDescriptor.source_path).
//   Without the model, loadModel() returns false.
//   Tokenizer: Full BPE vocabulary loading deferred (GGUF string arrays
//   require extended parser); encode() uses placeholder IDs.

#pragma once

#include "engine/IExecutionEngine.h"
#include "hydroxl/HydroXLModel.h"
#include "hydroxl/HydroXLTokenizer.h"
#include "hydroxl/HydroXLKVCache.h"
#include "hydroxl/HydroXLForwardPass.h"
#include "hydroxl/HydroXLSampler.h"

#include <memory>
#include <mutex>
#include <string>

namespace agr {

/// HydroXL native inference engine — registers as "hydroxl".
class HydroXLEngine final : public IExecutionEngine {
public:
    HydroXLEngine();
    ~HydroXLEngine() override;

    // Non-copyable
    HydroXLEngine(const HydroXLEngine&) = delete;
    HydroXLEngine& operator=(const HydroXLEngine&) = delete;

    // -------------------------------------------------------------------
    // Identity & capabilities
    // -------------------------------------------------------------------
    EngineIdentity      identity()     const override;
    EngineCapabilitySet capabilities() const override;

    // -------------------------------------------------------------------
    // Engine lifecycle
    // -------------------------------------------------------------------
    bool initialize()      override;
    void shutdown()        override;
    bool isInitialized()   const override;

    // -------------------------------------------------------------------
    // Model lifecycle
    // -------------------------------------------------------------------
    bool           loadModel(const ModelDescriptor& descriptor) override;
    void           unloadModel()          override;
    bool           isModelLoaded()        const override;
    ModelDescriptor currentModel()        const override;

    // -------------------------------------------------------------------
    // Execution
    // -------------------------------------------------------------------
    ExecutionResult execute(const ExecutionContext& ctx)                         override;
    ExecutionResult executeStreaming(const ExecutionContext& ctx, TokenCallback cb) override;

    // -------------------------------------------------------------------
    // Error handling
    // -------------------------------------------------------------------
    EngineError lastEngineError() const override;

    // -------------------------------------------------------------------
    // Registration helper — call once at process start
    // -------------------------------------------------------------------
    static void registerInRegistry();

private:
    ExecutionResult doGenerate(const ExecutionContext& ctx, TokenCallback cb);

    mutable std::mutex mutex_;
    bool initialized_ = false;
    bool model_loaded_ = false;

    ModelDescriptor current_descriptor_;
    EngineError     last_error_;

    std::unique_ptr<hydroxl::HydroXLModel>       model_;
    std::unique_ptr<hydroxl::HydroXLTokenizer>   tokenizer_;
    std::unique_ptr<hydroxl::HydroXLKVCache>     kv_cache_;
    std::unique_ptr<hydroxl::HydroXLForwardPass> forward_pass_;
};

} // namespace agr
