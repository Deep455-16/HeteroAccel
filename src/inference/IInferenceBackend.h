// src/inference/IInferenceBackend.h
// Phase 7: Abstract interface for inference backends.
// LlamaCppBackend implements this. Future backends (ExLlama, ONNX, etc.) would too.
#pragma once

#include "inference/InferenceTypes.h"
#include <memory>
#include <string>

namespace agr {

class IInferenceBackend {
public:
    virtual ~IInferenceBackend() = default;

    /// Load a model from disk. Returns false on failure.
    virtual bool loadModel(const std::string& path, int n_gpu_layers, bool cpu_only) = 0;

    /// Create (or recreate) the inference context. Must call loadModel first.
    virtual bool createContext(const GenerationOptions& opts) = 0;

    /// Query metadata about the currently loaded model.
    virtual ModelInfo modelInfo() const = 0;

    /// Current model state.
    virtual ModelState state() const = 0;

    /// Run one full inference (blocking). Returns result when generation is done.
    virtual InferenceResult generate(const InferenceRequest& req) = 0;

    /// Stream tokens as they are generated. Calls cb for every piece.
    /// Returns the complete result at the end.
    virtual InferenceResult generateStreaming(const InferenceRequest& req,
                                              TokenCallback cb) = 0;

    /// Release all model and context resources.
    virtual void unload() = 0;

    /// Human-readable backend name, e.g. "LlamaCppBackend".
    virtual std::string backendName() const = 0;

    /// Last error message.
    virtual std::string lastError() const = 0;
};

} // namespace agr
