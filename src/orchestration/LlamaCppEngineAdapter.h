// src/orchestration/LlamaCppEngineAdapter.h
// Phase 26: IInferenceEngine adapter over llama.cpp (agr::LlamaCppEngine) with
// *explicit* backend selection. The "cuda" orchestrator backend is this adapter
// pinned to a GGML_CUDA device; it is the real CUDA inference path.
//
// It deliberately reuses the existing LlamaCppEngine rather than adding another
// inference implementation. Requests carry vocabulary token ids from the shared
// GGUF tokenizer, which are identical to llama.cpp's ids for the same model.

#pragma once

#include "orchestration/IInferenceEngine.h"
#include "llm/LlamaCppEngine.h"

#include <memory>
#include <mutex>
#include <string>

namespace agr {
namespace orchestration {

class LlamaCppEngineAdapter : public IInferenceEngine {
public:
    /// @param backend_id   orchestrator id reported by backend_id() ("cuda", ...)
    /// @param preference   llama.cpp device kind: "cuda" | "vulkan" | "cpu"
    /// @param device_index index among devices of that kind
    LlamaCppEngineAdapter(std::string backend_id, std::string preference, int device_index = 0);
    ~LlamaCppEngineAdapter() override;

    /// Fails (returning false + reason) when the requested device kind is not
    /// compiled in / not present / out of memory, or when llama.cpp loaded the
    /// model but offloaded zero layers to the requested GPU. The orchestrator
    /// then falls back to CPU and records this reason.
    bool initialize(const std::string& model_path, std::string& error) override;
    InferenceResult run(const InferenceRequest& req) override;
    bool run_stream(const InferenceRequest& req,
                    StreamCallback cb,
                    InferenceResult& result_out) override;
    bool supports_streaming() const override { return true; }
    void shutdown() override;
    std::string backend_id() const override { return backend_id_; }
    bool is_ready() const override;

    // Diagnostics for reporting.
    std::string device_name() const;
    size_t device_total_bytes() const;
    size_t device_free_bytes() const;
    int gpu_layers() const;

    /// Context window used by initialize(); set before initialize(). Default 2048.
    void set_context_size(int n_ctx) { n_ctx_ = n_ctx; }
    /// When true, run() copies first-token logits into last_first_logits().
    void set_capture_logits(bool v) { capture_logits_ = v; }
    const std::vector<float>& last_first_logits() const { return last_logits_; }

private:
    std::string backend_id_;
    std::string preference_;
    int         device_index_;
    int         n_ctx_ = 2048;
    bool        capture_logits_ = false;

    mutable std::mutex mu_;
    std::unique_ptr<LlamaCppEngine> engine_;
    bool        ready_ = false;
    std::string device_name_;
    size_t      dev_total_ = 0;
    size_t      dev_free_ = 0;
    int         gpu_layers_ = 0;
    std::vector<float> last_logits_;
};

} // namespace orchestration
} // namespace agr
