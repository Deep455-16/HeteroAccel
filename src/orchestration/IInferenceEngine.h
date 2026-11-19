// src/orchestration/IInferenceEngine.h
// Phase 21: Inference Engine Abstraction

#pragma once

#include <atomic>
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace agr {
namespace orchestration {

// Token callback: return false to cancel generation
using StreamCallback = std::function<bool(const std::string& token_text, int token_id)>;

struct InferenceRequest {
    std::string prompt;
    std::vector<int> prompt_token_ids;
    int max_new_tokens = 16;
    float temperature = 0.0f;
    float top_p = 0.95f;
    bool stream = false;

    // Phase 33 generation controls (all optional; defaults preserve old behaviour)
    std::atomic<bool>* cancel_flag = nullptr;  ///< Polled between tokens
    int  time_budget_ms = 0;                   ///< 0 = no wall-clock limit
    std::vector<std::string> stop_sequences;   ///< Generation stops before these
    bool parse_special = false;                ///< Treat <|...|> control markers in prompt as tokens
    float repeat_penalty = 1.0f;               ///< 1.0 = disabled (preserves pure greedy equivalence)
};

struct InferenceResult {
    std::vector<int> output_token_ids;
    std::string output_text;
    int prompt_tokens = 0;
    int output_tokens = 0;
    float tokens_per_second = 0.0f;
    size_t peak_memory_bytes = 0;
    bool gpu_used = false;
    std::string backend_used;
    bool success = false;
    std::string error;
    /// Why generation ended: "eos", "max_tokens", "stop_sequence",
    /// "time_budget", "cancelled" or "context_limit" (empty if unknown).
    std::string finish_reason;
};


class IInferenceEngine {
public:
    virtual ~IInferenceEngine() = default;
    virtual bool initialize(const std::string& model_path, std::string& error) = 0;
    virtual InferenceResult run(const InferenceRequest& req) = 0;
    virtual bool run_stream(const InferenceRequest& req,
                            StreamCallback cb,
                            InferenceResult& result_out) {
        (void)req;
        (void)cb;
        (void)result_out;
        return false;
    }
    virtual bool supports_streaming() const { return false; }
    virtual void shutdown() = 0;
    virtual std::string backend_id() const = 0;
    virtual bool is_ready() const = 0;
};

} // namespace orchestration
} // namespace agr
