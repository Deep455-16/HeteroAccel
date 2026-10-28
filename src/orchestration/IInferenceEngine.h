// src/orchestration/IInferenceEngine.h
// Phase 21: Inference Engine Abstraction

#pragma once

#include <vector>
#include <string>
#include <memory>

namespace agr {
namespace orchestration {

struct InferenceRequest {
    std::vector<int> prompt_token_ids;
    int max_new_tokens = 16;
    float temperature = 0.0f;
    float top_p = 0.95f;
    bool stream = false;
};

struct InferenceResult {
    std::vector<int> output_token_ids;
    std::string output_text;
    int prompt_tokens = 0;
    int output_tokens = 0;
    float tokens_per_second = 0.0f;
    bool gpu_used = false;
    std::string backend_used;
    bool success = false;
    std::string error;
};

class IInferenceEngine {
public:
    virtual ~IInferenceEngine() = default;
    virtual bool initialize(const std::string& model_path, std::string& error) = 0;
    virtual InferenceResult run(const InferenceRequest& req) = 0;
    virtual void shutdown() = 0;
    virtual std::string backend_id() const = 0;
    virtual bool is_ready() const = 0;
};

} // namespace orchestration
} // namespace agr
