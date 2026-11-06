// src/orchestration/LlamaCppEngineAdapter.cpp
#include "orchestration/LlamaCppEngineAdapter.h"

namespace agr {
namespace orchestration {

LlamaCppEngineAdapter::LlamaCppEngineAdapter(std::string backend_id, std::string preference, int device_index)
    : backend_id_(std::move(backend_id)), preference_(std::move(preference)), device_index_(device_index) {}

LlamaCppEngineAdapter::~LlamaCppEngineAdapter() { shutdown(); }

bool LlamaCppEngineAdapter::initialize(const std::string& model_path, std::string& error) {
    std::lock_guard<std::mutex> lk(mu_);
    if (ready_) return true;

    LlmConfig cfg;
    cfg.model_path         = model_path;
    cfg.backend_preference = preference_;
    cfg.gpu_device_index   = device_index_;
    cfg.n_gpu_layers       = (preference_ == "cpu") ? 0 : 999;   // offload every layer
    cfg.n_ctx              = n_ctx_;
    cfg.capture_first_logits = capture_logits_;

    auto eng = std::make_unique<LlamaCppEngine>();
    if (!eng->initialize(cfg)) {
        error = "llama.cpp (" + preference_ + ") initialization failed: " + eng->lastError();
        return false;
    }

    if (preference_ != "cpu" && eng->gpuLayersActual() <= 0) {
        error = "llama.cpp loaded the model but offloaded no layers to the " + preference_ +
                " device; refusing to report a GPU backend that is not doing the work";
        eng->shutdown();
        return false;
    }

    device_name_ = eng->activeDeviceName();
    dev_total_   = eng->deviceTotalBytes();
    dev_free_    = eng->deviceFreeBytes();
    gpu_layers_  = eng->gpuLayersActual();
    engine_      = std::move(eng);
    ready_       = true;
    return true;
}

InferenceResult LlamaCppEngineAdapter::run(const InferenceRequest& req) {
    std::lock_guard<std::mutex> lk(mu_);
    InferenceResult out;
    out.backend_used = backend_id_;
    if (!ready_ || !engine_) {
        out.error = "llama.cpp adapter is not initialized";
        return out;
    }
    engine_->setGeneration(req.max_new_tokens > 0 ? req.max_new_tokens : 16, req.temperature, 42u);
    LlmResult r = req.prompt_token_ids.empty()
                      ? engine_->infer("The capital of France is")
                      : engine_->inferTokens(req.prompt_token_ids);
    last_logits_ = r.first_token_logits;

    out.success           = r.success;
    out.error             = r.error;
    out.output_text       = r.output;
    out.output_token_ids  = r.output_token_ids;
    out.prompt_tokens     = r.prompt_tokens;
    out.output_tokens     = static_cast<int>(r.output_token_ids.size());
    out.tokens_per_second = static_cast<float>(r.tokens_per_sec);
    out.gpu_used          = (r.backend == "CUDA" || r.backend == "Vulkan");
    if (r.device_total_bytes > 0) {
        // Device-resident bytes in use after load (weights + KV + compute buffers).
        out.peak_memory_bytes = r.device_total_bytes > r.device_free_bytes
                                    ? r.device_total_bytes - r.device_free_bytes : 0;
    }
    return out;
}

void LlamaCppEngineAdapter::shutdown() {
    std::lock_guard<std::mutex> lk(mu_);
    if (engine_) {
        engine_->shutdown();
        engine_.reset();
    }
    ready_ = false;
}

bool LlamaCppEngineAdapter::is_ready() const {
    std::lock_guard<std::mutex> lk(mu_);
    return ready_;
}

std::string LlamaCppEngineAdapter::device_name() const { std::lock_guard<std::mutex> lk(mu_); return device_name_; }
size_t LlamaCppEngineAdapter::device_total_bytes() const { std::lock_guard<std::mutex> lk(mu_); return dev_total_; }
size_t LlamaCppEngineAdapter::device_free_bytes() const { std::lock_guard<std::mutex> lk(mu_); return dev_free_; }
int LlamaCppEngineAdapter::gpu_layers() const { std::lock_guard<std::mutex> lk(mu_); return gpu_layers_; }

} // namespace orchestration
} // namespace agr
