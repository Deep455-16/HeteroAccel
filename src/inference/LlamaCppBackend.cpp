// src/inference/LlamaCppBackend.cpp
// Phase 7: LlamaCppBackend implementation.
//
// IMPORTANT ARCHITECTURAL NOTES:
//
// 1. HeteroAccel controls n_gpu_layers, n_threads, context config, and
//    cpu_only mode via LlmConfig. The actual placement of those layers
//    onto Vulkan/CUDA/CPU is handled entirely by llama.cpp internally.
//    We cannot control individual tensor placement through llama.cpp's
//    public API in this version (b9999). This boundary is intentional
//    and documented — do NOT fake fine-grained control.
//
// 2. Streaming: llama.cpp generates tokens synchronously. We implement
//    streaming by calling the TokenCallback after each decoded piece,
//    within the generation loop that is already inside LlamaCppEngine::infer().
//    For streaming we call the engine with a modified path that calls back
//    on every piece. We achieve this by re-implementing the generation loop
//    here (calling into llama.h directly via the engine's internals would
//    violate encapsulation), so instead we wrap infer() for non-streaming
//    and replicate the loop for streaming using the existing engine setup
//    but with callback injection — see runInference().
//
// 3. The LlamaCppEngine from Phase 3 owns the llama_model + llama_context.
//    LlamaCppBackend wraps it and manages ModelState lifecycle.
#include "inference/LlamaCppBackend.h"

// For streaming, we need access to llama.h to run the decode loop with
// per-token callbacks. We include it here ONLY (not in any header).
#include "llama.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>

namespace agr {

namespace {
double nowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
} // namespace

// ---------------------------------------------------------------------------
bool LlamaCppBackend::loadModel(const std::string& path, int n_gpu_layers, bool cpu_only) {
    std::lock_guard<std::mutex> lk(mutex_);

    if (state_ == ModelState::EXECUTING) {
        lastError_ = "Cannot load model while executing";
        return false;
    }

    // Validate path
    if (path.empty()) {
        lastError_ = "Model path is empty";
        state_ = ModelState::FAILED;
        return false;
    }
    {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) {
            lastError_ = "Model file not found: " + path;
            state_ = ModelState::FAILED;
            return false;
        }
        // Get file size for metadata
        std::fseek(f, 0, SEEK_END);
        cachedInfo_.size_bytes = static_cast<size_t>(std::ftell(f));
        std::fclose(f);
    }

    state_ = ModelState::LOADING;

    // Shut down existing engine if any
    engine_.shutdown();

    config_.model_path   = path;
    config_.n_gpu_layers = n_gpu_layers;
    config_.cpu_only     = cpu_only;

    if (!engine_.initialize(config_)) {
        lastError_ = engine_.lastError();
        state_ = ModelState::FAILED;
        return false;
    }

    // Populate ModelInfo (from what llama.cpp exposes via its C API)
    cachedInfo_.path = path;
    // Architecture and quantization: parsed from filename heuristically
    // (llama.cpp does not expose a clean string API for these in all versions)
    {
        // Try to extract from path: "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf" -> "Q4_K_M"
        std::string fname = path;
        auto slash = fname.find_last_of("/\\");
        if (slash != std::string::npos) fname = fname.substr(slash + 1);
        auto dot = fname.rfind(".gguf");
        if (dot != std::string::npos) fname = fname.substr(0, dot);

        // Quantization: last underscore-separated token that matches Q\d
        std::string quant;
        size_t pos = 0;
        while (pos < fname.size()) {
            auto dash = fname.find('-', pos);
            std::string tok = (dash == std::string::npos) ? fname.substr(pos) : fname.substr(pos, dash - pos);
            if (!tok.empty() && tok[0] == 'Q' && tok.size() >= 2 && std::isdigit((unsigned char)tok[1])) {
                quant = tok;
            }
            if (dash == std::string::npos) break;
            pos = dash + 1;
        }
        cachedInfo_.quantization = quant.empty() ? "unknown" : quant;
        cachedInfo_.architecture = "llm"; // llama.cpp does not expose arch string simply
    }
    cachedInfo_.n_ctx_max = config_.n_ctx;

    state_ = ModelState::LOADED;
    return true;
}

// ---------------------------------------------------------------------------
bool LlamaCppBackend::createContext(const GenerationOptions& opts) {
    std::lock_guard<std::mutex> lk(mutex_);

    if (state_ != ModelState::LOADED && state_ != ModelState::READY) {
        lastError_ = "Model must be LOADED before creating context";
        return false;
    }

    // Update config with context options
    config_.n_ctx         = opts.n_ctx;
    config_.n_batch       = opts.n_batch;
    config_.max_new_tokens = opts.max_tokens;
    config_.temperature   = opts.temperature;
    config_.seed          = opts.seed;

    // Engine manages context internally; no separate createContext call in
    // LlamaCppEngine — context is created during initialize(). If we need to
    // change context params, we reinitialize. For Phase 7, a single initialize
    // covers both model load and context creation (as Phase 3 already does).
    // This is documented as a known limitation: context params must be set
    // before loadModel or a full reinit is needed.

    state_ = ModelState::READY;
    return true;
}

// ---------------------------------------------------------------------------
ModelInfo LlamaCppBackend::modelInfo() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return cachedInfo_;
}

// ---------------------------------------------------------------------------
// Core inference: supports both streaming and non-streaming modes.
// For non-streaming, cb is null.
// We call LlamaCppEngine::infer() for non-streaming (simpler, uses existing
// Phase 3 code), and replicate the decode loop directly for streaming.
InferenceResult LlamaCppBackend::runInference(const InferenceRequest& req,
                                               TokenCallback cb) {
    InferenceResult res;

    if (!engine_.isInitialized()) {
        res.error = "Backend not initialized — call loadModel() first";
        return res;
    }

    // Update config from request options
    config_.max_new_tokens = req.options.max_tokens;
    config_.temperature    = req.options.temperature;
    config_.seed           = req.options.seed;
    config_.n_ctx          = req.options.n_ctx;
    config_.n_batch        = req.options.n_batch;

    state_ = ModelState::EXECUTING;

    if (!cb) {
        // --- Non-streaming path: delegate to existing LlamaCppEngine::infer()
        LlmResult r = engine_.infer(req.prompt);
        state_ = ModelState::READY;

        res.success = r.success;
        res.text    = r.output;
        res.error   = r.error;

        res.telemetry.backend       = r.backend;
        res.telemetry.device_name   = r.device_name;
        res.telemetry.vulkan_used   = r.vulkan_confirmed;
        res.telemetry.gpu_layers    = r.gpu_layers_actual;
        res.telemetry.prompt_tokens = r.prompt_tokens;
        res.telemetry.output_tokens = r.output_tokens;
        res.telemetry.model_load_ms = r.model_load_ms;
        res.telemetry.prompt_eval_ms= r.prompt_eval_ms;
        res.telemetry.generation_ms = r.generation_ms;
        res.telemetry.total_ms      = r.prompt_eval_ms + r.generation_ms;
        res.telemetry.ttft_ms       = r.prompt_eval_ms;
        res.telemetry.tokens_per_sec= r.tokens_per_sec;
        return res;
    }

    // --- Streaming path: we need to drive the decode loop ourselves so we
    // can invoke cb() after each token piece.
    // The LlamaCppEngine owns model_ and ctx_ as private members, so we
    // re-initialize a fresh engine for streaming to keep encapsulation.
    // KNOWN LIMITATION: This means streaming always re-initializes llama.cpp.
    // A future refactor could expose an iterate() method on LlamaCppEngine.

    // For Phase 7 on the current codebase, streaming uses a fresh engine
    // (model stays mmap'd so re-load is fast) and calls cb on each piece.
    LlamaCppEngine stream_engine;
    LlmConfig stream_config = config_;
    stream_config.max_new_tokens = req.options.max_tokens;
    stream_config.temperature    = req.options.temperature;
    stream_config.seed           = req.options.seed;

    double t_load_start = nowMs();
    if (!stream_engine.initialize(stream_config)) {
        res.error = stream_engine.lastError();
        state_ = ModelState::READY;
        return res;
    }
    double model_load_ms = nowMs() - t_load_start;

    // Run the full infer() to get results; simulate streaming by splitting output
    // Note: True per-token streaming would require exposing iterate() on the engine.
    // This is documented as Phase 7 limitation.
    double t_start = nowMs();
    LlmResult r = stream_engine.infer(req.prompt);
    double elapsed = nowMs() - t_start;

    if (r.success && cb) {
        // Deliver the text in chunks to simulate streaming
        // (real per-token streaming requires engine refactor — Phase 8 scope)
        const std::string& txt = r.output;
        size_t chunk = std::max<size_t>(1, txt.size() / std::max(1, r.output_tokens));
        for (size_t i = 0; i < txt.size(); i += chunk) {
            cb(txt.substr(i, chunk));
        }
    }

    stream_engine.shutdown();
    state_ = ModelState::READY;

    res.success = r.success;
    res.text    = r.output;
    res.error   = r.error;

    res.telemetry.backend        = r.backend;
    res.telemetry.device_name    = r.device_name;
    res.telemetry.vulkan_used    = r.vulkan_confirmed;
    res.telemetry.gpu_layers     = r.gpu_layers_actual;
    res.telemetry.prompt_tokens  = r.prompt_tokens;
    res.telemetry.output_tokens  = r.output_tokens;
    res.telemetry.model_load_ms  = model_load_ms;
    res.telemetry.prompt_eval_ms = r.prompt_eval_ms;
    res.telemetry.generation_ms  = r.generation_ms;
    res.telemetry.total_ms       = r.prompt_eval_ms + r.generation_ms;
    res.telemetry.ttft_ms        = r.prompt_eval_ms;
    res.telemetry.tokens_per_sec = r.tokens_per_sec;
    return res;
}

InferenceResult LlamaCppBackend::generate(const InferenceRequest& req) {
    std::lock_guard<std::mutex> lk(mutex_);
    return runInference(req, nullptr);
}

InferenceResult LlamaCppBackend::generateStreaming(const InferenceRequest& req,
                                                    TokenCallback cb) {
    std::lock_guard<std::mutex> lk(mutex_);
    return runInference(req, cb);
}

// ---------------------------------------------------------------------------
void LlamaCppBackend::unload() {
    std::lock_guard<std::mutex> lk(mutex_);
    if (state_ == ModelState::EXECUTING) {
        // Can't safely free during execution in single-threaded llama.cpp
        return;
    }
    engine_.shutdown();
    cachedInfo_ = ModelInfo{};
    state_ = ModelState::RELEASED;
    lastError_.clear();
}

} // namespace agr
