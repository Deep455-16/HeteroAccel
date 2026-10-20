// src/hydroxl/HydroXLEngine.cpp
// Phase 17: HydroXL IExecutionEngine implementation.

#include "hydroxl/HydroXLEngine.h"
#include "engine/ExecutionEngineRegistry.h"
#include "hydroxl/GGUFLoader.h"

#include <chrono>
#include <iostream>

namespace agr {

// ---------------------------------------------------------------------------
// Static registration
// ---------------------------------------------------------------------------
void HydroXLEngine::registerInRegistry() {
    ExecutionEngineRegistry::instance().registerFactory(
        "hydroxl",
        "HydroXL native inference engine (Phase 17)",
        []() { return std::make_shared<HydroXLEngine>(); }
    );
}

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------
HydroXLEngine::HydroXLEngine() {
    model_        = std::make_unique<hydroxl::HydroXLModel>();
    tokenizer_    = std::make_unique<hydroxl::HydroXLTokenizer>();
    kv_cache_     = std::make_unique<hydroxl::HydroXLKVCache>();
    forward_pass_ = std::make_unique<hydroxl::HydroXLForwardPass>();
}

HydroXLEngine::~HydroXLEngine() {
    shutdown();
}

// ---------------------------------------------------------------------------
// Identity & capabilities
// ---------------------------------------------------------------------------
EngineIdentity HydroXLEngine::identity() const {
    return {
        /* name         */ "HydroXL",
        /* version      */ "0.1.0-phase17",
        /* type         */ "LLM",
        /* author       */ "HeteroAccel / Phase 17",
        /* registry_key */ "hydroxl"
    };
}

EngineCapabilitySet HydroXLEngine::capabilities() const {
    return EngineCapabilitySet{
        EngineCapability::TEXT_GENERATION,
        EngineCapability::BACKEND_CPU,       // CPU only in Phase 17
        EngineCapability::STREAMING,
        EngineCapability::CANCELLATION,
        EngineCapability::DYNAMIC_LOADING,
    };
    // NOT advertised: BACKEND_VULKAN, BACKEND_CUDA, BACKEND_NPU.
    // We only claim CPU because that is what Phase 17 executes on.
}

// ---------------------------------------------------------------------------
// Engine lifecycle
// ---------------------------------------------------------------------------
bool HydroXLEngine::initialize() {
    std::lock_guard<std::mutex> lk(mutex_);
    if (initialized_) return true;
    last_error_ = EngineError::success();
    initialized_ = true;
    return true;
}

void HydroXLEngine::shutdown() {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!initialized_) return;
    if (model_)    model_->~HydroXLModel(),    new(model_.get())    hydroxl::HydroXLModel();
    if (kv_cache_) kv_cache_->reset();
    model_loaded_ = false;
    initialized_  = false;
}

bool HydroXLEngine::isInitialized() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return initialized_;
}

// ---------------------------------------------------------------------------
// Model lifecycle
// ---------------------------------------------------------------------------
bool HydroXLEngine::loadModel(const ModelDescriptor& descriptor) {
    std::lock_guard<std::mutex> lk(mutex_);

    if (!initialized_) {
        last_error_ = EngineError::make(EngineErrorKind::ENGINE_UNAVAILABLE,
                                        "HydroXL not initialized");
        return false;
    }

    if (!descriptor.isValid()) {
        last_error_ = EngineError::make(EngineErrorKind::INVALID_REQUEST,
                                        "ModelDescriptor is invalid (empty id or path)");
        return false;
    }

    // Unload any previously loaded model
    model_        = std::make_unique<hydroxl::HydroXLModel>();
    tokenizer_    = std::make_unique<hydroxl::HydroXLTokenizer>();
    kv_cache_     = std::make_unique<hydroxl::HydroXLKVCache>();
    forward_pass_ = std::make_unique<hydroxl::HydroXLForwardPass>();
    model_loaded_ = false;

    auto t0 = std::chrono::steady_clock::now();

    // --- Load model weights ---
    if (!model_->load(descriptor.source_path)) {
        last_error_ = EngineError::make(EngineErrorKind::MODEL_LOAD_FAILED,
                                        model_->lastError());
        return false;
    }

    // --- Load tokenizer from same GGUF ---
    hydroxl::GGUFLoader tok_gguf;
    if (tok_gguf.open(descriptor.source_path)) {
        if (!tokenizer_->loadFromGGUF(tok_gguf)) {
            last_error_ = EngineError::make(EngineErrorKind::MODEL_LOAD_FAILED,
                                            tokenizer_->lastError());
            return false;
        }
    } else {
        // Tokenizer load is non-fatal: we still have a placeholder
        tokenizer_->loadFromGGUF(tok_gguf); // safe with invalid gguf
    }

    // --- Allocate KV cache ---
    const auto& hp = model_->hparams();
    // Limit max_seq_len to 2048 for memory: the full 32768 would require
    // 24 * 2 * 2 * 2048 * 64 * 4 bytes ≈ 50 MB which is fine.
    uint32_t max_seq = std::min(hp.n_ctx_max, 2048u);

    if (!kv_cache_->allocate(hp.n_layers, hp.n_kv_heads,
                             max_seq, hp.n_head_dim)) {
        last_error_ = EngineError::make(EngineErrorKind::RESOURCE_UNAVAILABLE,
                                        "Failed to allocate KV cache");
        return false;
    }

    // --- Attach forward pass ---
    forward_pass_->attach(model_.get(), kv_cache_.get());

    auto t1 = std::chrono::steady_clock::now();
    static_cast<void>(t1); // load timing reserved for telemetry

    current_descriptor_ = descriptor;
    model_loaded_       = true;
    last_error_         = EngineError::success();

    return true;
}

void HydroXLEngine::unloadModel() {
    std::lock_guard<std::mutex> lk(mutex_);
    model_        = std::make_unique<hydroxl::HydroXLModel>();
    tokenizer_    = std::make_unique<hydroxl::HydroXLTokenizer>();
    kv_cache_     = std::make_unique<hydroxl::HydroXLKVCache>();
    forward_pass_ = std::make_unique<hydroxl::HydroXLForwardPass>();
    model_loaded_ = false;
    current_descriptor_ = {};
}

bool HydroXLEngine::isModelLoaded() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return model_loaded_;
}

ModelDescriptor HydroXLEngine::currentModel() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return current_descriptor_;
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------
ExecutionResult HydroXLEngine::execute(const ExecutionContext& ctx) {
    std::lock_guard<std::mutex> lk(mutex_);
    return doGenerate(ctx, nullptr);
}

ExecutionResult HydroXLEngine::executeStreaming(const ExecutionContext& ctx,
                                                TokenCallback cb) {
    std::lock_guard<std::mutex> lk(mutex_);
    return doGenerate(ctx, std::move(cb));
}

ExecutionResult HydroXLEngine::doGenerate(const ExecutionContext& ctx,
                                           TokenCallback cb) {
    ExecutionResult result;
    result.telemetry.backend_type = ExecutionBackendType::CPU;
    result.telemetry.backend_name = "HydroXL";
    result.telemetry.device_name  = "CPU (HydroXL native)";
    result.telemetry.gpu_used     = false;

    if (!model_loaded_) {
        result.success = false;
        result.error   = "No model loaded in HydroXLEngine";
        return result;
    }

    auto t_start = std::chrono::steady_clock::now();

    // ---- Tokenize prompt ----
    auto t_tok0 = std::chrono::steady_clock::now();
    // BOS behaviour follows the tokenizer metadata: Qwen2's gpt2-style
    // tokenizer does not prepend BOS, matching llama.cpp for this arch.
    std::vector<hydroxl::TokenID> prompt_tokens =
        tokenizer_->encode(ctx.prompt, tokenizer_->addsBos());
    result.telemetry.prompt_tokens = static_cast<int>(prompt_tokens.size());

    auto t_tok1 = std::chrono::steady_clock::now();

    // ---- Reset KV cache for new conversation ----
    kv_cache_->reset();

    // ---- Prefill (prompt tokens) ----
    std::vector<float> logits;
    uint32_t pos = 0;

    auto t_prefill0 = std::chrono::steady_clock::now();
    for (hydroxl::TokenID tok : prompt_tokens) {
        if (ctx.cancel_flag && ctx.cancel_flag->load()) {
            result.success = false;
            result.error   = "Cancelled during prefill";
            last_error_    = EngineError::make(EngineErrorKind::CANCELLED, result.error);
            return result;
        }
        if (pos >= kv_cache_->maxSeqLen()) {
            result.success = false;
            result.error   = "Prompt exceeds max_seq_len";
            return result;
        }
        if (!forward_pass_->forward(static_cast<uint32_t>(tok), pos, logits)) {
            result.success = false;
            result.error   = "Forward pass failed: " + forward_pass_->lastError();
            return result;
        }
        ++pos;
    }
    auto t_prefill1 = std::chrono::steady_clock::now();
    result.telemetry.prompt_eval_ms =
        std::chrono::duration<double, std::milli>(t_prefill1 - t_prefill0).count();

    // Guard: with no prompt tokens there are no logits to sample from.
    if (prompt_tokens.empty() || logits.empty()) {
        result.success = false;
        result.error   = "Prompt encoded to zero tokens — nothing to condition on";
        return result;
    }

    // ---- Autoregressive generation ----
    hydroxl::SamplerParams sp;
    sp.temperature = ctx.temperature;
    sp.top_k       = 40;
    sp.top_p       = 0.95f;
    sp.seed        = ctx.seed;
    hydroxl::HydroXLSampler sampler(sp);

    std::vector<hydroxl::TokenID> generated_tokens;
    int max_new = ctx.max_tokens;
    if (max_new <= 0) max_new = 256;

    std::string output_text;

    auto t_gen0 = std::chrono::steady_clock::now();
    for (int i = 0; i < max_new; ++i) {
        if (ctx.cancel_flag && ctx.cancel_flag->load()) {
            result.success = false;
            result.error   = "Cancelled during generation";
            last_error_    = EngineError::make(EngineErrorKind::CANCELLED, result.error);
            return result;
        }
        if (pos >= kv_cache_->maxSeqLen()) break;

        // Sample next token
        hydroxl::TokenID next_tok = sampler.sample(
            logits.data(),
            static_cast<uint32_t>(logits.size()),
            generated_tokens);

        if (next_tok == tokenizer_->eosToken() || next_tok == hydroxl::TOKEN_INVALID) break;

        generated_tokens.push_back(next_tok);

        // Decode token to text
        std::string piece = tokenizer_->decodeToken(next_tok);
        output_text += piece;

        // Streaming callback
        if (cb) cb(piece);

        // Forward pass for the new token
        if (!forward_pass_->forward(static_cast<uint32_t>(next_tok), pos, logits)) {
            result.success = false;
            result.error   = "Forward pass failed at token " + std::to_string(i)
                           + ": " + forward_pass_->lastError();
            return result;
        }
        ++pos;
    }
    auto t_gen1 = std::chrono::steady_clock::now();

    // ---- Fill result ----
    result.success           = true;
    result.output            = output_text;
    result.telemetry.output_tokens = static_cast<int>(generated_tokens.size());
    result.telemetry.generation_ms =
        std::chrono::duration<double, std::milli>(t_gen1 - t_gen0).count();
    result.telemetry.total_ms =
        result.telemetry.prompt_eval_ms + result.telemetry.generation_ms;
    result.telemetry.ttft_ms = result.telemetry.prompt_eval_ms;
    if (result.telemetry.generation_ms > 0.0 && result.telemetry.output_tokens > 0) {
        result.telemetry.tokens_per_sec =
            result.telemetry.output_tokens / (result.telemetry.generation_ms / 1000.0);
    }
    result.telemetry.scheduler_reason = "HydroXL native CPU engine";

    return result;
}

EngineError HydroXLEngine::lastEngineError() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return last_error_;
}

} // namespace agr
