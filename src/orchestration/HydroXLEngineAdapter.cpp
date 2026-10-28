// src/orchestration/HydroXLEngineAdapter.cpp
// Phase 21: Native HydroXL adapters implementing IInferenceEngine

#include "orchestration/HydroXLEngineAdapter.h"
#include "hydroxl/GGUFLoader.h"
#include "hydroxl/HydroXLOps.h"

#include <chrono>
#include <algorithm>
#include <cmath>

namespace agr {
namespace orchestration {

HydroXLCpuEngineAdapter::HydroXLCpuEngineAdapter() {
    model_        = std::make_unique<hydroxl::HydroXLModel>();
    tokenizer_    = std::make_unique<hydroxl::HydroXLTokenizer>();
    kv_cache_     = std::make_unique<hydroxl::HydroXLKVCache>();
    forward_pass_ = std::make_unique<hydroxl::HydroXLForwardPass>();
}

HydroXLCpuEngineAdapter::~HydroXLCpuEngineAdapter() {
    shutdown();
}

bool HydroXLCpuEngineAdapter::initialize(const std::string& model_path, std::string& error) {
    std::lock_guard<std::mutex> lk(mu_);

    if (model_path.empty()) {
        error = "Model path is empty";
        return false;
    }

    model_        = std::make_unique<hydroxl::HydroXLModel>();
    tokenizer_    = std::make_unique<hydroxl::HydroXLTokenizer>();
    kv_cache_     = std::make_unique<hydroxl::HydroXLKVCache>();
    forward_pass_ = std::make_unique<hydroxl::HydroXLForwardPass>();

    if (!model_->load(model_path)) {
        error = "Failed to load GGUF model: " + model_->lastError();
        return false;
    }

    hydroxl::GGUFLoader tok_gguf;
    if (tok_gguf.open(model_path)) {
        tokenizer_->loadFromGGUF(tok_gguf);
    }

    const auto& hp = model_->hparams();
    uint32_t max_seq = std::min(hp.n_ctx_max, 2048u);
    if (max_seq == 0) max_seq = 512;

    if (!kv_cache_->allocate(hp.n_layers, hp.n_kv_heads, max_seq, hp.n_head_dim)) {
        error = "Failed to allocate KV cache";
        return false;
    }

    forward_pass_->attach(model_.get(), kv_cache_.get());
    model_path_ = model_path;
    is_ready_ = true;
    return true;
}

InferenceResult HydroXLCpuEngineAdapter::run(const InferenceRequest& req) {
    std::lock_guard<std::mutex> lk(mu_);

    InferenceResult res;
    res.backend_used = "hydroxl_cpu";
    res.gpu_used = false;

    if (!is_ready_ || !forward_pass_ || !forward_pass_->isReady()) {
        res.success = false;
        res.error = "Engine not ready or not initialized";
        return res;
    }

    std::vector<hydroxl::TokenID> prompt_tokens;
    if (!req.prompt_token_ids.empty()) {
        for (int tid : req.prompt_token_ids) {
            prompt_tokens.push_back(static_cast<hydroxl::TokenID>(tid));
        }
    } else {
        prompt_tokens = tokenizer_->encode("The capital of France is", tokenizer_->addsBos());
    }

    if (prompt_tokens.empty()) {
        prompt_tokens.push_back(1); // minimal token fallback
    }

    res.prompt_tokens = static_cast<int>(prompt_tokens.size());
    kv_cache_->reset();

    std::vector<float> logits;
    uint32_t pos = 0;

    for (auto tok : prompt_tokens) {
        if (!forward_pass_->forward(static_cast<uint32_t>(tok), pos, logits)) {
            res.success = false;
            res.error = "Prefill failed: " + forward_pass_->lastError();
            return res;
        }
        ++pos;
    }

    hydroxl::SamplerParams sp;
    sp.temperature = req.temperature;
    sp.top_k = 40;
    sp.top_p = req.top_p;
    sp.seed = 42;
    hydroxl::HydroXLSampler sampler(sp);

    std::vector<int> out_ids;
    std::string out_text;
    int max_new = req.max_new_tokens > 0 ? req.max_new_tokens : 16;

    auto t_gen_start = std::chrono::steady_clock::now();
    for (int i = 0; i < max_new; ++i) {
        if (pos >= kv_cache_->maxSeqLen()) break;

        hydroxl::TokenID next_tok = sampler.sample(
            logits.data(),
            static_cast<uint32_t>(logits.size()),
            prompt_tokens
        );

        if (next_tok == tokenizer_->eosToken() || next_tok == hydroxl::TOKEN_INVALID) {
            break;
        }

        out_ids.push_back(static_cast<int>(next_tok));
        std::string piece = tokenizer_->decodeToken(next_tok);
        out_text += piece;

        if (!forward_pass_->forward(static_cast<uint32_t>(next_tok), pos, logits)) {
            res.success = false;
            res.error = "Forward pass failed during generation: " + forward_pass_->lastError();
            return res;
        }
        ++pos;
    }
    auto t_gen_end = std::chrono::steady_clock::now();

    res.output_token_ids = out_ids;
    res.output_text = out_text;
    res.output_tokens = static_cast<int>(out_ids.size());
    double gen_sec = std::chrono::duration<double>(t_gen_end - t_gen_start).count();
    if (gen_sec > 0.0 && res.output_tokens > 0) {
        res.tokens_per_second = static_cast<float>(res.output_tokens / gen_sec);
    }
    res.success = true;
    return res;
}

void HydroXLCpuEngineAdapter::shutdown() {
    std::lock_guard<std::mutex> lk(mu_);
    model_.reset();
    tokenizer_.reset();
    kv_cache_.reset();
    forward_pass_.reset();
    is_ready_ = false;
    model_path_.clear();
}

std::string HydroXLCpuEngineAdapter::backend_id() const {
    return "hydroxl_cpu";
}

bool HydroXLCpuEngineAdapter::is_ready() const {
    std::lock_guard<std::mutex> lk(mu_);
    return is_ready_;
}

// ---------------------------------------------------------------------------
// HydroXLVulkanEngineAdapter
// ---------------------------------------------------------------------------
HydroXLVulkanEngineAdapter::HydroXLVulkanEngineAdapter() = default;

HydroXLVulkanEngineAdapter::~HydroXLVulkanEngineAdapter() {
    shutdown();
}

bool HydroXLVulkanEngineAdapter::initialize(const std::string& model_path, std::string& error) {
    std::lock_guard<std::mutex> lk(mu_);

    if (model_path.empty()) {
        error = "Model path is empty";
        return false;
    }

    if (!vk_ops_.initialize(error)) {
        error = "HydroXLVulkanEngineAdapter: failed to initialize Vulkan: " + error;
        is_ready_ = false;
        return false;
    }

    model_     = std::make_unique<hydroxl::HydroXLModel>();
    tokenizer_ = std::make_unique<hydroxl::HydroXLTokenizer>();
    kv_cache_  = std::make_unique<hydroxl::HydroXLKVCache>();

    if (!model_->load(model_path)) {
        error = "Failed to load GGUF model: " + model_->lastError();
        model_.reset();
        vk_ops_.shutdown();
        is_ready_ = false;
        return false;
    }

    hydroxl::GGUFLoader tok_gguf;
    if (tok_gguf.open(model_path)) {
        tokenizer_->loadFromGGUF(tok_gguf);
    }

    const auto& hp = model_->hparams();
    uint32_t max_seq = std::min(hp.n_ctx_max, 2048u);
    if (max_seq == 0) max_seq = 512;

    if (!kv_cache_->allocate(hp.n_layers, hp.n_kv_heads, max_seq, hp.n_head_dim)) {
        error = "Failed to allocate KV cache";
        model_.reset();
        tokenizer_.reset();
        vk_ops_.shutdown();
        is_ready_ = false;
        return false;
    }

    // Upload output norm and LM head
    if (!model_->outputNorm().empty()) {
        if (!vk_ops_.uploadTensor("output_norm", model_->outputNorm().data(), static_cast<uint32_t>(model_->outputNorm().size()))) {
            error = "Failed to upload output_norm: " + vk_ops_.lastError();
            model_.reset(); tokenizer_.reset(); kv_cache_.reset(); vk_ops_.shutdown(); is_ready_ = false;
            return false;
        }
    }
    if (!model_->lmHead().empty()) {
        if (!vk_ops_.uploadTensor("output.weight", model_->lmHead().data(), static_cast<uint32_t>(model_->lmHead().size()))) {
            error = "Failed to upload output.weight: " + vk_ops_.lastError();
            model_.reset(); tokenizer_.reset(); kv_cache_.reset(); vk_ops_.shutdown(); is_ready_ = false;
            return false;
        }
    }

    // Upload layer weights
    for (uint32_t l = 0; l < hp.n_layers; ++l) {
        const auto& w = model_->layer(l);
        std::string pfx = "blk." + std::to_string(l) + ".";
        if (!w.attn_norm.empty()) {
            if (!vk_ops_.uploadTensor(pfx + "attn_norm", w.attn_norm.data(), static_cast<uint32_t>(w.attn_norm.size()))) {
                error = "Failed to upload " + pfx + "attn_norm: " + vk_ops_.lastError();
                model_.reset(); tokenizer_.reset(); kv_cache_.reset(); vk_ops_.shutdown(); is_ready_ = false;
                return false;
            }
        }
        if (!w.attn_q.empty()) {
            if (!vk_ops_.uploadTensor(pfx + "attn_q", w.attn_q.data(), static_cast<uint32_t>(w.attn_q.size()))) {
                error = "Failed to upload " + pfx + "attn_q: " + vk_ops_.lastError();
                model_.reset(); tokenizer_.reset(); kv_cache_.reset(); vk_ops_.shutdown(); is_ready_ = false;
                return false;
            }
        }
        if (!w.attn_k.empty()) {
            if (!vk_ops_.uploadTensor(pfx + "attn_k", w.attn_k.data(), static_cast<uint32_t>(w.attn_k.size()))) {
                error = "Failed to upload " + pfx + "attn_k: " + vk_ops_.lastError();
                model_.reset(); tokenizer_.reset(); kv_cache_.reset(); vk_ops_.shutdown(); is_ready_ = false;
                return false;
            }
        }
        if (!w.attn_v.empty()) {
            if (!vk_ops_.uploadTensor(pfx + "attn_v", w.attn_v.data(), static_cast<uint32_t>(w.attn_v.size()))) {
                error = "Failed to upload " + pfx + "attn_v: " + vk_ops_.lastError();
                model_.reset(); tokenizer_.reset(); kv_cache_.reset(); vk_ops_.shutdown(); is_ready_ = false;
                return false;
            }
        }
        if (!w.attn_out.empty()) {
            if (!vk_ops_.uploadTensor(pfx + "attn_output", w.attn_out.data(), static_cast<uint32_t>(w.attn_out.size()))) {
                error = "Failed to upload " + pfx + "attn_output: " + vk_ops_.lastError();
                model_.reset(); tokenizer_.reset(); kv_cache_.reset(); vk_ops_.shutdown(); is_ready_ = false;
                return false;
            }
        }
        if (!w.ffn_norm.empty()) {
            if (!vk_ops_.uploadTensor(pfx + "ffn_norm", w.ffn_norm.data(), static_cast<uint32_t>(w.ffn_norm.size()))) {
                error = "Failed to upload " + pfx + "ffn_norm: " + vk_ops_.lastError();
                model_.reset(); tokenizer_.reset(); kv_cache_.reset(); vk_ops_.shutdown(); is_ready_ = false;
                return false;
            }
        }
        if (!w.ffn_gate.empty()) {
            if (!vk_ops_.uploadTensor(pfx + "ffn_gate", w.ffn_gate.data(), static_cast<uint32_t>(w.ffn_gate.size()))) {
                error = "Failed to upload " + pfx + "ffn_gate: " + vk_ops_.lastError();
                model_.reset(); tokenizer_.reset(); kv_cache_.reset(); vk_ops_.shutdown(); is_ready_ = false;
                return false;
            }
        }
        if (!w.ffn_up.empty()) {
            if (!vk_ops_.uploadTensor(pfx + "ffn_up", w.ffn_up.data(), static_cast<uint32_t>(w.ffn_up.size()))) {
                error = "Failed to upload " + pfx + "ffn_up: " + vk_ops_.lastError();
                model_.reset(); tokenizer_.reset(); kv_cache_.reset(); vk_ops_.shutdown(); is_ready_ = false;
                return false;
            }
        }
        if (!w.ffn_down.empty()) {
            if (!vk_ops_.uploadTensor(pfx + "ffn_down", w.ffn_down.data(), static_cast<uint32_t>(w.ffn_down.size()))) {
                error = "Failed to upload " + pfx + "ffn_down: " + vk_ops_.lastError();
                model_.reset(); tokenizer_.reset(); kv_cache_.reset(); vk_ops_.shutdown(); is_ready_ = false;
                return false;
            }
        }
    }

    uint32_t n_embd   = hp.n_embd;
    uint32_t n_heads  = hp.n_heads;
    uint32_t n_kv     = hp.n_kv_heads;
    uint32_t head_dim = hp.n_head_dim;
    uint32_t n_ff     = hp.n_ff;

    hidden_.resize(n_embd);
    normed_.resize(n_embd);
    q_buf_.resize(n_heads * head_dim);
    k_buf_.resize(n_kv * head_dim);
    v_buf_.resize(n_kv * head_dim);
    attn_out_.resize(n_heads * head_dim);
    proj_out_.resize(n_embd);
    gate_buf_.resize(n_ff);
    up_buf_.resize(n_ff);
    ffn_buf_.resize(n_ff);
    ffn_out_.resize(n_embd);
    attn_scores_.resize(max_seq);

    model_path_ = model_path;
    is_ready_ = true;
    return true;
}

bool HydroXLVulkanEngineAdapter::forward_token(uint32_t token_id, uint32_t pos,
                                               std::vector<float>& logits,
                                               std::string& err) {
    if (!model_ || !kv_cache_ || !vk_ops_.isReady()) {
        err = "Model or Vulkan ops not ready";
        return false;
    }

    const auto& hp = model_->hparams();
    uint32_t n_embd    = hp.n_embd;
    uint32_t n_heads   = hp.n_heads;
    uint32_t n_kv      = hp.n_kv_heads;
    uint32_t head_dim  = hp.n_head_dim;
    uint32_t n_ff      = hp.n_ff;
    uint32_t vocab     = hp.vocab_size;
    float    rope_base = hp.rope_freq_base;
    float    attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    if (logits.size() != vocab) {
        logits.resize(vocab);
    }

    // Step 1: Embedding lookup on CPU
    if (!hydroxl::embed_lookup(model_->tokenEmbedding().data(), token_id,
                               vocab, n_embd, hidden_.data())) {
        err = "Invalid token_id=" + std::to_string(token_id);
        return false;
    }

    // Step 2: Transformer layers
    for (uint32_t l = 0; l < hp.n_layers; ++l) {
        const auto& w = model_->layer(l);
        std::string pfx = "blk." + std::to_string(l) + ".";

        // 2a. Attention RMSNorm on GPU
        if (!vk_ops_.rmsnorm(hidden_.data(), pfx + "attn_norm", normed_.data(), n_embd, hp.rms_norm_eps)) {
            err = "GPU rmsnorm failed at layer " + std::to_string(l) + ": " + vk_ops_.lastError();
            return false;
        }

        // 2b. QKV projections on GPU
        if (!vk_ops_.matvecHost(pfx + "attn_q", normed_.data(), q_buf_.data(), n_heads * head_dim, n_embd)) {
            err = "GPU matvec attn_q failed at layer " + std::to_string(l) + ": " + vk_ops_.lastError();
            return false;
        }
        if (!vk_ops_.matvecHost(pfx + "attn_k", normed_.data(), k_buf_.data(), n_kv * head_dim, n_embd)) {
            err = "GPU matvec attn_k failed at layer " + std::to_string(l) + ": " + vk_ops_.lastError();
            return false;
        }
        if (!vk_ops_.matvecHost(pfx + "attn_v", normed_.data(), v_buf_.data(), n_kv * head_dim, n_embd)) {
            err = "GPU matvec attn_v failed at layer " + std::to_string(l) + ": " + vk_ops_.lastError();
            return false;
        }

        // Add biases if present (Qwen2 has QKV biases)
        if (!w.attn_q_bias.empty()) {
            for (uint32_t i = 0; i < n_heads * head_dim; ++i)
                q_buf_[i] += w.attn_q_bias[i];
        }
        if (!w.attn_k_bias.empty()) {
            for (uint32_t i = 0; i < n_kv * head_dim; ++i)
                k_buf_[i] += w.attn_k_bias[i];
        }
        if (!w.attn_v_bias.empty()) {
            for (uint32_t i = 0; i < n_kv * head_dim; ++i)
                v_buf_[i] += w.attn_v_bias[i];
        }

        // 2c. RoPE on CPU
        hydroxl::rope_apply(q_buf_.data(), n_heads,
                            k_buf_.data(), n_kv,
                            head_dim, pos, rope_base, 1.0f);

        // 2d. Write K, V to cache
        if (!kv_cache_->writeKV(l, pos, k_buf_.data(), v_buf_.data())) {
            err = "KV cache write failed at layer " + std::to_string(l);
            return false;
        }

        // 2e. Causal attention (GQA) on CPU
        uint32_t seq_len = pos + 1;
        if (!hydroxl::causal_attention(q_buf_.data(),
                                      kv_cache_->kLayer(l),
                                      kv_cache_->vLayer(l),
                                      attn_out_.data(),
                                      seq_len,
                                      n_heads, n_kv, head_dim,
                                      kv_cache_->maxSeqLen(),
                                      attn_scale,
                                      attn_scores_.data())) {
            err = "causal_attention failed at layer " + std::to_string(l);
            return false;
        }

        // 2f. Output projection on GPU
        if (!vk_ops_.matvecHost(pfx + "attn_output", attn_out_.data(), proj_out_.data(), n_embd, n_heads * head_dim)) {
            err = "GPU matvec attn_output failed at layer " + std::to_string(l) + ": " + vk_ops_.lastError();
            return false;
        }

        // 2g. Residual add on GPU
        if (!vk_ops_.elemwise(hidden_.data(), proj_out_.data(), hidden_.data(), n_embd, 0)) {
            err = "GPU residual add failed at layer " + std::to_string(l) + ": " + vk_ops_.lastError();
            return false;
        }

        // 2h. FFN RMSNorm on GPU
        if (!vk_ops_.rmsnorm(hidden_.data(), pfx + "ffn_norm", normed_.data(), n_embd, hp.rms_norm_eps)) {
            err = "GPU ffn rmsnorm failed at layer " + std::to_string(l) + ": " + vk_ops_.lastError();
            return false;
        }

        // 2i. SwiGLU FFN on GPU
        if (!vk_ops_.matvecHost(pfx + "ffn_gate", normed_.data(), gate_buf_.data(), n_ff, n_embd)) {
            err = "GPU matvec ffn_gate failed at layer " + std::to_string(l) + ": " + vk_ops_.lastError();
            return false;
        }
        if (!vk_ops_.matvecHost(pfx + "ffn_up", normed_.data(), up_buf_.data(), n_ff, n_embd)) {
            err = "GPU matvec ffn_up failed at layer " + std::to_string(l) + ": " + vk_ops_.lastError();
            return false;
        }
        if (!vk_ops_.elemwise(gate_buf_.data(), up_buf_.data(), ffn_buf_.data(), n_ff, 2)) {
            err = "GPU swiglu failed at layer " + std::to_string(l) + ": " + vk_ops_.lastError();
            return false;
        }
        if (!vk_ops_.matvecHost(pfx + "ffn_down", ffn_buf_.data(), ffn_out_.data(), n_embd, n_ff)) {
            err = "GPU matvec ffn_down failed at layer " + std::to_string(l) + ": " + vk_ops_.lastError();
            return false;
        }

        // 2j. Residual add on GPU
        if (!vk_ops_.elemwise(hidden_.data(), ffn_out_.data(), hidden_.data(), n_embd, 0)) {
            err = "GPU residual add ffn failed at layer " + std::to_string(l) + ": " + vk_ops_.lastError();
            return false;
        }
    }

    // Step 3: Final RMSNorm on GPU
    if (!vk_ops_.rmsnorm(hidden_.data(), "output_norm", normed_.data(), n_embd, hp.rms_norm_eps)) {
        err = "GPU final rmsnorm failed: " + vk_ops_.lastError();
        return false;
    }

    // Step 4: LM head on GPU
    if (!vk_ops_.matvecHost("output.weight", normed_.data(), logits.data(), vocab, n_embd)) {
        err = "GPU LM head matvec failed: " + vk_ops_.lastError();
        return false;
    }

    return true;
}

InferenceResult HydroXLVulkanEngineAdapter::run(const InferenceRequest& req) {
    std::lock_guard<std::mutex> lk(mu_);

    InferenceResult res;
    res.backend_used = "hydroxl_vulkan";
    res.gpu_used = true;

    if (!is_ready_ || !model_ || !tokenizer_ || !kv_cache_ || !vk_ops_.isReady()) {
        res.success = false;
        res.error = "Engine not ready or not initialized";
        return res;
    }

    std::vector<hydroxl::TokenID> prompt_tokens;
    if (!req.prompt_token_ids.empty()) {
        for (int tid : req.prompt_token_ids) {
            prompt_tokens.push_back(static_cast<hydroxl::TokenID>(tid));
        }
    } else {
        prompt_tokens = tokenizer_->encode("The capital of France is", tokenizer_->addsBos());
    }

    if (prompt_tokens.empty()) {
        prompt_tokens.push_back(1); // minimal token fallback
    }

    res.prompt_tokens = static_cast<int>(prompt_tokens.size());
    kv_cache_->reset();

    std::vector<float> logits;
    std::string err;
    uint32_t pos = 0;

    for (auto tok : prompt_tokens) {
        if (!forward_token(static_cast<uint32_t>(tok), pos, logits, err)) {
            res.success = false;
            res.error = "Prefill failed: " + err;
            return res;
        }
        ++pos;
    }

    hydroxl::SamplerParams sp;
    sp.temperature = req.temperature;
    sp.top_k = 40;
    sp.top_p = req.top_p;
    sp.seed = 42;
    hydroxl::HydroXLSampler sampler(sp);

    std::vector<int> out_ids;
    std::string out_text;
    int max_new = req.max_new_tokens > 0 ? req.max_new_tokens : 16;

    auto t_gen_start = std::chrono::steady_clock::now();
    for (int i = 0; i < max_new; ++i) {
        if (pos >= kv_cache_->maxSeqLen()) break;

        hydroxl::TokenID next_tok = sampler.sample(
            logits.data(),
            static_cast<uint32_t>(logits.size()),
            prompt_tokens
        );

        if (next_tok == tokenizer_->eosToken() || next_tok == hydroxl::TOKEN_INVALID) {
            break;
        }

        out_ids.push_back(static_cast<int>(next_tok));
        std::string piece = tokenizer_->decodeToken(next_tok);
        out_text += piece;

        if (!forward_token(static_cast<uint32_t>(next_tok), pos, logits, err)) {
            res.success = false;
            res.error = "Forward pass failed during generation: " + err;
            return res;
        }
        ++pos;
    }
    auto t_gen_end = std::chrono::steady_clock::now();

    res.output_token_ids = out_ids;
    res.output_text = out_text;
    res.output_tokens = static_cast<int>(out_ids.size());
    double gen_sec = std::chrono::duration<double>(t_gen_end - t_gen_start).count();
    if (gen_sec > 0.0 && res.output_tokens > 0) {
        res.tokens_per_second = static_cast<float>(res.output_tokens / gen_sec);
    }
    res.success = true;
    return res;
}

void HydroXLVulkanEngineAdapter::shutdown() {
    std::lock_guard<std::mutex> lk(mu_);
    vk_ops_.shutdown();
    model_.reset();
    tokenizer_.reset();
    kv_cache_.reset();
    is_ready_ = false;
    model_path_.clear();
}

std::string HydroXLVulkanEngineAdapter::backend_id() const {
    return "hydroxl_vulkan";
}

bool HydroXLVulkanEngineAdapter::is_ready() const {
    std::lock_guard<std::mutex> lk(mu_);
    return is_ready_;
}

} // namespace orchestration
} // namespace agr
