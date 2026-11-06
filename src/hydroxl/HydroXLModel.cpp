// src/hydroxl/HydroXLModel.cpp
// Phase 17 / Phase 27: Universal model weight loader and dequantizer for HydroXL.

#include "hydroxl/HydroXLModel.h"
#include "hydroxl/HydroXLDequant.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>

namespace agr {
namespace hydroxl {

// ---------------------------------------------------------------------------
// Load hyperparameters from GGUF metadata
// ---------------------------------------------------------------------------
bool HydroXLModel::loadHParams(GGUFLoader& gguf) {
    hparams_ = Qwen2HParams{};

    std::string raw_arch = gguf.kvString("general.architecture", "");
    std::string arch = raw_arch.empty() ? "qwen2" : raw_arch;

    ModelArch parsed_arch = parse_model_arch(arch);
    if (parsed_arch == ModelArch::UNKNOWN) {
        last_error_ = "Unsupported architecture '" + arch +
                      "' (supported: llama, mistral, mistral3, qwen2, phi3, gemma, gemma2)";
        return false;
    }

    hparams_.arch      = parsed_arch;
    hparams_.arch_name = arch;

    // Canonical metadata prefix per architecture family
    std::string primary_prefix = arch;
    if (arch == "qwen2_vl") primary_prefix = "qwen2";

    auto kvU32 = [&](const std::string& suffix, uint32_t def) -> uint32_t {
        std::string k1 = primary_prefix + "." + suffix;
        if (gguf.kv(k1)) return gguf.kvUInt32(k1, def);
        if (arch != primary_prefix) {
            std::string k2 = arch + "." + suffix;
            if (gguf.kv(k2)) return gguf.kvUInt32(k2, def);
        }
        if (parsed_arch == ModelArch::MISTRAL) {
            if (gguf.kv("mistral." + suffix))  return gguf.kvUInt32("mistral." + suffix, def);
            if (gguf.kv("mistral3." + suffix)) return gguf.kvUInt32("mistral3." + suffix, def);
            if (gguf.kv("llama." + suffix))    return gguf.kvUInt32("llama." + suffix, def);
        }
        if (parsed_arch == ModelArch::PHI3) {
            if (gguf.kv("phi3." + suffix)) return gguf.kvUInt32("phi3." + suffix, def);
            if (gguf.kv("phi2." + suffix)) return gguf.kvUInt32("phi2." + suffix, def);
        }
        return def;
    };

    auto kvF32 = [&](const std::string& suffix, float def) -> float {
        std::string k1 = primary_prefix + "." + suffix;
        if (gguf.kv(k1)) return gguf.kvFloat(k1, def);
        if (arch != primary_prefix) {
            std::string k2 = arch + "." + suffix;
            if (gguf.kv(k2)) return gguf.kvFloat(k2, def);
        }
        if (parsed_arch == ModelArch::MISTRAL) {
            if (gguf.kv("mistral." + suffix))  return gguf.kvFloat("mistral." + suffix, def);
            if (gguf.kv("mistral3." + suffix)) return gguf.kvFloat("mistral3." + suffix, def);
            if (gguf.kv("llama." + suffix))    return gguf.kvFloat("llama." + suffix, def);
        }
        if (parsed_arch == ModelArch::PHI3) {
            if (gguf.kv("phi3." + suffix)) return gguf.kvFloat("phi3." + suffix, def);
            if (gguf.kv("phi2." + suffix)) return gguf.kvFloat("phi2." + suffix, def);
        }
        return def;
    };

    // Determine vocab_size from explicit metadata, tokenizer array, or token_embd tensor shape
    hparams_.vocab_size = gguf.kvUInt32("llm.vocab_size", 0);
    if (hparams_.vocab_size == 0) {
        hparams_.vocab_size = kvU32("vocab_size", 0);
    }
    if (hparams_.vocab_size == 0) {
        if (const auto* tok_desc = gguf.findTensor("token_embd.weight")) {
            if (tok_desc->shape.size() >= 2) {
                hparams_.vocab_size = static_cast<uint32_t>(tok_desc->shape[1]);
            }
        }
    }
    if (hparams_.vocab_size == 0) {
        auto toks = gguf.kvStringArray("tokenizer.ggml.tokens");
        if (!toks.empty()) {
            hparams_.vocab_size = static_cast<uint32_t>(toks.size());
        }
    }
    if (hparams_.vocab_size == 0 && parsed_arch == ModelArch::QWEN2) {
        hparams_.vocab_size = 151936;
    }

    // Core dimensions (preserve Qwen2 fallback defaults when arch is qwen2 and keys are omitted)
    const bool is_qwen_fallback = (parsed_arch == ModelArch::QWEN2);
    hparams_.n_embd     = kvU32("embedding_length",         is_qwen_fallback ? 896u   : 0u);
    hparams_.n_heads    = kvU32("attention.head_count",     is_qwen_fallback ? 14u    : 0u);
    hparams_.n_kv_heads = kvU32("attention.head_count_kv",  is_qwen_fallback ? 2u     : hparams_.n_heads);
    hparams_.n_layers   = kvU32("block_count",              is_qwen_fallback ? 24u    : 0u);
    hparams_.n_ff       = kvU32("feed_forward_length",      is_qwen_fallback ? 4864u  : 0u);
    hparams_.n_ctx_max  = kvU32("context_length",           is_qwen_fallback ? 32768u : 4096u);

    // Architecture-specific default norms, RoPE, and activations
    float default_eps       = 1e-6f;
    float default_rope_base = 10000.0f;

    switch (parsed_arch) {
        case ModelArch::LLAMA:
            hparams_.rope_type       = RoPEType::NORM;
            hparams_.ffn_activation  = FFNActivation::SWIGLU;
            hparams_.embedding_scale = 1.0f;
            default_eps              = 1e-5f;
            default_rope_base        = 10000.0f;
            break;
        case ModelArch::MISTRAL:
            hparams_.rope_type       = RoPEType::NORM;
            hparams_.ffn_activation  = FFNActivation::SWIGLU;
            hparams_.embedding_scale = 1.0f;
            default_eps              = 1e-5f;
            default_rope_base        = 10000.0f;
            hparams_.sliding_window  = kvU32("attention.sliding_window", 0);
            break;
        case ModelArch::QWEN2:
            hparams_.rope_type       = RoPEType::NEOX;
            hparams_.ffn_activation  = FFNActivation::SWIGLU;
            hparams_.embedding_scale = 1.0f;
            default_eps              = 1e-6f;
            default_rope_base        = 1000000.0f;
            break;
        case ModelArch::PHI3:
            hparams_.rope_type       = RoPEType::NEOX;
            hparams_.ffn_activation  = FFNActivation::SWIGLU;
            hparams_.embedding_scale = 1.0f;
            default_eps              = 1e-5f;
            default_rope_base        = 10000.0f;
            hparams_.sliding_window  = kvU32("attention.sliding_window", 0);
            break;
        case ModelArch::GEMMA:
            hparams_.rope_type       = RoPEType::NEOX;
            hparams_.ffn_activation  = FFNActivation::GEGLU;
            hparams_.embedding_scale = std::sqrt(static_cast<float>(hparams_.n_embd));
            default_eps              = 1e-6f;
            default_rope_base        = 10000.0f;
            break;
        case ModelArch::GEMMA2:
            hparams_.rope_type           = RoPEType::NEOX;
            hparams_.ffn_activation      = FFNActivation::GEGLU;
            hparams_.embedding_scale     = std::sqrt(static_cast<float>(hparams_.n_embd));
            default_eps                  = 1e-6f;
            default_rope_base            = 10000.0f;
            hparams_.attn_logit_softcap  = kvF32("attn_logit_softcapping", 0.0f);
            hparams_.final_logit_softcap = kvF32("final_logit_softcapping", 0.0f);
            hparams_.sliding_window      = kvU32("attention.sliding_window", 0);
            break;
        default:
            break;
    }

    hparams_.rms_norm_eps   = kvF32("attention.layer_norm_rms_epsilon", default_eps);
    hparams_.rope_freq_base = kvF32("rope.freq_base", default_rope_base);

    float scaling_factor = kvF32("rope.scaling.factor", 0.0f);
    if (scaling_factor <= 0.0f) {
        scaling_factor = kvF32("rope.scale_linear", 0.0f);
    }
    hparams_.rope_freq_scale = (scaling_factor > 0.0f) ? (1.0f / scaling_factor) : 1.0f;

    uint32_t default_head_dim = (hparams_.n_heads > 0) ? (hparams_.n_embd / hparams_.n_heads) : 0;
    hparams_.n_head_dim = kvU32("attention.key_length", default_head_dim);
    hparams_.n_rope_dim = kvU32("rope.dimension_count", hparams_.n_head_dim);
    if (hparams_.n_rope_dim == 0 || hparams_.n_rope_dim > hparams_.n_head_dim) {
        hparams_.n_rope_dim = hparams_.n_head_dim;
    }

    if (!hparams_.isValid() || hparams_.n_head_dim == 0 || hparams_.n_rope_dim % 2 != 0) {
        last_error_ = "Invalid hyperparameters for architecture '" + arch + "'";
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Generic tensor load + dequantize helper
// ---------------------------------------------------------------------------
bool HydroXLModel::loadTensor(GGUFLoader& gguf, const std::string& name,
                               uint64_t expected_elems, std::vector<float>& out) {
    const GGUFTensorDesc* desc = gguf.findTensor(name);
    if (!desc) {
        return false;
    }

    if (desc->shape.empty()) {
        last_error_ = "Tensor '" + name + "' has empty shape";
        return false;
    }

    uint64_t n_elems = 1;
    for (auto d : desc->shape) {
        if (d == 0) {
            last_error_ = "Tensor '" + name + "' has 0 dimension";
            return false;
        }
        if (n_elems > UINT64_MAX / d) {
            last_error_ = "Tensor '" + name + "' element count overflow";
            return false;
        }
        n_elems *= d;
    }

    if (expected_elems > 0 && n_elems != expected_elems) {
        last_error_ = "Tensor '" + name + "' has " + std::to_string(n_elems) +
                      " elements, expected " + std::to_string(expected_elems);
        return false;
    }

    if (!is_quant_type_supported(desc->type)) {
        last_error_ = "Tensor '" + name + "' has unsupported type " + toString(desc->type);
        return false;
    }

    const uint32_t block_size  = ggml_type_block_size(desc->type);
    const uint32_t block_bytes = ggml_type_block_bytes(desc->type);

    if (n_elems % block_size != 0) {
        last_error_ = "Tensor '" + name + "' n_elems=" + std::to_string(n_elems) +
                      " not divisible by block size " + std::to_string(block_size) +
                      " for type " + toString(desc->type);
        return false;
    }

    const uint64_t n_blocks = n_elems / block_size;
    if (n_blocks > UINT64_MAX / block_bytes) {
        last_error_ = "Tensor '" + name + "' byte size overflow";
        return false;
    }
    const uint64_t min_bytes = n_blocks * block_bytes;
    if (desc->n_bytes < min_bytes) {
        last_error_ = "Tensor '" + name + "' disk size " + std::to_string(desc->n_bytes) +
                      " < minimum expected " + std::to_string(min_bytes);
        return false;
    }

    out.resize(static_cast<size_t>(n_elems));

    if (raw_buf_.size() < desc->n_bytes) {
        raw_buf_.resize(static_cast<size_t>(desc->n_bytes));
    }
    if (!gguf.readTensorData(*desc, raw_buf_.data(), desc->n_bytes)) {
        last_error_ = "IO error reading tensor '" + name + "'";
        return false;
    }

    if (!dequantize_tensor(desc->type, raw_buf_.data(), out.data(), n_elems)) {
        last_error_ = "Dequantization failed for tensor '" + name + "' (" +
                      toString(desc->type) + ")";
        return false;
    }

    memory_bytes_ += n_elems * sizeof(float);
    return true;
}

// ---------------------------------------------------------------------------
// Main load entry point
// ---------------------------------------------------------------------------
bool HydroXLModel::load(const std::string& path) {
    path_ = path;
    loaded_ = false;
    memory_bytes_ = 0;
    last_error_.clear();

    if (!std::filesystem::exists(path)) {
        last_error_ = "Model file not found: " + path;
        return false;
    }

    GGUFLoader gguf;
    if (!gguf.open(path)) {
        last_error_ = "Failed to open GGUF: " + gguf.lastError();
        return false;
    }

    if (!loadHParams(gguf)) return false;

    const auto& hp = hparams_;
    uint32_t head_dim = hp.n_head_dim;

    // --- Embeddings ---
    if (!loadTensor(gguf, "token_embd.weight",
                    static_cast<uint64_t>(hp.vocab_size) * hp.n_embd,
                    tok_embd_)) {
        if (last_error_.empty()) last_error_ = "Missing required tensor 'token_embd.weight'";
        return false;
    }

    // --- Output norm ---
    if (!loadTensor(gguf, "output_norm.weight", hp.n_embd, output_norm_)) {
        if (last_error_.empty()) last_error_ = "Missing required tensor 'output_norm.weight'";
        return false;
    }

    // --- LM head ---
    if (!loadTensor(gguf, "output.weight",
                    static_cast<uint64_t>(hp.vocab_size) * hp.n_embd,
                    lm_head_)) {
        // Tied embedding: share tok_embd_ without duplicating memory
        lm_head_tied_ = true;
        lm_head_.clear();
        lm_head_.shrink_to_fit();
    } else {
        lm_head_tied_ = false;
    }

    // --- Transformer layers ---
    layers_.clear();
    layers_.resize(hp.n_layers);
    for (uint32_t l = 0; l < hp.n_layers; ++l) {
        std::string prefix = "blk." + std::to_string(l) + ".";
        auto& w = layers_[l];

        // Norms
        if (!loadTensor(gguf, prefix + "attn_norm.weight", hp.n_embd, w.attn_norm)) {
            if (last_error_.empty()) last_error_ = "Missing " + prefix + "attn_norm.weight";
            return false;
        }
        if (!loadTensor(gguf, prefix + "ffn_norm.weight", hp.n_embd, w.ffn_norm)) {
            if (last_error_.empty()) last_error_ = "Missing " + prefix + "ffn_norm.weight";
            return false;
        }

        // Optional Gemma 2 post-attention and post-FFN norms
        if (gguf.findTensor(prefix + "post_attention_norm.weight")) {
            if (!loadTensor(gguf, prefix + "post_attention_norm.weight", hp.n_embd, w.post_attn_norm)) {
                return false;
            }
        }
        if (gguf.findTensor(prefix + "post_ffw_norm.weight")) {
            if (!loadTensor(gguf, prefix + "post_ffw_norm.weight", hp.n_embd, w.post_ffn_norm)) {
                return false;
            }
        }

        // QKV projections (separate attn_q/k/v OR fused attn_qkv as in Phi-3)
        uint64_t q_rows  = static_cast<uint64_t>(hp.n_heads)    * head_dim;
        uint64_t kv_rows = static_cast<uint64_t>(hp.n_kv_heads) * head_dim;
        uint64_t q_size  = q_rows  * hp.n_embd;
        uint64_t kv_size = kv_rows * hp.n_embd;

        if (gguf.findTensor(prefix + "attn_q.weight")) {
            if (!loadTensor(gguf, prefix + "attn_q.weight", q_size,  w.attn_q)) return false;
            if (!loadTensor(gguf, prefix + "attn_k.weight", kv_size, w.attn_k)) return false;
            if (!loadTensor(gguf, prefix + "attn_v.weight", kv_size, w.attn_v)) return false;
        } else if (gguf.findTensor(prefix + "attn_qkv.weight")) {
            std::vector<float> qkv_fused;
            uint64_t qkv_size = q_size + 2 * kv_size;
            if (!loadTensor(gguf, prefix + "attn_qkv.weight", qkv_size, qkv_fused)) {
                return false;
            }
            w.attn_q.assign(qkv_fused.begin(),
                            qkv_fused.begin() + static_cast<std::ptrdiff_t>(q_size));
            w.attn_k.assign(qkv_fused.begin() + static_cast<std::ptrdiff_t>(q_size),
                            qkv_fused.begin() + static_cast<std::ptrdiff_t>(q_size + kv_size));
            w.attn_v.assign(qkv_fused.begin() + static_cast<std::ptrdiff_t>(q_size + kv_size),
                            qkv_fused.end());
        } else {
            last_error_ = "Missing QKV weight tensors at layer " + std::to_string(l);
            return false;
        }

        // Attention output projection
        uint64_t o_size = static_cast<uint64_t>(hp.n_embd) * hp.n_heads * head_dim;
        if (!loadTensor(gguf, prefix + "attn_output.weight", o_size, w.attn_out)) {
            if (last_error_.empty()) last_error_ = "Missing " + prefix + "attn_output.weight";
            return false;
        }

        // Optional QKV biases (separate or fused)
        if (gguf.findTensor(prefix + "attn_q.bias")) {
            loadTensor(gguf, prefix + "attn_q.bias", q_rows,  w.attn_q_bias);
            loadTensor(gguf, prefix + "attn_k.bias", kv_rows, w.attn_k_bias);
            loadTensor(gguf, prefix + "attn_v.bias", kv_rows, w.attn_v_bias);
        } else if (gguf.findTensor(prefix + "attn_qkv.bias")) {
            std::vector<float> qkv_bias;
            if (loadTensor(gguf, prefix + "attn_qkv.bias", q_rows + 2 * kv_rows, qkv_bias)) {
                w.attn_q_bias.assign(qkv_bias.begin(),
                                     qkv_bias.begin() + static_cast<std::ptrdiff_t>(q_rows));
                w.attn_k_bias.assign(qkv_bias.begin() + static_cast<std::ptrdiff_t>(q_rows),
                                     qkv_bias.begin() + static_cast<std::ptrdiff_t>(q_rows + kv_rows));
                w.attn_v_bias.assign(qkv_bias.begin() + static_cast<std::ptrdiff_t>(q_rows + kv_rows),
                                     qkv_bias.end());
            }
        }

        // FFN projections (separate ffn_gate + ffn_up OR fused ffn_up as in Phi-3)
        uint64_t ff_in  = static_cast<uint64_t>(hp.n_ff) * hp.n_embd;
        uint64_t ff_out = static_cast<uint64_t>(hp.n_embd) * hp.n_ff;

        if (gguf.findTensor(prefix + "ffn_gate.weight")) {
            if (!loadTensor(gguf, prefix + "ffn_gate.weight", ff_in, w.ffn_gate)) return false;
            if (!loadTensor(gguf, prefix + "ffn_up.weight",   ff_in, w.ffn_up))   return false;
        } else if (gguf.findTensor(prefix + "ffn_up.weight")) {
            std::vector<float> gate_up_fused;
            if (!loadTensor(gguf, prefix + "ffn_up.weight", 2 * ff_in, gate_up_fused)) {
                return false;
            }
            w.ffn_gate.assign(gate_up_fused.begin(),
                              gate_up_fused.begin() + static_cast<std::ptrdiff_t>(ff_in));
            w.ffn_up.assign(gate_up_fused.begin() + static_cast<std::ptrdiff_t>(ff_in),
                            gate_up_fused.end());
        } else {
            last_error_ = "Missing FFN gate/up weight tensors at layer " + std::to_string(l);
            return false;
        }

        if (!loadTensor(gguf, prefix + "ffn_down.weight", ff_out, w.ffn_down)) {
            if (last_error_.empty()) last_error_ = "Missing " + prefix + "ffn_down.weight";
            return false;
        }

        if (!w.isComplete()) {
            last_error_ = "Layer " + std::to_string(l) + " is incomplete after loading";
            return false;
        }
    }

    raw_buf_.clear();
    raw_buf_.shrink_to_fit();

    loaded_ = true;
    return true;
}

} // namespace hydroxl
} // namespace agr
