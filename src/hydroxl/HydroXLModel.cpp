// src/hydroxl/HydroXLModel.cpp
// Phase 17: Model weight loader and dequantizer for HydroXL.

#include "hydroxl/HydroXLModel.h"
#include "hydroxl/HydroXLDequant.h"

#include <cstring>
#include <filesystem>
#include <iostream>

namespace agr {
namespace hydroxl {

// ---------------------------------------------------------------------------
// Load hyperparameters from GGUF metadata
// ---------------------------------------------------------------------------
bool HydroXLModel::loadHParams(GGUFLoader& gguf) {
    // Architecture check
    std::string arch = gguf.kvString("general.architecture", "");
    if (arch != "qwen2" && arch != "qwen2_vl") {
        // Still attempt to load; some GGUF files omit the architecture field
        // or use slightly different names.
        if (!arch.empty() && arch.find("qwen") == std::string::npos) {
            last_error_ = "Unsupported architecture '" + arch + "' (expected qwen2)";
            return false;
        }
    }

    // Qwen2 GGUF uses prefix "qwen2." for hyperparams
    hparams_.vocab_size   = gguf.kvUInt32("llm.vocab_size", 0);
    if (hparams_.vocab_size == 0) {
        hparams_.vocab_size = gguf.kvUInt32("qwen2.vocab_size", 151936);
    }

    hparams_.n_embd     = gguf.kvUInt32("qwen2.embedding_length",   896);
    hparams_.n_heads    = gguf.kvUInt32("qwen2.attention.head_count", 14);
    hparams_.n_kv_heads = gguf.kvUInt32("qwen2.attention.head_count_kv", 2);
    hparams_.n_layers   = gguf.kvUInt32("qwen2.block_count", 24);
    hparams_.n_ff       = gguf.kvUInt32("qwen2.feed_forward_length", 4864);
    hparams_.n_ctx_max  = gguf.kvUInt32("qwen2.context_length", 32768);
    hparams_.rms_norm_eps   = gguf.kvFloat("qwen2.attention.layer_norm_rms_epsilon", 1e-6f);
    hparams_.rope_freq_base = gguf.kvFloat("qwen2.rope.freq_base", 1000000.0f);

    if (!hparams_.isValid()) {
        last_error_ = "Invalid hyperparameters (zero embedding/heads/layers)";
        return false;
    }

    hparams_.n_head_dim = hparams_.n_embd / hparams_.n_heads;
    return true;
}

// ---------------------------------------------------------------------------
// Generic tensor load + dequantize helper
// ---------------------------------------------------------------------------
bool HydroXLModel::loadTensor(GGUFLoader& gguf, const std::string& name,
                               uint64_t expected_elems, std::vector<float>& out) {
    const GGUFTensorDesc* desc = gguf.findTensor(name);
    if (!desc) {
        // Not all tensors are mandatory; caller decides whether to fail
        return false;
    }

    // Compute actual element count
    uint64_t n_elems = 1;
    for (auto d : desc->shape) n_elems *= d;

    if (expected_elems > 0 && n_elems != expected_elems) {
        last_error_ = "Tensor '" + name + "' has " + std::to_string(n_elems) +
                      " elements, expected " + std::to_string(expected_elems);
        return false;
    }

    // Allocate output
    out.resize(static_cast<size_t>(n_elems));

    switch (desc->type) {
    case GGMLType::F32: {
        std::vector<uint8_t> raw(desc->n_bytes);
        if (!gguf.readTensorData(*desc, raw.data(), desc->n_bytes)) {
            last_error_ = "IO error reading tensor '" + name + "'";
            return false;
        }
        std::memcpy(out.data(), raw.data(), n_elems * 4);
        break;
    }
    case GGMLType::F16: {
        std::vector<uint8_t> raw(desc->n_bytes);
        if (!gguf.readTensorData(*desc, raw.data(), desc->n_bytes)) {
            last_error_ = "IO error reading tensor '" + name + "' (F16)";
            return false;
        }
        dequant_f16(raw.data(), out.data(), n_elems);
        break;
    }
    case GGMLType::Q4_K:
    {
        if (n_elems % Q4K_BLOCK_SIZE != 0) {
            last_error_ = "Tensor '" + name + "' n_elems=" + std::to_string(n_elems)
                        + " not divisible by " + std::to_string(Q4K_BLOCK_SIZE);
            return false;
        }
        std::vector<uint8_t> raw(desc->n_bytes);
        if (!gguf.readTensorData(*desc, raw.data(), desc->n_bytes)) {
            last_error_ = "IO error reading tensor '" + name + "' (Q4_K)";
            return false;
        }
        if (!dequant_q4k(raw.data(), out.data(), n_elems)) {
            last_error_ = "Dequant failed for tensor '" + name + "'";
            return false;
        }
        break;
    }
    case GGMLType::Q8_0: {
        if (n_elems % Q8_0_BLOCK_SIZE != 0) {
            last_error_ = "Tensor '" + name + "' n_elems not divisible by " +
                          std::to_string(Q8_0_BLOCK_SIZE);
            return false;
        }
        std::vector<uint8_t> raw(desc->n_bytes);
        if (!gguf.readTensorData(*desc, raw.data(), desc->n_bytes)) {
            last_error_ = "IO error reading tensor '" + name + "' (Q8_0)";
            return false;
        }
        if (!dequant_q8_0(raw.data(), out.data(), n_elems)) {
            last_error_ = "Q8_0 dequant failed for tensor '" + name + "'";
            return false;
        }
        break;
    }
    case GGMLType::Q6_K: {
        if (n_elems % 256 != 0) {
            last_error_ = "Tensor '" + name + "' n_elems not divisible by 256";
            return false;
        }
        std::vector<uint8_t> raw(desc->n_bytes);
        if (!gguf.readTensorData(*desc, raw.data(), desc->n_bytes)) {
            last_error_ = "IO error reading tensor '" + name + "' (Q6_K)";
            return false;
        }
        if (!dequant_q6_k(raw.data(), out.data(), n_elems)) {
            last_error_ = "Q6_K dequant failed for tensor '" + name + "'";
            return false;
        }
        break;
    }
    case GGMLType::Q5_0: {
        if (n_elems % 32 != 0) {
            last_error_ = "Tensor '" + name + "' n_elems not divisible by 32";
            return false;
        }
        std::vector<uint8_t> raw(desc->n_bytes);
        if (!gguf.readTensorData(*desc, raw.data(), desc->n_bytes)) {
            last_error_ = "IO error reading tensor '" + name + "' (Q5_0)";
            return false;
        }
        if (!dequant_q5_0(raw.data(), out.data(), n_elems)) {
            last_error_ = "Q5_0 dequant failed for tensor '" + name + "'";
            return false;
        }
        break;
    }
    case GGMLType::Q5_1: {
        if (n_elems % 32 != 0) {
            last_error_ = "Tensor '" + name + "' n_elems not divisible by 32";
            return false;
        }
        std::vector<uint8_t> raw(desc->n_bytes);
        if (!gguf.readTensorData(*desc, raw.data(), desc->n_bytes)) {
            last_error_ = "IO error reading tensor '" + name + "' (Q5_1)";
            return false;
        }
        if (!dequant_q5_1(raw.data(), out.data(), n_elems)) {
            last_error_ = "Q5_1 dequant failed for tensor '" + name + "'";
            return false;
        }
        break;
    }
    case GGMLType::Q3_K: {
        if (n_elems % Q3K_BLOCK_SIZE != 0) {
            last_error_ = "Tensor '" + name + "' n_elems not divisible by " +
                          std::to_string(Q3K_BLOCK_SIZE);
            return false;
        }
        std::vector<uint8_t> raw(desc->n_bytes);
        if (!gguf.readTensorData(*desc, raw.data(), desc->n_bytes)) {
            last_error_ = "IO error reading tensor '" + name + "' (Q3_K)";
            return false;
        }
        if (!dequant_q3k(raw.data(), out.data(), n_elems)) {
            last_error_ = "Q3_K dequant failed for tensor '" + name + "'";
            return false;
        }
        break;
    }
    default:
        last_error_ = "Tensor '" + name + "' has unsupported type " +
                      toString(desc->type);
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

    // --- Check file exists ---
    if (!std::filesystem::exists(path)) {
        last_error_ = "Model file not found: " + path;
        return false;
    }

    // --- Parse GGUF header ---
    GGUFLoader gguf;
    if (!gguf.open(path)) {
        last_error_ = "Failed to open GGUF: " + gguf.lastError();
        return false;
    }

    // --- Hyperparameters ---
    if (!loadHParams(gguf)) return false;

    const auto& hp = hparams_;
    uint32_t head_dim = hp.n_head_dim;

    // --- Embeddings ---
    if (!loadTensor(gguf, "token_embd.weight",
                    static_cast<uint64_t>(hp.vocab_size) * hp.n_embd,
                    tok_embd_)) {
        return false;
    }

    // --- Output norm ---
    if (!loadTensor(gguf, "output_norm.weight", hp.n_embd, output_norm_)) {
        return false;
    }

    // --- LM head ---
    // Some GGUF files tie the LM head to token_embd; if "output.weight" is absent, share it.
    if (!loadTensor(gguf, "output.weight",
                    static_cast<uint64_t>(hp.vocab_size) * hp.n_embd,
                    lm_head_)) {
        // Tied embedding
        lm_head_ = tok_embd_;
    }

    // --- Transformer layers ---
    layers_.resize(hp.n_layers);
    for (uint32_t l = 0; l < hp.n_layers; ++l) {
        std::string prefix = "blk." + std::to_string(l) + ".";
        auto& w = layers_[l];

        // Norms
        if (!loadTensor(gguf, prefix + "attn_norm.weight", hp.n_embd, w.attn_norm)) return false;
        if (!loadTensor(gguf, prefix + "ffn_norm.weight",  hp.n_embd, w.ffn_norm))  return false;

        // QKV
        uint64_t q_size  = static_cast<uint64_t>(hp.n_heads)    * head_dim * hp.n_embd;
        uint64_t kv_size = static_cast<uint64_t>(hp.n_kv_heads) * head_dim * hp.n_embd;

        if (!loadTensor(gguf, prefix + "attn_q.weight", q_size,  w.attn_q)) return false;
        if (!loadTensor(gguf, prefix + "attn_k.weight", kv_size, w.attn_k)) return false;
        if (!loadTensor(gguf, prefix + "attn_v.weight", kv_size, w.attn_v)) return false;

        // Attention output projection
        uint64_t o_size = static_cast<uint64_t>(hp.n_embd) * hp.n_heads * head_dim;
        if (!loadTensor(gguf, prefix + "attn_output.weight", o_size, w.attn_out)) return false;

        // Biases (Qwen2 has QKV biases; they are optional)
        loadTensor(gguf, prefix + "attn_q.bias",
                   static_cast<uint64_t>(hp.n_heads)    * head_dim, w.attn_q_bias);
        loadTensor(gguf, prefix + "attn_k.bias",
                   static_cast<uint64_t>(hp.n_kv_heads) * head_dim, w.attn_k_bias);
        loadTensor(gguf, prefix + "attn_v.bias",
                   static_cast<uint64_t>(hp.n_kv_heads) * head_dim, w.attn_v_bias);

        // FFN
        uint64_t ff_in   = static_cast<uint64_t>(hp.n_ff) * hp.n_embd;
        uint64_t ff_out  = static_cast<uint64_t>(hp.n_embd) * hp.n_ff;

        if (!loadTensor(gguf, prefix + "ffn_gate.weight", ff_in,  w.ffn_gate)) return false;
        if (!loadTensor(gguf, prefix + "ffn_up.weight",   ff_in,  w.ffn_up))   return false;
        if (!loadTensor(gguf, prefix + "ffn_down.weight", ff_out, w.ffn_down))  return false;

        if (!w.isComplete()) {
            last_error_ = "Layer " + std::to_string(l) + " is incomplete after loading";
            return false;
        }
    }

    loaded_ = true;
    return true;
}

} // namespace hydroxl
} // namespace agr
