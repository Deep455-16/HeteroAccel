// src/hydroxl/HydroXLModel.h
// Phase 17 / Phase 27: Loaded model weights for HydroXL.
//
// HydroXLModel holds the dequantized FLOAT32 weight tensors for a transformer
// model loaded from a GGUF file (Llama/Llama 3, Mistral, Qwen2/Qwen2.5, Phi-3,
// Gemma, Gemma 2). It owns the weight buffers.
//
// All tensors are stored as flat std::vector<float> after dequantization.
// Standard GGUF tensor names:
//   token_embd.weight                 : [vocab_size, n_embd]
//   output_norm.weight                : [n_embd]
//   output.weight                     : [vocab_size, n_embd]  (LM head, or tied to token_embd)
//   blk.{l}.attn_norm.weight          : [n_embd]
//   blk.{l}.ffn_norm.weight           : [n_embd]
//   blk.{l}.post_attention_norm.weight: [n_embd]              (Gemma 2)
//   blk.{l}.post_ffw_norm.weight      : [n_embd]              (Gemma 2)
//   blk.{l}.attn_q.weight             : [n_heads*head_dim, n_embd]
//   blk.{l}.attn_k.weight             : [n_kv_heads*head_dim, n_embd]
//   blk.{l}.attn_v.weight             : [n_kv_heads*head_dim, n_embd]
//   blk.{l}.attn_qkv.weight           : [(n_heads + 2*n_kv_heads)*head_dim, n_embd] (Phi-3 fused, split on load)
//   blk.{l}.attn_output.weight        : [n_embd, n_heads*head_dim]
//   blk.{l}.ffn_gate.weight           : [n_ff, n_embd]
//   blk.{l}.ffn_up.weight             : [n_ff, n_embd] (or [2*n_ff, n_embd] fused in Phi-3, split on load)
//   blk.{l}.ffn_down.weight           : [n_embd, n_ff]
//   blk.{l}.attn_q/k/v.bias           : optional QKV biases (Qwen2, Phi-3)

#pragma once

#include "hydroxl/HydroXLTypes.h"
#include "hydroxl/GGUFLoader.h"

#include <cstdint>
#include <string>
#include <vector>

namespace agr {
namespace hydroxl {

struct Qwen2LayerWeights {
    std::vector<float> attn_norm;       ///< [n_embd]
    std::vector<float> ffn_norm;        ///< [n_embd]
    std::vector<float> post_attn_norm;  ///< [n_embd] (Gemma 2 post-attention RMSNorm, optional)
    std::vector<float> post_ffn_norm;   ///< [n_embd] (Gemma 2 post-FFN RMSNorm, optional)

    std::vector<float> attn_q;          ///< [n_heads*head_dim, n_embd]  row-major
    std::vector<float> attn_k;          ///< [n_kv_heads*head_dim, n_embd]
    std::vector<float> attn_v;          ///< [n_kv_heads*head_dim, n_embd]
    std::vector<float> attn_out;        ///< [n_embd, n_heads*head_dim]

    std::vector<float> attn_q_bias;     ///< [n_heads*head_dim]   (may be empty)
    std::vector<float> attn_k_bias;     ///< [n_kv_heads*head_dim]
    std::vector<float> attn_v_bias;     ///< [n_kv_heads*head_dim]

    std::vector<float> ffn_gate;        ///< [n_ff, n_embd]
    std::vector<float> ffn_up;          ///< [n_ff, n_embd]
    std::vector<float> ffn_down;        ///< [n_embd, n_ff]

    bool isComplete() const {
        return !attn_norm.empty() && !ffn_norm.empty()
            && !attn_q.empty() && !attn_k.empty() && !attn_v.empty()
            && !attn_out.empty()
            && !ffn_gate.empty() && !ffn_up.empty() && !ffn_down.empty();
    }
};

using TransformerLayerWeights = Qwen2LayerWeights;

class HydroXLModel {
public:
    HydroXLModel() = default;
    ~HydroXLModel() = default;

    // Non-copyable (large weight tensors), movable
    HydroXLModel(const HydroXLModel&) = delete;
    HydroXLModel& operator=(const HydroXLModel&) = delete;
    HydroXLModel(HydroXLModel&&) = default;
    HydroXLModel& operator=(HydroXLModel&&) = default;

    /// Load and dequantize all weights from a GGUF file.
    /// Returns false on any error. lastError() contains the reason.
    bool load(const std::string& path);

    bool isLoaded() const { return loaded_; }

    const Qwen2HParams&          hparams() const { return hparams_; }
    const std::vector<float>&    tokenEmbedding() const { return tok_embd_; }
    const std::vector<float>&    outputNorm()    const { return output_norm_; }
    const std::vector<float>&    lmHead()        const { return lm_head_tied_ ? tok_embd_ : lm_head_; }
    bool                         isLmHeadTied()  const { return lm_head_tied_; }
    const Qwen2LayerWeights&     layer(uint32_t l) const { return layers_.at(l); }

    const std::string& path()      const { return path_; }
    std::string        lastError() const { return last_error_; }

    // Memory footprint of all dequantized weights (bytes)
    size_t memoryBytes() const { return memory_bytes_; }

private:
    bool loadHParams(GGUFLoader& gguf);
    bool loadTensor(GGUFLoader& gguf, const std::string& name,
                    uint64_t expected_elems, std::vector<float>& out);

    std::string path_;
    bool loaded_ = false;
    bool lm_head_tied_ = false;
    std::vector<uint8_t> raw_buf_;

    Qwen2HParams hparams_;
    std::vector<float> tok_embd_;     ///< [vocab_size * n_embd]
    std::vector<float> output_norm_;  ///< [n_embd]
    std::vector<float> lm_head_;      ///< [vocab_size * n_embd]

    std::vector<Qwen2LayerWeights> layers_;

    size_t memory_bytes_ = 0;
    mutable std::string last_error_;
};

} // namespace hydroxl
} // namespace agr
