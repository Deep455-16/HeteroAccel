// src/hydroxl/HydroXLTypes.h
// Phase 19 / Phase 27: HydroXL Native Inference Engine — shared types.
//
// Supported architectures:
//   - Llama / Llama 3   ("llama")
//   - Mistral           ("mistral", "mistral3")
//   - Qwen2 / Qwen2.5   ("qwen2", "qwen2_vl")
//   - Phi-3             ("phi3", "phi2")
//   - Gemma / Gemma 2   ("gemma", "gemma2")
//
// Supported quantization (13 GGML tensor types):
//   F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
//   Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K
//
// Supported modality: TEXT_GENERATION

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace agr {
namespace hydroxl {

// ---------------------------------------------------------------------------
// GGUF scalar value types (from the GGUF spec)
// ---------------------------------------------------------------------------
enum class GGUFValueType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

// ---------------------------------------------------------------------------
// GGUF tensor data types — values MUST match the real GGML enum in ggml.h
//
//   F32=0  F16=1  Q4_0=2  Q4_1=3
//   (4,5 removed from ggml)
//   Q5_0=6  Q5_1=7  Q8_0=8  Q8_1=9
//   Q2_K=10  Q3_K=11  Q4_K=12  Q5_K=13  Q6_K=14  Q8_K=15
// ---------------------------------------------------------------------------
enum class GGMLType : uint32_t {
    F32   = 0,
    F16   = 1,
    Q4_0  = 2,
    Q4_1  = 3,
    // 4, 5 — removed from GGML
    Q5_0  = 6,
    Q5_1  = 7,
    Q8_0  = 8,
    Q8_1  = 9,
    Q2_K  = 10,
    Q3_K  = 11,   ///< Q3_K_{S,M,L} — same 110-byte block, type id 11
    Q4_K  = 12,   ///< Q4_K_{S,M}   — same 144-byte block, type id 12
    Q5_K  = 13,
    Q6_K  = 14,
    Q8_K  = 15,
    UNKNOWN = 0xFFFF,
};

inline const char* toString(GGMLType t) {
    switch (t) {
        case GGMLType::F32:   return "F32";
        case GGMLType::F16:   return "F16";
        case GGMLType::Q4_0:  return "Q4_0";
        case GGMLType::Q4_1:  return "Q4_1";
        case GGMLType::Q5_0:  return "Q5_0";
        case GGMLType::Q5_1:  return "Q5_1";
        case GGMLType::Q8_0:  return "Q8_0";
        case GGMLType::Q8_1:  return "Q8_1";
        case GGMLType::Q2_K:  return "Q2_K";
        case GGMLType::Q3_K:  return "Q3_K";
        case GGMLType::Q4_K:  return "Q4_K";
        case GGMLType::Q5_K:  return "Q5_K";
        case GGMLType::Q6_K:  return "Q6_K";
        case GGMLType::Q8_K:  return "Q8_K";
        default:              return "Other";
    }
}

// ---------------------------------------------------------------------------
// Quantization block format constants
// ---------------------------------------------------------------------------
static constexpr int Q4_0_BLOCK_SIZE  = 32;
static constexpr int Q4_0_BLOCK_BYTES = 18;

static constexpr int Q4_1_BLOCK_SIZE  = 32;
static constexpr int Q4_1_BLOCK_BYTES = 20;

static constexpr int Q5_0_BLOCK_SIZE  = 32;
static constexpr int Q5_0_BLOCK_BYTES = 22;

static constexpr int Q5_1_BLOCK_SIZE  = 32;
static constexpr int Q5_1_BLOCK_BYTES = 24;

static constexpr int Q8_0_BLOCK_SIZE  = 32;
static constexpr int Q8_0_BLOCK_BYTES = 34;

static constexpr int Q2K_BLOCK_SIZE      = 256;
static constexpr int Q2K_RAW_BLOCK_BYTES = 84;

static constexpr int Q3K_BLOCK_SIZE      = 256;
static constexpr int Q3K_RAW_BLOCK_BYTES = 110;

static constexpr int Q4K_BLOCK_SIZE      = 256;
static constexpr int Q4K_N_SCALE_GROUPS  = 8;
static constexpr int Q4K_RAW_BLOCK_BYTES = 144;

static constexpr int Q5K_BLOCK_SIZE      = 256;
static constexpr int Q5K_RAW_BLOCK_BYTES = 176;

static constexpr int Q6K_BLOCK_SIZE      = 256;
static constexpr int Q6K_RAW_BLOCK_BYTES = 210;

static constexpr int Q8K_BLOCK_SIZE      = 256;
static constexpr int Q8K_RAW_BLOCK_BYTES = 292;

/// Elements per block for a GGML tensor type (1 for scalar F32/F16, 0 if unsupported).
inline uint32_t ggml_type_block_size(GGMLType t) noexcept {
    switch (t) {
        case GGMLType::F32:  return 1;
        case GGMLType::F16:  return 1;
        case GGMLType::Q4_0: return Q4_0_BLOCK_SIZE;
        case GGMLType::Q4_1: return Q4_1_BLOCK_SIZE;
        case GGMLType::Q5_0: return Q5_0_BLOCK_SIZE;
        case GGMLType::Q5_1: return Q5_1_BLOCK_SIZE;
        case GGMLType::Q8_0: return Q8_0_BLOCK_SIZE;
        case GGMLType::Q2_K: return Q2K_BLOCK_SIZE;
        case GGMLType::Q3_K: return Q3K_BLOCK_SIZE;
        case GGMLType::Q4_K: return Q4K_BLOCK_SIZE;
        case GGMLType::Q5_K: return Q5K_BLOCK_SIZE;
        case GGMLType::Q6_K: return Q6K_BLOCK_SIZE;
        case GGMLType::Q8_K: return Q8K_BLOCK_SIZE;
        default:             return 0;
    }
}

/// Raw bytes per block for a GGML tensor type (0 if unsupported).
inline uint32_t ggml_type_block_bytes(GGMLType t) noexcept {
    switch (t) {
        case GGMLType::F32:  return 4;
        case GGMLType::F16:  return 2;
        case GGMLType::Q4_0: return Q4_0_BLOCK_BYTES;
        case GGMLType::Q4_1: return Q4_1_BLOCK_BYTES;
        case GGMLType::Q5_0: return Q5_0_BLOCK_BYTES;
        case GGMLType::Q5_1: return Q5_1_BLOCK_BYTES;
        case GGMLType::Q8_0: return Q8_0_BLOCK_BYTES;
        case GGMLType::Q2_K: return Q2K_RAW_BLOCK_BYTES;
        case GGMLType::Q3_K: return Q3K_RAW_BLOCK_BYTES;
        case GGMLType::Q4_K: return Q4K_RAW_BLOCK_BYTES;
        case GGMLType::Q5_K: return Q5K_RAW_BLOCK_BYTES;
        case GGMLType::Q6_K: return Q6K_RAW_BLOCK_BYTES;
        case GGMLType::Q8_K: return Q8K_RAW_BLOCK_BYTES;
        default:             return 0;
    }
}

/// Whether HydroXL implements a verified reference decoder for this GGML type.
inline bool is_quant_type_supported(GGMLType t) noexcept {
    return ggml_type_block_size(t) > 0 && ggml_type_block_bytes(t) > 0;
}

// ---------------------------------------------------------------------------
// Architecture enums and traits (Phase 27)
// ---------------------------------------------------------------------------
enum class ModelArch : uint32_t {
    UNKNOWN = 0,
    LLAMA   = 1,
    MISTRAL = 2,
    QWEN2   = 3,
    PHI3    = 4,
    GEMMA   = 5,
    GEMMA2  = 6,
};

enum class RoPEType : uint32_t {
    NORM = 0, ///< Interleaved consecutive pairs (2*i, 2*i+1) — Llama, Mistral
    NEOX = 1, ///< Split-half pairs (i, i + rope_dim/2) — Qwen2, Phi-3, Gemma, Gemma 2
};

enum class FFNActivation : uint32_t {
    SWIGLU = 0, ///< silu(gate) * up — Llama, Mistral, Qwen2, Phi-3
    GEGLU  = 1, ///< gelu(gate) * up — Gemma, Gemma 2
};

inline ModelArch parse_model_arch(const std::string& arch) noexcept {
    if (arch == "llama")                           return ModelArch::LLAMA;
    if (arch == "mistral" || arch == "mistral3")   return ModelArch::MISTRAL;
    if (arch == "qwen2"   || arch == "qwen2_vl")   return ModelArch::QWEN2;
    if (arch == "phi3"    || arch == "phi2")       return ModelArch::PHI3;
    if (arch == "gemma")                           return ModelArch::GEMMA;
    if (arch == "gemma2")                          return ModelArch::GEMMA2;
    return ModelArch::UNKNOWN;
}

inline bool is_arch_supported(const std::string& arch) noexcept {
    return parse_model_arch(arch) != ModelArch::UNKNOWN;
}

// ---------------------------------------------------------------------------
// GGUF tensor descriptor (header metadata only, no data pointer)
// ---------------------------------------------------------------------------
struct GGUFTensorDesc {
    std::string  name;
    GGMLType     type    = GGMLType::UNKNOWN;
    std::vector<uint64_t> shape;   ///< dim[0..n_dims-1], innermost first
    uint64_t     offset  = 0;      ///< Byte offset from start of data section
    uint64_t     n_bytes = 0;      ///< Raw byte size on disk
};

// ---------------------------------------------------------------------------
// Universal model hyperparameters parsed from GGUF metadata
// (Retains struct name Qwen2HParams for SDK/API backward compatibility,
//  with ModelHParams alias)
// ---------------------------------------------------------------------------
struct Qwen2HParams {
    ModelArch     arch              = ModelArch::QWEN2;
    std::string   arch_name         = "qwen2";
    RoPEType      rope_type         = RoPEType::NEOX;
    FFNActivation ffn_activation    = FFNActivation::SWIGLU;

    uint32_t vocab_size             = 0;
    uint32_t n_embd                 = 0;   ///< Embedding / hidden size
    uint32_t n_heads                = 0;   ///< Number of attention heads
    uint32_t n_kv_heads             = 0;   ///< Number of key/value heads (GQA)
    uint32_t n_layers               = 0;   ///< Number of transformer layers
    uint32_t n_ff                   = 0;   ///< FFN intermediate size
    uint32_t n_ctx_max              = 0;   ///< Max supported context length
    float    rms_norm_eps           = 1e-6f;
    float    rope_freq_base         = 1000000.0f;
    float    rope_freq_scale        = 1.0f;
    uint32_t n_head_dim             = 0;   ///< key_length or n_embd / n_heads
    uint32_t n_rope_dim             = 0;   ///< RoPE rotary dimensions (<= n_head_dim)
    float    embedding_scale        = 1.0f;///< sqrt(n_embd) for Gemma/Gemma2, 1.0f otherwise
    float    attn_logit_softcap     = 0.0f;///< Gemma 2 attention logit softcapping (0 = disabled)
    float    final_logit_softcap    = 0.0f;///< Gemma 2 final logit softcapping (0 = disabled)
    uint32_t sliding_window         = 0;   ///< Sliding window attention size (0 = full causal)

    bool isValid() const {
        return vocab_size > 0 && n_embd > 0 && n_heads > 0 &&
               n_kv_heads > 0 && n_layers > 0 && n_ff > 0 &&
               n_heads % n_kv_heads == 0 &&
               (n_head_dim == 0 || n_head_dim % 2 == 0);
    }
};

using ModelHParams = Qwen2HParams;

// ---------------------------------------------------------------------------
// BPE token ID type and special tokens
// ---------------------------------------------------------------------------
using TokenID = int32_t;
static constexpr TokenID TOKEN_INVALID = -1;

} // namespace hydroxl
} // namespace agr
