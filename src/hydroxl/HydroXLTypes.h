// src/hydroxl/HydroXLTypes.h
// Phase 17: HydroXL Native Inference Engine — shared types.
//
// This header defines the core data types, metadata structures, and
// constants used across all HydroXL subsystems (loader, tokenizer,
// operators, KV-cache, sampler, and engine).
//
// HydroXL is an INDEPENDENT native inference engine. It does NOT
// delegate to llama.cpp for model execution, tokenization, or sampling.
//
// Supported architecture (Phase 17): Qwen2 (qwen2)
// Supported quantization: F32 (FLOAT32 dequantized from Q4_K_M)
// Supported modality: TEXT_GENERATION
//
// Real-model test requires:
//   C:\HeteroAccel\Models\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
//   (absent on build system; real-model test exits 77 = SKIP)

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
// GGUF tensor data types (ggml_type equivalent — only what HydroXL needs)
// ---------------------------------------------------------------------------
enum class GGMLType : uint32_t {
    F32      = 0,
    F16      = 1,
    Q4_0     = 2,
    Q4_1     = 3,
    Q5_0     = 6,
    Q5_1     = 7,
    Q8_0     = 8,
    Q8_1     = 9,
    Q2_K     = 10,
    Q3_K_S   = 11,
    Q3_K_M   = 12,
    Q3_K_L   = 13,
    Q4_K_S   = 14,
    Q4_K_M   = 15,   ///< Block-quantized 4-bit, medium mode (Phase 17 target)
    Q5_K_S   = 16,
    Q5_K_M   = 17,
    Q6_K     = 18,
    Q8_K     = 19,
    UNKNOWN  = 0xFFFF,
};

inline const char* toString(GGMLType t) {
    switch (t) {
        case GGMLType::F32:    return "F32";
        case GGMLType::F16:    return "F16";
        case GGMLType::Q4_0:   return "Q4_0";
        case GGMLType::Q4_K_S: return "Q4_K_S";
        case GGMLType::Q4_K_M: return "Q4_K_M";
        case GGMLType::Q8_0:   return "Q8_0";
        case GGMLType::Q6_K:   return "Q6_K";
        default:               return "Other";
    }
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
// Qwen2 model hyperparameters parsed from GGUF metadata
// ---------------------------------------------------------------------------
struct Qwen2HParams {
    uint32_t vocab_size      = 0;
    uint32_t n_embd          = 0;   ///< Embedding / hidden size
    uint32_t n_heads         = 0;   ///< Number of attention heads
    uint32_t n_kv_heads      = 0;   ///< Number of key/value heads (GQA)
    uint32_t n_layers        = 0;   ///< Number of transformer layers
    uint32_t n_ff            = 0;   ///< FFN intermediate size
    uint32_t n_ctx_max       = 0;   ///< Max supported context length
    float    rms_norm_eps    = 1e-6f;
    float    rope_freq_base  = 1000000.0f;  ///< RoPE frequency base (Qwen2 uses 1M)
    float    rope_freq_scale = 1.0f;
    uint32_t n_head_dim      = 0;   ///< n_embd / n_heads

    bool isValid() const {
        return vocab_size > 0 && n_embd > 0 && n_heads > 0 &&
               n_kv_heads > 0 && n_layers > 0 && n_ff > 0;
    }
};

// ---------------------------------------------------------------------------
// Q4_K block format constants (GGML_TYPE_Q4_K)
//
// Q4_K_M packs 256 float32 values into one block:
//   - 2 FP16 super-scale and super-min  (d, dmin)
//   - 12 bytes: 8 x 6-bit scale/min pairs encoded in 96 bits
//   - 128 bytes: 256 x 4-bit values (2 per byte)
// Total block size: 144 bytes for 256 elements
// ---------------------------------------------------------------------------
static constexpr int Q4K_BLOCK_SIZE       = 256;   ///< Elements per block
static constexpr int Q4K_N_SCALE_GROUPS   = 8;     ///< Sub-groups per block
static constexpr int Q4K_RAW_BLOCK_BYTES  = 144;   ///< 4 + 12 + 128 bytes

// ---------------------------------------------------------------------------
// Q8_0 block format constants
// 32 elements per block: 2-byte scale (FP16) + 32 x INT8
// Block = 34 bytes
// ---------------------------------------------------------------------------
static constexpr int Q8_0_BLOCK_SIZE     = 32;
static constexpr int Q8_0_BLOCK_BYTES    = 34;

// ---------------------------------------------------------------------------
// BPE token ID type and special tokens
// ---------------------------------------------------------------------------
using TokenID = int32_t;
static constexpr TokenID TOKEN_INVALID = -1;

} // namespace hydroxl
} // namespace agr
