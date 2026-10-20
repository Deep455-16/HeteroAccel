// src/hydroxl/HydroXLTypes.h
// Phase 19: HydroXL Native Inference Engine — shared types.
//
// Supported architectures: Qwen2 (qwen2)
// Supported quantization: F32, F16, Q8_0, Q4_K, Q3_K, Q5_0, Q5_1, Q6_K
//   (Q4_K_M / Q3_K_M / etc. are per-tensor mixes of these block types;
//    every tensor is decoded according to its own GGML type id)
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
//
// The _S / _M / _L suffixes (Q3_K_M, Q4_K_M, …) are quantization preset
// names used in model filenames. On disk all variants of a K-quant share
// the same block layout and the same GGML type integer.
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
    float    rope_freq_base  = 1000000.0f;
    float    rope_freq_scale = 1.0f;
    uint32_t n_head_dim      = 0;   ///< n_embd / n_heads

    bool isValid() const {
        return vocab_size > 0 && n_embd > 0 && n_heads > 0 &&
               n_kv_heads > 0 && n_layers > 0 && n_ff > 0;
    }
};

// ---------------------------------------------------------------------------
// Q4_0 block format constants  (GGML_TYPE_Q4_0 = 2)
// 32 elements / 18 bytes per block
//   Bytes  0..1  : d   (FP16) scale
//   Bytes  2..17 : 16 bytes — 32 x 4-bit values (unsigned, offset by 8)
// ---------------------------------------------------------------------------
static constexpr int Q4_0_BLOCK_SIZE  = 32;
static constexpr int Q4_0_BLOCK_BYTES = 18;

// ---------------------------------------------------------------------------
// Q4_1 block format constants  (GGML_TYPE_Q4_1 = 3)
// 32 elements / 20 bytes per block
//   Bytes  0..1  : d   (FP16) scale
//   Bytes  2..3  : m   (FP16) additive minimum
//   Bytes  4..19 : 16 bytes — 32 x 4-bit values
// ---------------------------------------------------------------------------
static constexpr int Q4_1_BLOCK_SIZE  = 32;
static constexpr int Q4_1_BLOCK_BYTES = 20;

// ---------------------------------------------------------------------------
// Q4_K block format constants  (GGML_TYPE_Q4_K = 12)
// 256 elements / 144 bytes per block
//   Bytes  0..1  : d    (FP16) super-scale
//   Bytes  2..3  : dmin (FP16) super-min
//   Bytes  4..15 : 12 bytes — 8 x (6-bit scale, 6-bit min)
//   Bytes 16..143: 128 bytes — 256 x 4-bit values
// ---------------------------------------------------------------------------
static constexpr int Q4K_BLOCK_SIZE      = 256;
static constexpr int Q4K_N_SCALE_GROUPS  = 8;
static constexpr int Q4K_RAW_BLOCK_BYTES = 144;

// ---------------------------------------------------------------------------
// Q3_K block format constants  (GGML_TYPE_Q3_K = 11)
// 256 elements / 110 bytes per block
//   Bytes   0..31 : hmask  (32 bytes) — bit 2 (high bit) per element
//   Bytes  32..95 : qs     (64 bytes) — bits 0-1 per element (2 per byte)
//   Bytes  96..107: scales (12 bytes) — 16 x 6-bit scales
//   Bytes 108..109: d      (FP16)  super-scale
// ---------------------------------------------------------------------------
static constexpr int Q3K_BLOCK_SIZE      = 256;
static constexpr int Q3K_RAW_BLOCK_BYTES = 110;

// ---------------------------------------------------------------------------
// Q8_0 block format constants  (GGML_TYPE_Q8_0 = 8)
// 32 elements / 34 bytes per block
//   Bytes  0..1  : d   (FP16) block scale
//   Bytes  2..33 : 32 x INT8 quantized values
// ---------------------------------------------------------------------------
static constexpr int Q8_0_BLOCK_SIZE  = 32;
static constexpr int Q8_0_BLOCK_BYTES = 34;

// ---------------------------------------------------------------------------
// BPE token ID type and special tokens
// ---------------------------------------------------------------------------
using TokenID = int32_t;
static constexpr TokenID TOKEN_INVALID = -1;

} // namespace hydroxl
} // namespace agr
