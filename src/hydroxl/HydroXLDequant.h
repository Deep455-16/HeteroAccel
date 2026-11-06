// src/hydroxl/HydroXLDequant.h
// Phase 19 / Phase 27: Quantized tensor dequantization for HydroXL.
//
// Implements reference-equivalent dequantization for all 13 prioritized GGML
// tensor formats:
//   - F32, F16
//   - Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
//   - Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K
//
// Reference: ggml/src/ggml-quants.h, ggml/src/ggml-quants.c (bundled).
// Numerical equivalence with that reference is enforced by
// tests/test_p19_dequant_ref.cpp and tests/test_p27_quant_decoders.cpp.

#pragma once

#include "hydroxl/HydroXLTypes.h"

#include <cstddef>
#include <cstdint>

namespace agr {
namespace hydroxl {

/// Convert an IEEE 754 FP16 (half-precision) value to float32.
float fp16_to_float(uint16_t h) noexcept;

/// Dequantize an F16 tensor (fp16->f32 conversion).
void dequant_f16(const void* raw_data, float* dst, uint64_t n_elements) noexcept;

// Q4_0 dequantization (GGML_TYPE_Q4_0 = 2)
// 32 elements per block, 18 bytes per block
// Block: [d_fp16 (2)][qs nibbles (16)]
// x[j] = d * ((qs[j] & 0xF) - 8), x[j+16] = d * ((qs[j] >> 4) - 8)
void dequant_q4_0_block(const uint8_t* raw, float* dst) noexcept;
bool dequant_q4_0(const void* raw_data, float* dst, uint64_t n_elements) noexcept;

// Q4_1 dequantization (GGML_TYPE_Q4_1 = 3)
// 32 elements per block, 20 bytes per block
// Block: [d_fp16 (2)][m_fp16 (2)][qs nibbles (16)]
// x[j] = d * (qs[j] & 0xF) + m, x[j+16] = d * (qs[j] >> 4) + m
void dequant_q4_1_block(const uint8_t* raw, float* dst) noexcept;
bool dequant_q4_1(const void* raw_data, float* dst, uint64_t n_elements) noexcept;

// Q5_0 dequantization (GGML_TYPE_Q5_0 = 6)
// 32 elements per block, 22 bytes per block
// Block: [d_fp16 (2)][qh uint32 (4) high bits][qs nibbles (16)]
// x[j] = d * (((qs[j] & 0xF) | highbit(j)) - 16), high bit plane in qh
void dequant_q5_0_block(const uint8_t* raw, float* dst) noexcept;
bool dequant_q5_0(const void* raw_data, float* dst, uint64_t n_elements) noexcept;

// Q5_1 dequantization (GGML_TYPE_Q5_1 = 7)
// 32 elements per block, 24 bytes per block
// Block: [d_fp16 (2)][m_fp16 (2)][qh uint32 (4)][qs nibbles (16)]
// x[j] = d * ((qs[j] & 0xF) | highbit(j)) + m  (additive FP16 minimum)
void dequant_q5_1_block(const uint8_t* raw, float* dst) noexcept;
bool dequant_q5_1(const void* raw_data, float* dst, uint64_t n_elements) noexcept;

// Q8_0 dequantization (GGML_TYPE_Q8_0 = 8)
// 32 elements per block, 34 bytes per block
void dequant_q8_0_block(const uint8_t* raw, float* dst) noexcept;
bool dequant_q8_0(const void* raw_data, float* dst, uint64_t n_elements) noexcept;

// Q2_K dequantization (GGML_TYPE_Q2_K = 10)
// 256 elements per block, 84 bytes per block
// Block: [scales (16)][qs (64)][d_fp16 (2)][dmin_fp16 (2)]
void dequant_q2_k_block(const uint8_t* raw, float* dst) noexcept;
bool dequant_q2_k(const void* raw_data, float* dst, uint64_t n_elements) noexcept;

// Q3_K dequantization (GGML_TYPE_Q3_K = 11)
// 256 elements per block, 110 bytes per block
void dequant_q3k_block(const uint8_t* raw, float* dst) noexcept;
bool dequant_q3k(const void* raw_data, float* dst, uint64_t n_elements) noexcept;

// Q4_K dequantization (GGML_TYPE_Q4_K = 12)
// 256 elements per block, 144 bytes per block
void dequant_q4k_block(const uint8_t* raw, float* dst) noexcept;
bool dequant_q4k(const void* raw_data, float* dst, uint64_t n_elements) noexcept;

// Q5_K dequantization (GGML_TYPE_Q5_K = 13)
// 256 elements per block, 176 bytes per block
// Block: [d_fp16 (2)][dmin_fp16 (2)][scales (12)][qh (32)][qs (128)]
void dequant_q5_k_block(const uint8_t* raw, float* dst) noexcept;
bool dequant_q5_k(const void* raw_data, float* dst, uint64_t n_elements) noexcept;

// Q6_K dequantization (GGML_TYPE_Q6_K = 14)
// 256 elements per block, 210 bytes per block
// Block: [ql lower-4-bits (128)][qh upper-2-bits (64)][int8 scales (16)][d_fp16 (2)]
void dequant_q6_k_block(const uint8_t* raw, float* dst) noexcept;
bool dequant_q6_k(const void* raw_data, float* dst, uint64_t n_elements) noexcept;

// Q8_K dequantization (GGML_TYPE_Q8_K = 15)
// 256 elements per block, 292 bytes per block
// Block: [d_fp32 (4)][qs int8 (256)][bsums int16 (32)]
void dequant_q8_k_block(const uint8_t* raw, float* dst) noexcept;
bool dequant_q8_k(const void* raw_data, float* dst, uint64_t n_elements) noexcept;

/// Unified dispatcher for any supported GGMLType.
/// Validates block-size divisibility and non-null pointers.
bool dequantize_tensor(GGMLType type, const void* raw_data,
                       float* dst, uint64_t n_elements) noexcept;

} // namespace hydroxl
} // namespace agr
