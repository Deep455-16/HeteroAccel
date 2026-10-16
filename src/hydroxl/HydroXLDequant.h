// src/hydroxl/HydroXLDequant.h
// Phase 17: Quantized tensor dequantization for HydroXL.
//
// Implements dequantization for Q4_K, Q3_K, Q8_0 block formats.
// All functions produce FLOAT32 output from the raw quantized bytes.
//
// Reference: ggml/src/ggml-quants.h, ggml/src/ggml-quants.c

#pragma once

#include "hydroxl/HydroXLTypes.h"

#include <cstddef>
#include <cstdint>

namespace agr {
namespace hydroxl {

/// Convert an IEEE 754 FP16 (half-precision) value to float32.
float fp16_to_float(uint16_t h) noexcept;

/// Dequantize one Q4_K block (256 elements) into dst.
/// raw must point to exactly Q4K_RAW_BLOCK_BYTES (144) bytes.
void dequant_q4k_block(const uint8_t* raw, float* dst) noexcept;

/// Dequantize a full Q4_K tensor (type id 12, covers Q4_K_S and Q4_K_M).
bool dequant_q4k(const void* raw_data, float* dst, uint64_t n_elements) noexcept;

/// Dequantize one Q3_K block (256 elements) into dst.
/// raw must point to exactly Q3K_RAW_BLOCK_BYTES (110) bytes.
void dequant_q3k_block(const uint8_t* raw, float* dst) noexcept;

/// Dequantize a full Q3_K tensor (type id 11, covers Q3_K_S/M/L).
bool dequant_q3k(const void* raw_data, float* dst, uint64_t n_elements) noexcept;

/// Dequantize one Q8_0 block (32 elements).
void dequant_q8_0_block(const uint8_t* raw, float* dst) noexcept;

/// Dequantize a full Q8_0 tensor.
bool dequant_q8_0(const void* raw_data, float* dst, uint64_t n_elements) noexcept;

/// Dequantize a full Q8_0 tensor.
bool dequant_q8_0(const void* raw_data, float* dst, uint64_t n_elements) noexcept;

/// Dequantize an F16 tensor (fp16->f32 conversion).
void dequant_f16(const void* raw_data, float* dst, uint64_t n_elements) noexcept;

// Q5_0 dequantization (GGML_TYPE_Q5_0 = 6)
// 32 elements per block, 22 bytes per block
// Block: [d_fp16 (2 bytes)][q0..q31 packed 5-bit values (20 bytes)]
// Dequantization: x[i] = d * (q[i] - 16.0f)  (centered, zero at q=16)
void dequant_q5_0_block(const uint8_t* raw, float* dst) noexcept;

/// Dequantize a full Q5_0 tensor (type id 6).
bool dequant_q5_0(const void* raw_data, float* dst, uint64_t n_elements) noexcept;

// Q5_1 dequantization (GGML_TYPE_Q5_1 = 7)
// 32 elements per block, 22 bytes per block (same block layout as Q5_0,
//  but different scaling/offset for the quantized values).
void dequant_q5_1_block(const uint8_t* raw, float* dst) noexcept;

/// Dequantize a full Q5_1 tensor (type id 7).
bool dequant_q5_1(const void* raw_data, float* dst, uint64_t n_elements) noexcept;

// Q6_K dequantization (GGML_TYPE_Q6_K = 14)
// 256 elements per block, 210 bytes per block
// Block: [d_fp16 (2 bytes)][q0..q255 packed 6-bit values (210 bytes)]
// Dequantization: x[i] = d * q[i]  (unsigned, 0-63 range per value)
void dequant_q6_k_block(const uint8_t* raw, float* dst) noexcept;

/// Dequantize a full Q6_K tensor (type id 14).
bool dequant_q6_k(const void* raw_data, float* dst, uint64_t n_elements) noexcept;

// ---------------------------------------------------------------------------
/// End of Q5_0/Q5_1/Q6_K dequantization implementations
// ---------------------------------------------------------------------------

} // namespace hydroxl
} // namespace agr
