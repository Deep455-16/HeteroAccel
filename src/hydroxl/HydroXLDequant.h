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

/// Dequantize an F16 tensor (fp16->f32 conversion).
void dequant_f16(const void* raw_data, float* dst, uint64_t n_elements) noexcept;

} // namespace hydroxl
} // namespace agr
