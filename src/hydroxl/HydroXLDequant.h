// src/hydroxl/HydroXLDequant.h
// Phase 17: Quantized tensor dequantization for HydroXL.
//
// Implements dequantization for Q4_K_M (Q4_K) and Q8_0 block formats.
// All functions produce FLOAT32 output from the raw quantized bytes.
//
// Q4_K_M block structure (256 elements, 144 bytes):
//   Bytes  0..1  : d     (FP16) — super-block scale
//   Bytes  2..3  : dmin  (FP16) — super-block min
//   Bytes  4..15 : 12 bytes encoding 8 x (6-bit scale, 6-bit min) pairs
//   Bytes 16..143: 128 bytes of 4-bit quantized values (2 per byte)
//
// Q8_0 block structure (32 elements, 34 bytes):
//   Bytes 0..1 : d   (FP16) — block scale
//   Bytes 2..33: 32 x INT8 quantized values
//
// Reference: ggml/src/ggml-quants.h, ggml/src/ggml-quants.c
// These implementations follow the reference exactly.

#pragma once

#include "hydroxl/HydroXLTypes.h"

#include <cstddef>
#include <cstdint>

namespace agr {
namespace hydroxl {

/// Convert an IEEE 754 FP16 (half-precision) value to float32.
/// Implemented without platform-specific intrinsics.
float fp16_to_float(uint16_t h) noexcept;

/// Dequantize one Q4_K block (256 elements) into dst.
/// raw must point to exactly Q4K_RAW_BLOCK_BYTES (144) bytes.
/// dst must have room for Q4K_BLOCK_SIZE (256) floats.
void dequant_q4k_block(const uint8_t* raw, float* dst) noexcept;

/// Dequantize a full Q4_K tensor.
/// raw_data : raw quantized bytes (n_blocks * 144 bytes)
/// dst      : output float32 array (n_elements floats)
/// n_elements must be divisible by Q4K_BLOCK_SIZE.
/// Returns false if n_elements is not a multiple of Q4K_BLOCK_SIZE.
bool dequant_q4k(const void* raw_data, float* dst,
                 uint64_t n_elements) noexcept;

/// Dequantize one Q8_0 block (32 elements).
void dequant_q8_0_block(const uint8_t* raw, float* dst) noexcept;

/// Dequantize a full Q8_0 tensor.
bool dequant_q8_0(const void* raw_data, float* dst,
                  uint64_t n_elements) noexcept;

/// Dequantize an F16 tensor (just fp16->f32 conversion).
void dequant_f16(const void* raw_data, float* dst,
                 uint64_t n_elements) noexcept;

} // namespace hydroxl
} // namespace agr
