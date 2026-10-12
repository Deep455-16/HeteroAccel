// src/primitives/cpu/CpuOps.h
// Phase 16: CPU tensor operation declarations.
//
// All functions operate on host-accessible (CPU) memory only.
//
// Supported element types per operation:
//   copy      — all DataTypes with known elementSize
//   fill      — FLOAT32
//   add       — FLOAT32
//   multiply  — FLOAT32
//   scale     — FLOAT32
//   sum       — FLOAT32 (returns double result in OpResult::value field — stored out-param)
//   maxVal    — FLOAT32 (returns double result via out-param)
//
// Invalid shape, mismatched shapes, wrong dtype, and nullptr tensors all
// produce OpResult::fail() with a descriptive message.
//
// Thread-safety: functions are stateless; concurrent calls on DIFFERENT
// tensors are safe. Same-tensor concurrent access is not safe.

#pragma once

#include "primitives/Tensor.h"
#include "primitives/OpResult.h"
#include <cstddef>

namespace agr {
namespace primitives {
namespace cpu {

// ---------------------------------------------------------------------------
// copy — byte-wise copy of src into dst.
// dst and src must have identical shapes (dtype + dims).
// ---------------------------------------------------------------------------
OpResult copy(Tensor& dst, const Tensor& src);

// ---------------------------------------------------------------------------
// fill — set every FLOAT32 element to `value`.
// ---------------------------------------------------------------------------
OpResult fill(Tensor& dst, float value);

// ---------------------------------------------------------------------------
// add — elementwise dst[i] = a[i] + b[i].  FLOAT32 only.
// All three tensors must have the same shape.
// dst may alias a or b (result is correct).
// ---------------------------------------------------------------------------
OpResult add(Tensor& dst, const Tensor& a, const Tensor& b);

// ---------------------------------------------------------------------------
// multiply — elementwise dst[i] = a[i] * b[i].  FLOAT32 only.
// ---------------------------------------------------------------------------
OpResult multiply(Tensor& dst, const Tensor& a, const Tensor& b);

// ---------------------------------------------------------------------------
// scale — elementwise dst[i] = src[i] * scalar.  FLOAT32 only.
// dst may alias src.
// ---------------------------------------------------------------------------
OpResult scale(Tensor& dst, const Tensor& src, float scalar);

// ---------------------------------------------------------------------------
// sum — compute the sum of all FLOAT32 elements.
// Result is written to *out_sum. Returns fail if tensor is invalid/empty.
// ---------------------------------------------------------------------------
OpResult sum(const Tensor& src, double* out_sum);

// ---------------------------------------------------------------------------
// maxVal — find the maximum FLOAT32 element.
// Result is written to *out_max. Returns fail if tensor is invalid/empty.
// ---------------------------------------------------------------------------
OpResult maxVal(const Tensor& src, float* out_max);

} // namespace cpu
} // namespace primitives
} // namespace agr
