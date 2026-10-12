// src/primitives/cpu/CpuOps.cpp
// Phase 16: CPU tensor operation implementations.
//
// All implementations are scalar (no SIMD) for maximum portability and
// easy verification. Correctness is the invariant; performance optimization
// is left for Phase 17 if it uses these primitives directly.

#include "primitives/cpu/CpuOps.h"

#include <algorithm>
#include <cfloat>
#include <cstring>
#include <limits>

namespace agr {
namespace primitives {
namespace cpu {

// Internal validation helpers

static bool shapesMatch(const TensorShape& a, const TensorShape& b) noexcept {
    return a == b;
}

static bool isFloat32(const Tensor& t) noexcept {
    return t.isValid() && t.dtype() == DataType::FLOAT32;
}

// ---------------------------------------------------------------------------
// copy
// ---------------------------------------------------------------------------
OpResult copy(Tensor& dst, const Tensor& src) {
    if (!dst.isValid())   return OpResult::fail("copy: dst is invalid");
    if (!src.isValid())   return OpResult::fail("copy: src is invalid");
    if (!shapesMatch(dst.shape(), src.shape()))
        return OpResult::fail("copy: shape mismatch: dst=" + dst.shape().describe() +
                              " src=" + src.shape().describe());

    uint64_t bytes = src.byteSize();
    if (bytes == 0) return OpResult::ok(); // empty is a no-op

    const void* sp = src.hostBuffer().data(0, static_cast<size_t>(bytes));
    if (!sp) return OpResult::fail("copy: src data pointer is null");

    bool ok = dst.hostBuffer().write(0, sp, static_cast<size_t>(bytes));
    if (!ok) return OpResult::fail("copy: write to dst failed");

    return OpResult::ok();
}

// ---------------------------------------------------------------------------
// fill
// ---------------------------------------------------------------------------
OpResult fill(Tensor& dst, float value) {
    if (!isFloat32(dst)) return OpResult::fail("fill: tensor must be FLOAT32, got " +
                                               std::string(toString(dst.dtype())));

    uint64_t count = dst.elementCount();
    if (count == 0)      return OpResult::fail("fill: tensor is empty");

    float* p = dst.dataAs<float>();
    if (!p)              return OpResult::fail("fill: data pointer is null");

    for (uint64_t i = 0; i < count; ++i) p[i] = value;
    return OpResult::ok();
}

// ---------------------------------------------------------------------------
// add
// ---------------------------------------------------------------------------
OpResult add(Tensor& dst, const Tensor& a, const Tensor& b) {
    if (!isFloat32(dst)) return OpResult::fail("add: dst must be FLOAT32");
    if (!isFloat32(a))   return OpResult::fail("add: a must be FLOAT32");
    if (!isFloat32(b))   return OpResult::fail("add: b must be FLOAT32");
    if (!shapesMatch(dst.shape(), a.shape()))
        return OpResult::fail("add: dst/a shape mismatch");
    if (!shapesMatch(a.shape(), b.shape()))
        return OpResult::fail("add: a/b shape mismatch");

    uint64_t count = a.elementCount();
    if (count == 0) return OpResult::fail("add: tensors are empty");

    const float* ap = a.dataAs<const float>();
    const float* bp = b.dataAs<const float>();
    float*       dp = dst.dataAs<float>();

    if (!ap || !bp || !dp) return OpResult::fail("add: null data pointer");

    // Correct even when dst aliases a or b: read both first, write dst.
    // Note: because ap/bp/dp may alias, we compute inline. This is correct
    // for same-pointer aliasing (a[i]+b[i] evaluated before dst[i] written).
    for (uint64_t i = 0; i < count; ++i) {
        float va = ap[i];
        float vb = bp[i];
        dp[i] = va + vb;
    }
    return OpResult::ok();
}

// ---------------------------------------------------------------------------
// multiply
// ---------------------------------------------------------------------------
OpResult multiply(Tensor& dst, const Tensor& a, const Tensor& b) {
    if (!isFloat32(dst)) return OpResult::fail("multiply: dst must be FLOAT32");
    if (!isFloat32(a))   return OpResult::fail("multiply: a must be FLOAT32");
    if (!isFloat32(b))   return OpResult::fail("multiply: b must be FLOAT32");
    if (!shapesMatch(dst.shape(), a.shape()))
        return OpResult::fail("multiply: dst/a shape mismatch");
    if (!shapesMatch(a.shape(), b.shape()))
        return OpResult::fail("multiply: a/b shape mismatch");

    uint64_t count = a.elementCount();
    if (count == 0) return OpResult::fail("multiply: tensors are empty");

    const float* ap = a.dataAs<const float>();
    const float* bp = b.dataAs<const float>();
    float*       dp = dst.dataAs<float>();

    if (!ap || !bp || !dp) return OpResult::fail("multiply: null data pointer");

    for (uint64_t i = 0; i < count; ++i) {
        float va = ap[i];
        float vb = bp[i];
        dp[i] = va * vb;
    }
    return OpResult::ok();
}

// ---------------------------------------------------------------------------
// scale
// ---------------------------------------------------------------------------
OpResult scale(Tensor& dst, const Tensor& src, float scalar) {
    if (!isFloat32(dst)) return OpResult::fail("scale: dst must be FLOAT32");
    if (!isFloat32(src)) return OpResult::fail("scale: src must be FLOAT32");
    if (!shapesMatch(dst.shape(), src.shape()))
        return OpResult::fail("scale: dst/src shape mismatch");

    uint64_t count = src.elementCount();
    if (count == 0) return OpResult::fail("scale: tensors are empty");

    const float* sp = src.dataAs<const float>();
    float*       dp = dst.dataAs<float>();

    if (!sp || !dp) return OpResult::fail("scale: null data pointer");

    for (uint64_t i = 0; i < count; ++i) {
        float v = sp[i]; // read before write (handles aliasing)
        dp[i] = v * scalar;
    }
    return OpResult::ok();
}

// ---------------------------------------------------------------------------
// sum
// ---------------------------------------------------------------------------
OpResult sum(const Tensor& src, double* out_sum) {
    if (!out_sum) return OpResult::fail("sum: out_sum is null");
    if (!isFloat32(src)) return OpResult::fail("sum: src must be FLOAT32");

    uint64_t count = src.elementCount();
    if (count == 0) return OpResult::fail("sum: tensor is empty");

    const float* p = src.dataAs<const float>();
    if (!p) return OpResult::fail("sum: null data pointer");

    double acc = 0.0;
    for (uint64_t i = 0; i < count; ++i) acc += static_cast<double>(p[i]);
    *out_sum = acc;
    return OpResult::ok();
}

// ---------------------------------------------------------------------------
// maxVal
// ---------------------------------------------------------------------------
OpResult maxVal(const Tensor& src, float* out_max) {
    if (!out_max) return OpResult::fail("maxVal: out_max is null");
    if (!isFloat32(src)) return OpResult::fail("maxVal: src must be FLOAT32");

    uint64_t count = src.elementCount();
    if (count == 0) return OpResult::fail("maxVal: tensor is empty");

    const float* p = src.dataAs<const float>();
    if (!p) return OpResult::fail("maxVal: null data pointer");

    float best = p[0];
    for (uint64_t i = 1; i < count; ++i) {
        if (p[i] > best) best = p[i];
    }
    *out_max = best;
    return OpResult::ok();
}

} // namespace cpu
} // namespace primitives
} // namespace agr
