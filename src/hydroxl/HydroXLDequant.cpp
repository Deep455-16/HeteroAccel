// src/hydroxl/HydroXLDequant.cpp
// Phase 19: quantized tensor dequantization for HydroXL.
//
// Block decoders for Q4_K, Q3_K, Q5_0, Q5_1, Q6_K and Q8_0 follow the
// bundled GGML reference (ggml/src/ggml-quants.c) exactly — same field
// offsets, same bit unpacking, same arithmetic order — so that decoded
// values match the reference bit-for-bit. Numerical equivalence is
// enforced by tests/test_p19_dequant_ref.cpp, which quantizes with the
// real GGML quantize_row_*_ref() and compares against the real GGML
// dequantize_row_*() on the same blocks.

#include "hydroxl/HydroXLDequant.h"

#include <algorithm>
#include <cstring>
#include <future>
#include <thread>
#include <vector>

namespace agr {
namespace hydroxl {

// ---------------------------------------------------------------------------
// FP16 → FP32 conversion (IEEE 754). Exact: every FP16 value is exactly
// representable in FP32, so no rounding is involved.
// ---------------------------------------------------------------------------
float fp16_to_float(uint16_t h) noexcept {
    // Sign, exponent, mantissa extraction
    uint32_t sign     = (h >> 15) & 0x1u;
    uint32_t exponent = (h >> 10) & 0x1Fu;
    uint32_t mantissa =  h        & 0x3FFu;

    uint32_t f32;
    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            f32 = sign << 31;
        } else {
            // Subnormal: normalise
            exponent = 1;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FFu;
            f32 = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        // Inf / NaN
        f32 = (sign << 31) | 0x7F800000u | (mantissa << 13);
    } else {
        f32 = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &f32, 4);
    return result;
}

// ---------------------------------------------------------------------------
// Q4_K block dequantization — reference: dequantize_row_q4_K (ggml-quants.c)
//
// Block layout (144 bytes per 256-element block):
//   [0..1]   d    (FP16): super-block scale
//   [2..3]   dmin (FP16): super-block min scale
//   [4..15]  scales: 8 x (6-bit scale, 6-bit min) pairs, packed per
//             get_scale_min_k4()
//   [16..143] qs: 256 x 4-bit values. For each 64-element group (8 groups),
//             the low nibbles of 32 bytes give elements [0..31] and the high
//             nibbles give elements [32..63].
//
// x[i] = d * scale[g] * q[i] - dmin * min[g]
// ---------------------------------------------------------------------------
static inline void get_scale_min_k4(int j, const uint8_t* q,
                                    uint8_t& d, uint8_t& m) noexcept {
    if (j < 4) {
        d = q[j]     & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

void dequant_q4k_block(const uint8_t* raw, float* dst) noexcept {
    uint16_t d_raw, dmin_raw;
    std::memcpy(&d_raw,    raw + 0, 2);
    std::memcpy(&dmin_raw, raw + 2, 2);
    const float d   = fp16_to_float(d_raw);
    const float min = fp16_to_float(dmin_raw);

    const uint8_t* scales = raw + 4;
    const uint8_t* q      = raw + 16;

    int is = 0;
    uint8_t sc, m;
    for (int j = 0; j < Q4K_BLOCK_SIZE; j += 64) {
        get_scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * sc;
        const float m1 = min * m;
        get_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * sc;
        const float m2 = min * m;
        for (int l = 0; l < 32; ++l) *dst++ = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) *dst++ = d2 * (q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}


// ---------------------------------------------------------------------------
// Threaded dequantization helper for large tensors (>128k elements)
// ---------------------------------------------------------------------------
template <typename BlockFn>
static inline bool parallel_dequant(const void* raw_data, float* dst,
                                     uint64_t n_elements, size_t block_size,
                                     size_t raw_block_bytes, BlockFn&& fn) {
    if (!raw_data || !dst || n_elements == 0) return false;
    if (n_elements % block_size != 0) return false;

    uint64_t n_blocks = n_elements / block_size;
    const uint8_t* src = static_cast<const uint8_t*>(raw_data);

    unsigned int hw = std::thread::hardware_concurrency();
    uint32_t num_threads = std::min(4u, hw == 0 ? 1u : hw);

    if (num_threads <= 1 || n_blocks < 256) {
        for (uint64_t b = 0; b < n_blocks; ++b) {
            fn(src + b * raw_block_bytes, dst + b * block_size);
        }
    } else {
        uint64_t blocks_per_thread = (n_blocks + num_threads - 1) / num_threads;
        std::vector<std::future<void>> futures;
        futures.reserve(num_threads - 1);
        for (uint32_t t = 1; t < num_threads; ++t) {
            uint64_t start_b = t * blocks_per_thread;
            uint64_t end_b   = std::min(n_blocks, start_b + blocks_per_thread);
            if (start_b < end_b) {
                futures.push_back(std::async(std::launch::async, [src, dst, start_b, end_b, raw_block_bytes, block_size, &fn]() {
                    for (uint64_t b = start_b; b < end_b; ++b) {
                        fn(src + b * raw_block_bytes, dst + b * block_size);
                    }
                }));
            }
        }
        uint64_t end_0 = std::min(n_blocks, blocks_per_thread);
        for (uint64_t b = 0; b < end_0; ++b) {
            fn(src + b * raw_block_bytes, dst + b * block_size);
        }
        for (auto& f : futures) {
            f.get();
        }
    }
    return true;
}
bool dequant_q4k(const void* raw_data, float* dst, uint64_t n_elements) noexcept {
    return parallel_dequant(raw_data, dst, n_elements,
                            Q4K_BLOCK_SIZE, Q4K_RAW_BLOCK_BYTES,
                            dequant_q4k_block);
}

// ---------------------------------------------------------------------------
// Q3_K block dequantization — reference: dequantize_row_q3_K (ggml-quants.c)
//
// Block layout (110 bytes per 256-element block):
//   [0..31]   hmask: 1 high bit per element
//   [32..95]  qs: 2 low bits per element
//   [96..107] scales: 16 x 6-bit scales, interleaved-packed into 12 bytes
//             (unpacked below exactly as the reference does with kmask1/2)
//   [108..109] d (FP16) super-scale
//
// x[i] = d * (scale[g] - 32) * (q2[i] - (hmask_bit ? 0 : 4))
// ---------------------------------------------------------------------------
void dequant_q3k_block(const uint8_t* raw, float* dst) noexcept {
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;

    const uint8_t* hm = raw;          // hmask  [0..31]
    const uint8_t* q  = raw + 32;     // qs     [32..95]
    const uint8_t* sc_raw = raw + 96; // scales [96..107]
    uint16_t d_raw;
    std::memcpy(&d_raw, raw + 108, 2);
    const float d_all = fp16_to_float(d_raw);

    // Unpack 12 packed bytes into 16 signed per-sub-block scales (int8).
    uint32_t aux[4] = {0, 0, 0, 0};
    std::memcpy(aux, sc_raw, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);

    int is = 0;
    float dl;
    uint8_t m = 1;
    for (int n = 0; n < Q3K_BLOCK_SIZE; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l) {
                *dst++ = dl * (static_cast<int8_t>((q[l + 0] >> shift) & 3)
                               - ((hm[l + 0] & m) ? 0 : 4));
            }

            dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l) {
                *dst++ = dl * (static_cast<int8_t>((q[l + 16] >> shift) & 3)
                               - ((hm[l + 16] & m) ? 0 : 4));
            }

            shift += 2;
            m <<= 1;
        }
        q += 32;
        // hm advances implicitly: 32 hmask bytes serve 256 elements via the
        // per-128 m-bit selection (reference reuses hm[l] with m<<shift).
        (void)n;
    }
}

bool dequant_q3k(const void* raw_data, float* dst, uint64_t n_elements) noexcept {
    return parallel_dequant(raw_data, dst, n_elements,
                            Q3K_BLOCK_SIZE, Q3K_RAW_BLOCK_BYTES,
                            dequant_q3k_block);
}

// ---------------------------------------------------------------------------
// Q8_0 block dequantization (32 elements, 34 bytes)
// Block: [d_fp16][q0..q31];  x[i] = d * q[i]
// ---------------------------------------------------------------------------
void dequant_q8_0_block(const uint8_t* raw, float* dst) noexcept {
    uint16_t d_raw;
    std::memcpy(&d_raw, raw, 2);
    float d = fp16_to_float(d_raw);

    const int8_t* q = reinterpret_cast<const int8_t*>(raw + 2);
    for (int i = 0; i < Q8_0_BLOCK_SIZE; ++i) {
        dst[i] = d * static_cast<float>(q[i]);
    }
}

bool dequant_q8_0(const void* raw_data, float* dst, uint64_t n_elements) noexcept {
    return parallel_dequant(raw_data, dst, n_elements,
                            Q8_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES,
                            dequant_q8_0_block);
}

// ---------------------------------------------------------------------------
// F16 dequantization (element-wise)
// ---------------------------------------------------------------------------
void dequant_f16(const void* raw_data, float* dst, uint64_t n_elements) noexcept {
    if (!raw_data || !dst || n_elements == 0) return;
    const uint16_t* src = static_cast<const uint16_t*>(raw_data);
    unsigned int hw = std::thread::hardware_concurrency();
    uint32_t num_threads = std::min(4u, hw == 0 ? 1u : hw);

    if (num_threads <= 1 || n_elements < 131072) {
        for (uint64_t i = 0; i < n_elements; ++i) {
            dst[i] = fp16_to_float(src[i]);
        }
    } else {
        uint64_t chunk = (n_elements + num_threads - 1) / num_threads;
        std::vector<std::future<void>> futures;
        futures.reserve(num_threads - 1);
        for (uint32_t t = 1; t < num_threads; ++t) {
            uint64_t start = t * chunk;
            uint64_t end   = std::min(n_elements, start + chunk);
            if (start < end) {
                futures.push_back(std::async(std::launch::async, [src, dst, start, end]() {
                    for (uint64_t i = start; i < end; ++i) {
                        dst[i] = fp16_to_float(src[i]);
                    }
                }));
            }
        }
        uint64_t end_0 = std::min(n_elements, chunk);
        for (uint64_t i = 0; i < end_0; ++i) {
            dst[i] = fp16_to_float(src[i]);
        }
        for (auto& f : futures) {
            f.get();
        }
    }
}

// ---------------------------------------------------------------------------
// Q5_0 block dequantization — reference: dequantize_row_q5_0 (ggml-quants.c)
//
// Block layout (22 bytes per 32-element block):
//   [0..1]  d   (FP16) scale
//   [2..5]  qh  (uint32 LE) 5th bit of each quant, bit j for element j (0..15)
//           and bit j+12 for element j+16
//   [6..21] qs  16 bytes: low nibble → element j, high nibble → element j+16
//
// x[j]    = d * (((qs[j] & 0x0F) | highbit0) - 16)
// x[j+16] = d * (((qs[j] >> 4)  | highbit1) - 16)
// where highbit0 = ((qh >> j) << 4) & 0x10, highbit1 = (qh >> (j+12)) & 0x10
// ---------------------------------------------------------------------------
void dequant_q5_0_block(const uint8_t* raw, float* dst) noexcept {
    uint16_t d_raw;
    std::memcpy(&d_raw, raw, 2);
    const float d = fp16_to_float(d_raw);

    uint32_t qh;
    std::memcpy(&qh, raw + 2, 4);
    const uint8_t* qs = raw + 6;

    for (int j = 0; j < 16; ++j) {
        const uint8_t xh_0 = static_cast<uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
        const uint8_t xh_1 = static_cast<uint8_t>((qh >> (j + 12)) & 0x10);

        const int x0 = ((qs[j] & 0x0F) | xh_0) - 16;
        const int x1 = ((qs[j] >> 4)   | xh_1) - 16;

        dst[j]      = x0 * d;
        dst[j + 16] = x1 * d;
    }
}

bool dequant_q5_0(const void* raw_data, float* dst, uint64_t n_elements) noexcept {
    return parallel_dequant(raw_data, dst, n_elements,
                            32, 22,
                            dequant_q5_0_block);
}

// ---------------------------------------------------------------------------
// Q5_1 block dequantization — reference: dequantize_row_q5_1 (ggml-quants.c)
//
// Block layout (24 bytes per 32-element block):
//   [0..1]  d (FP16) scale
//   [2..3]  m (FP16) additive minimum
//   [4..7]  qh (uint32 LE) 5th bits, same mapping as Q5_0
//   [8..23] qs 16 bytes nibbles, same mapping as Q5_0
//
// x[j]    = d * ((qs[j] & 0x0F) | highbit0) + m
// x[j+16] = d * ((qs[j] >> 4)  | highbit1) + m
// ---------------------------------------------------------------------------
void dequant_q5_1_block(const uint8_t* raw, float* dst) noexcept {
    uint16_t d_raw, m_raw;
    std::memcpy(&d_raw, raw + 0, 2);
    std::memcpy(&m_raw, raw + 2, 2);
    const float d = fp16_to_float(d_raw);
    const float m = fp16_to_float(m_raw);

    uint32_t qh;
    std::memcpy(&qh, raw + 4, 4);
    const uint8_t* qs = raw + 8;

    for (int j = 0; j < 16; ++j) {
        const uint8_t xh_0 = static_cast<uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
        const uint8_t xh_1 = static_cast<uint8_t>((qh >> (j + 12)) & 0x10);

        const int x0 = (qs[j] & 0x0F) | xh_0;
        const int x1 = (qs[j] >> 4)   | xh_1;

        dst[j]      = x0 * d + m;
        dst[j + 16] = x1 * d + m;
    }
}

bool dequant_q5_1(const void* raw_data, float* dst, uint64_t n_elements) noexcept {
    return parallel_dequant(raw_data, dst, n_elements,
                            32, 24,
                            dequant_q5_1_block);
}

// ---------------------------------------------------------------------------
// Q6_K block dequantization — reference: dequantize_row_q6_K (ggml-quants.c)
//
// Block layout (210 bytes per 256-element block):
//   [0..127]   ql: lower 4 bits of each quant
//   [128..191] qh: upper 2 bits of each quant (2 bits per element)
//   [192..207] scales: 16 x INT8 sub-block scales (signed, 8-bit)
//   [208..209] d (FP16) super-scale
//
// Per 128-element half-block, for l in 0..31 (is = l/16):
//   y[l+0]   = d * sc[is+0] * q1,  q1 = (ql[l+0] & 0xF) | ((qh[l]>>0 & 3)<<4) - 32
//   y[l+32]  = d * sc[is+2] * q2,  q2 = (ql[l+32] & 0xF) | ((qh[l]>>2 & 3)<<4) - 32
//   y[l+64]  = d * sc[is+4] * q3,  q3 = (ql[l+0] >> 4) | ((qh[l]>>4 & 3)<<4) - 32
//   y[l+96]  = d * sc[is+6] * q4,  q4 = (ql[l+32] >> 4) | ((qh[l]>>6 & 3)<<4) - 32
// ---------------------------------------------------------------------------
void dequant_q6_k_block(const uint8_t* raw, float* dst) noexcept {
    const uint8_t* ql = raw;            // [0..127]
    const uint8_t* qh = raw + 128;      // [128..191]
    const int8_t*  sc = reinterpret_cast<const int8_t*>(raw + 192); // [192..207]
    uint16_t d_raw;
    std::memcpy(&d_raw, raw + 208, 2);
    const float d = fp16_to_float(d_raw);

    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int8_t q1 = static_cast<int8_t>((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = static_cast<int8_t>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = static_cast<int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            dst[l + 0]   = d * sc[is + 0] * q1;
            dst[l + 32]  = d * sc[is + 2] * q2;
            dst[l + 64]  = d * sc[is + 4] * q3;
            dst[l + 96]  = d * sc[is + 6] * q4;
        }
        dst += 128;
        ql  += 64;
        qh  += 32;
        sc  += 8;
    }
}

bool dequant_q6_k(const void* raw_data, float* dst, uint64_t n_elements) noexcept {
    return parallel_dequant(raw_data, dst, n_elements,
                            256, 210,
                            dequant_q6_k_block);
}

} // namespace hydroxl
} // namespace agr
