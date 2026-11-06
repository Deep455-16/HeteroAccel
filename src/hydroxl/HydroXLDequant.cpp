// src/hydroxl/HydroXLDequant.cpp
// Phase 19 / Phase 27: quantized tensor dequantization for HydroXL.
//
// Block decoders for Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K,
// Q6_K, and Q8_K follow the bundled GGML reference (ggml/src/ggml-quants.c)
// exactly — same field offsets, same bit unpacking, same arithmetic order —
// so that decoded values match the reference bit-for-bit. Numerical
// equivalence is enforced by tests/test_p19_dequant_ref.cpp and
// tests/test_p27_quant_decoders.cpp.

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
    uint32_t sign     = (h >> 15) & 0x1u;
    uint32_t exponent = (h >> 10) & 0x1Fu;
    uint32_t mantissa =  h        & 0x3FFu;

    uint32_t f32;
    if (exponent == 0) {
        if (mantissa == 0) {
            f32 = sign << 31;
        } else {
            exponent = 1;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FFu;
            f32 = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        f32 = (sign << 31) | 0x7F800000u | (mantissa << 13);
    } else {
        f32 = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &f32, 4);
    return result;
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
// Q4_0 block dequantization — reference: dequantize_row_q4_0 (ggml-quants.c)
//
// Block layout (18 bytes per 32-element block):
//   [0..1]  d   (FP16) scale
//   [2..17] qs  16 bytes: low nibble → element j, high nibble → element j+16
//
// x[j]    = ((qs[j] & 0x0F) - 8) * d
// x[j+16] = ((qs[j] >> 4)   - 8) * d
// ---------------------------------------------------------------------------
void dequant_q4_0_block(const uint8_t* raw, float* dst) noexcept {
    uint16_t d_raw;
    std::memcpy(&d_raw, raw, 2);
    const float d = fp16_to_float(d_raw);
    const uint8_t* qs = raw + 2;

    for (int j = 0; j < Q4_0_BLOCK_SIZE / 2; ++j) {
        const int x0 = (qs[j] & 0x0F) - 8;
        const int x1 = (qs[j] >> 4)   - 8;

        dst[j + 0]                  = x0 * d;
        dst[j + Q4_0_BLOCK_SIZE / 2] = x1 * d;
    }
}

bool dequant_q4_0(const void* raw_data, float* dst, uint64_t n_elements) noexcept {
    return parallel_dequant(raw_data, dst, n_elements,
                            Q4_0_BLOCK_SIZE, Q4_0_BLOCK_BYTES,
                            dequant_q4_0_block);
}

// ---------------------------------------------------------------------------
// Q4_1 block dequantization — reference: dequantize_row_q4_1 (ggml-quants.c)
//
// Block layout (20 bytes per 32-element block):
//   [0..1]  d  (FP16) scale
//   [2..3]  m  (FP16) additive minimum
//   [4..19] qs 16 bytes: low nibble → element j, high nibble → element j+16
//
// x[j]    = (qs[j] & 0x0F) * d + m
// x[j+16] = (qs[j] >> 4)   * d + m
// ---------------------------------------------------------------------------
void dequant_q4_1_block(const uint8_t* raw, float* dst) noexcept {
    uint16_t d_raw, m_raw;
    std::memcpy(&d_raw, raw + 0, 2);
    std::memcpy(&m_raw, raw + 2, 2);
    const float d = fp16_to_float(d_raw);
    const float m = fp16_to_float(m_raw);
    const uint8_t* qs = raw + 4;

    for (int j = 0; j < Q4_1_BLOCK_SIZE / 2; ++j) {
        const int x0 = qs[j] & 0x0F;
        const int x1 = qs[j] >> 4;

        dst[j + 0]                  = x0 * d + m;
        dst[j + Q4_1_BLOCK_SIZE / 2] = x1 * d + m;
    }
}

bool dequant_q4_1(const void* raw_data, float* dst, uint64_t n_elements) noexcept {
    return parallel_dequant(raw_data, dst, n_elements,
                            Q4_1_BLOCK_SIZE, Q4_1_BLOCK_BYTES,
                            dequant_q4_1_block);
}

// ---------------------------------------------------------------------------
// Q5_0 block dequantization — reference: dequantize_row_q5_0 (ggml-quants.c)
//
// Block layout (22 bytes per 32-element block):
//   [0..1]  d   (FP16) scale
//   [2..5]  qh  (uint32 LE) 5th bit of each quant, bit j for element j (0..15)
//           and bit j+12 for element j+16
//   [6..21] qs  16 bytes: low nibble → element j, high nibble → element j+16
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
                            Q5_0_BLOCK_SIZE, Q5_0_BLOCK_BYTES,
                            dequant_q5_0_block);
}

// ---------------------------------------------------------------------------
// Q5_1 block dequantization — reference: dequantize_row_q5_1 (ggml-quants.c)
//
// Block layout (24 bytes per 32-element block):
//   [0..1]  d (FP16) scale
//   [2..3]  m (FP16) additive minimum
//   [4..7]  qh (uint32 LE) 5th bits
//   [8..23] qs 16 bytes nibbles
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
                            Q5_1_BLOCK_SIZE, Q5_1_BLOCK_BYTES,
                            dequant_q5_1_block);
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
// Q2_K block dequantization — reference: dequantize_row_q2_K (ggml-quants.c)
//
// Block layout (84 bytes per 256-element block):
//   [0..15]  scales: 16 bytes — low 4 bits = scale, high 4 bits = min
//   [16..79] qs:     64 bytes — 2-bit quants (4 per byte)
//   [80..81] d    (FP16) super-block scale
//   [82..83] dmin (FP16) super-block min
// ---------------------------------------------------------------------------
void dequant_q2_k_block(const uint8_t* raw, float* dst) noexcept {
    const uint8_t* scales = raw + 0;
    const uint8_t* q      = raw + 16;
    uint16_t d_raw, dmin_raw;
    std::memcpy(&d_raw,    raw + 80, 2);
    std::memcpy(&dmin_raw, raw + 82, 2);
    const float d   = fp16_to_float(d_raw);
    const float min = fp16_to_float(dmin_raw);

    int is = 0;
    for (int n = 0; n < Q2K_BLOCK_SIZE; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            uint8_t sc = scales[is++];
            float dl = d * (sc & 0xF);
            float ml = min * (sc >> 4);
            for (int l = 0; l < 16; ++l) {
                *dst++ = dl * static_cast<int8_t>((q[l] >> shift) & 3) - ml;
            }

            sc = scales[is++];
            dl = d * (sc & 0xF);
            ml = min * (sc >> 4);
            for (int l = 0; l < 16; ++l) {
                *dst++ = dl * static_cast<int8_t>((q[l + 16] >> shift) & 3) - ml;
            }

            shift += 2;
        }
        q += 32;
        (void)n;
    }
}

bool dequant_q2_k(const void* raw_data, float* dst, uint64_t n_elements) noexcept {
    return parallel_dequant(raw_data, dst, n_elements,
                            Q2K_BLOCK_SIZE, Q2K_RAW_BLOCK_BYTES,
                            dequant_q2_k_block);
}

// ---------------------------------------------------------------------------
// Q3_K block dequantization — reference: dequantize_row_q3_K (ggml-quants.c)
//
// Block layout (110 bytes per 256-element block):
//   [0..31]   hmask: 1 high bit per element
//   [32..95]  qs: 2 low bits per element
//   [96..107] scales: 16 x 6-bit scales, interleaved-packed into 12 bytes
//   [108..109] d (FP16) super-scale
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
        (void)n;
    }
}

bool dequant_q3k(const void* raw_data, float* dst, uint64_t n_elements) noexcept {
    return parallel_dequant(raw_data, dst, n_elements,
                            Q3K_BLOCK_SIZE, Q3K_RAW_BLOCK_BYTES,
                            dequant_q3k_block);
}

// ---------------------------------------------------------------------------
// Q4_K / Q5_K shared scale unpacking helper
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

// ---------------------------------------------------------------------------
// Q4_K block dequantization — reference: dequantize_row_q4_K (ggml-quants.c)
//
// Block layout (144 bytes per 256-element block):
//   [0..1]   d    (FP16): super-block scale
//   [2..3]   dmin (FP16): super-block min scale
//   [4..15]  scales: 8 x (6-bit scale, 6-bit min) pairs
//   [16..143] qs: 256 x 4-bit values
// ---------------------------------------------------------------------------
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

bool dequant_q4k(const void* raw_data, float* dst, uint64_t n_elements) noexcept {
    return parallel_dequant(raw_data, dst, n_elements,
                            Q4K_BLOCK_SIZE, Q4K_RAW_BLOCK_BYTES,
                            dequant_q4k_block);
}

// ---------------------------------------------------------------------------
// Q5_K block dequantization — reference: dequantize_row_q5_K (ggml-quants.c)
//
// Block layout (176 bytes per 256-element block):
//   [0..1]   d    (FP16): super-block scale
//   [2..3]   dmin (FP16): super-block min scale
//   [4..15]  scales: 8 x (6-bit scale, 6-bit min) pairs
//   [16..47] qh: 32 bytes — 5th bit per element
//   [48..175] qs: 128 bytes — low 4 bits per element
// ---------------------------------------------------------------------------
void dequant_q5_k_block(const uint8_t* raw, float* dst) noexcept {
    uint16_t d_raw, dmin_raw;
    std::memcpy(&d_raw,    raw + 0, 2);
    std::memcpy(&dmin_raw, raw + 2, 2);
    const float d   = fp16_to_float(d_raw);
    const float min = fp16_to_float(dmin_raw);

    const uint8_t* scales = raw + 4;
    const uint8_t* qh     = raw + 16;
    const uint8_t* ql     = raw + 48;

    int is = 0;
    uint8_t sc, m;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < Q5K_BLOCK_SIZE; j += 64) {
        get_scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * sc;
        const float m1 = min * m;
        get_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * sc;
        const float m2 = min * m;
        for (int l = 0; l < 32; ++l) {
            *dst++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
        }
        for (int l = 0; l < 32; ++l) {
            *dst++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
        }
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

bool dequant_q5_k(const void* raw_data, float* dst, uint64_t n_elements) noexcept {
    return parallel_dequant(raw_data, dst, n_elements,
                            Q5K_BLOCK_SIZE, Q5K_RAW_BLOCK_BYTES,
                            dequant_q5_k_block);
}

// ---------------------------------------------------------------------------
// Q6_K block dequantization — reference: dequantize_row_q6_K (ggml-quants.c)
//
// Block layout (210 bytes per 256-element block):
//   [0..127]   ql: lower 4 bits of each quant
//   [128..191] qh: upper 2 bits of each quant (2 bits per element)
//   [192..207] scales: 16 x INT8 sub-block scales (signed, 8-bit)
//   [208..209] d (FP16) super-scale
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
                            Q6K_BLOCK_SIZE, Q6K_RAW_BLOCK_BYTES,
                            dequant_q6_k_block);
}

// ---------------------------------------------------------------------------
// Q8_K block dequantization — reference: dequantize_row_q8_K (ggml-quants.c)
//
// Block layout (292 bytes per 256-element block):
//   [0..3]     d     (FP32): block scale
//   [4..259]   qs    (256 x INT8): quantized values
//   [260..291] bsums (16 x INT16): sub-block sums (not needed for dequant)
// ---------------------------------------------------------------------------
void dequant_q8_k_block(const uint8_t* raw, float* dst) noexcept {
    float d;
    std::memcpy(&d, raw, 4);
    const int8_t* qs = reinterpret_cast<const int8_t*>(raw + 4);
    for (int j = 0; j < Q8K_BLOCK_SIZE; ++j) {
        dst[j] = d * static_cast<float>(qs[j]);
    }
}

bool dequant_q8_k(const void* raw_data, float* dst, uint64_t n_elements) noexcept {
    return parallel_dequant(raw_data, dst, n_elements,
                            Q8K_BLOCK_SIZE, Q8K_RAW_BLOCK_BYTES,
                            dequant_q8_k_block);
}

// ---------------------------------------------------------------------------
// Unified tensor dequantization dispatcher
// ---------------------------------------------------------------------------
bool dequantize_tensor(GGMLType type, const void* raw_data,
                       float* dst, uint64_t n_elements) noexcept {
    if (!raw_data || !dst || n_elements == 0) return false;

    switch (type) {
        case GGMLType::F32:
            std::memcpy(dst, raw_data, static_cast<size_t>(n_elements) * sizeof(float));
            return true;
        case GGMLType::F16:
            dequant_f16(raw_data, dst, n_elements);
            return true;
        case GGMLType::Q4_0: return dequant_q4_0(raw_data, dst, n_elements);
        case GGMLType::Q4_1: return dequant_q4_1(raw_data, dst, n_elements);
        case GGMLType::Q5_0: return dequant_q5_0(raw_data, dst, n_elements);
        case GGMLType::Q5_1: return dequant_q5_1(raw_data, dst, n_elements);
        case GGMLType::Q8_0: return dequant_q8_0(raw_data, dst, n_elements);
        case GGMLType::Q2_K: return dequant_q2_k(raw_data, dst, n_elements);
        case GGMLType::Q3_K: return dequant_q3k(raw_data, dst, n_elements);
        case GGMLType::Q4_K: return dequant_q4k(raw_data, dst, n_elements);
        case GGMLType::Q5_K: return dequant_q5_k(raw_data, dst, n_elements);
        case GGMLType::Q6_K: return dequant_q6_k(raw_data, dst, n_elements);
        case GGMLType::Q8_K: return dequant_q8_k(raw_data, dst, n_elements);
        default:
            return false;
    }
}

} // namespace hydroxl
} // namespace agr
