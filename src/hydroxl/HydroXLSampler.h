// src/hydroxl/HydroXLSampler.h
// Phase 17: Logits sampling for HydroXL token generation.
//
// Supports: temperature, top-k, top-p (nucleus), repetition penalty.
// All sampling is done on FLOAT32 logits.
// Uses a seeded LCG (linear congruential generator) for reproducibility.

#pragma once

#include "hydroxl/HydroXLTypes.h"

#include <cstdint>
#include <vector>

namespace agr {
namespace hydroxl {

struct SamplerParams {
    float    temperature  = 1.0f;   ///< 0.0 = greedy, >0 = stochastic
    int      top_k        = 40;     ///< 0 = disabled
    float    top_p        = 0.95f;  ///< 1.0 = disabled
    float    repeat_penalty = 1.1f; ///< 1.0 = disabled
    uint32_t seed         = 42;
};

class HydroXLSampler {
public:
    explicit HydroXLSampler(const SamplerParams& p = {});

    void reset(uint32_t seed);

    /// Sample next token from logits[0..vocab_size-1].
    /// past_tokens: token IDs used for repetition penalty.
    /// Returns TOKEN_INVALID on parameter error.
    TokenID sample(const float* logits, uint32_t vocab_size,
                   const std::vector<TokenID>& past_tokens);

private:
    SamplerParams params_;
    uint64_t      rng_state_;

    float nextRandom();  ///< Uniform [0, 1)
};

} // namespace hydroxl
} // namespace agr
