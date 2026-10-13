// src/hydroxl/HydroXLSampler.cpp
// Phase 17: Logits sampler implementation.

#include "hydroxl/HydroXLSampler.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>

namespace agr {
namespace hydroxl {

HydroXLSampler::HydroXLSampler(const SamplerParams& p)
    : params_(p), rng_state_(p.seed)
{}

void HydroXLSampler::reset(uint32_t seed) {
    rng_state_ = seed;
    params_.seed = seed;
}

// LCG random number generator (Knuth/MMIX constants)
float HydroXLSampler::nextRandom() {
    rng_state_ = rng_state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t bits = static_cast<uint32_t>(rng_state_ >> 33);
    return static_cast<float>(bits) / static_cast<float>(UINT32_MAX);
}

TokenID HydroXLSampler::sample(const float* logits, uint32_t vocab_size,
                                const std::vector<TokenID>& past_tokens) {
    if (!logits || vocab_size == 0) return TOKEN_INVALID;

    // ---- Make a mutable copy ----
    std::vector<float> scores(logits, logits + vocab_size);

    // ---- Repetition penalty ----
    if (params_.repeat_penalty != 1.0f && !past_tokens.empty()) {
        for (TokenID t : past_tokens) {
            if (t >= 0 && static_cast<uint32_t>(t) < vocab_size) {
                float& s = scores[static_cast<uint32_t>(t)];
                if (s > 0.0f) s /= params_.repeat_penalty;
                else          s *= params_.repeat_penalty;
            }
        }
    }

    // ---- Greedy (temperature == 0) ----
    if (params_.temperature <= 0.0f) {
        return static_cast<TokenID>(
            std::distance(scores.begin(),
                          std::max_element(scores.begin(), scores.end())));
    }

    // ---- Temperature scaling ----
    float inv_temp = 1.0f / params_.temperature;
    for (auto& s : scores) s *= inv_temp;

    // ---- Softmax ----
    float max_s = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    for (auto& s : scores) { s = std::exp(s - max_s); sum += s; }
    for (auto& s : scores) s /= sum;

    // ---- Top-K filtering ----
    // Build index vector sorted by descending probability
    std::vector<uint32_t> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0u);

    int k = (params_.top_k > 0 && params_.top_k < static_cast<int>(vocab_size))
                ? params_.top_k
                : static_cast<int>(vocab_size);

    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](uint32_t a, uint32_t b){ return scores[a] > scores[b]; });
    indices.resize(static_cast<size_t>(k));

    // ---- Top-P (nucleus) filtering ----
    if (params_.top_p < 1.0f) {
        float cumsum = 0.0f;
        size_t keep = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            cumsum += scores[indices[i]];
            ++keep;
            if (cumsum >= params_.top_p) break;
        }
        indices.resize(keep);
    }

    // ---- Sample from the remaining candidates ----
    // Re-normalise their probabilities
    float sub_sum = 0.0f;
    for (uint32_t idx : indices) sub_sum += scores[idx];

    float r = nextRandom() * sub_sum;
    float running = 0.0f;
    for (uint32_t idx : indices) {
        running += scores[idx];
        if (running >= r) return static_cast<TokenID>(idx);
    }

    // Fallback: return top-1
    return static_cast<TokenID>(indices[0]);
}

} // namespace hydroxl
} // namespace agr
