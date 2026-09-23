// src/mem/EvictionPolicy.cpp
#include "mem/EvictionPolicy.h"
#include <algorithm>

namespace agr {

double EvictionPolicy::scoreBlock(const MemoryBlock& block) const {
    // Lower score means it should be evicted FIRST.
    double score = 0.0;

    // 1. Priority is the dominant factor.
    switch (block.priority) {
        case MemoryPriority::LOW:      score += 0.0;     break;
        case MemoryPriority::NORMAL:   score += 1000.0;  break;
        case MemoryPriority::HIGH:     score += 10000.0; break;
        case MemoryPriority::CRITICAL: score += 100000.0; break;
    }

    // 2. LRU: older blocks get a lower score.
    auto now = std::chrono::steady_clock::now();
    auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - block.last_accessed).count();
    
    // Cap age influence so it doesn't override priority
    // Older = larger age = subtract from score
    double agePenalty = static_cast<double>(age) * 0.001; 
    if (agePenalty > 900.0) agePenalty = 900.0; 

    score -= agePenalty;

    return score;
}

std::vector<uint64_t> EvictionPolicy::selectForEviction(const std::vector<MemoryBlock>& candidates, size_t count) const {
    if (count == 0 || candidates.empty()) return {};

    struct ScoredBlock {
        uint64_t id;
        double score;
        MemoryPriority priority;
    };

    std::vector<ScoredBlock> scored;
    scored.reserve(candidates.size());
    for (const auto& b : candidates) {
        scored.push_back({b.id, scoreBlock(b), b.priority});
    }

    // Sort ascending by score (lowest score first = evict first)
    std::sort(scored.begin(), scored.end(), [](const ScoredBlock& a, const ScoredBlock& b) {
        return a.score < b.score;
    });

    std::vector<uint64_t> result;
    result.reserve(count);
    for (size_t i = 0; i < scored.size() && result.size() < count; ++i) {
        // Do not evict critical blocks in Phase 4.
        if (scored[i].priority == MemoryPriority::CRITICAL) continue;
        result.push_back(scored[i].id);
    }

    return result;
}

} // namespace agr
