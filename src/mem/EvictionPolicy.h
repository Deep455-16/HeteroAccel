// src/mem/EvictionPolicy.h
#pragma once
#include "mem/MemoryTypes.h"
#include <vector>

namespace agr {

/// Ranks blocks for eviction using Priority + LRU.
class EvictionPolicy {
public:
    /// Select up to `count` block IDs from the provided candidates that should
    /// be evicted. Excludes CRITICAL priority unless absolutely necessary (not currently implemented to evict CRITICAL).
    std::vector<uint64_t> selectForEviction(const std::vector<MemoryBlock>& candidates, size_t count) const;

    /// Score a single block. Lower score = evict first.
    double scoreBlock(const MemoryBlock& block) const;
};

} // namespace agr
