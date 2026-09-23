// src/mem/ResidencyManager.h
#pragma once
#include "mem/MemoryTypes.h"
#include <mutex>
#include <unordered_map>

namespace agr {

/// Tracks and updates the residency state of tracked blocks.
class ResidencyManager {
public:
    void registerBlock(uint64_t id, Residency initialResidency);
    void unregisterBlock(uint64_t id);

    void promote(uint64_t id, MemoryLocation target);
    void demote(uint64_t id, MemoryLocation from);

    Residency getResidency(uint64_t id) const;
    void touch(uint64_t id); ///< Updates last accessed and increment access count

    bool getBlockInfo(uint64_t id, std::chrono::steady_clock::time_point& last_accessed, uint64_t& access_count) const;

private:
    struct Meta {
        Residency residency = Residency::CPU;
        std::chrono::steady_clock::time_point last_accessed{};
        uint64_t access_count = 0;
    };

    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, Meta> records_;
};

} // namespace agr
