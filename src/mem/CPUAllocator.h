// src/mem/CPUAllocator.h
#pragma once
#include "mem/IMemoryAllocator.h"
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace agr {

/// Allocates ordinary host (CPU) RAM via std::malloc.
/// Thread-safe via internal mutex.
class CPUAllocator : public IMemoryAllocator {
public:
    CPUAllocator();
    ~CPUAllocator() override;

    MemoryBlock allocate(size_t size) override;
    void free(MemoryBlock& block) override;

    size_t used()     const override;
    size_t available() const override;
    size_t capacity()  const override; ///< Physical RAM total in bytes

    std::string name() const override { return "CPUAllocator"; }

    size_t peakUsed()        const;
    size_t allocationCount() const;

private:
    struct Record {
        void*  ptr  = nullptr;
        size_t size = 0;
    };

    mutable std::mutex              mutex_;
    std::unordered_map<uint64_t, Record> records_; // block.id -> Record
    uint64_t nextId_          = 1;
    size_t   usedBytes_       = 0;
    size_t   peakBytes_       = 0;
    size_t   allocationCount_ = 0;
    size_t   totalRamBytes_   = 0; // detected once at construction
};

} // namespace agr
