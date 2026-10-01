// src/mem/CPUAllocator.cpp
#include "mem/CPUAllocator.h"

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

namespace agr {

static size_t detectTotalRam() {
#ifdef _WIN32
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms))
        return static_cast<size_t>(ms.ullTotalPhys);
#endif
    return 0; // unknown fallback
}

CPUAllocator::CPUAllocator()
    : totalRamBytes_(detectTotalRam()) {}

CPUAllocator::~CPUAllocator() {
    // Free any leaked allocations (should not happen in correct usage)
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& [id, rec] : records_) {
        if (rec.ptr) std::free(rec.ptr);
    }
}

MemoryBlock CPUAllocator::allocate(size_t size) {
    MemoryBlock block;
    if (size == 0) return block; // id==0, invalid

    void* ptr = std::malloc(size);
    if (!ptr) return block; // allocation failed

    std::lock_guard<std::mutex> lk(mutex_);
    uint64_t id = nextId_++;
    records_[id] = {ptr, size};
    usedBytes_ += size;
    if (usedBytes_ > peakBytes_) peakBytes_ = usedBytes_;
    ++allocationCount_;

    block.id             = id;
    block.size           = size;
    block.location       = MemoryLocation::CPU;
    block.state          = MemoryState::RESIDENT;
    block.residency      = Residency::CPU;
    block.backend_handle = reinterpret_cast<uint64_t>(ptr);
    block.last_accessed  = std::chrono::steady_clock::now();
    return block;
}

void CPUAllocator::free(MemoryBlock& block) {
    if (!block.isValid()) return;

    std::lock_guard<std::mutex> lk(mutex_);
    auto it = records_.find(block.id);
    if (it == records_.end()) return; // double-free or unknown id — ignored

    std::free(it->second.ptr);
    usedBytes_ -= it->second.size;
    records_.erase(it);

    block = MemoryBlock{}; // zero out caller's handle
}

size_t CPUAllocator::used()     const { std::lock_guard<std::mutex> lk(mutex_); return usedBytes_; }
size_t CPUAllocator::capacity() const { return totalRamBytes_; }
size_t CPUAllocator::available() const {
#ifdef _WIN32
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms))
        return static_cast<size_t>(ms.ullAvailPhys);
#endif
    if (totalRamBytes_ > used()) return totalRamBytes_ - used();
    return 0;
}

size_t CPUAllocator::peakUsed()        const { std::lock_guard<std::mutex> lk(mutex_); return peakBytes_; }
size_t CPUAllocator::allocationCount() const { std::lock_guard<std::mutex> lk(mutex_); return allocationCount_; }

} // namespace agr
