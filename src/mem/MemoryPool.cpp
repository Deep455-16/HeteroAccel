// src/mem/MemoryPool.cpp
#include "mem/MemoryPool.h"

namespace agr {

MemoryPool::MemoryPool(IMemoryAllocator& allocator, size_t blockSize, size_t maxFree)
    : allocator_(allocator), blockSize_(blockSize), maxFree_(maxFree) {}

MemoryPool::~MemoryPool() {
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& b : freeList_) allocator_.free(b);
    freeList_.clear();
}

MemoryBlock MemoryPool::acquire() {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!freeList_.empty()) {
        MemoryBlock b = freeList_.back();
        freeList_.pop_back();
        b.state       = MemoryState::RESIDENT;
        b.access_count = 0;
        ++reuseCount_;
        ++activeCount_;
        return b;
    }
    // Allocate fresh
    MemoryBlock b = allocator_.allocate(blockSize_);
    if (b.isValid()) {
        ++activeCount_;
        ++totalCreated_;
    }
    return b;
}

void MemoryPool::release(MemoryBlock& block) {
    if (!block.isValid()) return;
    std::lock_guard<std::mutex> lk(mutex_);
    --activeCount_;
    if (freeList_.size() < maxFree_) {
        block.state = MemoryState::FREE;
        freeList_.push_back(block);
    } else {
        allocator_.free(block);
    }
    block = MemoryBlock{};
}

size_t MemoryPool::reuseCount()   const { std::lock_guard<std::mutex> lk(mutex_); return reuseCount_; }
size_t MemoryPool::activeCount()  const { std::lock_guard<std::mutex> lk(mutex_); return activeCount_; }
size_t MemoryPool::freeCount()    const { std::lock_guard<std::mutex> lk(mutex_); return freeList_.size(); }
size_t MemoryPool::totalCreated() const { std::lock_guard<std::mutex> lk(mutex_); return totalCreated_; }

} // namespace agr
