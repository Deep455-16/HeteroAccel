// src/mem/MemoryPool.h
//
// A simple free-list pool that reuses MemoryBlocks of matching size
// to reduce repeated allocation/free overhead.
//
#pragma once
#include "mem/IMemoryAllocator.h"
#include "mem/MemoryTypes.h"

#include <mutex>
#include <vector>

namespace agr {

/// Simple free-list pool sitting on top of an IMemoryAllocator.
/// Blocks are returned to the free list on release rather than freed.
class MemoryPool {
public:
    /// @param allocator Underlying allocator (CPU or Vulkan). Must outlive the pool.
    /// @param blockSize Fixed block size this pool manages (bytes).
    /// @param maxFree   Maximum number of blocks to keep in the free list.
    explicit MemoryPool(IMemoryAllocator& allocator, size_t blockSize, size_t maxFree = 16);
    ~MemoryPool();

    /// Obtain a block: reuses from free list if available, else allocates fresh.
    MemoryBlock acquire();

    /// Return a block to the free list (if capacity allows) or free it.
    void release(MemoryBlock& block);

    size_t blockSize()   const { return blockSize_; }
    size_t reuseCount()  const;
    size_t activeCount() const;
    size_t freeCount()   const;
    size_t totalCreated() const;

private:
    IMemoryAllocator& allocator_;
    size_t            blockSize_;
    size_t            maxFree_;

    mutable std::mutex       mutex_;
    std::vector<MemoryBlock> freeList_;
    size_t reuseCount_   = 0;
    size_t activeCount_  = 0;
    size_t totalCreated_ = 0;
};

} // namespace agr
