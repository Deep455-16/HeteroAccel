// src/mem/IMemoryAllocator.h
#pragma once
#include "mem/MemoryTypes.h"
#include <cstddef>
#include <string>

namespace agr {

/// Abstract allocator interface — every backend implements this.
class IMemoryAllocator {
public:
    virtual ~IMemoryAllocator() = default;

    /// Allocate `size` bytes and return a valid MemoryBlock.
    /// Returns an invalid block (id==0) on failure.
    virtual MemoryBlock allocate(size_t size) = 0;

    /// Release a previously allocated block. Safe to call on invalid blocks.
    virtual void free(MemoryBlock& block) = 0;

    /// Bytes currently allocated.
    virtual size_t used() const = 0;

    /// Bytes still available (best estimate).
    virtual size_t available() const = 0;

    /// Total capacity of this allocator (0 if unknown).
    virtual size_t capacity() const = 0;

    /// Human-readable name for diagnostics.
    virtual std::string name() const = 0;
};

} // namespace agr
