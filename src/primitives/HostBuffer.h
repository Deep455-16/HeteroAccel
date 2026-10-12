// src/primitives/HostBuffer.h
// Phase 16: CPU-accessible buffer abstraction.
//
// HostBuffer owns a heap allocation of a specified byte count.
// It is non-copyable, movable, and RAII-based.
//
// Bounds-checking:
//   - data(offset, size) validates offset+size <= capacity_.
//   - Zero-size buffers are valid (data() returns nullptr safely).
//
// Alignment:
//   - Allocations are aligned to at least AGR_HOSTBUFFER_ALIGN bytes
//     (default: 64 bytes for SIMD compatibility). This is done via
//     _aligned_malloc on Windows or std::aligned_alloc on POSIX.
//
// Thread-safety:
//   - Individual HostBuffer instances are NOT thread-safe. External
//     synchronization required for concurrent access.
//
// Error model:
//   - Constructor throws std::bad_alloc on allocation failure.
//   - data(offset, size) returns nullptr on out-of-bounds access.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace agr {
namespace primitives {

static constexpr size_t kHostBufferAlign = 64; ///< bytes

class HostBuffer {
public:
    /// Construct an unallocated (empty) buffer.
    HostBuffer() = default;

    /// Allocate `bytes` bytes, aligned to kHostBufferAlign.
    /// Throws std::bad_alloc if allocation fails.
    /// A zero-byte buffer is valid: ptr_ stays nullptr but capacity_ == 0.
    explicit HostBuffer(size_t bytes) : capacity_(bytes) {
        if (bytes == 0) return;
#ifdef _WIN32
        ptr_ = _aligned_malloc(bytes, kHostBufferAlign);
#else
        ptr_ = std::aligned_alloc(kHostBufferAlign, bytes);
#endif
        if (!ptr_) throw std::bad_alloc();
    }

    // Non-copyable
    HostBuffer(const HostBuffer&)            = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;

    // Movable
    HostBuffer(HostBuffer&& o) noexcept
        : ptr_(o.ptr_), capacity_(o.capacity_) {
        o.ptr_      = nullptr;
        o.capacity_ = 0;
    }
    HostBuffer& operator=(HostBuffer&& o) noexcept {
        if (this != &o) {
            release();
            ptr_      = o.ptr_;
            capacity_ = o.capacity_;
            o.ptr_      = nullptr;
            o.capacity_ = 0;
        }
        return *this;
    }

    ~HostBuffer() { release(); }

    /// Byte capacity.
    size_t capacity() const noexcept { return capacity_; }

    /// True when capacity > 0 and allocation succeeded.
    bool isValid() const noexcept { return ptr_ != nullptr || capacity_ == 0; }

    /// Raw pointer to byte offset `offset`, validated that offset+size <= capacity_.
    /// Returns nullptr on any out-of-bounds condition.
    void* data(size_t offset = 0, size_t size = 0) const noexcept {
        if (!ptr_ || capacity_ == 0) return nullptr;
        if (offset >= capacity_) return nullptr;
        size_t needed = (size == 0) ? (capacity_ - offset) : size;
        if (needed > capacity_ - offset) return nullptr;
        return static_cast<uint8_t*>(ptr_) + offset;
    }

    /// Fill the entire buffer with the given byte value.
    void fill(uint8_t value) noexcept {
        if (ptr_ && capacity_) std::memset(ptr_, static_cast<int>(value), capacity_);
    }

    /// Copy `size` bytes from `src` into this buffer at `offset`.
    /// Returns false on out-of-bounds.
    bool write(size_t offset, const void* src, size_t size) noexcept {
        if (size == 0) return true;
        void* dst = data(offset, size);
        if (!dst || !src) return false;
        std::memcpy(dst, src, size);
        return true;
    }

    /// Copy `size` bytes from this buffer at `offset` into `dst`.
    /// Returns false on out-of-bounds.
    bool read(size_t offset, void* dst, size_t size) const noexcept {
        if (size == 0) return true;
        const void* src_ptr = data(offset, size);
        if (!src_ptr || !dst) return false;
        std::memcpy(dst, src_ptr, size);
        return true;
    }

private:
    void release() noexcept {
        if (ptr_) {
#ifdef _WIN32
            _aligned_free(ptr_);
#else
            std::free(ptr_);
#endif
            ptr_      = nullptr;
            capacity_ = 0;
        }
    }

    void*  ptr_      = nullptr;
    size_t capacity_ = 0;
};

} // namespace primitives
} // namespace agr
