// src/primitives/Tensor.h
// Phase 16: Tensor — owned, typed, multi-dimensional array on CPU host memory.
//
// A Tensor owns a HostBuffer. It stores shape metadata (TensorShape) and
// provides typed, bounds-checked element access.
//
// Ownership rules:
//   - Tensor owns its data — non-copyable, movable.
//   - TensorView is a non-owning view — lifetime bounded by source Tensor.
//   - Use Tensor::view() to create a TensorView that borrows from this Tensor.
//
// Construction:
//   - Tensor::allocate(shape)  — allocates a new buffer, data is uninitialised.
//   - Tensor::zeros(shape)     — allocates and zeroes the buffer.
//
// Validation:
//   - Returns invalid Tensor (isValid()==false) for UNKNOWN dtype, overflow,
//     or allocation failure.
//
// Access:
//   - dataAs<T>()  — raw typed pointer to the entire buffer (nullptr if invalid).
//   - at<T>(i)     — bounds-checked element at flat index i.
//   - Operator[] is NOT provided intentionally to force explicit bounds checks.

#pragma once

#include "primitives/TensorTypes.h"
#include "primitives/HostBuffer.h"
#include <cassert>
#include <cstring>
#include <new>

namespace agr {
namespace primitives {

// Forward declare
class TensorView;

// ---------------------------------------------------------------------------
// Tensor — owns a HostBuffer; non-copyable, movable
// ---------------------------------------------------------------------------
class Tensor {
public:
    Tensor() = default;

    // Non-copyable
    Tensor(const Tensor&)            = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Movable
    Tensor(Tensor&& o) noexcept
        : shape_(std::move(o.shape_)),
          buffer_(std::move(o.buffer_)),
          valid_(o.valid_) {
        o.valid_ = false;
    }

    Tensor& operator=(Tensor&& o) noexcept {
        if (this != &o) {
            shape_  = std::move(o.shape_);
            buffer_ = std::move(o.buffer_);
            valid_  = o.valid_;
            o.valid_ = false;
        }
        return *this;
    }

    // Factory: allocate uninitialised storage.
    // Returns an invalid (default-constructed) Tensor on validation failure
    // or allocation error, never throws to caller.
    static Tensor allocate(const TensorShape& shape) noexcept {
        if (!shape.isValid()) return Tensor{};
        uint64_t bytes = shape.byteSize();
        if (bytes == 0 || bytes == UINT64_MAX) return Tensor{};
        // Reject tensors larger than 32 GiB for safety.
        if (bytes > (static_cast<uint64_t>(32) << 30)) return Tensor{};
        try {
            Tensor t;
            t.shape_  = shape;
            t.buffer_ = HostBuffer(static_cast<size_t>(bytes));
            t.valid_  = true;
            return t;
        } catch (const std::bad_alloc&) {
            return Tensor{};
        }
    }

    // Factory: allocate and zero-initialise.
    static Tensor zeros(const TensorShape& shape) noexcept {
        Tensor t = allocate(shape);
        if (t.isValid()) t.buffer_.fill(0);
        return t;
    }

    bool isValid() const noexcept { return valid_ && buffer_.isValid() && shape_.isValid(); }

    const TensorShape& shape() const noexcept { return shape_; }
    DataType dtype()           const noexcept { return shape_.dtype; }
    uint64_t elementCount()    const noexcept { return shape_.elementCount(); }
    uint64_t byteSize()        const noexcept { return shape_.byteSize(); }

    /// Raw typed pointer — nullptr when !isValid().
    template<typename T>
    T* dataAs() noexcept {
        return static_cast<T*>(buffer_.data(0, static_cast<size_t>(byteSize())));
    }

    template<typename T>
    const T* dataAs() const noexcept {
        return static_cast<const T*>(buffer_.data(0, static_cast<size_t>(byteSize())));
    }

    /// Bounds-checked element access by flat index.
    /// Returns nullptr on out-of-bounds or type mismatch.
    template<typename T>
    T* at(uint64_t idx) noexcept {
        if (!isValid() || idx >= elementCount()) return nullptr;
        size_t offset = static_cast<size_t>(idx) * elementSize(shape_.dtype);
        return static_cast<T*>(buffer_.data(offset, sizeof(T)));
    }

    template<typename T>
    const T* at(uint64_t idx) const noexcept {
        if (!isValid() || idx >= elementCount()) return nullptr;
        size_t offset = static_cast<size_t>(idx) * elementSize(shape_.dtype);
        return static_cast<const T*>(buffer_.data(offset, sizeof(T)));
    }

    /// Fill the tensor buffer with raw bytes.
    void fillBytes(uint8_t value) noexcept {
        if (isValid()) buffer_.fill(value);
    }

    /// Access the underlying HostBuffer (for transfer operations).
    const HostBuffer& hostBuffer() const noexcept { return buffer_; }
    HostBuffer&       hostBuffer()       noexcept { return buffer_; }

private:
    TensorShape shape_;
    HostBuffer  buffer_;
    bool        valid_ = false;
};

// ---------------------------------------------------------------------------
// TensorView — non-owning view over a Tensor's data
// Lifetime MUST NOT exceed the source Tensor.
// ---------------------------------------------------------------------------
class TensorView {
public:
    TensorView() = default;

    explicit TensorView(Tensor& t) noexcept
        : shape_(&t.shape()), ptr_(t.dataAs<uint8_t>()), size_bytes_(static_cast<size_t>(t.byteSize())) {}

    bool isValid() const noexcept { return ptr_ != nullptr && shape_ != nullptr && shape_->isValid(); }

    const TensorShape& shape()      const noexcept { return *shape_; }
    DataType           dtype()      const noexcept { return shape_->dtype; }
    uint64_t           elementCount() const noexcept { return shape_->elementCount(); }
    size_t             byteSize()   const noexcept { return size_bytes_; }

    template<typename T>
    T* dataAs() noexcept { return static_cast<T*>(static_cast<void*>(ptr_)); }

    template<typename T>
    const T* dataAs() const noexcept { return static_cast<const T*>(static_cast<const void*>(ptr_)); }

    template<typename T>
    T* at(uint64_t idx) noexcept {
        if (!isValid() || idx >= elementCount()) return nullptr;
        size_t offset = static_cast<size_t>(idx) * elementSize(shape_->dtype);
        if (offset + sizeof(T) > size_bytes_) return nullptr;
        return reinterpret_cast<T*>(ptr_ + offset);
    }

    template<typename T>
    const T* at(uint64_t idx) const noexcept {
        if (!isValid() || idx >= elementCount()) return nullptr;
        size_t offset = static_cast<size_t>(idx) * elementSize(shape_->dtype);
        if (offset + sizeof(T) > size_bytes_) return nullptr;
        return reinterpret_cast<const T*>(ptr_ + offset);
    }

private:
    const TensorShape* shape_      = nullptr;
    uint8_t*           ptr_        = nullptr;
    size_t             size_bytes_ = 0;
};

} // namespace primitives
} // namespace agr
