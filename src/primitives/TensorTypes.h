// src/primitives/TensorTypes.h
// Phase 16: Native HeteroAccel Execution Primitives
//
// Defines the supported data types and core tensor shape/metadata for
// Phase 16. Only data types that can be correctly stored, sized, and
// validated are listed here. Quantized types (Q4, Q8, etc.) require
// special packing semantics that are deferred to Phase 17.
//
// Ownership model:
//   - TensorShape  — value type, no allocation. Always safe to copy.
//   - Tensor       — owns its backing HostBuffer. Non-copyable, movable.
//   - TensorView   — non-owning span over a Tensor's data. Lifetime
//                    bounded by the Tensor it was created from.
//
// Thread-safety:
//   - Tensor is NOT thread-safe. External synchronization is required
//     if multiple threads share a Tensor.
//   - TensorView inherits the same rule.
//   - All static helpers (elementSize, etc.) are stateless and safe.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace agr {
namespace primitives {

// ---------------------------------------------------------------------------
// DataType — scalar element types supported by Phase 16 primitives
// ---------------------------------------------------------------------------
enum class DataType {
    FLOAT32,   //  4 bytes — primary working type
    FLOAT16,   //  2 bytes — storage / reduced-precision
    INT32,     //  4 bytes
    INT16,     //  2 bytes
    INT8,      //  1 byte — quantization staging / integer kernels
    UINT8,     //  1 byte
    BOOL,      //  1 byte (stored as uint8, 0=false 1=true)

    UNKNOWN,   // Sentinel — invalid; operations must reject this
};

/// Returns bytes per scalar element, or 0 for UNKNOWN.
inline size_t elementSize(DataType dt) noexcept {
    switch (dt) {
        case DataType::FLOAT32: return 4;
        case DataType::FLOAT16: return 2;
        case DataType::INT32:   return 4;
        case DataType::INT16:   return 2;
        case DataType::INT8:    return 1;
        case DataType::UINT8:   return 1;
        case DataType::BOOL:    return 1;
        default:                return 0;
    }
}

inline const char* toString(DataType dt) noexcept {
    switch (dt) {
        case DataType::FLOAT32: return "FLOAT32";
        case DataType::FLOAT16: return "FLOAT16";
        case DataType::INT32:   return "INT32";
        case DataType::INT16:   return "INT16";
        case DataType::INT8:    return "INT8";
        case DataType::UINT8:   return "UINT8";
        case DataType::BOOL:    return "BOOL";
        default:                return "UNKNOWN";
    }
}

// ---------------------------------------------------------------------------
// TensorShape — immutable shape/metadata value type (no data allocation)
// ---------------------------------------------------------------------------
struct TensorShape {
    std::vector<size_t> dims; ///< Dimension sizes, e.g. {batch, rows, cols}
    DataType            dtype = DataType::UNKNOWN;

    TensorShape() = default;
    TensorShape(std::vector<size_t> d, DataType t) : dims(std::move(d)), dtype(t) {}

    /// Number of dimensions (rank).
    size_t rank() const noexcept { return dims.size(); }

    /// True when shape is explicitly zero-element (any dim is 0).
    bool isEmpty() const noexcept {
        for (auto d : dims) if (d == 0) return true;
        return false;
    }

    bool isValid() const noexcept {
        return dtype != DataType::UNKNOWN && elementSize(dtype) > 0;
    }

    /// Total element count. Returns 0 on empty, UINT64_MAX on overflow.
    /// Caller must check the returned value before multiplying by elementSize.
    uint64_t elementCount() const noexcept {
        if (dims.empty()) return 0;
        uint64_t count = 1;
        for (size_t d : dims) {
            // Overflow guard: if count > UINT64_MAX / d, would overflow
            if (d != 0 && count > (UINT64_MAX / d)) return UINT64_MAX;
            count *= d;
        }
        return count;
    }

    /// Total byte size. Returns 0 on empty, UINT64_MAX on overflow.
    uint64_t byteSize() const noexcept {
        if (!isValid()) return 0;
        uint64_t elems = elementCount();
        if (elems == UINT64_MAX) return UINT64_MAX;
        size_t esz = elementSize(dtype);
        if (esz != 0 && elems > (UINT64_MAX / static_cast<uint64_t>(esz)))
            return UINT64_MAX;
        return elems * static_cast<uint64_t>(esz);
    }

    bool operator==(const TensorShape& o) const noexcept {
        return dtype == o.dtype && dims == o.dims;
    }
    bool operator!=(const TensorShape& o) const noexcept { return !(*this == o); }

    std::string describe() const {
        std::string s = std::string(toString(dtype)) + "[";
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i) s += "x";
            s += std::to_string(dims[i]);
        }
        s += "]";
        return s;
    }
};

} // namespace primitives
} // namespace agr
