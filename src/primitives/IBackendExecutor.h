// src/primitives/IBackendExecutor.h
// Phase 16: Backend Execution Interface
//
// IBackendExecutor is the generic abstraction through which graph nodes and
// the Phase 15 execution graph dispatch tensor operations to a backend.
//
// Backends implemented in Phase 16:
//   CpuBackendExecutor   — complete, real implementation using CpuOps.
//
// Backends NOT yet implemented:
//   VulkanBackendExecutor — buffer allocation and host<->device transfer
//                           are provided; tensor compute is not yet exposed
//                           through this interface (the existing VulkanBackend
//                           provides vector-add; wiring it into this interface
//                           is deferred to Phase 17 native engine work).
//   CudaBackendExecutor   — not built; CUDA is disabled in this project.
//
// Error contract:
//   - Every operation returns BackendOpResult with success=false and a
//     descriptive error when the operation is unsupported, the device is
//     unavailable, or inputs are invalid.
//   - Never silently succeeds on invalid inputs.
//
// Thread-safety:
//   - Implementations document their own thread-safety.
//   - CpuBackendExecutor is stateless — concurrent calls on different tensors
//     are safe. Same-tensor concurrent access is not safe.

#pragma once

#include "primitives/Tensor.h"
#include "primitives/OpResult.h"
#include <string>

namespace agr {
namespace primitives {

/// Result of a backend operation — same shape as OpResult, typed separately
/// so callers can distinguish backend-level errors from op-level errors.
using BackendOpResult = OpResult;

// ---------------------------------------------------------------------------
// Backend capability flags
// ---------------------------------------------------------------------------
struct BackendCapabilities {
    bool copy      = false;
    bool fill      = false;
    bool add       = false;
    bool multiply  = false;
    bool scale     = false;
    bool sum       = false;
    bool maxVal    = false;

    // Device-level
    bool deviceBufferAllocation = false; ///< Can allocate device-side buffers
    bool hostToDevice           = false; ///< Can transfer host->device
    bool deviceToHost           = false; ///< Can transfer device->host

    std::string backendName;
    bool        available = false;
};

// ---------------------------------------------------------------------------
// IBackendExecutor — abstract backend interface
// ---------------------------------------------------------------------------
class IBackendExecutor {
public:
    virtual ~IBackendExecutor() = default;

    /// Human-readable backend name.
    virtual std::string name() const = 0;

    /// Whether this backend is usable on the current system.
    virtual bool isAvailable() const = 0;

    /// Query what this backend can do.
    virtual BackendCapabilities capabilities() const = 0;

    // -----------------------------------------------------------------------
    // Tensor operations
    // -----------------------------------------------------------------------

    /// Copy src into dst (same shape required).
    virtual BackendOpResult copy(Tensor& dst, const Tensor& src) = 0;

    /// Fill dst with scalar value (FLOAT32 only).
    virtual BackendOpResult fill(Tensor& dst, float value) = 0;

    /// Elementwise dst[i] = a[i] + b[i] (FLOAT32 only).
    virtual BackendOpResult add(Tensor& dst, const Tensor& a, const Tensor& b) = 0;

    /// Elementwise dst[i] = a[i] * b[i] (FLOAT32 only).
    virtual BackendOpResult multiply(Tensor& dst, const Tensor& a, const Tensor& b) = 0;

    /// Elementwise dst[i] = src[i] * scalar (FLOAT32 only).
    virtual BackendOpResult scale(Tensor& dst, const Tensor& src, float scalar) = 0;

    /// Sum of all elements (FLOAT32 only). Result in *out_sum.
    virtual BackendOpResult sum(const Tensor& src, double* out_sum) = 0;

    /// Maximum element (FLOAT32 only). Result in *out_max.
    virtual BackendOpResult maxVal(const Tensor& src, float* out_max) = 0;
};

} // namespace primitives
} // namespace agr
