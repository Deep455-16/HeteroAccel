#pragma once
#include <cstdint>
#include <cstddef>
#include <string>

namespace agr {

// Opaque, backend-defined buffer handle. Application code never sees a
// VkBuffer/VkDeviceMemory directly -- that stays inside VulkanBackend.
struct Buffer {
    uint64_t id = 0;        // 0 == invalid/unallocated
    size_t size_bytes = 0;
};

// Vendor-neutral GPU compute backend interface.
//
// Phase 2 implements exactly one concrete backend (VulkanBackend) and
// exactly one real operation (vector addition). The interface is kept
// generic enough that later phases can add operations without breaking
// this contract, but nothing beyond vector-add is implemented yet.
class GPUBackend {
public:
    virtual ~GPUBackend() = default;

    // Selects a compute-capable device dynamically and brings up the
    // instance/device/queue/command pool. Returns false on any failure;
    // call lastError() for why. Never throws.
    virtual bool initialize() = 0;

    virtual bool isAvailable() const = 0;
    virtual std::string deviceName() const = 0;
    virtual std::string lastError() const = 0;

    // Allocates a GPU-visible buffer of sizeBytes. Returns an invalid
    // Buffer (id == 0) on failure.
    virtual Buffer createBuffer(size_t sizeBytes) = 0;
    virtual void destroyBuffer(Buffer& buffer) = 0;

    // Host -> GPU and GPU -> Host transfers. If outMs is non-null, the
    // wall-clock time of the transfer (including any staging-buffer copy)
    // is written there. Returns false on failure.
    virtual bool upload(const Buffer& buffer, const float* data, size_t count, double* outMs = nullptr) = 0;
    virtual bool download(const Buffer& buffer, float* data, size_t count, double* outMs = nullptr) = 0;

    // Dispatches C[i] = A[i] + B[i] for elementCount elements and blocks
    // until the GPU has finished. If outExecuteMs is non-null, the
    // GPU-side execution time (submit -> fence signaled) is written there
    // -- this deliberately excludes upload/download time so the three
    // phases can be reported separately and honestly.
    virtual bool executeVectorAdd(const Buffer& a, const Buffer& b, Buffer& c,
                                   uint32_t elementCount, double* outExecuteMs = nullptr) = 0;

    virtual void shutdown() = 0;
};

} // namespace agr
