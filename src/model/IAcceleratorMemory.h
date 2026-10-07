#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace agr {

/// Generic accelerator-visible memory. CPU, Vulkan, CUDA, and future NPUs
/// all implement this. CUDA may be unavailable at runtime.
class IAcceleratorMemory {
public:
    virtual ~IAcceleratorMemory() = default;

    virtual std::string name() const = 0;
    virtual bool isAvailable() const = 0;
    virtual uint64_t capacity() const = 0;
    virtual uint64_t available() const = 0;
    virtual std::string lastError() const = 0;

    /// Opaque allocation. Not necessarily a host-mapped pointer.
    virtual void* allocate(size_t bytes) = 0;
    virtual void release(void* ptr) = 0;

    /// Host -> accelerator. Returns false on failure; dest is then invalid
    /// for residency purposes and the caller must release it.
    virtual bool upload(const void* source, void* destination, size_t bytes) = 0;

    /// Optional download. Default: unsupported.
    virtual bool download(void* /*destination*/, const void* /*source*/, size_t /*bytes*/) {
        return false;
    }
};

} // namespace agr
