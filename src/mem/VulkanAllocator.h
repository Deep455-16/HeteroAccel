// src/mem/VulkanAllocator.h
//
// Vulkan memory allocator for Phase 4.
// Wraps the existing VulkanBackend — does NOT create a second VkInstance.
//
#pragma once
#include "mem/IMemoryAllocator.h"
#include "gpu/VulkanBackend.h"

#include <mutex>
#include <unordered_map>

namespace agr {

#if AGR_HAVE_VULKAN

/// Allocates Vulkan GPU memory by delegating to the shared VulkanBackend.
class VulkanAllocator : public IMemoryAllocator {
public:
    /// @param backend  Must already be initialized (isAvailable() == true).
    explicit VulkanAllocator(VulkanBackend& backend);
    ~VulkanAllocator() override;

    MemoryBlock allocate(size_t size) override;
    void free(MemoryBlock& block) override;

    size_t used()      const override;
    size_t available() const override;
    size_t capacity()  const override; ///< Device-local heap size in bytes

    std::string name() const override { return "VulkanAllocator"; }

    size_t peakUsed()             const;
    size_t allocationCount()      const;
    size_t deviceLocalBytes()     const; ///< Total device-local heap capacity
    size_t hostVisibleBytes()     const; ///< Total host-visible heap capacity

private:
    void queryHeapSizes();

    VulkanBackend& backend_;

    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, Buffer> records_; // block.id -> GPU Buffer
    uint64_t nextId_          = 1;
    size_t   usedBytes_       = 0;
    size_t   peakBytes_       = 0;
    size_t   allocationCount_ = 0;

    size_t   deviceLocalHeapBytes_  = 0;
    size_t   hostVisibleHeapBytes_  = 0;
};

#else // AGR_HAVE_VULKAN == 0

/// Stub when Vulkan is not available at compile time.
class VulkanAllocator : public IMemoryAllocator {
public:
    explicit VulkanAllocator(VulkanBackend& /*backend*/) {}
    MemoryBlock allocate(size_t) override { return MemoryBlock{}; }
    void free(MemoryBlock&) override {}
    size_t used()      const override { return 0; }
    size_t available() const override { return 0; }
    size_t capacity()  const override { return 0; }
    std::string name() const override { return "VulkanAllocator(stub)"; }
    size_t peakUsed()         const { return 0; }
    size_t allocationCount()  const { return 0; }
    size_t deviceLocalBytes() const { return 0; }
    size_t hostVisibleBytes() const { return 0; }
};

#endif // AGR_HAVE_VULKAN

} // namespace agr
