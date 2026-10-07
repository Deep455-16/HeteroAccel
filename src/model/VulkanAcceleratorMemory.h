#pragma once

#include "model/IAcceleratorMemory.h"
#include "gpu/VulkanBackend.h"

#include <mutex>
#include <string>
#include <unordered_map>

namespace agr {

/// Real Vulkan-backed accelerator memory using the existing VulkanBackend.
/// Allocations and uploads complete before residency is reported.
class VulkanAcceleratorMemory : public IAcceleratorMemory {
public:
    VulkanAcceleratorMemory(VulkanBackend* backend, uint64_t budget_bytes);
    ~VulkanAcceleratorMemory() override;

    std::string name() const override { return "VulkanAcceleratorMemory"; }
    bool isAvailable() const override;
    uint64_t capacity() const override { return budget_; }
    uint64_t available() const override;
    std::string lastError() const override;

    void* allocate(size_t bytes) override;
    void release(void* ptr) override;
    bool upload(const void* source, void* destination, size_t bytes) override;

private:
    VulkanBackend* backend_ = nullptr;
    uint64_t budget_ = 0;
    uint64_t used_ = 0;
    uint64_t next_key_ = 1;
    mutable std::mutex mutex_;
    std::string last_error_;
    std::unordered_map<uint64_t, Buffer> buffers_;
};

} // namespace agr
