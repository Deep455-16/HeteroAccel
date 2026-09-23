#pragma once
#include "gpu/GPUBackend.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#if AGR_HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace agr {

#if AGR_HAVE_VULKAN

// Real Vulkan compute backend. Selects a compute-capable physical device
// dynamically (never hardcoded to a vendor/model), builds the vector-add
// pipeline from a SPIR-V module compiled at build time, and executes
// real GPU dispatches.
class VulkanBackend : public GPUBackend {
public:
    VulkanBackend() = default;
    ~VulkanBackend() override;

    bool initialize() override;
    bool isAvailable() const override { return available_; }
    std::string deviceName() const override { return deviceName_; }
    std::string lastError() const override { return lastError_; }

    // Phase 4: expose the physical device so VulkanAllocator can query
    // memory heap sizes without creating a second Vulkan instance/device.
    VkPhysicalDevice physicalDevice() const { return physicalDevice_; }

    Buffer createBuffer(size_t sizeBytes) override;
    void destroyBuffer(Buffer& buffer) override;

    bool upload(const Buffer& buffer, const float* data, size_t count, double* outMs) override;
    bool download(const Buffer& buffer, float* data, size_t count, double* outMs) override;

    bool executeVectorAdd(const Buffer& a, const Buffer& b, Buffer& c,
                           uint32_t elementCount, double* outExecuteMs) override;

    void shutdown() override;

private:
    struct BufferRecord {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        // True if this buffer's memory is both DEVICE_LOCAL and
        // HOST_VISIBLE (typical on integrated GPUs with unified memory,
        // e.g. Intel Iris Xe) -- in that case upload/download map it
        // directly with no separate staging buffer or copy command.
        bool unifiedMemory = false;
        VkMemoryPropertyFlags memoryFlags = 0;
    };

    bool createInstance();
    bool selectPhysicalDevice();
    bool createLogicalDeviceAndQueue();
    bool createCommandPoolAndBuffer();
    bool loadShaderAndCreatePipeline();
    bool createDescriptorInfrastructure();

    int32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags desired, bool* exactMatch) const;
    bool copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size);

    void fail(const std::string& what);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue computeQueue_ = VK_NULL_HANDLE;
    uint32_t computeQueueFamilyIndex_ = 0;

    VkCommandPool commandPool_ = VK_NULL_HANDLE;

    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline computePipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;

    std::unordered_map<uint64_t, BufferRecord> buffers_;
    uint64_t nextBufferId_ = 1;

    bool available_ = false;
    std::string deviceName_;
    std::string lastError_;
    std::string shaderPath_;
};

#else // AGR_HAVE_VULKAN == 0

// Vulkan loader/headers were not present at build time. This stub keeps
// the rest of the application (and its build) working; every call
// reports the same honest "unavailable" state as VulkanDetector does.
class VulkanBackend : public GPUBackend {
public:
    bool initialize() override { return false; }
    bool isAvailable() const override { return false; }
    std::string deviceName() const override { return ""; }
    std::string lastError() const override { return "Built without Vulkan loader/headers (AGR_HAVE_VULKAN=0)"; }
    Buffer createBuffer(size_t) override { return Buffer{}; }
    void destroyBuffer(Buffer&) override {}
    bool upload(const Buffer&, const float*, size_t, double*) override { return false; }
    bool download(const Buffer&, float*, size_t, double*) override { return false; }
    bool executeVectorAdd(const Buffer&, const Buffer&, Buffer&, uint32_t, double*) override { return false; }
    void shutdown() override {}
};

#endif

} // namespace agr
