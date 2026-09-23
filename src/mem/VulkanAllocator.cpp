// src/mem/VulkanAllocator.cpp
#include "mem/VulkanAllocator.h"

#if AGR_HAVE_VULKAN

#include <chrono>

namespace agr {

VulkanAllocator::VulkanAllocator(VulkanBackend& backend)
    : backend_(backend)
{
    queryHeapSizes();
}

VulkanAllocator::~VulkanAllocator() {
    // Free any leaked GPU allocations
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& [id, buf] : records_) {
        Buffer b = buf;
        backend_.destroyBuffer(b);
    }
}

void VulkanAllocator::queryHeapSizes() {
    VkPhysicalDevice pd = backend_.physicalDevice();
    if (pd == VK_NULL_HANDLE) return;

    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(pd, &props);

    for (uint32_t i = 0; i < props.memoryHeapCount; ++i) {
        VkMemoryHeapFlags flags = props.memoryHeaps[i].flags;
        size_t heapSize = static_cast<size_t>(props.memoryHeaps[i].size);

        if (flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            deviceLocalHeapBytes_ += heapSize;
        }
        // Check if any memory type for this heap is HOST_VISIBLE
        for (uint32_t j = 0; j < props.memoryTypeCount; ++j) {
            if (props.memoryTypes[j].heapIndex == i &&
                (props.memoryTypes[j].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                hostVisibleHeapBytes_ += heapSize;
                break;
            }
        }
    }
}

MemoryBlock VulkanAllocator::allocate(size_t size) {
    MemoryBlock block;
    if (!backend_.isAvailable() || size == 0) return block;

    Buffer buf = backend_.createBuffer(size);
    if (!buf.id) return block; // GPU allocation failed

    std::lock_guard<std::mutex> lk(mutex_);
    uint64_t id = nextId_++;
    records_[id] = buf;
    usedBytes_ += size;
    if (usedBytes_ > peakBytes_) peakBytes_ = usedBytes_;
    ++allocationCount_;

    block.id             = id;
    block.size           = size;
    block.location       = MemoryLocation::GPU;
    block.state          = MemoryState::RESIDENT;
    block.residency      = Residency::GPU;
    block.backend_handle = buf.id;
    block.last_accessed  = std::chrono::steady_clock::now();
    return block;
}

void VulkanAllocator::free(MemoryBlock& block) {
    if (!block.isValid()) return;

    std::lock_guard<std::mutex> lk(mutex_);
    auto it = records_.find(block.id);
    if (it == records_.end()) return; // unknown or already freed

    Buffer b = it->second;
    backend_.destroyBuffer(b);
    usedBytes_ -= block.size;
    records_.erase(it);

    block = MemoryBlock{};
}

size_t VulkanAllocator::used()     const { std::lock_guard<std::mutex> lk(mutex_); return usedBytes_; }
size_t VulkanAllocator::capacity() const { return deviceLocalHeapBytes_; }
size_t VulkanAllocator::available() const {
    size_t u = used();
    return (deviceLocalHeapBytes_ > u) ? (deviceLocalHeapBytes_ - u) : 0;
}
size_t VulkanAllocator::peakUsed()        const { std::lock_guard<std::mutex> lk(mutex_); return peakBytes_; }
size_t VulkanAllocator::allocationCount() const { std::lock_guard<std::mutex> lk(mutex_); return allocationCount_; }
size_t VulkanAllocator::deviceLocalBytes() const { return deviceLocalHeapBytes_; }
size_t VulkanAllocator::hostVisibleBytes() const { return hostVisibleHeapBytes_; }

} // namespace agr

#endif // AGR_HAVE_VULKAN
