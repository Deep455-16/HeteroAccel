// src/mem/MemoryManager.h
#pragma once
#include "mem/MemoryTypes.h"
#include "mem/CPUAllocator.h"
#include "mem/VulkanAllocator.h"
#include "mem/MemoryPool.h"
#include "mem/TransferManager.h"
#include "mem/ResidencyManager.h"
#include "mem/PressureMonitor.h"
#include "mem/EvictionPolicy.h"
#include "gpu/VulkanBackend.h"

#include <memory>
#include <mutex>
#include <vector>

namespace agr {

/// Unified memory manager for CPU and GPU resources.
class MemoryManager {
public:
    explicit MemoryManager(VulkanBackend& backend);
    ~MemoryManager();

    /// Allocate memory at the preferred location.
    MemoryBlock allocate(size_t size, MemoryLocation preferred, MemoryPriority priority = MemoryPriority::NORMAL);

    /// Release memory back to the manager/allocator.
    void release(MemoryBlock& block);

    /// Move data between CPU and GPU.
    bool move(MemoryBlock& block, MemoryLocation destination);

    /// Check memory pressure. Evict if necessary based on policy.
    void enforcePressureLimits();

    MemoryStats statistics() const;

private:
    void syncBlockResidency(MemoryBlock& block);
    MemoryBlock allocateInternal(size_t size, MemoryLocation loc);
    void freeInternal(MemoryBlock& block);

    VulkanBackend& backend_;

    CPUAllocator cpuAllocator_;
    VulkanAllocator gpuAllocator_;
    
    // We maintain multiple pools for common block sizes (e.g. 1MB, 4MB, 16MB)
    // For simplicity in Phase 4, we use one generic pool for CPU and one for GPU
    // specifically tuned for a standard buffer size (e.g., 4MB) if requested size matches.
    std::unique_ptr<MemoryPool> cpuPool4MB_;
    std::unique_ptr<MemoryPool> gpuPool4MB_;

    TransferManager  transferManager_;
    ResidencyManager residencyManager_;
    PressureMonitor  cpuPressure_;
    PressureMonitor  gpuPressure_;
    EvictionPolicy   evictionPolicy_;

    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, MemoryBlock> activeBlocks_;
    size_t evictionCount_ = 0;
};

} // namespace agr
