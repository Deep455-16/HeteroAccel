// src/mem/MemoryManager.cpp
#include "mem/MemoryManager.h"
#include <iostream>

namespace agr {

static constexpr size_t POOL_BLOCK_SIZE = 4 * 1024 * 1024; // 4 MB

MemoryManager::MemoryManager(VulkanBackend& backend)
    : backend_(backend),
      cpuAllocator_(),
      gpuAllocator_(backend),
      transferManager_(backend)
{
    cpuPool4MB_ = std::make_unique<MemoryPool>(cpuAllocator_, POOL_BLOCK_SIZE);
    gpuPool4MB_ = std::make_unique<MemoryPool>(gpuAllocator_, POOL_BLOCK_SIZE);
}

MemoryManager::~MemoryManager() {
    // Blocks should be released by callers. Any remaining are forcefully freed.
    for (auto& [id, block] : activeBlocks_) {
        freeInternal(block);
    }
}

void MemoryManager::syncBlockResidency(MemoryBlock& block) {
    residencyManager_.touch(block.id);
    std::chrono::steady_clock::time_point la;
    uint64_t ac = 0;
    residencyManager_.getBlockInfo(block.id, la, ac);
    block.last_accessed = la;
    block.access_count = ac;
    block.residency = residencyManager_.getResidency(block.id);
}

MemoryBlock MemoryManager::allocateInternal(size_t size, MemoryLocation loc) {
    if (size == POOL_BLOCK_SIZE) {
        if (loc == MemoryLocation::CPU) return cpuPool4MB_->acquire();
        if (loc == MemoryLocation::GPU) return gpuPool4MB_->acquire();
    }
    if (loc == MemoryLocation::CPU) return cpuAllocator_.allocate(size);
    if (loc == MemoryLocation::GPU) return gpuAllocator_.allocate(size);
    return MemoryBlock{};
}

void MemoryManager::freeInternal(MemoryBlock& block) {
    if (block.size == POOL_BLOCK_SIZE) {
        if (block.location == MemoryLocation::CPU) cpuPool4MB_->release(block);
        else if (block.location == MemoryLocation::GPU) gpuPool4MB_->release(block);
    } else {
        if (block.location == MemoryLocation::CPU) cpuAllocator_.free(block);
        else if (block.location == MemoryLocation::GPU) gpuAllocator_.free(block);
    }
}

MemoryBlock MemoryManager::allocate(size_t size, MemoryLocation preferred, MemoryPriority priority) {
    std::lock_guard<std::mutex> lk(mutex_);
    
    // Check if we need to evict before allocating
    // (A real implementation might do this dynamically, but for Phase 4 we just allocate and monitor)
    
    MemoryBlock block = allocateInternal(size, preferred);
    if (!block.isValid() && preferred == MemoryLocation::GPU) {
        // Fallback to CPU if GPU allocation fails
        block = allocateInternal(size, MemoryLocation::CPU);
    }
    
    if (block.isValid()) {
        block.priority = priority;
        residencyManager_.registerBlock(block.id, block.location == MemoryLocation::GPU ? Residency::GPU : Residency::CPU);
        syncBlockResidency(block);
        activeBlocks_[block.id] = block;
    }
    
    return block;
}

void MemoryManager::release(MemoryBlock& block) {
    if (!block.isValid()) return;
    
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = activeBlocks_.find(block.id);
    if (it != activeBlocks_.end()) {
        residencyManager_.unregisterBlock(block.id);
        freeInternal(it->second);
        activeBlocks_.erase(it);
    }
    block = MemoryBlock{};
}

bool MemoryManager::move(MemoryBlock& block, MemoryLocation destination) {
    if (!block.isValid() || block.location == destination) return true;
    
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = activeBlocks_.find(block.id);
    if (it == activeBlocks_.end()) return false;

    MemoryBlock& tracked = it->second;
    syncBlockResidency(tracked);
    
    if (tracked.location == MemoryLocation::CPU && destination == MemoryLocation::GPU) {
        MemoryBlock newGpuBlock = allocateInternal(tracked.size, MemoryLocation::GPU);
        if (!newGpuBlock.isValid()) return false;
        
        TransferHandle th = transferManager_.upload(reinterpret_cast<const void*>(tracked.backend_handle), newGpuBlock, tracked.size);
        if (!th.valid()) {
            freeInternal(newGpuBlock);
            return false;
        }
        
        residencyManager_.promote(tracked.id, MemoryLocation::GPU);
        freeInternal(tracked);
        
        tracked.location = MemoryLocation::GPU;
        tracked.backend_handle = newGpuBlock.backend_handle;
        syncBlockResidency(tracked);
        block = tracked;
        return true;
    }
    else if (tracked.location == MemoryLocation::GPU && destination == MemoryLocation::CPU) {
        MemoryBlock newCpuBlock = allocateInternal(tracked.size, MemoryLocation::CPU);
        if (!newCpuBlock.isValid()) return false;
        
        TransferHandle th = transferManager_.download(tracked, reinterpret_cast<void*>(newCpuBlock.backend_handle), tracked.size);
        if (!th.valid()) {
            freeInternal(newCpuBlock);
            return false;
        }
        
        residencyManager_.promote(tracked.id, MemoryLocation::CPU);
        freeInternal(tracked);
        
        tracked.location = MemoryLocation::CPU;
        tracked.backend_handle = newCpuBlock.backend_handle;
        syncBlockResidency(tracked);
        block = tracked;
        return true;
    }
    
    return false;
}

void MemoryManager::enforcePressureLimits() {
    std::lock_guard<std::mutex> lk(mutex_);
    
    PressureLevel gpuLvl = gpuPressure_.assess(gpuAllocator_.used(), gpuAllocator_.capacity());
    if (gpuLvl == PressureLevel::NORMAL) return;
    
    // We are under pressure. Find GPU blocks to evict.
    std::vector<MemoryBlock> gpuCandidates;
    for (const auto& [id, b] : activeBlocks_) {
        if (b.location == MemoryLocation::GPU) {
            MemoryBlock copy = b;
            syncBlockResidency(copy); // update LRU before scoring
            gpuCandidates.push_back(copy);
        }
    }
    
    // Evict up to 4 blocks at a time when under pressure
    std::vector<uint64_t> toEvict = evictionPolicy_.selectForEviction(gpuCandidates, 4);
    
    for (uint64_t id : toEvict) {
        auto it = activeBlocks_.find(id);
        if (it != activeBlocks_.end()) {
            // Unlock to call move, which will re-lock. 
            // Better to do move logic inline or just release if CPU memory is also full.
            // For Phase 4, let's just release to simulate disk swap/drop.
            residencyManager_.unregisterBlock(id);
            freeInternal(it->second);
            activeBlocks_.erase(it);
            evictionCount_++;
        }
    }
}

MemoryStats MemoryManager::statistics() const {
    std::lock_guard<std::mutex> lk(mutex_);
    MemoryStats stats;
    
    stats.cpu_total_bytes = cpuAllocator_.capacity();
    stats.cpu_used_bytes  = cpuAllocator_.used();
    stats.cpu_peak_bytes  = cpuAllocator_.peakUsed();
    
    stats.gpu_device_local_bytes = gpuAllocator_.capacity();
    stats.gpu_used_bytes         = gpuAllocator_.used();
    stats.gpu_peak_bytes         = gpuAllocator_.peakUsed();
    
    stats.allocation_count = cpuAllocator_.allocationCount() + gpuAllocator_.allocationCount();
    stats.transfer_count   = transferManager_.transferCount();
    stats.pool_reuse_count = cpuPool4MB_->reuseCount() + gpuPool4MB_->reuseCount();
    stats.eviction_count   = evictionCount_;
    
    stats.bytes_uploaded   = transferManager_.totalBytesUploaded();
    stats.bytes_downloaded = transferManager_.totalBytesDownloaded();
    
    stats.upload_bandwidth_gbps   = transferManager_.avgUploadBandwidthGbps();
    stats.download_bandwidth_gbps = transferManager_.avgDownloadBandwidthGbps();
    
    stats.cpu_pressure = cpuPressure_.assess(stats.cpu_used_bytes, stats.cpu_total_bytes);
    stats.gpu_pressure = gpuPressure_.assess(stats.gpu_used_bytes, stats.gpu_device_local_bytes);
    
    return stats;
}

} // namespace agr
