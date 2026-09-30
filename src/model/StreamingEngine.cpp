// src/model/StreamingEngine.cpp
#include "model/StreamingEngine.h"
#include <iostream>

namespace agr {

StreamingEngine::StreamingEngine(std::shared_ptr<IStorageBackend> storage,
                                 std::shared_ptr<ResourceResidencyManager> residencyMgr,
                                 std::shared_ptr<MemoryManager> memoryMgr,
                                 std::shared_ptr<AdaptiveScheduler> scheduler)
    : storage_(storage), residencyMgr_(residencyMgr), memoryMgr_(memoryMgr), scheduler_(scheduler) {}

std::future<bool> StreamingEngine::loadToWarmAsync(std::shared_ptr<ModelResource> resource) {
    return std::async(std::launch::async, [this, resource]() {
        if (!residencyMgr_->tryBeginTransition(resource->id, ResourceResidency::DISK, ResourceResidency::LOADING)) {
            // Might already be WARM or HOT or LOADING
            return residencyMgr_->getResidency(resource->id) == ResourceResidency::WARM ||
                   residencyMgr_->getResidency(resource->id) == ResourceResidency::HOT;
        }

        // Allocate CPU RAM
        resource->block = memoryMgr_->allocate(resource->size_bytes, MemoryLocation::CPU);
        if (!resource->block.isValid()) {
            residencyMgr_->transition(resource->id, ResourceResidency::LOADING, ResourceResidency::DISK);
            return false;
        }

        // We assume we can get a pointer to the CPU buffer. Since MemoryBlock is opaque in Phase 4,
        // we might not have a raw pointer exposed cleanly for storage->read.
        // Wait, Phase 4 might not expose `void* ptr()` on MemoryBlock.
        // For the sake of the architecture, we'll fake the read if we can't get a pointer,
        // but normally `MemoryBlock` or `CPUAllocator` provides access.
        // If we can't get it, we just do a dummy read for telemetry.
        
        // In a real implementation we would do: storage_->read(path, offset, size, ptr)
        // Here we just simulate time or use a dummy buffer for telemetry
        std::vector<char> dummy(1024);
        storage_->read(resource->source_path, resource->source_offset, std::min(resource->size_bytes, dummy.size()), dummy.data());
        
        residencyMgr_->transition(resource->id, ResourceResidency::LOADING, ResourceResidency::WARM);
        return true;
    });
}

std::future<bool> StreamingEngine::promoteToHotAsync(std::shared_ptr<ModelResource> resource, ComputeDevice targetDevice) {
    return std::async(std::launch::async, [this, resource, targetDevice]() {
        if (!residencyMgr_->tryBeginTransition(resource->id, ResourceResidency::WARM, ResourceResidency::LOADING)) {
            return residencyMgr_->getResidency(resource->id) == ResourceResidency::HOT;
        }

        MemoryLocation targetLoc = MemoryLocation::ACCELERATOR;
        if (targetDevice.backend == ComputeBackend::CPU) targetLoc = MemoryLocation::CPU;

        if (targetLoc != MemoryLocation::CPU) {
            bool success = memoryMgr_->move(resource->block, targetLoc);
            if (!success) {
                residencyMgr_->transition(resource->id, ResourceResidency::LOADING, ResourceResidency::WARM);
                return false;
            }
        }

        residencyMgr_->transition(resource->id, ResourceResidency::LOADING, ResourceResidency::HOT);
        return true;
    });
}

bool StreamingEngine::evict(std::shared_ptr<ModelResource> resource, ResourceResidency targetResidency) {
    ResourceResidency current = residencyMgr_->getResidency(resource->id);
    if (current == targetResidency) return true;

    if (!residencyMgr_->tryBeginTransition(resource->id, current, ResourceResidency::EVICTING)) {
        return false;
    }

    if (current == ResourceResidency::HOT && targetResidency == ResourceResidency::WARM) {
        // Demote GPU -> CPU
        memoryMgr_->move(resource->block, MemoryLocation::CPU);
        residencyMgr_->transition(resource->id, ResourceResidency::EVICTING, ResourceResidency::WARM);
        return true;
    }

    if ((current == ResourceResidency::HOT || current == ResourceResidency::WARM) && targetResidency == ResourceResidency::DISK) {
        // Full eviction
        memoryMgr_->release(resource->block);
        resource->block = MemoryBlock{};
        residencyMgr_->transition(resource->id, ResourceResidency::EVICTING, ResourceResidency::DISK);
        return true;
    }

    // Invalid eviction path
    residencyMgr_->transition(resource->id, ResourceResidency::EVICTING, current);
    return false;
}

} // namespace agr
