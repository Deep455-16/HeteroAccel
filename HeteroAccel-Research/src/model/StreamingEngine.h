// src/model/StreamingEngine.h
#pragma once

#include "model/ModelTypes.h"
#include "model/IStorageBackend.h"
#include "model/ResourceResidencyManager.h"
#include "mem/MemoryManager.h"
#include "scheduler/AdaptiveScheduler.h"
#include <future>
#include <memory>

namespace agr {

/// Handles the movement of resources between DISK, RAM (WARM), and ACCELERATOR (HOT).
class StreamingEngine {
public:
    StreamingEngine(std::shared_ptr<IStorageBackend> storage,
                    std::shared_ptr<ResourceResidencyManager> residencyMgr,
                    std::shared_ptr<MemoryManager> memoryMgr,
                    std::shared_ptr<AdaptiveScheduler> scheduler);

    /// Asynchronously load a resource from DISK to WARM (System RAM).
    std::future<bool> loadToWarmAsync(std::shared_ptr<ModelResource> resource);

    /// Asynchronously promote a resource from WARM to HOT (Device).
    /// Target device is selected by Phase 5 Scheduler.
    std::future<bool> promoteToHotAsync(std::shared_ptr<ModelResource> resource, ComputeDevice targetDevice);

    /// Demote a resource back to WARM or DISK and free memory.
    bool evict(std::shared_ptr<ModelResource> resource, ResourceResidency targetResidency);

private:
    std::shared_ptr<IStorageBackend> storage_;
    std::shared_ptr<ResourceResidencyManager> residencyMgr_;
    std::shared_ptr<MemoryManager> memoryMgr_;
    std::shared_ptr<AdaptiveScheduler> scheduler_;
};

} // namespace agr
