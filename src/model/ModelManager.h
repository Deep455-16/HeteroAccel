// src/model/ModelManager.h
#pragma once

#include "model/ModelTypes.h"
#include "model/LayerManager.h"
#include "model/ResourceResidencyManager.h"
#include "model/StreamingEngine.h"
#include "model/PrefetchEngine.h"
#include "scheduler/AdaptiveScheduler.h"

#include <memory>
#include <unordered_map>
#include <mutex>
#include <string>

namespace agr {

/// High-level entry point for managing models and coordinating subsystems.
class ModelManager {
public:
    ModelManager(std::shared_ptr<AdaptiveScheduler> scheduler,
                 std::shared_ptr<StreamingEngine> streamingEngine,
                 std::shared_ptr<PrefetchEngine> prefetchEngine,
                 std::shared_ptr<LayerManager> layerMgr,
                 std::shared_ptr<ResourceResidencyManager> residencyMgr);

    uint64_t registerModel(const std::string& name);
    void unregisterModel(uint64_t modelId);

    void registerResource(std::shared_ptr<ModelResource> resource);
    std::shared_ptr<ModelResource> getResource(uint64_t resourceId) const;

    /// Executes a layer.
    /// This triggers residency checks, prefetching, and scheduler dispatch.
    /// Returns the result of the scheduled execution.
    TaskResult executeLayer(uint64_t layerId);

private:
    std::shared_ptr<AdaptiveScheduler> scheduler_;
    std::shared_ptr<StreamingEngine> streamingEngine_;
    std::shared_ptr<PrefetchEngine> prefetchEngine_;
    std::shared_ptr<LayerManager> layerMgr_;
    std::shared_ptr<ResourceResidencyManager> residencyMgr_;

    mutable std::mutex mutex_;
    uint64_t next_model_id_ = 1;
    
    std::unordered_map<uint64_t, std::string> models_;
    std::unordered_map<uint64_t, std::shared_ptr<ModelResource>> resources_;
};

} // namespace agr
