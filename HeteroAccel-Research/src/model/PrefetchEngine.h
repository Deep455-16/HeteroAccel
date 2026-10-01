// src/model/PrefetchEngine.h
#pragma once

#include "model/LayerManager.h"
#include "model/StreamingEngine.h"
#include "model/ResourceResidencyManager.h"
#include "mem/MemoryManager.h"
#include <memory>
#include <vector>
#include <future>
#include <algorithm>
#include <chrono>

namespace agr {

/// Predicts and requests resources that will be needed soon.
class PrefetchEngine {
public:
    PrefetchEngine(std::shared_ptr<LayerManager> layerMgr,
                   std::shared_ptr<StreamingEngine> streamingEngine,
                   std::shared_ptr<ResourceResidencyManager> residencyMgr,
                   std::shared_ptr<MemoryManager> memoryMgr);

    /// Set prefetch depth (0 = disable).
    void setPrefetchDistance(int distance);

    /// Call this when a layer is about to execute to trigger prefetch of successors.
    /// Returns the number of prefetch requests initiated.
    /// Needs a lookup callback for Resources to know sizes and locations.
    int onLayerRequested(uint64_t layerId, std::function<std::shared_ptr<ModelResource>(uint64_t)> resourceLookup);

private:
    std::shared_ptr<LayerManager> layerMgr_;
    std::shared_ptr<StreamingEngine> streamingEngine_;
    std::shared_ptr<ResourceResidencyManager> residencyMgr_;
    std::shared_ptr<MemoryManager> memoryMgr_;

    int prefetch_distance_ = 1;
    std::vector<std::future<bool>> active_prefetches_;
};

} // namespace agr
