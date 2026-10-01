// src/model/PrefetchEngine.cpp
#include "model/PrefetchEngine.h"

namespace agr {

PrefetchEngine::PrefetchEngine(std::shared_ptr<LayerManager> layerMgr,
                               std::shared_ptr<StreamingEngine> streamingEngine,
                               std::shared_ptr<ResourceResidencyManager> residencyMgr,
                               std::shared_ptr<MemoryManager> memoryMgr)
    : layerMgr_(layerMgr), streamingEngine_(streamingEngine), 
      residencyMgr_(residencyMgr), memoryMgr_(memoryMgr) {}

void PrefetchEngine::setPrefetchDistance(int distance) {
    prefetch_distance_ = distance > 0 ? distance : 0;
}

int PrefetchEngine::onLayerRequested(uint64_t layerId, std::function<std::shared_ptr<ModelResource>(uint64_t)> resourceLookup) {
    if (prefetch_distance_ <= 0) return 0;
    
    // Prune completed prefetches to avoid unbounded growth
    active_prefetches_.erase(
        std::remove_if(active_prefetches_.begin(), active_prefetches_.end(),
            [](const std::future<bool>& f) {
                return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
            }),
        active_prefetches_.end()
    );

    // Check global memory pressure. If CRITICAL or HIGH, don't speculative prefetch.
    auto stats = memoryMgr_->statistics();
    if (stats.cpu_pressure == PressureLevel::HIGH || stats.cpu_pressure == PressureLevel::CRITICAL) {
        return 0;
    }

    int initiated = 0;
    
    // Simple BFS for prefetch distance
    std::vector<std::shared_ptr<ModelLayer>> current_frontier;
    auto initial_layer = layerMgr_->getLayer(layerId);
    if (initial_layer) current_frontier.push_back(initial_layer);
    
    for (int d = 0; d < prefetch_distance_; ++d) {
        std::vector<std::shared_ptr<ModelLayer>> next_frontier;
        for (const auto& l : current_frontier) {
            auto successors = layerMgr_->getSuccessors(l->id);
            for (const auto& succ : successors) {
                next_frontier.push_back(succ);
                
                // Prefetch input resources for this successor
                for (uint64_t resId : succ->input_resources) {
                    auto res = resourceLookup(resId);
                    if (res) {
                        ResourceResidency state = residencyMgr_->getResidency(resId);
                        if (state == ResourceResidency::DISK) {
                            // Keep the future alive so the async task actually runs
                            active_prefetches_.push_back(streamingEngine_->loadToWarmAsync(res));
                            initiated++;
                        }
                    }
                }
            }
        }
        current_frontier = next_frontier;
        if (current_frontier.empty()) break;
    }

    return initiated;
}

} // namespace agr
