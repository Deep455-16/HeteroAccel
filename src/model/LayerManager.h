// src/model/LayerManager.h
#pragma once

#include "model/ModelTypes.h"
#include <unordered_map>
#include <memory>
#include <mutex>
#include <vector>

namespace agr {

/// Maintains layer metadata and dependency relationships.
class LayerManager {
public:
    void registerLayer(std::shared_ptr<ModelLayer> layer);
    void unregisterLayer(uint64_t layerId);

    std::shared_ptr<ModelLayer> getLayer(uint64_t layerId) const;

    /// Get the next sequential or dependent layers for prefetching.
    std::vector<std::shared_ptr<ModelLayer>> getSuccessors(uint64_t layerId) const;

    /// Get all layers for a specific model ID, typically sorted by ID/execution order.
    std::vector<std::shared_ptr<ModelLayer>> getLayersForModel(uint64_t modelId) const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, std::shared_ptr<ModelLayer>> layers_;
};

} // namespace agr
