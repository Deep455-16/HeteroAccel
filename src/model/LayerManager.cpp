// src/model/LayerManager.cpp
#include "model/LayerManager.h"
#include <algorithm>

namespace agr {

void LayerManager::registerLayer(std::shared_ptr<ModelLayer> layer) {
    if (!layer) return;
    std::lock_guard<std::mutex> lock(mutex_);
    layers_[layer->id] = layer;
}

void LayerManager::unregisterLayer(uint64_t layerId) {
    std::lock_guard<std::mutex> lock(mutex_);
    layers_.erase(layerId);
}

std::shared_ptr<ModelLayer> LayerManager::getLayer(uint64_t layerId) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = layers_.find(layerId);
    if (it != layers_.end()) return it->second;
    return nullptr;
}

std::vector<std::shared_ptr<ModelLayer>> LayerManager::getSuccessors(uint64_t layerId) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::shared_ptr<ModelLayer>> successors;
    
    // In a real graph, we would follow forward edges.
    // Here we find layers that explicitly declare `layerId` in `depends_on_layers`.
    // Alternatively, for sequential models, we assume layerId + 1 is the successor.
    
    auto current = layers_.find(layerId);
    if (current == layers_.end()) return successors;
    
    bool has_explicit_dependencies = false;
    for (const auto& [id, layer] : layers_) {
        if (layer->model_id == current->second->model_id) {
            auto it = std::find(layer->depends_on_layers.begin(), layer->depends_on_layers.end(), layerId);
            if (it != layer->depends_on_layers.end()) {
                successors.push_back(layer);
                has_explicit_dependencies = true;
            }
        }
    }
    
    // Fallback for sequential layers if no explicit dependency graph is built
    if (!has_explicit_dependencies) {
        auto next = layers_.find(layerId + 1);
        if (next != layers_.end() && next->second->model_id == current->second->model_id) {
            successors.push_back(next->second);
        }
    }
    
    return successors;
}

std::vector<std::shared_ptr<ModelLayer>> LayerManager::getLayersForModel(uint64_t modelId) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::shared_ptr<ModelLayer>> modelLayers;
    for (const auto& [id, layer] : layers_) {
        if (layer->model_id == modelId) {
            modelLayers.push_back(layer);
        }
    }
    // Sort by ID to assume sequence
    std::sort(modelLayers.begin(), modelLayers.end(), [](const auto& a, const auto& b) {
        return a->id < b->id;
    });
    return modelLayers;
}

} // namespace agr
