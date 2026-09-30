// src/model/ModelManager.cpp
#include "model/ModelManager.h"
#include <iostream>

namespace agr {

ModelManager::ModelManager(std::shared_ptr<AdaptiveScheduler> scheduler,
                           std::shared_ptr<StreamingEngine> streamingEngine,
                           std::shared_ptr<PrefetchEngine> prefetchEngine,
                           std::shared_ptr<LayerManager> layerMgr,
                           std::shared_ptr<ResourceResidencyManager> residencyMgr)
    : scheduler_(scheduler), streamingEngine_(streamingEngine), 
      prefetchEngine_(prefetchEngine), layerMgr_(layerMgr), residencyMgr_(residencyMgr) {}

uint64_t ModelManager::registerModel(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t id = next_model_id_++;
    models_[id] = name;
    return id;
}

void ModelManager::unregisterModel(uint64_t modelId) {
    std::lock_guard<std::mutex> lock(mutex_);
    models_.erase(modelId);
}

void ModelManager::registerResource(std::shared_ptr<ModelResource> resource) {
    if (!resource) return;
    std::lock_guard<std::mutex> lock(mutex_);
    resources_[resource->id] = resource;
    residencyMgr_->setResidency(resource->id, ResourceResidency::DISK);
}

std::shared_ptr<ModelResource> ModelManager::getResource(uint64_t resourceId) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = resources_.find(resourceId);
    if (it != resources_.end()) return it->second;
    return nullptr;
}

TaskResult ModelManager::executeLayer(uint64_t layerId) {
    auto layer = layerMgr_->getLayer(layerId);
    if (!layer) {
        return TaskResult{false, TaskStatus::FAILED, 0, 0, 0, 0, "Layer not found"};
    }

    // 1. Trigger Prefetch
    prefetchEngine_->onLayerRequested(layerId, [this](uint64_t resId) {
        return getResource(resId);
    });

    // 2. Build Workload abstraction for Scheduler
    Workload w;
    w.id = layer->id;
    w.name = layer->name;
    w.compute_ops_estimate = layer->estimated_compute_ops;
    w.compute_intensive = layer->prefer_gpu;
    
    // We sum up input sizes for scheduler cost model
    w.input_bytes = 0;
    bool inputs_are_hot = true;

    for (uint64_t resId : layer->input_resources) {
        auto res = getResource(resId);
        if (res) {
            w.input_bytes += res->size_bytes;
            ResourceResidency state = residencyMgr_->getResidency(resId);
            if (state != ResourceResidency::HOT) {
                inputs_are_hot = false;
                
                // If it's on DISK, we must load to WARM first
                if (state == ResourceResidency::DISK) {
                    auto future = streamingEngine_->loadToWarmAsync(res);
                    future.wait(); // Blocking wait for now for correctness
                }
            }
        }
    }

    // Set a dummy CPU callback (Phase 6 uses synthetic execution)
    w.cpu_execute = []() { return true; };
    w.vulkan_execute = []() { return true; };
    w.cuda_execute = []() { return true; };

    // Set input_block properties if available so Scheduler knows current location
    // The CostModel in Phase 5 expects `w.input_block.location` and `w.input_block.id`.
    if (!layer->input_resources.empty()) {
        auto firstRes = getResource(layer->input_resources[0]);
        if (firstRes) {
            w.input_block = firstRes->block;
            if (!w.input_block.isValid()) w.input_block.id = 1; // Fake ID so CostModel evaluates transfer cost
            
            ResourceResidency state = residencyMgr_->getResidency(firstRes->id);
            if (state == ResourceResidency::WARM) {
                w.input_block.location = MemoryLocation::CPU;
            } else if (state == ResourceResidency::HOT) {
                w.input_block.location = MemoryLocation::ACCELERATOR;
            } else {
                w.input_block.location = MemoryLocation::DISK;
            }
        }
    }

    // 3. Let Scheduler decide
    TaskHandle handle = scheduler_->schedule(w);
    
    // In a full implementation, we'd wait for scheduler's plan, promote inputs to HOT on chosen device, 
    // then execute. Our Phase 5 AdaptiveScheduler currently auto-submits to the worker internally.
    // To respect Phase 6 streaming pipeline, we simulate promotion before returning result.
    
    // Let's assume the scheduler chose the best device and the worker is doing it.
    // For Phase 6 abstraction, we just manually promote all to HOT before returning, 
    // waiting on the scheduler's dummy execution.
    
    TaskResult result = scheduler_->wait(handle);

    // Promote to HOT (simulating that the resource is now resident on the chosen device)
    // The target device backend is captured inside the Scheduler history, but we'll assume VULKAN or CPU.
    ComputeDevice mockTarget;
    mockTarget.backend = ComputeBackend::CPU; // Fallback simulation
    
    for (uint64_t resId : layer->input_resources) {
        auto res = getResource(resId);
        if (res && residencyMgr_->getResidency(resId) == ResourceResidency::WARM) {
            auto future = streamingEngine_->promoteToHotAsync(res, mockTarget);
            future.wait();
        }
    }

    // Telemetry updates on layers
    layer->access_count++;
    layer->last_access_time = std::chrono::steady_clock::now();

    return result;
}

} // namespace agr
