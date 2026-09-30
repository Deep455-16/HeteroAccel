// src/model/ResourceResidencyManager.cpp
#include "model/ResourceResidencyManager.h"

namespace agr {

ResourceResidency ResourceResidencyManager::getResidency(uint64_t resourceId) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = states_.find(resourceId);
    if (it != states_.end()) {
        return it->second;
    }
    return ResourceResidency::DISK; // Default assumption if not found
}

bool ResourceResidencyManager::transition(uint64_t resourceId, ResourceResidency from, ResourceResidency to) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = states_.find(resourceId);
    ResourceResidency current = (it != states_.end()) ? it->second : ResourceResidency::DISK;

    if (current != from) {
        return false; // Invalid transition start state
    }

    states_[resourceId] = to;
    return true;
}

void ResourceResidencyManager::setResidency(uint64_t resourceId, ResourceResidency state) {
    std::lock_guard<std::mutex> lock(mutex_);
    states_[resourceId] = state;
}

bool ResourceResidencyManager::tryBeginTransition(uint64_t resourceId, ResourceResidency expectedFrom, ResourceResidency inProgressState) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = states_.find(resourceId);
    ResourceResidency current = (it != states_.end()) ? it->second : ResourceResidency::DISK;

    if (current == expectedFrom) {
        states_[resourceId] = inProgressState;
        return true;
    }
    return false;
}

} // namespace agr
