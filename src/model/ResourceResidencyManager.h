// src/model/ResourceResidencyManager.h
#pragma once

#include "model/ModelTypes.h"
#include <mutex>
#include <unordered_map>

namespace agr {

/// Tracks and manages logical residency states of ModelResources.
class ResourceResidencyManager {
public:
    /// Check the current logical residency of a resource.
    ResourceResidency getResidency(uint64_t resourceId) const;

    /// Attempt to transition a resource's residency state.
    /// Returns true if the transition is valid and was recorded.
    bool transition(uint64_t resourceId, ResourceResidency from, ResourceResidency to);

    /// Force set the residency state (e.g., during initialization or error recovery).
    void setResidency(uint64_t resourceId, ResourceResidency state);

    /// Atomically check if transition is possible, and if so, apply it.
    /// Used for transitions like: if WARM, go to LOADING.
    bool tryBeginTransition(uint64_t resourceId, ResourceResidency expectedFrom, ResourceResidency inProgressState);

private:
    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, ResourceResidency> states_;
};

} // namespace agr
