// src/scheduler/WorkloadRegistry.h
// Phase 9: Tracks multiple concurrent workloads with priority, isolation, and cancellation.
#pragma once
#include "model/ModelTypes.h"
#include "inference/InferenceTypes.h"
#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace agr {

/// Priority levels for workloads
enum class WorkloadPriority {
    BACKGROUND = 0,
    NORMAL     = 1,
    HIGH       = 2,
    CRITICAL   = 3
};

inline const char* toString(WorkloadPriority p) {
    switch(p) {
        case WorkloadPriority::BACKGROUND: return "BACKGROUND";
        case WorkloadPriority::NORMAL:     return "NORMAL";
        case WorkloadPriority::HIGH:       return "HIGH";
        case WorkloadPriority::CRITICAL:   return "CRITICAL";
        default:                           return "UNKNOWN";
    }
}

/// State of a workload in the registry
enum class WorkloadState {
    QUEUED,
    RUNNING,
    COMPLETED,
    CANCELLED,
    FAILED
};

inline const char* toString(WorkloadState s) {
    switch(s) {
        case WorkloadState::QUEUED:    return "QUEUED";
        case WorkloadState::RUNNING:   return "RUNNING";
        case WorkloadState::COMPLETED: return "COMPLETED";
        case WorkloadState::CANCELLED: return "CANCELLED";
        case WorkloadState::FAILED:    return "FAILED";
        default:                       return "UNKNOWN";
    }
}

/// A registered workload entry
struct WorkloadEntry {
    uint64_t             id = 0;
    std::string          name;
    std::string          model_path;
    WorkloadPriority     priority  = WorkloadPriority::NORMAL;
    WorkloadClass        wclass    = WorkloadClass::DEFAULT;
    WorkloadState        state     = WorkloadState::QUEUED;
    std::chrono::steady_clock::time_point queued_at{};
    std::chrono::steady_clock::time_point started_at{};
    std::chrono::steady_clock::time_point finished_at{};
    std::atomic<bool>    cancel_requested{false};
    std::string          error;

    // Non-copyable/movable due to atomic<bool>
    WorkloadEntry() = default;
    WorkloadEntry(const WorkloadEntry& o)
        : id(o.id), name(o.name), model_path(o.model_path),
          priority(o.priority), wclass(o.wclass), state(o.state),
          queued_at(o.queued_at), started_at(o.started_at), finished_at(o.finished_at),
          error(o.error) {
        cancel_requested.store(o.cancel_requested.load(std::memory_order_relaxed),
                               std::memory_order_relaxed);
    }
    WorkloadEntry& operator=(const WorkloadEntry& o) {
        if (this != &o) {
            id = o.id; name = o.name; model_path = o.model_path;
            priority = o.priority; wclass = o.wclass; state = o.state;
            queued_at = o.queued_at; started_at = o.started_at; finished_at = o.finished_at;
            error = o.error;
            cancel_requested.store(o.cancel_requested.load(std::memory_order_relaxed),
                                   std::memory_order_relaxed);
        }
        return *this;
    }
};

/// Thread-safe registry of concurrent workloads.
/// Ensures isolation: one workload's failure does not affect others.
class WorkloadRegistry {
public:
    /// Register a new workload. Returns its ID.
    uint64_t registerWorkload(const std::string& name,
                               const std::string& model_path,
                               WorkloadPriority priority = WorkloadPriority::NORMAL,
                               WorkloadClass wclass = WorkloadClass::DEFAULT);

    /// Get cancel flag pointer for a workload (to pass into InferenceRequest).
    std::atomic<bool>* cancelFlag(uint64_t id);

    /// Signal cancellation of a workload.
    bool cancel(uint64_t id);

    /// Transition state
    void setRunning(uint64_t id);
    void setCompleted(uint64_t id);
    void setFailed(uint64_t id, const std::string& error);
    void setCancelled(uint64_t id);

    /// Query state
    WorkloadState getState(uint64_t id) const;
    bool isCancelled(uint64_t id) const;
    WorkloadClass getClass(uint64_t id) const;

    /// Get all active (queued or running) workloads sorted by priority descending
    std::vector<WorkloadEntry*> activeWorkloadsSorted();

    /// Get a snapshot of all entries (for CLI diagnostics)
    std::vector<WorkloadEntry> snapshot() const;

    /// Remove completed/failed/cancelled entries older than given seconds
    void pruneOld(double max_age_sec = 60.0);

    size_t totalCount() const;
    size_t activeCount() const;

private:
    mutable std::mutex mutex_;
    uint64_t next_id_ = 1;
    std::unordered_map<uint64_t, WorkloadEntry> entries_;
};

} // namespace agr
