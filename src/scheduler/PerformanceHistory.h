// src/scheduler/PerformanceHistory.h
// Phase 8: Hardware-Specific Performance History and Online Learning
#pragma once

#include "backend/ComputeDevice.h"
#include "profiler/IProfiler.h"
#include "scheduler/SchedulerTypes.h"
#include <unordered_map>
#include <mutex>
#include <string>
#include <vector>

namespace agr {

/// Key for hardware-specific, workload-specific performance tracking
struct ProfileKey {
    std::string hardware_id;
    ComputeBackend backend;
    std::string workload_type;
    std::string model_name;

    bool operator==(const ProfileKey& other) const {
        return hardware_id == other.hardware_id &&
               backend == other.backend &&
               workload_type == other.workload_type &&
               model_name == other.model_name;
    }
};

} // namespace agr

namespace std {
template <>
struct hash<agr::ProfileKey> {
    size_t operator()(const agr::ProfileKey& k) const {
        size_t h1 = hash<std::string>{}(k.hardware_id);
        size_t h2 = hash<int>{}(static_cast<int>(k.backend));
        size_t h3 = hash<std::string>{}(k.workload_type);
        size_t h4 = hash<std::string>{}(k.model_name);
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
    }
};
}

namespace agr {

/// Historical statistics for a specific workload on a specific backend
struct HistoricalStats {
    uint64_t sample_count = 0;
    double   ema_duration_ms = 0.0;
    double   variance = 0.0;
    double   min_duration_ms = -1.0;
    double   max_duration_ms = -1.0;
    
    uint64_t failure_count = 0;
    bool     regression_detected = false;
    
    // Derived confidence (0.0 to 1.0)
    double confidence() const {
        if (sample_count == 0) return 0.0;
        if (sample_count < 5) return 0.2 * sample_count;
        return 1.0;
    }
    
    // Reliability penalty based on failures
    double reliability() const {
        if (sample_count == 0) return 1.0;
        double fail_rate = static_cast<double>(failure_count) / sample_count;
        return 1.0 - fail_rate;
    }
};

/// Phase 8: Maintains hardware-specific EMA of backend performance metrics,
/// with regression detection, failure learning, and confidence.
class PerformanceHistory {
public:
    PerformanceHistory();

    /// Phase 5 legacy path
    void recordCompletion(ComputeBackend backend, const Workload& workload, const TaskResult& result);
    double getEstimatedOpsPerMs(ComputeBackend backend) const;
    double getEstimatedBandwidthBytesPerMs(ComputeBackend backend) const;

    /// Phase 8 path: Record an event from the Profiler
    void recordEvent(const ProfileKey& key, const ProfileEvent& event);

    /// Phase 8 path: Get the predicted duration for a workload
    /// Returns negative if no history.
    double predictDurationMs(const ProfileKey& key) const;

    /// Get full stats
    HistoricalStats getStats(const ProfileKey& key) const;
    
    /// Extract all stats for persistence
    std::unordered_map<ProfileKey, HistoricalStats> getAllStats() const;
    
    /// Restore stats from persistence
    void setAllStats(const std::unordered_map<ProfileKey, HistoricalStats>& stats);

private:
    void updateStats(HistoricalStats& stats, double duration_ms, bool success);

    mutable std::mutex mutex_;
    
    // Phase 5 legacy metrics
    struct BackendMetrics {
        double ema_ops_per_ms        = 1000.0;
        double ema_bandwidth_per_ms  = 1024.0 * 1024.0;
        uint64_t sample_count        = 0;
    };
    std::unordered_map<ComputeBackend, BackendMetrics> legacy_metrics_;
    
    // Phase 8 specific profiles
    std::unordered_map<ProfileKey, HistoricalStats> profiles_;
    
    double alpha_ = 0.2; 
};

} // namespace agr
