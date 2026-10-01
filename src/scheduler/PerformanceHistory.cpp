// src/scheduler/PerformanceHistory.cpp
// Phase 8: PerformanceHistory implementation
#include "scheduler/PerformanceHistory.h"
#include <cmath>
#include <algorithm>

namespace agr {

PerformanceHistory::PerformanceHistory() {
    // Seed initial defaults for Phase 5
    legacy_metrics_[ComputeBackend::CPU].ema_ops_per_ms = 500.0;
    legacy_metrics_[ComputeBackend::VULKAN].ema_ops_per_ms = 2500.0;
    legacy_metrics_[ComputeBackend::CUDA].ema_ops_per_ms = 4000.0;

    legacy_metrics_[ComputeBackend::CPU].ema_bandwidth_per_ms = 20.0 * 1024.0 * 1024.0;
    legacy_metrics_[ComputeBackend::VULKAN].ema_bandwidth_per_ms = 10.0 * 1024.0 * 1024.0;
    legacy_metrics_[ComputeBackend::CUDA].ema_bandwidth_per_ms = 12.0 * 1024.0 * 1024.0; 
}

void PerformanceHistory::recordCompletion(ComputeBackend backend, const Workload& workload, const TaskResult& result) {
    if (!result.success || result.compute_ms <= 0.0) return;

    std::lock_guard<std::mutex> lock(mutex_);
    auto& m = legacy_metrics_[backend];

    double current_ops_per_ms = static_cast<double>(workload.compute_ops_estimate) / result.compute_ms;
    
    double total_bytes = static_cast<double>(workload.input_bytes + workload.output_bytes);
    double current_bw_per_ms = (result.transfer_ms > 0.0 && total_bytes > 0.0) 
                             ? (total_bytes / result.transfer_ms) 
                             : m.ema_bandwidth_per_ms;

    if (m.sample_count == 0) {
        m.ema_ops_per_ms = current_ops_per_ms;
        m.ema_bandwidth_per_ms = current_bw_per_ms;
    } else {
        m.ema_ops_per_ms = (alpha_ * current_ops_per_ms) + ((1.0 - alpha_) * m.ema_ops_per_ms);
        m.ema_bandwidth_per_ms = (alpha_ * current_bw_per_ms) + ((1.0 - alpha_) * m.ema_bandwidth_per_ms);
    }
    m.sample_count++;
}

double PerformanceHistory::getEstimatedOpsPerMs(ComputeBackend backend) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = legacy_metrics_.find(backend);
    if (it != legacy_metrics_.end()) return it->second.ema_ops_per_ms;
    return 1000.0;
}

double PerformanceHistory::getEstimatedBandwidthBytesPerMs(ComputeBackend backend) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = legacy_metrics_.find(backend);
    if (it != legacy_metrics_.end()) return it->second.ema_bandwidth_per_ms;
    return 1024.0 * 1024.0;
}

// Phase 8 additions
void PerformanceHistory::recordEvent(const ProfileKey& key, const ProfileEvent& event) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& stats = profiles_[key];
    updateStats(stats, event.execution_duration_ms, event.success);
}

void PerformanceHistory::updateStats(HistoricalStats& stats, double duration_ms, bool success) {
    stats.sample_count++;
    if (!success) {
        stats.failure_count++;
        return;
    }

    if (stats.sample_count == 1 || stats.ema_duration_ms == 0.0) {
        stats.ema_duration_ms = duration_ms;
        stats.min_duration_ms = duration_ms;
        stats.max_duration_ms = duration_ms;
        stats.variance = 0.0;
    } else {
        // Welford's online variance algorithm adapted for EMA
        double old_ema = stats.ema_duration_ms;
        double diff = duration_ms - old_ema;
        stats.ema_duration_ms += alpha_ * diff;
        stats.variance = (1.0 - alpha_) * (stats.variance + alpha_ * diff * diff);

        if (duration_ms < stats.min_duration_ms) stats.min_duration_ms = duration_ms;
        if (duration_ms > stats.max_duration_ms) stats.max_duration_ms = duration_ms;

        // Regression detection: compare against pre-update EMA to avoid dampening
        double stddev = std::sqrt(stats.variance);
        if (stats.sample_count > 5) {
            if (stddev > 0.0 && duration_ms > old_ema + 2.0 * stddev) {
                stats.regression_detected = true;
            } else if (stddev == 0.0 && duration_ms > old_ema * 1.5) {
                // Edge case: variance is exactly 0 (all same values), any big deviation counts
                stats.regression_detected = true;
            } else if (duration_ms < old_ema + stddev) {
                stats.regression_detected = false; // Recovered
            }
        }
    }
}

double PerformanceHistory::predictDurationMs(const ProfileKey& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = profiles_.find(key);
    if (it != profiles_.end() && it->second.sample_count > 0) {
        return it->second.ema_duration_ms;
    }
    return -1.0; // No history
}

HistoricalStats PerformanceHistory::getStats(const ProfileKey& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = profiles_.find(key);
    if (it != profiles_.end()) {
        return it->second;
    }
    return HistoricalStats{};
}

std::unordered_map<ProfileKey, HistoricalStats> PerformanceHistory::getAllStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return profiles_;
}

void PerformanceHistory::setAllStats(const std::unordered_map<ProfileKey, HistoricalStats>& stats) {
    std::lock_guard<std::mutex> lock(mutex_);
    profiles_ = stats;
}

} // namespace agr
