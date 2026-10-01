// src/scheduler/PerformanceHistory.cpp
#include "scheduler/PerformanceHistory.h"
#include <iostream>

namespace agr {

PerformanceHistory::PerformanceHistory() {
    // Seed initial defaults
    metrics_[ComputeBackend::CPU].ema_ops_per_ms = 500.0;
    metrics_[ComputeBackend::VULKAN].ema_ops_per_ms = 2500.0;
    metrics_[ComputeBackend::CUDA].ema_ops_per_ms = 4000.0;

    // Seed bandwidth defaults (MB/ms)
    metrics_[ComputeBackend::CPU].ema_bandwidth_per_ms = 20.0 * 1024.0 * 1024.0; // CPU is fast to itself
    metrics_[ComputeBackend::VULKAN].ema_bandwidth_per_ms = 10.0 * 1024.0 * 1024.0; // PCIE overhead
    metrics_[ComputeBackend::CUDA].ema_bandwidth_per_ms = 12.0 * 1024.0 * 1024.0; 
}

void PerformanceHistory::recordCompletion(ComputeBackend backend, const Workload& workload, const TaskResult& result) {
    if (!result.success || result.compute_ms <= 0.0) return;

    std::lock_guard<std::mutex> lock(mutex_);
    auto& m = metrics_[backend];

    // Compute ops/ms
    double current_ops_per_ms = static_cast<double>(workload.compute_ops_estimate) / result.compute_ms;
    
    // Compute bandwidth if there was a transfer
    double total_bytes = static_cast<double>(workload.input_bytes + workload.output_bytes);
    double current_bw_per_ms = (result.transfer_ms > 0.0 && total_bytes > 0.0) 
                             ? (total_bytes / result.transfer_ms) 
                             : m.ema_bandwidth_per_ms;

    // EMA Update
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
    auto it = metrics_.find(backend);
    if (it != metrics_.end()) return it->second.ema_ops_per_ms;
    return 1000.0;
}

double PerformanceHistory::getEstimatedBandwidthBytesPerMs(ComputeBackend backend) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = metrics_.find(backend);
    if (it != metrics_.end()) return it->second.ema_bandwidth_per_ms;
    return 1024.0 * 1024.0; // 1 GB/s default
}

} // namespace agr
