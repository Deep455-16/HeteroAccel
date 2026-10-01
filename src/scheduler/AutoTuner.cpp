// src/scheduler/AutoTuner.cpp
#include "scheduler/AutoTuner.h"
#include <iostream>
#include <algorithm>

namespace agr {

AutoTuner::AutoTuner(const PerformanceHistory& history, const HardwareInfo& hw)
    : history_(history), hw_(hw) {}

TuningConfig AutoTuner::suggestConfiguration(const ProfileKey& key, bool gpu_available, int max_threads) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Check if we have state
    auto it = states_.find(key);
    if (it == states_.end()) {
        // Cold start
        WorkloadState state;
        state.current_best.n_gpu_layers = gpu_available ? 99 : 0;
        state.current_best.n_threads = max_threads / 2 > 0 ? max_threads / 2 : 1;
        state.stable_samples = 0;
        states_[key] = state;
        return state.current_best;
    }

    auto& state = it->second;

    HistoricalStats hist = history_.getStats(key);

    // If regression detected in history, trigger re-exploration
    if (hist.regression_detected && state.stable_samples > min_samples_before_tune_) {
        state.stable_samples = 0;
        state.is_exploring = false;
        state.current_best_duration = -1.0;
    }

    if (state.is_exploring) {
        return state.exploring;
    }

    // Should we explore?
    if (state.stable_samples >= min_samples_before_tune_) {
        state.is_exploring = true;
        state.exploring = state.current_best;

        // Simple exploration strategy: try changing thread count
        // (In a real system, we might explore layer counts if memory allows)
        if (state.exploring.n_threads < max_threads) {
            state.exploring.n_threads++;
        } else if (state.exploring.n_threads > 1) {
            state.exploring.n_threads--;
        }
        
        return state.exploring;
    }

    return state.current_best;
}

void AutoTuner::recordResult(const ProfileKey& key, const TuningConfig& config, double duration_ms, bool success) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = states_.find(key);
    if (it == states_.end()) return;
    auto& state = it->second;

    if (!success) {
        if (state.is_exploring) {
            // Rollback immediately
            state.is_exploring = false;
            state.stable_samples = 0; // cooldown
        }
        return;
    }

    if (state.is_exploring) {
        if (state.current_best_duration < 0) {
            state.current_best_duration = duration_ms;
            state.is_exploring = false;
        } else {
            // Compare
            double improvement = (state.current_best_duration - duration_ms) / state.current_best_duration;
            if (improvement > improvement_threshold_) {
                // Keep the new config
                state.current_best = state.exploring;
                state.current_best_duration = duration_ms;
            } else {
                // Rollback
                // state.current_best remains the same
            }
            state.is_exploring = false;
            state.stable_samples = 0; // reset counter after tuning
        }
    } else {
        // Update baseline
        if (state.current_best_duration < 0 || state.stable_samples == 0) {
            state.current_best_duration = duration_ms;
        } else {
            // EMA for the baseline duration
            state.current_best_duration = 0.2 * duration_ms + 0.8 * state.current_best_duration;
        }
        state.stable_samples++;
    }
}

} // namespace agr
