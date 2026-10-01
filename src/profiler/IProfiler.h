// src/profiler/IProfiler.h
// Phase 8: Unified Profiling System interface
#pragma once

#include "backend/ComputeDevice.h"
#include <string>
#include <cstdint>
#include <vector>

namespace agr {

/// Characteristics of a workload to allow performance generalization.
struct WorkloadCharacteristics {
    std::string type;         ///< "llm-inference", "vector-add", etc.
    std::string model_name;   ///< e.g., "Qwen2.5-0.5B"
    size_t      input_size  = 0;
    size_t      output_size = 0;
    int         token_count = 0;
};

/// A complete profile record for a single execution event.
struct ProfileEvent {
    size_t      workload_id = 0;
    std::string workload_type;
    std::string model_name;
    
    ComputeBackend backend = ComputeBackend::CPU;
    std::string    device_name;
    
    double start_timestamp_ms = 0.0;
    double end_timestamp_ms   = 0.0;
    
    double execution_duration_ms = 0.0;
    double queue_time_ms         = 0.0;
    double transfer_time_ms      = 0.0;
    
    size_t memory_usage_bytes    = 0;
    
    // LLM-specific telemetry
    int    input_tokens          = 0;
    int    output_tokens         = 0;
    double prompt_processing_ms  = 0.0;
    double generation_ms         = 0.0;
    double ttft_ms               = 0.0;
    double tokens_per_sec        = 0.0;
    
    // Auto-tuning & Scheduler
    int    scheduler_gpu_layers  = 0;
    int    scheduler_threads     = 0;
    double predicted_cost_ms     = 0.0;
    double actual_cost_ms        = 0.0;
    
    bool   success               = false;
    bool   fallback_occurred     = false;
    std::string error_message;
};

class IProfiler {
public:
    virtual ~IProfiler() = default;

    /// Record a single execution event.
    virtual void recordEvent(const ProfileEvent& event) = 0;

    /// Retrieve all recorded events.
    virtual std::vector<ProfileEvent> getEvents() const = 0;

    /// Save the current performance profile to disk.
    virtual bool saveProfile(const std::string& path) = 0;

    /// Load the performance profile from disk.
    virtual bool loadProfile(const std::string& path) = 0;
    
    /// Reset the profile.
    virtual void reset() = 0;
};

} // namespace agr
