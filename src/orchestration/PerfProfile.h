// src/orchestration/PerfProfile.h
// Phase 24: Persistent Performance Profiles for HeteroAccel

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace agr {
namespace orchestration {

struct PerfRecord {
    std::string backend_id;
    std::string model_hash;    // SHA-like: file size + first 64 bytes hash
    uint32_t    quant_type = 0;
    uint32_t    n_layers = 0;
    uint32_t    n_embd = 0;
    float       tokens_per_second = 0.0f;
    uint64_t    timestamp_unix = 0;  // seconds since epoch
    uint32_t    sample_count = 0;
};

class PerfProfile {
public:
    PerfProfile() = default;

    // Load from disk (returns empty profile if file missing/corrupt)
    static PerfProfile load(const std::string& path);

    // Save to disk (line-oriented text format)
    bool save(const std::string& path) const;

    // Record a new measurement (updates existing record if same key)
    void record(const std::string& backend_id, const std::string& model_hash,
                uint32_t quant_type, uint32_t n_layers, uint32_t n_embd,
                float tokens_per_second);

    // Query best known backend for a workload (empty = unknown)
    std::string best_backend(const std::string& model_hash,
                             uint32_t quant_type, uint32_t n_layers) const;

    // Clear all records
    void reset();

    size_t record_count() const { return records_.size(); }

    const std::vector<PerfRecord>& records() const { return records_; }

    static std::string compute_model_hash(const std::string& path);
    static std::string default_profile_path();

private:
    std::vector<PerfRecord> records_;
};

} // namespace orchestration
} // namespace agr
