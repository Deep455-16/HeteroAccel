// src/orchestration/EngineOrchestrator.h
// Phase 21: Engine Orchestrator

#pragma once

#include "orchestration/BackendCapabilities.h"
#include "orchestration/HardwareDetector.h"
#include "orchestration/IInferenceEngine.h"

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <iostream>

namespace agr {
namespace orchestration {

class EngineOrchestrator {
public:
    struct SelectionRequest {
        std::string model_path;
        std::string model_arch;       // from GGUF metadata or empty
        int quant_type = 0;           // GGML type ID
        size_t model_size_bytes = 0;
        size_t n_ctx = 512;
        bool prefer_latency = false;  // true = minimize TTFT, false = maximize throughput
        bool allow_cpu_fallback = true;
        std::string forced_backend;   // empty = auto-select
    };

    struct SelectionResult {
        std::string selected_backend;
        std::string reason;           // human-readable explanation
        std::vector<std::string> rejected_backends;  // with reasons
        bool is_fallback = false;
        bool used_benchmark_history = false;
    };

    using EngineFactory = std::function<std::shared_ptr<IInferenceEngine>()>;

    static EngineOrchestrator& instance();

    void register_backend(BackendCapabilities caps);
    void register_engine_factory(const std::string& backend_id, EngineFactory factory);

    SelectionResult select_backend(const SelectionRequest& req) const;

    // Create and initialize the selected engine
    // Returns nullptr + populates error on failure
    std::shared_ptr<IInferenceEngine> create_engine(
        const SelectionResult& sel,
        const SelectionRequest& req,
        std::string& error_out
    );

    // Record observed performance for future selection decisions
    void record_performance(const std::string& backend_id,
                            float tokens_per_second,
                            size_t peak_rss_bytes);

    void print_diagnostics(std::ostream& out) const;

    // Testing helpers
    void reset();
    std::vector<BackendCapabilities> backends() const;
    std::map<std::string, float> perf_history() const;

private:
    EngineOrchestrator();

    std::vector<BackendCapabilities> backends_;
    std::map<std::string, float> perf_history_;  // backend_id -> best observed tok/s
    std::map<std::string, size_t> rss_history_;   // backend_id -> peak rss
    std::map<std::string, EngineFactory> factories_;
    mutable std::mutex mu_;
};

} // namespace orchestration
} // namespace agr
