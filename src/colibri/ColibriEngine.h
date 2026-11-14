// src/colibri/ColibriEngine.h
// Specialized Structural, Complexity, and Verification Engine for HeteroAccel
//
// Colibri is the third engine in HeteroAccel's collaborative multi-engine
// architecture alongside HydroXL (native engine) and llama.cpp (compatibility
// engine). Unlike general neural autoregressive engines, Colibri is a
// lightweight, deterministic CPU-based structural analysis, complexity
// estimation, static verification, and cross-engine consistency engine.
//
// Honest Capability Reporting:
// - Supported Hardware Backend: CPU only (does NOT claim Vulkan or CUDA kernels)
// - Model Inspection: Parses GGUF header, architecture hyperparameters, and
//   tokenizer metadata via GGUFLoader without duplicating multi-GB neural weights
// - Specialized Workloads:
//   1. Code & algorithmic complexity analysis (time/space complexity, loop
//      nesting, memory access patterns, synchronization overhead)
//   2. Static verification & edge-case auditing (bounds checks, resource
//      lifetimes, numerical stability hazards, balanced syntax)
//   3. Cross-engine intermediate result validation & structured summary

#pragma once

#include "engine/IExecutionEngine.h"
#include "orchestration/IInferenceEngine.h"
#include "hydroxl/HydroXLTokenizer.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace agr {

struct ColibriAnalysisReport {
    std::string task_category;               // "complexity_optimization", "verification", "structural_summary"
    std::string estimated_time_complexity;   // e.g. "O(N)", "O(N^2)", "O(N log N)"
    std::string estimated_space_complexity;  // e.g. "O(1)", "O(N)"
    int         max_loop_nesting_depth = 0;
    bool        syntax_balanced = true;
    std::vector<std::string> key_findings;
    std::vector<std::string> supporting_evidence;
    std::vector<std::string> detected_risks;
    std::vector<std::string> optimization_actions;
    float       confidence_score = 0.0f;
    bool        validated = false;
    std::string formatted_summary;
};

class ColibriEngine final : public IExecutionEngine {
public:
    static void registerInRegistry();

    ColibriEngine();
    ~ColibriEngine() override;

    ColibriEngine(const ColibriEngine&) = delete;
    ColibriEngine& operator=(const ColibriEngine&) = delete;

    // Identity & capabilities
    EngineIdentity identity() const override;
    EngineCapabilitySet capabilities() const override;

    // Engine lifecycle
    bool initialize() override;
    void shutdown() override;
    bool isInitialized() const override;

    // Model lifecycle (lightweight GGUF metadata + tokenizer inspection)
    bool loadModel(const ModelDescriptor& descriptor) override;
    void unloadModel() override;
    bool isModelLoaded() const override;
    ModelDescriptor currentModel() const override;

    // Execution
    ExecutionResult execute(const ExecutionContext& ctx) override;
    ExecutionResult executeStreaming(const ExecutionContext& ctx,
                                     TokenCallback cb) override;

    // Direct structured analysis entry point for CollaborativeExecutionOrchestrator
    ColibriAnalysisReport analyze(const std::string& query,
                                  const std::string& upstream_context = "") const;

    EngineError lastEngineError() const override;

    // Inspected model metadata accessors
    std::string modelArchitecture() const;
    uint32_t modelLayers() const;
    uint32_t modelEmbeddingDim() const;
    uint32_t modelVocabSize() const;
    size_t estimatedWorkspaceBytes() const;

private:
    mutable std::mutex mutex_;
    bool initialized_ = false;
    bool model_loaded_ = false;
    ModelDescriptor current_descriptor_;
    EngineError last_error_;

    std::string arch_;
    uint32_t n_layers_ = 0;
    uint32_t n_embd_ = 0;
    uint32_t n_heads_ = 0;
    uint32_t n_kv_heads_ = 0;
    uint32_t n_ff_ = 0;
    uint32_t vocab_size_ = 0;
    uint32_t context_len_ = 0;
    size_t file_size_bytes_ = 0;
    std::unique_ptr<hydroxl::HydroXLTokenizer> tokenizer_;
};

namespace orchestration {

/// IInferenceEngine adapter exposing ColibriEngine to EngineOrchestrator and
/// CollaborativeExecutionOrchestrator under backend_id "colibri_cpu".
class ColibriEngineAdapter final : public IInferenceEngine {
public:
    ColibriEngineAdapter();
    ~ColibriEngineAdapter() override;

    bool initialize(const std::string& model_path, std::string& error) override;
    InferenceResult run(const InferenceRequest& req) override;
    bool run_stream(const InferenceRequest& req,
                    StreamCallback cb,
                    InferenceResult& result_out) override;
    bool supports_streaming() const override { return true; }
    void shutdown() override;
    std::string backend_id() const override { return "colibri_cpu"; }
    bool is_ready() const override;

    const ColibriEngine& engine() const { return engine_; }
    ColibriAnalysisReport last_report() const;

private:
    mutable std::mutex mu_;
    ColibriEngine engine_;
    ColibriAnalysisReport last_report_;
};

} // namespace orchestration
} // namespace agr
