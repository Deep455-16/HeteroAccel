// src/collaboration/CollaborativeTypes.h
// Unified Collaborative Multi-Engine Inference Runtime Contracts
//
// Defines the shared request/response, task DAG, intermediate-result exchange,
// engine vs. hardware backend separation, resource management telemetry, and
// result validation/fusion data structures for HeteroAccel.

#pragma once

#include "engine/EngineCapability.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace agr {
namespace collaboration {

/// Distinct inference engines in the collaborative runtime.
/// Kept strictly separate from HardwareBackend (Section 10).
enum class EngineId {
    HYDROXL,    ///< Native HeteroAccel engine ("hydroxl")
    LLAMA_CPP,  ///< Compatibility GGUF engine ("llama.cpp")
    COLIBRI     ///< Specialized structural, complexity & verification engine ("colibri")
};

inline const char* engineIdToString(EngineId id) noexcept {
    switch (id) {
        case EngineId::HYDROXL:   return "hydroxl";
        case EngineId::LLAMA_CPP: return "llama.cpp";
        case EngineId::COLIBRI:   return "colibri";
    }
    return "unknown";
}

/// Hardware execution backends.
/// Kept strictly separate from EngineId (Section 10).
enum class HardwareBackend {
    CPU,
    VULKAN,
    CUDA
};

inline const char* hardwareBackendToString(HardwareBackend b) noexcept {
    switch (b) {
        case HardwareBackend::CPU:    return "CPU";
        case HardwareBackend::VULKAN: return "Vulkan";
        case HardwareBackend::CUDA:   return "CUDA";
    }
    return "CPU";
}

/// High-level query classification driving task decomposition.
enum class QueryCategory {
    CODE_ANALYSIS_AND_OPTIMIZATION,
    COMPLEX_TECHNICAL_QUESTION,
    DEBUGGING_AND_VERIFICATION,
    GENERAL_REASONING_AND_SYNTHESIS,
    SIMPLE_FACTUAL_QA,
    MODEL_METADATA_INSPECTION,
    STATIC_COMPLEXITY_ONLY
};

inline const char* queryCategoryToString(QueryCategory c) noexcept {
    switch (c) {
        case QueryCategory::CODE_ANALYSIS_AND_OPTIMIZATION:
            return "code_analysis_and_optimization";
        case QueryCategory::COMPLEX_TECHNICAL_QUESTION:
            return "complex_technical_question";
        case QueryCategory::DEBUGGING_AND_VERIFICATION:
            return "debugging_and_verification";
        case QueryCategory::GENERAL_REASONING_AND_SYNTHESIS:
            return "general_reasoning_and_synthesis";
        case QueryCategory::SIMPLE_FACTUAL_QA:
            return "simple_factual_qa";
        case QueryCategory::MODEL_METADATA_INSPECTION:
            return "model_metadata_inspection";
        case QueryCategory::STATIC_COMPLEXITY_ONLY:
            return "static_complexity_only";
    }
    return "general_reasoning_and_synthesis";
}

/// Decomposed subtask specialization type.
enum class TaskType {
    STRUCTURAL_AND_REASONING,       ///< Primary fit: HydroXL
    IMPLEMENTATION_AND_EDGE_CASES,  ///< Primary fit: llama.cpp
    COMPLEXITY_AND_OPTIMIZATION,    ///< Primary fit: Colibri
    CONCEPTUAL_EXPLANATION,         ///< Primary fit: HydroXL
    PRACTICAL_EXAMPLES,             ///< Primary fit: llama.cpp
    VERIFICATION_AND_SUMMARY        ///< Primary fit: Colibri
};

inline const char* taskTypeToString(TaskType t) noexcept {
    switch (t) {
        case TaskType::STRUCTURAL_AND_REASONING:      return "structural_and_reasoning";
        case TaskType::IMPLEMENTATION_AND_EDGE_CASES: return "implementation_and_edge_cases";
        case TaskType::COMPLEXITY_AND_OPTIMIZATION:   return "complexity_and_optimization";
        case TaskType::CONCEPTUAL_EXPLANATION:        return "conceptual_explanation";
        case TaskType::PRACTICAL_EXAMPLES:            return "practical_examples";
        case TaskType::VERIFICATION_AND_SUMMARY:      return "verification_and_summary";
    }
    return "structural_and_reasoning";
}

/// Lifecycle state of a task inside the collaborative DAG.
enum class TaskState {
    PENDING,
    READY,
    RUNNING,
    COMPLETED,
    FAILED,
    CANCELLED
};

inline const char* taskStateToString(TaskState s) noexcept {
    switch (s) {
        case TaskState::PENDING:   return "PENDING";
        case TaskState::READY:     return "READY";
        case TaskState::RUNNING:   return "RUNNING";
        case TaskState::COMPLETED: return "COMPLETED";
        case TaskState::FAILED:    return "FAILED";
        case TaskState::CANCELLED: return "CANCELLED";
    }
    return "PENDING";
}

/// Standardized Cross-Engine Intermediate-Result Contract (Section 8).
struct IntermediateResult {
    std::string task_id;
    std::string engine_id;                ///< "hydroxl", "llama.cpp", or "colibri"
    std::string backend_used;             ///< "CPU", "Vulkan", or "CUDA"
    std::string orchestrator_backend_id;  ///< e.g. "hydroxl_cpu", "llamacpp_vulkan", "colibri_cpu"
    std::string structured_result;        ///< Normalized output text
    std::vector<std::string> key_findings;
    std::vector<std::string> supporting_evidence;
    float       confidence_score = 0.0f;  ///< [0.0, 1.0]
    bool        validated = false;
    std::string validation_notes;
    int         prompt_tokens = 0;
    int         tokens_generated = 0;
    float       tokens_per_second = 0.0f;
    double      execution_ms = 0.0;
    size_t      peak_memory_bytes = 0;
    bool        gpu_used = false;
    bool        success = false;
    std::string error;
};

/// Single task node in the collaborative dependency DAG (Section 7).
struct CollaborativeTask {
    std::string task_id;
    std::string description;
    TaskType    task_type = TaskType::STRUCTURAL_AND_REASONING;
    std::string input_context;
    EngineCapabilitySet required_capabilities;

    EngineId        assigned_engine  = EngineId::HYDROXL;
    HardwareBackend assigned_backend = HardwareBackend::CPU;
    std::string     orchestrator_backend_id;
    std::string     assignment_reason;
    std::vector<std::string> rejected_candidates;

    std::vector<std::string> dependencies; ///< Predecessor task_ids
    size_t      wave_index = 0;            ///< Topological wave index (0 = root wave)
    TaskState   state = TaskState::PENDING;

    IntermediateResult output;

    double      queue_wait_ms = 0.0;
    double      execution_ms = 0.0;
    size_t      estimated_memory_bytes = 0;
    size_t      peak_memory_bytes = 0;
    bool        fallback_occurred = false;
    std::string fallback_reason;
};

/// Per-engine participation summary in a collaborative execution.
struct EngineContributionSummary {
    std::string engine_id;
    std::string primary_backend_used;
    int         tasks_assigned = 0;
    int         tasks_completed = 0;
    int         tasks_failed = 0;
    int         total_tokens_generated = 0;
    double      total_execution_ms = 0.0;
    float       average_confidence = 0.0f;
    size_t      peak_memory_bytes = 0;
};

/// Result validation and fusion report (Section 12).
struct FusionReport {
    std::vector<std::string> deduplicated_findings;
    std::vector<std::string> supporting_evidence;
    std::vector<std::string> resolved_conflicts;
    std::vector<std::string> preserved_disagreements;
    int                      raw_findings_count = 0;
    int                      deduplicated_findings_count = 0;
    int                      conflicts_detected = 0;
    bool                     validation_passed = false;
    float                    overall_confidence = 0.0f;
    std::vector<EngineContributionSummary> engine_contributions;
    std::string              synthesized_response;
};

/// Shared resource and memory coordination telemetry (Sections 9 & 11).
struct ResourceCoordinationTelemetry {
    size_t available_ram_bytes = 0;
    size_t effective_ram_budget_bytes = 0;
    size_t estimated_concurrent_neural_bytes = 0;
    size_t peak_resident_bytes = 0;
    size_t actual_process_rss_bytes = 0;      ///< Actual OS process working-set RSS
    size_t actual_process_peak_rss_bytes = 0; ///< Actual OS process peak working-set RSS
    bool   memory_constrained_mode = false;   ///< True when sequential model swapping was used
    bool   concurrent_wave_executed = false;  ///< True when >= 2 tasks ran in parallel
    int    models_loaded_count = 0;
    int    models_evicted_count = 0;
    int    gpu_lock_acquisitions = 0;
};

/// Unified collaborative request contract (Section 3).
struct CollaborativeRequest {
    std::string model_path;
    std::string prompt;
    std::vector<std::pair<std::string, std::string>> conversation_history; ///< (role, content)
    int         max_new_tokens_per_task = 24;
    float       temperature = 0.0f;
    float       top_p = 0.95f;
    int         n_ctx = 512;

    // Optional execution & resource constraints
    size_t      max_ram_budget_bytes = 0;     ///< 0 = auto (80% of available system RAM)
    bool        allow_parallel_waves = true;  ///< False forces strictly sequential execution
    bool        allow_engine_fallback = true; ///< Fallback to surviving engines if one fails
    std::string preferred_gpu_backend;        ///< "" = auto, "cpu", "vulkan", "cuda"

    std::atomic<bool>* cancel_flag = nullptr;
};

/// Streaming callback receiving the unified integrated response chunks.
/// Return false to cancel streaming.
using CollaborativeStreamCallback = std::function<bool(const std::string& text_chunk, int chunk_index)>;

/// Unified collaborative response contract (Section 3).
struct CollaborativeResult {
    bool        success = false;
    std::string error;

    QueryCategory query_category = QueryCategory::GENERAL_REASONING_AND_SYNTHESIS;
    std::string   unified_response;

    std::vector<CollaborativeTask> tasks;
    std::vector<std::vector<std::string>> execution_waves; ///< Task IDs grouped by DAG wave
    FusionReport  fusion;
    ResourceCoordinationTelemetry resource_telemetry;

    double      planning_ms = 0.0;
    double      execution_ms = 0.0;
    double      fusion_ms = 0.0;
    double      total_ms = 0.0;

    /// Helper: returns true if the given engine ("hydroxl", "llama.cpp", "colibri")
    /// completed at least one task in this collaborative run.
    bool engineParticipated(const std::string& engine_id) const {
        for (const auto& c : fusion.engine_contributions) {
            if (c.engine_id == engine_id && c.tasks_completed > 0) {
                return true;
            }
        }
        return false;
    }

    /// Formats a human-readable telemetry & collaboration report.
    std::string formatTelemetryReport() const;
};

} // namespace collaboration
} // namespace agr
