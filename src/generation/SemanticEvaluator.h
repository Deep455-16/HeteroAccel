// src/generation/SemanticEvaluator.h
// Phase 33: Property- and rubric-based semantic evaluation harness.
//
// Evaluates correctness, completeness, relevance, formatting and
// refusal/uncertainty behavior (not brittle exact-string snapshots) and records
// full reproducibility metadata (model, backend, hardware, template, budget).

#pragma once

#include "generation/GenerationBudget.h"

#include <string>
#include <vector>

namespace agr {
namespace generation {

struct ConceptGroup {
    std::string              label;
    std::vector<std::string> any_of_lower;
    double                   weight = 1.0;
};

struct SemanticCaseSpec {
    std::string               id;
    std::string               category;       ///< "algorithm", "explanation", "codegen", "refusal"
    TaskProfile               profile = TaskProfile::EXPLANATION;
    std::string               prompt;
    std::string               untrusted_context;
    bool                      expects_code = false;
    std::string               code_language;  ///< e.g. "python"
    std::vector<ConceptGroup> concept_groups;
    std::vector<std::string>  forbidden_lower;
    int                       min_non_ws_chars = 24;
    double                    pass_threshold = 0.70;
};

struct RubricCheck {
    std::string name;
    bool        passed = false;
    double      weight = 1.0;
    std::string detail;
};

struct SemanticCaseResult {
    std::string case_id;
    std::string category;
    std::string backend_id;
    std::string model_path;
    std::string template_id;
    int         template_version = 0;
    std::string template_family;
    std::string finish_reason;
    int         prompt_tokens = 0;
    int         output_tokens = 0;
    double      latency_ms = 0.0;
    float       tokens_per_second = 0.0f;
    std::string output_text;
    std::string extracted_code;
    bool        exec_attempted = false;
    bool        exec_passed = false;
    std::string exec_detail;
    std::vector<RubricCheck> checks;
    double      score = 0.0;
    bool        passed = false;
};

struct SemanticRunSummary {
    std::string model_path;
    std::string backend_id;
    std::string hardware_summary;
    double      temperature = 0.0;
    int         total_cases = 0;
    int         passed_cases = 0;
    double      average_score = 0.0;
    double      regression_threshold = 0.70;
    bool        meets_threshold = false;
    std::vector<SemanticCaseResult> cases;
};

class SemanticEvaluator {
public:
    /// Curated Phase 33 golden task set covering binary search, merge sort,
    /// Fibonacci, nested-loop complexity, safe codegen, and ungrounded refusal.
    static std::vector<SemanticCaseSpec> goldenSuite();

    /// Evaluates one response against a case specification's rubric.
    static SemanticCaseResult evaluate(
        const SemanticCaseSpec& spec,
        const std::string& output_text,
        const std::string& finish_reason,
        const std::string& backend_id = "",
        const std::string& model_path = "",
        const std::string& template_id = "",
        int template_version = 0,
        const std::string& template_family = "");

    /// Extracts the first fenced code block (preferring `lang` if given).
    static std::string extractCodeBlock(const std::string& text,
                                        const std::string& lang = "");

    /// Summarizes a batch of case results and checks the overall threshold.
    static SemanticRunSummary summarize(
        const std::string& model_path,
        const std::string& backend_id,
        const std::string& hardware_summary,
        double temperature,
        std::vector<SemanticCaseResult> cases,
        double regression_threshold = 0.70);

    /// Serializes a list of run summaries (e.g. CPU vs Vulkan comparison) to JSON.
    static std::string toJson(const std::vector<SemanticRunSummary>& runs);
};

} // namespace generation
} // namespace agr
