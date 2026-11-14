// src/collaboration/ResultFusionEngine.h
// Cross-Engine Intermediate Result Validation, Deduplication, Conflict
// Resolution, and Coherent Response Synthesis (Section 12).

#pragma once

#include "collaboration/CollaborativeTypes.h"

#include <string>
#include <vector>

namespace agr {
namespace collaboration {

class ResultFusionEngine {
public:
    ResultFusionEngine() = default;

    /// Populates key_findings, supporting_evidence, confidence_score, and
    /// validated status on an IntermediateResult from raw engine output.
    static void enrichIntermediateResult(const CollaborativeTask& task,
                                         IntermediateResult& result);

    /// Validates, deduplicates, resolves conflicts, and fuses intermediate
    /// results from HydroXL, llama.cpp, and Colibri into a single coherent
    /// FusionReport and unified response.
    static FusionReport fuse(QueryCategory category,
                             const std::string& original_prompt,
                             const std::vector<CollaborativeTask>& tasks);

    /// Computes word-level Jaccard similarity in [0.0, 1.0] between two strings.
    static float computeSimilarity(const std::string& a, const std::string& b);
};

} // namespace collaboration
} // namespace agr
