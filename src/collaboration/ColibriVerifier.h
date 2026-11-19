// src/collaboration/ColibriVerifier.h
// Phase 33: Optional Colibri adapter health gate, query relevance check,
// and evidence-based verification policy.
//
// Colibri is strictly optional: if disabled, unavailable, irrelevant to the
// query, timed out, or lacking concrete evidence, it abstains rather than
// failing core inference or inventing unsupported claims.

#pragma once

#include "collaboration/CollaborativeTypes.h"
#include "colibri/ColibriEngine.h"

#include <string>
#include <vector>

namespace agr {
namespace collaboration {

struct ColibriHealthStatus {
    bool        available = false;
    bool        enabled   = true;
    std::string status_detail;
    double      probe_ms  = 0.0;
};

struct ColibriRelevanceDecision {
    bool        relevant        = false;
    std::string reason;
    float       signal_strength = 0.0f;  ///< [0.0, 1.0]
};

struct ColibriVerificationResult {
    bool                     accepted = false;
    bool                     abstained = false;
    std::string              abstention_reason;
    std::string              provenance;
    std::vector<std::string> verified_claims;
    std::vector<std::string> rejected_claims;
    float                    verified_confidence = 0.0f;
};

class ColibriVerifier {
public:
    /// Checks whether Colibri is enabled by policy AND registered/available.
    static ColibriHealthStatus checkHealth(bool enabled_by_request);

    /// Decides whether Colibri structural/algorithmic/metadata verification
    /// is relevant for this query before invoking it.
    static ColibriRelevanceDecision assessRelevance(
        const std::string& query,
        QueryCategory category,
        bool has_upstream_outputs);

    /// Verifies a ColibriAnalysisReport against available concrete evidence
    /// and a wall-clock timeout. Abstains cleanly when evidence is weak or
    /// when the timeout was exceeded.
    static ColibriVerificationResult verifyReport(
        const std::string& query,
        const std::string& upstream_context,
        const ColibriAnalysisReport& report,
        int timeout_ms,
        double elapsed_ms);
};

} // namespace collaboration
} // namespace agr
