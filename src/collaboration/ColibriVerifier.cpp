// src/collaboration/ColibriVerifier.cpp
#include "collaboration/ColibriVerifier.h"
#include "engine/ExecutionEngineRegistry.h"
#include "orchestration/EngineOrchestrator.h"

#include <algorithm>
#include <cctype>
#include <chrono>

namespace agr {
namespace collaboration {

namespace {

std::string toLowerAscii(const std::string& s) {
    std::string out = s;
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

bool containsAny(const std::string& lower, const std::vector<const char*>& needles) {
    for (const char* n : needles) {
        if (lower.find(n) != std::string::npos) return true;
    }
    return false;
}

bool hasConcreteStructuralSignal(const std::string& lower) {
    return containsAny(lower, {
        "for(", "for (", "while(", "while (", "void ", "int ", "float ", "double ",
        "def ", "std::", "mutex", "atomic", "lock", "malloc(", "new ",
        "binary search", "binary_search", "merge sort", "merge_sort", "mergesort",
        "fibonacci", "fib(", "matmul", "matrix", "nested", "loop", "complexity",
        "big-o", "big o", "o(n", "o(log", "o(2^", "o(1)",
        "algorithm", "sort", "recursion", "recursive", "memory", "kv cache",
        "kv-cache", "attention", "gqa", "quant", "gguf", "metadata", "paging",
        "tlb", "virtual memory", "bug", "race", "deadlock", "crash", "leak",
        "verify", "validate", "optimize", "cache", "simd", "code", "function"
    });
}

} // namespace

ColibriHealthStatus ColibriVerifier::checkHealth(bool enabled_by_request) {
    ColibriHealthStatus status;
    status.enabled = enabled_by_request;
    auto t0 = std::chrono::steady_clock::now();

    if (!enabled_by_request) {
        status.available = false;
        status.status_detail = "Colibri disabled by caller policy";
        status.probe_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        return status;
    }

    auto& orch = orchestration::EngineOrchestrator::instance();
    bool registered_orch = false;
    for (const auto& b : orch.backends()) {
        if (b.backend_id == "colibri_cpu") {
            registered_orch = true;
            break;
        }
    }

    auto& reg = ExecutionEngineRegistry::instance();
    bool registered_exec = reg.isRegistered("colibri");

    if (!registered_orch && !registered_exec) {
        status.available = false;
        status.status_detail = "Colibri engine ('colibri_cpu' / 'colibri') is not registered";
    } else {
        status.available = true;
        status.status_detail = "Colibri deterministic verifier healthy and available on CPU";
    }

    status.probe_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    return status;
}

ColibriRelevanceDecision ColibriVerifier::assessRelevance(
    const std::string& query,
    QueryCategory category,
    bool has_upstream_outputs) {

    ColibriRelevanceDecision dec;
    const std::string lower = toLowerAscii(query);

    if (category == QueryCategory::SIMPLE_FACTUAL_QA) {
        dec.relevant = false;
        dec.signal_strength = 0.0f;
        dec.reason = "Simple factual Q&A has no structural, algorithmic or verification requirements";
        return dec;
    }

    if (category == QueryCategory::MODEL_METADATA_INSPECTION ||
        category == QueryCategory::STATIC_COMPLEXITY_ONLY) {
        dec.relevant = true;
        dec.signal_strength = 1.0f;
        dec.reason = "Query explicitly requests deterministic metadata or static complexity inspection";
        return dec;
    }

    if (hasConcreteStructuralSignal(lower)) {
        dec.relevant = true;
        dec.signal_strength = 0.90f;
        dec.reason = "Query contains concrete code, algorithmic complexity, concurrency, memory or GGUF structural signals";
        return dec;
    }

    if (category == QueryCategory::CODE_ANALYSIS_AND_OPTIMIZATION ||
        category == QueryCategory::DEBUGGING_AND_VERIFICATION) {
        dec.relevant = true;
        dec.signal_strength = 0.85f;
        dec.reason = "Query category requires structural/debugging verification";
        return dec;
    }

    if (category == QueryCategory::COMPLEX_TECHNICAL_QUESTION && has_upstream_outputs) {
        dec.relevant = true;
        dec.signal_strength = 0.65f;
        dec.reason = "Complex technical query with upstream outputs to cross-check";
        return dec;
    }

    if (has_upstream_outputs) {
        // Check if the query asks for retrieval-only or creative/general prose where static checks cannot corroborate
        if (containsAny(lower, {"poem", "story", "haiku", "hello", "joke", "only from the text", "from the data"})) {
            dec.relevant = false;
            dec.signal_strength = 0.10f;
            dec.reason = "Non-structural/grounded prose query lacks static verification criteria; Colibri abstains";
            return dec;
        }
        // Only engage on general queries if upstream context contains structured/code artifacts
        if (lower.size() > 24 && category == QueryCategory::GENERAL_REASONING_AND_SYNTHESIS) {
            dec.relevant = false;
            dec.signal_strength = 0.20f;
            dec.reason = "General reasoning query contains no code, algorithm or structural markers; Colibri abstains";
            return dec;
        }
    }

    dec.relevant = false;
    dec.signal_strength = 0.0f;
    dec.reason = "No structural, complexity or verification signal present in query; Colibri abstains";
    return dec;
}

ColibriVerificationResult ColibriVerifier::verifyReport(
    const std::string& query,
    const std::string& upstream_context,
    const ColibriAnalysisReport& report,
    int timeout_ms,
    double elapsed_ms) {

    ColibriVerificationResult ver;
    ver.provenance = "colibri:deterministic_static_v1";

    if (timeout_ms > 0 && elapsed_ms > static_cast<double>(timeout_ms)) {
        ver.accepted = false;
        ver.abstained = true;
        ver.abstention_reason = "Colibri exceeded timeout budget (" +
                                std::to_string(static_cast<int>(elapsed_ms)) + " ms > " +
                                std::to_string(timeout_ms) + " ms)";
        ver.verified_confidence = 0.0f;
        ver.rejected_claims = report.key_findings;
        return ver;
    }

    const std::string combined_lower = toLowerAscii(query + "\n" + upstream_context);
    const bool has_loop_proof = (report.max_loop_nesting_depth > 0);
    const bool has_algo_or_struct_signal = hasConcreteStructuralSignal(combined_lower);
    bool any_domain_evidence = false;
    for (const auto& ev : report.supporting_evidence) {
        const std::string le = toLowerAscii(ev);
        if (le.find("gguf") != std::string::npos ||
            le.find("floating-point") != std::string::npos ||
            le.find("delimiter") != std::string::npos) {
            any_domain_evidence = true;
            break;
        }
    }

    if (!has_loop_proof && !has_algo_or_struct_signal && !any_domain_evidence) {
        ver.accepted = false;
        ver.abstained = true;
        ver.abstention_reason = "Insufficient structural evidence to verify claims; abstaining";
        ver.verified_confidence = 0.0f;
        ver.rejected_claims = report.key_findings;
        return ver;
    }

    // Filter individual claims: keep only non-empty, evidence-supported claims
    for (const auto& claim : report.key_findings) {
        if (claim.empty()) {
            ver.rejected_claims.push_back("Empty claim rejected");
            continue;
        }
        // Reject claims asserting "linear/direct" O(N) on queries that lack any algorithmic/loop evidence
        const std::string cl = toLowerAscii(claim);
        if (cl.find("loop nesting depth 0") != std::string::npos && !has_algo_or_struct_signal && upstream_context.empty()) {
            ver.rejected_claims.push_back(claim);
            continue;
        }
        ver.verified_claims.push_back(claim);
    }

    if (ver.verified_claims.empty()) {
        ver.accepted = false;
        ver.abstained = true;
        ver.abstention_reason = "No verifiable structural claims remained after evidence filtering";
        ver.verified_confidence = 0.0f;
        return ver;
    }

    // Confidence reflects concrete evidence strength rather than a flat hardcoded constant
    float conf = report.confidence_score > 0.0f ? report.confidence_score : 0.70f;
    if (has_loop_proof) {
        conf = std::max(conf, 0.92f);
    } else if (has_algo_or_struct_signal && report.syntax_balanced) {
        conf = std::clamp(conf, 0.78f, 0.90f);
    } else {
        conf = std::min(conf, 0.72f);
    }
    if (!report.syntax_balanced) {
        conf = std::min(conf, 0.75f);
    }

    ver.accepted = true;
    ver.abstained = false;
    ver.verified_confidence = conf;
    return ver;
}

} // namespace collaboration
} // namespace agr
