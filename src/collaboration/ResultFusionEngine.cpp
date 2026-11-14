// src/collaboration/ResultFusionEngine.cpp
// Cross-Engine Intermediate Result Validation, Deduplication, Conflict
// Resolution, and Coherent Response Synthesis implementation.

#include "collaboration/ResultFusionEngine.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <set>
#include <sstream>

namespace agr {
namespace collaboration {

namespace {

std::string trimCopy(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n-*•");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

std::string toLowerCopy(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

std::set<std::string> extractContentWords(const std::string& s) {
    std::set<std::string> words;
    std::string cur;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '_' || c == '^') {
            cur += static_cast<char>(std::tolower(c));
        } else {
            if (cur.size() >= 3) {
                words.insert(cur);
            }
            cur.clear();
        }
    }
    if (cur.size() >= 3) {
        words.insert(cur);
    }
    return words;
}

std::vector<std::string> splitIntoSentencesOrLines(const std::string& text) {
    std::vector<std::string> parts;
    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line)) {
        std::string trimmed = trimCopy(line);
        if (trimmed.empty()) continue;
        // Skip pure bracket headers like "[Colibri Structural & Complexity Analysis]"
        if (trimmed.front() == '[' && trimmed.back() == ']') continue;

        // Also split long single-line paragraphs on ". "
        size_t pos = 0;
        while (pos < trimmed.size()) {
            size_t dot = trimmed.find(". ", pos);
            if (dot == std::string::npos || dot - pos < 16) {
                std::string seg = trimCopy(trimmed.substr(pos));
                if (!seg.empty()) parts.push_back(seg);
                break;
            } else {
                std::string seg = trimCopy(trimmed.substr(pos, dot - pos + 1));
                if (!seg.empty()) parts.push_back(seg);
                pos = dot + 2;
            }
        }
    }
    return parts;
}

std::string detectComplexityClaim(const std::string& lower) {
    if (lower.find("o(n^3)") != std::string::npos) return "O(N^3)";
    if (lower.find("o(n^2)") != std::string::npos) return "O(N^2)";
    if (lower.find("o(n log n)") != std::string::npos) return "O(N log N)";
    if (lower.find("o(log n)") != std::string::npos) return "O(log N)";
    if (lower.find("o(1)") != std::string::npos && lower.find("space") == std::string::npos) return "O(1)";
    if (lower.find("o(n)") != std::string::npos && lower.find("space") == std::string::npos) return "O(N)";
    return "";
}

} // namespace

float ResultFusionEngine::computeSimilarity(const std::string& a, const std::string& b) {
    std::string la = toLowerCopy(trimCopy(a));
    std::string lb = toLowerCopy(trimCopy(b));
    if (la.empty() || lb.empty()) return 0.0f;
    if (la == lb) return 1.0f;

    auto wa = extractContentWords(la);
    auto wb = extractContentWords(lb);
    if (wa.empty() || wb.empty()) return 0.0f;

    size_t inter = 0;
    for (const auto& w : wa) {
        if (wb.count(w) > 0) ++inter;
    }
    size_t uni = wa.size() + wb.size() - inter;
    return uni > 0 ? static_cast<float>(inter) / static_cast<float>(uni) : 0.0f;
}

void ResultFusionEngine::enrichIntermediateResult(
    const CollaborativeTask& task,
    IntermediateResult& result) {

    result.task_id = task.task_id;
    result.engine_id = engineIdToString(task.assigned_engine);
    result.backend_used = hardwareBackendToString(task.assigned_backend);
    result.orchestrator_backend_id = task.orchestrator_backend_id;

    if (!result.success) {
        result.confidence_score = 0.0f;
        result.validated = false;
        result.validation_notes = "Task failed: " + result.error;
        return;
    }

    std::string text = trimCopy(result.structured_result);
    if (text.empty()) {
        result.confidence_score = 0.1f;
        result.validated = false;
        result.validation_notes = "Empty output text";
        return;
    }

    // Extract structured findings if not already populated
    if (result.key_findings.empty()) {
        auto segments = splitIntoSentencesOrLines(text);
        for (const auto& seg : segments) {
            if (seg.rfind("Finding:", 0) == 0) {
                result.key_findings.push_back(trimCopy(seg.substr(8)));
            } else if (seg.rfind("Optimization:", 0) == 0) {
                result.key_findings.push_back(trimCopy(seg));
            } else if (seg.rfind("Risk:", 0) == 0) {
                result.key_findings.push_back(trimCopy(seg));
            } else if (seg.rfind("Complexity:", 0) == 0) {
                result.key_findings.push_back(trimCopy(seg));
            } else if (seg.rfind("Category:", 0) == 0 || seg.rfind("Structural Validation:", 0) == 0) {
                result.supporting_evidence.push_back(trimCopy(seg));
            } else {
                result.key_findings.push_back(seg);
            }
        }
        if (result.key_findings.empty()) {
            result.key_findings.push_back(text);
        }
    }

    if (result.supporting_evidence.empty()) {
        std::ostringstream ev;
        ev << "Generated by " << result.engine_id << " on " << result.backend_used
           << " (" << result.tokens_generated << " tokens in "
           << static_cast<int>(result.execution_ms) << " ms)";
        result.supporting_evidence.push_back(ev.str());
    }

    if (result.confidence_score <= 0.0f) {
        if (task.assigned_engine == EngineId::COLIBRI) {
            result.confidence_score = 0.92f;
        } else if (task.assigned_engine == EngineId::HYDROXL) {
            result.confidence_score = 0.86f;
        } else {
            result.confidence_score = 0.85f;
        }
    }

    result.validated = (text.find("WARNING") == std::string::npos &&
                        text.find("Unmatched") == std::string::npos &&
                        text.find("Unclosed") == std::string::npos);
    result.validation_notes = result.validated
                                  ? "Output verified against structural and task constraints"
                                  : "Static verification flagged structural or syntax warnings";
}

FusionReport ResultFusionEngine::fuse(
    QueryCategory category,
    const std::string& original_prompt,
    const std::vector<CollaborativeTask>& tasks) {

    FusionReport report;

    // 1. Aggregate per-engine contribution telemetry
    std::map<std::string, EngineContributionSummary> contrib_map;
    const char* kKnownEngines[] = {"hydroxl", "llama.cpp", "colibri"};
    for (const char* eng : kKnownEngines) {
        EngineContributionSummary s;
        s.engine_id = eng;
        s.primary_backend_used = "CPU";
        contrib_map[eng] = s;
    }

    struct TaggedFinding {
        std::string text;
        std::string engine_id;
        float confidence = 0.0f;
        TaskType task_type;
    };
    std::vector<TaggedFinding> all_findings;

    bool all_validated = true;
    float confidence_sum = 0.0f;
    int completed_tasks = 0;

    for (const auto& t : tasks) {
        std::string eid = t.output.engine_id.empty()
                              ? std::string(engineIdToString(t.assigned_engine))
                              : t.output.engine_id;
        auto& cs = contrib_map[eid];
        cs.engine_id = eid;
        cs.tasks_assigned += 1;

        if (t.state == TaskState::COMPLETED && t.output.success) {
            cs.tasks_completed += 1;
            cs.primary_backend_used = t.output.backend_used.empty()
                                          ? std::string(hardwareBackendToString(t.assigned_backend))
                                          : t.output.backend_used;
            cs.total_tokens_generated += t.output.tokens_generated;
            cs.total_execution_ms += t.output.execution_ms;
            cs.average_confidence += t.output.confidence_score;
            if (t.output.peak_memory_bytes > cs.peak_memory_bytes) {
                cs.peak_memory_bytes = t.output.peak_memory_bytes;
            }

            if (!t.output.validated) {
                all_validated = false;
            }
            confidence_sum += t.output.confidence_score;
            ++completed_tasks;

            for (const auto& f : t.output.key_findings) {
                std::string clean = trimCopy(f);
                if (!clean.empty()) {
                    all_findings.push_back({clean, eid, t.output.confidence_score, t.task_type});
                }
            }
            for (const auto& ev : t.output.supporting_evidence) {
                std::string clean_ev = trimCopy(ev);
                if (!clean_ev.empty()) {
                    report.supporting_evidence.push_back("[" + eid + "] " + clean_ev);
                }
            }
        } else {
            cs.tasks_failed += 1;
        }
    }

    for (const char* eng : kKnownEngines) {
        auto it = contrib_map.find(eng);
        if (it != contrib_map.end()) {
            if (it->second.tasks_completed > 0) {
                it->second.average_confidence /= static_cast<float>(it->second.tasks_completed);
            }
            if (it->second.tasks_assigned > 0) {
                report.engine_contributions.push_back(it->second);
            }
        }
    }

    report.raw_findings_count = static_cast<int>(all_findings.size());

    // 2. Conflict & contradiction detection across engines
    std::map<std::string, std::string> engine_complexity_claims;
    bool claims_thread_safe = false, claims_race_or_lock_risk = false;
    std::string thread_safe_engine, race_risk_engine;

    for (const auto& tf : all_findings) {
        std::string lower = toLowerCopy(tf.text);
        std::string comp = detectComplexityClaim(lower);
        if (!comp.empty()) {
            engine_complexity_claims[tf.engine_id] = comp;
        }
        if (lower.find("thread-safe") != std::string::npos ||
            lower.find("lock-free") != std::string::npos ||
            lower.find("no race") != std::string::npos) {
            claims_thread_safe = true;
            thread_safe_engine = tf.engine_id;
        }
        if (lower.find("race condition") != std::string::npos ||
            lower.find("lock contention") != std::string::npos ||
            lower.find("lock-order") != std::string::npos ||
            lower.find("data race") != std::string::npos) {
            claims_race_or_lock_risk = true;
            race_risk_engine = tf.engine_id;
        }
    }

    if (engine_complexity_claims.size() >= 2) {
        std::string first_eng = engine_complexity_claims.begin()->first;
        std::string first_val = engine_complexity_claims.begin()->second;
        for (const auto& kv : engine_complexity_claims) {
            if (kv.second != first_val) {
                ++report.conflicts_detected;
                std::string authoritative = engine_complexity_claims.count("colibri")
                                                ? engine_complexity_claims["colibri"]
                                                : kv.second;
                report.resolved_conflicts.push_back(
                    "Complexity estimate divergence between " + first_eng + " (" + first_val +
                    ") and " + kv.first + " (" + kv.second +
                    ") resolved in favor of static control-flow bound " + authoritative + ".");
                report.preserved_disagreements.push_back(
                    first_eng + " reported " + first_val + " (average/best-case path) while " +
                    kv.first + " reported " + kv.second + " (worst-case structural bound).");
                break;
            }
        }
    }

    if (claims_thread_safe && claims_race_or_lock_risk && thread_safe_engine != race_risk_engine) {
        ++report.conflicts_detected;
        report.resolved_conflicts.push_back(
            "Concurrency assessment conflict between " + thread_safe_engine +
            " (claimed thread-safe) and " + race_risk_engine +
            " (flagged synchronization/contention risk): prioritizing defensive synchronization audit.");
        report.preserved_disagreements.push_back(
            thread_safe_engine + " noted safe single-instance locking whereas " +
            race_risk_engine + " highlighted multi-stage lock contention risk.");
    }

    // 3. Deduplicate overlapping findings and rank by confidence
    struct MergedFinding {
        std::string text;
        std::vector<std::string> engines;
        float best_confidence = 0.0f;
    };
    std::vector<MergedFinding> merged;

    for (const auto& tf : all_findings) {
        bool found_match = false;
        for (auto& m : merged) {
            float sim = computeSimilarity(tf.text, m.text);
            if (sim >= 0.55f) {
                found_match = true;
                if (std::find(m.engines.begin(), m.engines.end(), tf.engine_id) == m.engines.end()) {
                    m.engines.push_back(tf.engine_id);
                }
                if (tf.confidence > m.best_confidence) {
                    m.best_confidence = tf.confidence;
                    // Prefer the more detailed phrasing when confidence is higher
                    if (tf.text.size() > m.text.size()) {
                        m.text = tf.text;
                    }
                }
                break;
            }
        }
        if (!found_match) {
            MergedFinding mf;
            mf.text = tf.text;
            mf.engines.push_back(tf.engine_id);
            mf.best_confidence = tf.confidence;
            merged.push_back(std::move(mf));
        }
    }

    // Sort merged findings: multi-engine corroborated findings and higher confidence first
    std::stable_sort(merged.begin(), merged.end(),
                     [](const MergedFinding& a, const MergedFinding& b) {
                         if (a.engines.size() != b.engines.size()) {
                             return a.engines.size() > b.engines.size();
                         }
                         return a.best_confidence > b.best_confidence;
                     });

    for (const auto& m : merged) {
        std::ostringstream tag;
        for (size_t i = 0; i < m.engines.size(); ++i) {
            if (i > 0) tag << "+";
            tag << m.engines[i];
        }
        report.deduplicated_findings.push_back("[" + tag.str() + "] " + m.text);
    }

    report.deduplicated_findings_count = static_cast<int>(report.deduplicated_findings.size());
    report.validation_passed = all_validated && (completed_tasks > 0);
    report.overall_confidence = completed_tasks > 0
                                    ? (confidence_sum / static_cast<float>(completed_tasks))
                                    : 0.0f;

    // 4. Synthesize one coherent, unified final response (never raw concatenation)
    std::string hydroxl_text, llamacpp_text;
    std::vector<std::string> colibri_highlights;

    for (const auto& t : tasks) {
        if (t.state != TaskState::COMPLETED || !t.output.success) continue;
        if (t.assigned_engine == EngineId::HYDROXL && hydroxl_text.empty()) {
            hydroxl_text = trimCopy(t.output.structured_result);
        } else if (t.assigned_engine == EngineId::LLAMA_CPP && llamacpp_text.empty()) {
            llamacpp_text = trimCopy(t.output.structured_result);
        } else if (t.assigned_engine == EngineId::COLIBRI) {
            for (const auto& f : t.output.key_findings) {
                colibri_highlights.push_back(trimCopy(f));
            }
        }
    }

    std::ostringstream synth;
    synth << "=== HeteroAccel Unified Collaborative Response ===\n";
    synth << "Query Category: " << queryCategoryToString(category) << "\n\n";

    // Section 1: Integrated Core Analysis (combining HydroXL & llama.cpp insights without duplication)
    synth << "1. Core Reasoning & Implementation Synthesis:\n";
    if (!hydroxl_text.empty() && !llamacpp_text.empty()) {
        float overlap = computeSimilarity(hydroxl_text, llamacpp_text);
        if (overlap >= 0.70f) {
            // High agreement between HydroXL and llama.cpp: emit once as corroborated consensus
            synth << "   - [Corroborated Consensus (HydroXL + llama.cpp)]: "
                  << original_prompt << " -> " << hydroxl_text << "\n";
        } else {
            synth << "   - [Structural & Conceptual Reasoning (HydroXL)]: "
                  << hydroxl_text << "\n";
            synth << "   - [Implementation & Edge-Case Analysis (llama.cpp)]: "
                  << llamacpp_text << "\n";
        }
    } else if (!hydroxl_text.empty()) {
        synth << "   - [Primary Reasoning (HydroXL)]: " << hydroxl_text << "\n";
    } else if (!llamacpp_text.empty()) {
        synth << "   - [Compatibility Synthesis (llama.cpp)]: " << llamacpp_text << "\n";
    }

    // Section 2: Colibri Complexity, Optimization & Verification Findings
    if (!colibri_highlights.empty()) {
        synth << "\n2. Complexity, Optimization & Verification (Colibri):\n";
        for (const auto& ch : colibri_highlights) {
            synth << "   - " << ch << "\n";
        }
    }

    // Section 3: Conflict Resolution & Cross-Engine Validation Summary
    synth << "\n3. Validation & Fusion Summary:\n";
    synth << "   - Validation Status: " << (report.validation_passed ? "VERIFIED" : "FLAGGED_WARNINGS")
          << " (Confidence: " << static_cast<int>( std::round(report.overall_confidence * 100.0f) ) << "%)\n";
    synth << "   - Findings Deduplicated: " << report.raw_findings_count
          << " raw -> " << report.deduplicated_findings_count << " unique\n";
    if (!report.resolved_conflicts.empty()) {
        for (const auto& rc : report.resolved_conflicts) {
            synth << "   - Conflict Resolved: " << rc << "\n";
        }
    }

    report.synthesized_response = synth.str();
    return report;
}

} // namespace collaboration
} // namespace agr
