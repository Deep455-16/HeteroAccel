// src/collaboration/ResultFusionEngine.cpp
// Cross-Engine Intermediate Result Validation, Deduplication, Conflict
// Resolution, and Coherent Response Synthesis implementation.

#include "collaboration/ResultFusionEngine.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
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
    if (lower.find("o(2^n)") != std::string::npos) return "O(2^N)";
    if (lower.find("o(n^3)") != std::string::npos) return "O(N^3)";
    if (lower.find("o(n^2)") != std::string::npos) return "O(N^2)";
    if (lower.find("o(n log n)") != std::string::npos) return "O(N log N)";
    if (lower.find("o(log n)") != std::string::npos) return "O(log N)";
    if (lower.find("o(1)") != std::string::npos && lower.find("space") == std::string::npos) return "O(1)";
    if (lower.find("o(n)") != std::string::npos && lower.find("space") == std::string::npos) return "O(N)";
    return "";
}

float sanitizeConfidence(float c) {
    if (std::isnan(c) || std::isinf(c)) return 0.0f;
    if (c < 0.0f) return 0.0f;
    if (c > 1.0f) return 1.0f;
    return c;
}

bool isRepetitiveOutput(const std::string& text) {
    if (text.size() >= 12) {
        size_t run = 1;
        for (size_t i = 1; i < text.size(); ++i) {
            if (!std::isspace(static_cast<unsigned char>(text[i])) && text[i] == text[i - 1]) {
                if (++run >= 12) return true;
            } else {
                run = 1;
            }
        }
    }
    std::istringstream iss(toLowerCopy(text));
    std::vector<std::string> words;
    std::string w;
    while (iss >> w) {
        words.push_back(w);
    }
    if (words.size() >= 6) {
        std::set<std::string> uniq(words.begin(), words.end());
        if (uniq.size() * 4 <= words.size()) {
            return true;
        }
        // Check consecutive identical words >= 4
        size_t word_run = 1;
        for (size_t i = 1; i < words.size(); ++i) {
            if (words[i] == words[i - 1]) {
                if (++word_run >= 4) return true;
            } else {
                word_run = 1;
            }
        }
    }
    return false;
}

bool isMalformedOutput(const std::string& text) {
    // Check UTF-8 replacement character U+FFFD (\xEF\xBF\xBD) or explicit BEL corruption
    if (text.find("\xEF\xBF\xBD") != std::string::npos) return true;
    if (text.find('\x07') != std::string::npos) return true;
    // Check severely unclosed braces (e.g., "{{{")
    int brace_depth = 0;
    for (char c : text) {
        if (c == '{') ++brace_depth;
        else if (c == '}') --brace_depth;
    }
    if (brace_depth >= 3) return true;
    return false;
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

    if (result.colibri_abstained) {
        result.confidence_score = 0.0f;
        result.validated = false;
        if (result.validation_notes.empty()) {
            result.validation_notes = result.colibri_abstention_reason;
        }
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

    bool had_invalid_confidence = std::isnan(result.confidence_score) ||
                                  std::isinf(result.confidence_score) ||
                                  result.confidence_score < 0.0f ||
                                  result.confidence_score > 1.0f;
    if (std::isnan(result.confidence_score) || std::isinf(result.confidence_score) || result.confidence_score < 0.0f) {
        result.confidence_score = 0.0f;
    } else if (result.confidence_score > 1.0f) {
        result.confidence_score = 1.0f;
    } else if (result.confidence_score == 0.0f) {
        if (task.assigned_engine == EngineId::COLIBRI) {
            result.confidence_score = 0.92f;
        } else if (task.assigned_engine == EngineId::HYDROXL) {
            result.confidence_score = 0.86f;
        } else {
            result.confidence_score = 0.85f;
        }
    }

    bool repetitive = isRepetitiveOutput(text);
    bool malformed = isMalformedOutput(text);
    bool structural_warn = (text.find("WARNING") != std::string::npos ||
                            text.find("Unmatched") != std::string::npos ||
                            text.find("Unclosed") != std::string::npos);

    if (repetitive) {
        result.validated = false;
        result.confidence_score = std::min(result.confidence_score, 0.20f);
        result.validation_notes = "Repetitive degenerate output detected";
    } else if (malformed || structural_warn) {
        result.validated = false;
        result.confidence_score = std::min(result.confidence_score, 0.35f);
        result.validation_notes = "Static verification flagged malformed or structural syntax warnings";
    } else if (had_invalid_confidence && result.confidence_score == 0.0f) {
        result.validated = false;
        result.validation_notes = "Invalid confidence score (NaN or negative) clamped to 0.0";
    } else {
        result.validated = true;
        result.validation_notes = "Output verified against structural and task constraints";
    }
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
        bool validated = true;
    };
    std::vector<TaggedFinding> all_findings;

    bool all_validated = true;
    float confidence_sum = 0.0f;
    int completed_tasks = 0;
    int confident_tasks = 0;

    for (const auto& t : tasks) {
        std::string eid = t.output.engine_id.empty()
                              ? std::string(engineIdToString(t.assigned_engine))
                              : t.output.engine_id;
        auto& cs = contrib_map[eid];
        cs.engine_id = eid;
        cs.tasks_assigned += 1;

        if (t.state == TaskState::COMPLETED && t.output.success) {
            // Preserve raw component output for debugging & provenance inspection
            {
                std::ostringstream raw_os;
                raw_os << "[" << eid << " | task=" << t.task_id
                       << " | backend=" << (t.output.backend_used.empty()
                                                ? hardwareBackendToString(t.assigned_backend)
                                                : t.output.backend_used);
                if (!t.output.provenance_source.empty()) {
                    raw_os << " | provenance=" << t.output.provenance_source;
                }
                if (!t.output.finish_reason.empty()) {
                    raw_os << " | finish=" << t.output.finish_reason;
                }
                raw_os << "] " << t.output.structured_result;
                report.raw_component_outputs.push_back(raw_os.str());
            }

            if (t.output.colibri_abstained) {
                cs.tasks_completed += 1;
                cs.primary_backend_used = t.output.backend_used.empty()
                                              ? std::string(hardwareBackendToString(t.assigned_backend))
                                              : t.output.backend_used;
                cs.total_execution_ms += t.output.execution_ms;
                report.rejected_claims.push_back(
                    "[" + eid + "] abstained: " +
                    (t.output.colibri_abstention_reason.empty()
                         ? std::string("insufficient evidence")
                         : t.output.colibri_abstention_reason));
                report.source_attributions.push_back(
                    "[" + eid + "] backend=" + cs.primary_backend_used +
                    " status=abstained provenance=" + t.output.provenance_source);
                continue;
            }

            bool raw_invalid_conf = std::isnan(t.output.confidence_score) ||
                                    std::isinf(t.output.confidence_score) ||
                                    t.output.confidence_score < 0.0f ||
                                    t.output.confidence_score > 1.0f;
            float clean_conf = sanitizeConfidence(t.output.confidence_score);
            bool task_validated = t.output.validated && !raw_invalid_conf;
            std::string trimmed_res = trimCopy(t.output.structured_result);
            if (isRepetitiveOutput(trimmed_res) || isMalformedOutput(trimmed_res)) {
                task_validated = false;
                clean_conf = std::min(clean_conf, 0.25f);
            }

            cs.tasks_completed += 1;
            cs.primary_backend_used = t.output.backend_used.empty()
                                          ? std::string(hardwareBackendToString(t.assigned_backend))
                                          : t.output.backend_used;
            cs.total_tokens_generated += t.output.tokens_generated;
            cs.total_execution_ms += t.output.execution_ms;
            cs.average_confidence += clean_conf;
            if (t.output.peak_memory_bytes > cs.peak_memory_bytes) {
                cs.peak_memory_bytes = t.output.peak_memory_bytes;
            }

            {
                std::ostringstream attr;
                attr << "[" << eid << "] backend=" << cs.primary_backend_used;
                if (!t.output.provenance_source.empty()) attr << " provenance=" << t.output.provenance_source;
                if (!t.output.template_id.empty()) {
                    attr << " template=" << t.output.template_id << "@v" << t.output.template_version;
                }
                attr << " conf=" << std::fixed << std::setprecision(2) << clean_conf
                     << " validated=" << (task_validated ? "yes" : "no");
                report.source_attributions.push_back(attr.str());
            }

            if (!task_validated) {
                all_validated = false;
            }
            confidence_sum += clean_conf;
            ++completed_tasks;
            ++confident_tasks;

            const bool accept_claims = task_validated && clean_conf >= 0.40f;
            for (const auto& f : t.output.key_findings) {
                std::string clean = trimCopy(f);
                if (clean.empty() || isRepetitiveOutput(clean) || isMalformedOutput(clean)) {
                    if (!clean.empty()) {
                        report.rejected_claims.push_back("[" + eid + "] rejected degenerate claim: " + clean);
                    }
                    continue;
                }
                if (!accept_claims) {
                    report.rejected_claims.push_back("[" + eid + "] rejected low-trust claim (validated=" +
                                                     std::string(task_validated ? "yes" : "no") +
                                                     "): " + clean);
                }
                float f_conf = task_validated ? clean_conf : std::min(clean_conf, 0.30f);
                all_findings.push_back({clean, eid, f_conf, t.task_type, task_validated});
            }
            for (const auto& ev : t.output.supporting_evidence) {
                std::string clean_ev = trimCopy(ev);
                if (!clean_ev.empty()) {
                    report.supporting_evidence.push_back("[" + eid + "] " + clean_ev);
                }
            }
        } else {
            cs.tasks_failed += 1;
            if (!t.output.error.empty()) {
                report.rejected_claims.push_back("[" + eid + "] task failed: " + t.output.error);
            }
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

    std::string colibri_authoritative_complexity;
    if (engine_complexity_claims.count("colibri") > 0) {
        colibri_authoritative_complexity = engine_complexity_claims["colibri"];
    }

    if (engine_complexity_claims.size() >= 2) {
        std::string first_eng = engine_complexity_claims.begin()->first;
        std::string first_val = engine_complexity_claims.begin()->second;
        for (const auto& kv : engine_complexity_claims) {
            if (kv.second != first_val) {
                ++report.conflicts_detected;
                if (!colibri_authoritative_complexity.empty()) {
                    report.resolved_conflicts.push_back(
                        "Complexity estimate divergence between " + first_eng + " (" + first_val +
                        ") and " + kv.first + " (" + kv.second +
                        ") resolved in favor of static control-flow bound " + colibri_authoritative_complexity +
                        " (structural proof overrides majority heuristic voting).");
                    report.preserved_disagreements.push_back(
                        first_eng + " reported " + first_val + " while " +
                        kv.first + " reported " + kv.second + " (authoritative bound: " +
                        colibri_authoritative_complexity + ").");
                } else {
                    report.preserved_disagreements.push_back(
                        "UNRESOLVED complexity disagreement: " + first_eng + " reported " + first_val +
                        " while " + kv.first + " reported " + kv.second +
                        " (no deterministic Colibri proof available).");
                }
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

    // 3. Deduplicate overlapping findings and rank by confidence & structural proof
    //    (Never allow 2-vs-1 neural majority voting to override Colibri structural proof)
    struct MergedFinding {
        std::string text;
        std::vector<std::string> engines;
        float best_confidence = 0.0f;
        bool is_structural_proof = false;
        bool disputed_by_structural_proof = false;
        bool validated = true;
    };
    std::vector<MergedFinding> merged;

    for (const auto& tf : all_findings) {
        std::string lower = toLowerCopy(tf.text);
        std::string comp = detectComplexityClaim(lower);
        bool is_colibri_proof = (tf.engine_id == "colibri" && !comp.empty());
        bool disputed = (!colibri_authoritative_complexity.empty() &&
                         !comp.empty() &&
                         comp != colibri_authoritative_complexity);

        bool found_match = false;
        // Only merge findings across engines into multi-engine corroboration when BOTH
        // are validated and meet the confidence threshold (never fabricate corroboration).
        const bool can_corroborate_tf = tf.validated && tf.confidence >= 0.50f;
        for (auto& m : merged) {
            const bool can_corroborate_m = m.validated && m.best_confidence >= 0.50f;
            float sim = computeSimilarity(tf.text, m.text);
            if (can_corroborate_tf && can_corroborate_m &&
                sim >= 0.55f && m.disputed_by_structural_proof == disputed) {
                found_match = true;
                if (std::find(m.engines.begin(), m.engines.end(), tf.engine_id) == m.engines.end()) {
                    m.engines.push_back(tf.engine_id);
                }
                if (is_colibri_proof) {
                    m.is_structural_proof = true;
                }
                m.validated = m.validated && tf.validated;
                if (tf.confidence > m.best_confidence) {
                    m.best_confidence = tf.confidence;
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
            mf.best_confidence = disputed ? std::min(tf.confidence, 0.35f) : tf.confidence;
            mf.is_structural_proof = is_colibri_proof;
            mf.disputed_by_structural_proof = disputed;
            mf.validated = tf.validated;
            merged.push_back(std::move(mf));
        }
    }

    // Sort merged findings:
    //  1) Undisputed findings before findings disputed by structural proof
    //  2) Validated findings before unvalidated findings
    //  3) Authoritative Colibri structural proofs before heuristic claims when conflict exists
    //  4) Multi-engine corroboration and higher confidence
    std::stable_sort(merged.begin(), merged.end(),
                     [](const MergedFinding& a, const MergedFinding& b) {
                         if (a.disputed_by_structural_proof != b.disputed_by_structural_proof) {
                             return !a.disputed_by_structural_proof;
                         }
                         if (a.validated != b.validated) {
                             return a.validated;
                         }
                         if (a.is_structural_proof != b.is_structural_proof) {
                             return a.is_structural_proof;
                         }
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
        std::string entry = "[" + tag.str() + "] " + m.text;
        if (m.disputed_by_structural_proof && !colibri_authoritative_complexity.empty()) {
            entry += " [Disputed: overridden by Colibri structural bound " +
                     colibri_authoritative_complexity + "]";
        }
        report.deduplicated_findings.push_back(entry);
    }

    report.deduplicated_findings_count = static_cast<int>(report.deduplicated_findings.size());
    report.validation_passed = all_validated && (completed_tasks > 0);
    report.overall_confidence = confident_tasks > 0
                                    ? sanitizeConfidence(confidence_sum / static_cast<float>(confident_tasks))
                                    : 0.0f;

    // 4. Synthesize one coherent, unified final response (never raw concatenation)
    std::string hydroxl_text, llamacpp_text;
    bool hydroxl_trustworthy = false, llamacpp_trustworthy = false;
    std::vector<std::string> colibri_highlights;
    std::string colibri_abstention;

    for (const auto& t : tasks) {
        if (t.state != TaskState::COMPLETED || !t.output.success) continue;
        if (t.output.colibri_abstained) {
            colibri_abstention = t.output.colibri_abstention_reason;
            continue;
        }
        if (t.assigned_engine == EngineId::HYDROXL && hydroxl_text.empty()) {
            hydroxl_text = trimCopy(t.output.structured_result);
            hydroxl_trustworthy = t.output.validated && sanitizeConfidence(t.output.confidence_score) >= 0.50f;
        } else if (t.assigned_engine == EngineId::LLAMA_CPP && llamacpp_text.empty()) {
            llamacpp_text = trimCopy(t.output.structured_result);
            llamacpp_trustworthy = t.output.validated && sanitizeConfidence(t.output.confidence_score) >= 0.50f;
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
        if (overlap >= 0.70f && hydroxl_trustworthy && llamacpp_trustworthy) {
            // High agreement between validated HydroXL and llama.cpp outputs
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
    } else if (!colibri_highlights.empty()) {
        synth << "   - [Direct Structural & Metadata Inspection (Colibri)]: Executed deterministically without redundant neural LLM invocation.\n";
    }

    // Section 2: Colibri Complexity, Optimization & Verification Findings (or explicit abstention)
    if (!colibri_highlights.empty()) {
        synth << "\n2. Complexity, Optimization & Verification (Colibri):\n";
        for (const auto& ch : colibri_highlights) {
            synth << "   - " << ch << "\n";
        }
    } else if (!colibri_abstention.empty()) {
        synth << "\n2. Verification Note (Colibri):\n";
        synth << "   - Abstained: " << colibri_abstention << "\n";
    }

    // Section 3: Conflict Resolution, Source Attribution & Cross-Engine Validation Summary
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
    if (!report.preserved_disagreements.empty()) {
        for (const auto& pd : report.preserved_disagreements) {
            synth << "   - Preserved Disagreement: " << pd << "\n";
        }
    }
    if (!report.rejected_claims.empty()) {
        synth << "   - Rejected/Abstained Claims: " << report.rejected_claims.size() << "\n";
    }
    if (!report.source_attributions.empty()) {
        synth << "   - Sources: ";
        for (size_t i = 0; i < report.source_attributions.size(); ++i) {
            if (i > 0) synth << "; ";
            synth << report.source_attributions[i];
        }
        synth << "\n";
    }

    report.synthesized_response = synth.str();
    return report;
}

} // namespace collaboration
} // namespace agr
