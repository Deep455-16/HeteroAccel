// src/generation/GenerationBudget.h
// Phase 33: Central generation budget policy.
//
// One policy object bounds every resource a generation may consume: context
// window, output tokens, wall-clock time, retries and stop conditions. Task
// profiles supply safe defaults; explicit caller overrides are validated and
// clamped to hard ceilings so no request can run unbounded.

#pragma once

#include <atomic>
#include <chrono>
#include <string>
#include <vector>

namespace agr {
namespace generation {

/// Per-task generation profile (Phase 33, work package 1).
enum class TaskProfile {
    AUTO,                ///< Resolved by the caller from the query category
    SHORT_ANSWER,        ///< Factual one-liners
    EXPLANATION,         ///< Conceptual explanations
    CODE,                ///< Complete runnable code + explanation
    LONG_FORM,           ///< Extended multi-section answers
    RETRIEVAL_GROUNDED   ///< Answers constrained to supplied (untrusted) context
};

const char* profileToString(TaskProfile p) noexcept;

/// Parses "auto", "short_answer", "explanation", "code", "long_form",
/// "retrieval_grounded" (case-insensitive). Returns false for unknown names.
bool profileFromString(const std::string& name, TaskProfile& out) noexcept;

/// Hard ceilings. No override may exceed these.
struct BudgetLimits {
    static constexpr int kMinContextTokens = 64;
    static constexpr int kMaxContextTokens = 32768;
    static constexpr int kMinOutputTokens  = 1;
    static constexpr int kMaxOutputTokens  = 4096;
    static constexpr int kMinTimeMs        = 100;
    static constexpr int kMaxTimeMs        = 600000;  // 10 minutes
    static constexpr int kMaxRetries       = 3;
    /// Tokens always reserved for the prompt in front of the output.
    static constexpr int kPromptReserveTokens = 64;
};

struct GenerationBudget {
    int context_tokens    = 2048;
    int max_output_tokens = 256;
    int time_budget_ms    = 120000;
    int max_retries       = 0;   ///< Bounded retries for empty output only
    std::vector<std::string> stop_sequences;
};

/// Optional caller overrides. A value <= 0 means "not set".
struct BudgetOverrides {
    int max_output_tokens = 0;
    int context_tokens    = 0;
    int time_budget_ms    = 0;
    int max_retries       = -1;  ///< -1 = not set (0 is a valid explicit value)
    std::vector<std::string> extra_stop_sequences;
};

struct BudgetResolution {
    GenerationBudget budget;
    std::vector<std::string> warnings;  ///< One entry per clamp / adjustment
    bool clamped = false;
};

/// Safe defaults for a profile. AUTO behaves like EXPLANATION.
GenerationBudget defaultsForProfile(TaskProfile profile);

/// Applies overrides to the profile defaults, validates every field and clamps
/// unsafe values. Never fails; every adjustment is reported in `warnings`.
BudgetResolution resolveBudget(TaskProfile profile, const BudgetOverrides& overrides);

/// Wall-clock deadline helper shared by every generation loop.
class Deadline {
public:
    /// time_budget_ms <= 0 disables the deadline.
    explicit Deadline(int time_budget_ms);
    bool enabled() const noexcept { return enabled_; }
    bool expired() const noexcept;
    double remainingMs() const noexcept;

private:
    bool enabled_ = false;
    std::chrono::steady_clock::time_point end_;
};

/// Holds back the tail of streamed text that could be the start of a stop
/// sequence, so a stop sequence is never partially emitted to the user.
class StopSequenceFilter {
public:
    explicit StopSequenceFilter(std::vector<std::string> stops);

    /// Feeds a decoded piece. Returns the text that is safe to emit now.
    /// `stopped` becomes true when a full stop sequence was matched; text
    /// preceding the sequence is returned and everything after is discarded.
    std::string feed(const std::string& piece, bool& stopped);

    /// Flushes held-back text (call when generation ends without a stop hit).
    std::string flush();

    bool hasStops() const noexcept { return !stops_.empty(); }

private:
    std::vector<std::string> stops_;
    std::string held_;
    bool stopped_ = false;
};

/// Trims any incomplete trailing UTF-8 sequence from `text` (used when a
/// token limit cuts generation in the middle of a multibyte character).
void trimIncompleteUtf8Tail(std::string& text);

/// Reason strings used for InferenceResult::finish_reason.
namespace finish {
constexpr const char* kEos          = "eos";
constexpr const char* kMaxTokens    = "max_tokens";
constexpr const char* kStopSequence = "stop_sequence";
constexpr const char* kTimeBudget   = "time_budget";
constexpr const char* kCancelled    = "cancelled";
constexpr const char* kContextLimit = "context_limit";
} // namespace finish

} // namespace generation
} // namespace agr
