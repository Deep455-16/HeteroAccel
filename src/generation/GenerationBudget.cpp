// src/generation/GenerationBudget.cpp
#include "generation/GenerationBudget.h"

#include <algorithm>
#include <cctype>

namespace agr {
namespace generation {

namespace {

std::string lowerCopy(const std::string& s) {
    std::string r = s;
    for (auto& c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return r;
}

void clampInt(int& value, int lo, int hi, const char* name,
              BudgetResolution& res) {
    if (value < lo) {
        res.warnings.push_back(std::string(name) + " " + std::to_string(value) +
                               " below minimum; raised to " + std::to_string(lo));
        value = lo;
        res.clamped = true;
    } else if (value > hi) {
        res.warnings.push_back(std::string(name) + " " + std::to_string(value) +
                               " above hard ceiling; lowered to " + std::to_string(hi));
        value = hi;
        res.clamped = true;
    }
}

} // namespace

const char* profileToString(TaskProfile p) noexcept {
    switch (p) {
        case TaskProfile::AUTO:               return "auto";
        case TaskProfile::SHORT_ANSWER:       return "short_answer";
        case TaskProfile::EXPLANATION:        return "explanation";
        case TaskProfile::CODE:               return "code";
        case TaskProfile::LONG_FORM:          return "long_form";
        case TaskProfile::RETRIEVAL_GROUNDED: return "retrieval_grounded";
    }
    return "auto";
}

bool profileFromString(const std::string& name, TaskProfile& out) noexcept {
    const std::string n = lowerCopy(name);
    if (n.empty() || n == "auto")      { out = TaskProfile::AUTO; return true; }
    if (n == "short_answer")           { out = TaskProfile::SHORT_ANSWER; return true; }
    if (n == "explanation")            { out = TaskProfile::EXPLANATION; return true; }
    if (n == "code")                   { out = TaskProfile::CODE; return true; }
    if (n == "long_form")              { out = TaskProfile::LONG_FORM; return true; }
    if (n == "retrieval_grounded")     { out = TaskProfile::RETRIEVAL_GROUNDED; return true; }
    return false;
}

GenerationBudget defaultsForProfile(TaskProfile profile) {
    GenerationBudget b;
    switch (profile) {
        case TaskProfile::SHORT_ANSWER:
            b.context_tokens = 1024; b.max_output_tokens = 64;
            b.time_budget_ms = 60000; b.max_retries = 0;
            break;
        case TaskProfile::CODE:
            b.context_tokens = 2048; b.max_output_tokens = 640;
            b.time_budget_ms = 240000; b.max_retries = 1;
            break;
        case TaskProfile::LONG_FORM:
            b.context_tokens = 4096; b.max_output_tokens = 1024;
            b.time_budget_ms = 420000; b.max_retries = 1;
            break;
        case TaskProfile::RETRIEVAL_GROUNDED:
            b.context_tokens = 4096; b.max_output_tokens = 512;
            b.time_budget_ms = 240000; b.max_retries = 0;
            break;
        case TaskProfile::AUTO:
        case TaskProfile::EXPLANATION:
        default:
            b.context_tokens = 2048; b.max_output_tokens = 384;
            b.time_budget_ms = 180000; b.max_retries = 0;
            break;
    }
    return b;
}

BudgetResolution resolveBudget(TaskProfile profile, const BudgetOverrides& ov) {
    BudgetResolution res;
    res.budget = defaultsForProfile(profile);
    GenerationBudget& b = res.budget;

    if (ov.max_output_tokens > 0) b.max_output_tokens = ov.max_output_tokens;
    if (ov.context_tokens > 0)    b.context_tokens    = ov.context_tokens;
    if (ov.time_budget_ms > 0)    b.time_budget_ms    = ov.time_budget_ms;
    if (ov.max_retries >= 0)      b.max_retries       = ov.max_retries;

    clampInt(b.max_output_tokens, BudgetLimits::kMinOutputTokens,
             BudgetLimits::kMaxOutputTokens, "max_output_tokens", res);
    clampInt(b.context_tokens, BudgetLimits::kMinContextTokens,
             BudgetLimits::kMaxContextTokens, "context_tokens", res);
    clampInt(b.time_budget_ms, BudgetLimits::kMinTimeMs,
             BudgetLimits::kMaxTimeMs, "time_budget_ms", res);
    clampInt(b.max_retries, 0, BudgetLimits::kMaxRetries, "max_retries", res);

    // The output must always leave room for a prompt inside the context.
    const int needed = b.max_output_tokens + BudgetLimits::kPromptReserveTokens;
    if (b.context_tokens < needed) {
        if (needed <= BudgetLimits::kMaxContextTokens && ov.context_tokens <= 0) {
            // Context came from the profile: grow it to fit the requested output.
            res.warnings.push_back("context_tokens " + std::to_string(b.context_tokens) +
                                   " too small for max_output_tokens " +
                                   std::to_string(b.max_output_tokens) +
                                   "; raised to " + std::to_string(needed));
            b.context_tokens = needed;
        } else {
            // Caller fixed the context: shrink the output instead of the prompt room.
            const int fit = std::max(BudgetLimits::kMinOutputTokens,
                                     b.context_tokens - BudgetLimits::kPromptReserveTokens);
            res.warnings.push_back("max_output_tokens " + std::to_string(b.max_output_tokens) +
                                   " leaves no prompt room in context_tokens " +
                                   std::to_string(b.context_tokens) +
                                   "; lowered to " + std::to_string(fit));
            b.max_output_tokens = fit;
        }
        res.clamped = true;
    }

    // Stop sequences: caller extras only; empty and duplicate entries dropped.
    for (const auto& s : ov.extra_stop_sequences) {
        if (s.empty()) {
            res.warnings.push_back("empty stop sequence ignored");
            continue;
        }
        if (std::find(b.stop_sequences.begin(), b.stop_sequences.end(), s) ==
            b.stop_sequences.end()) {
            b.stop_sequences.push_back(s);
        }
    }
    return res;
}

Deadline::Deadline(int time_budget_ms) {
    if (time_budget_ms > 0) {
        enabled_ = true;
        end_ = std::chrono::steady_clock::now() +
               std::chrono::milliseconds(time_budget_ms);
    }
}

bool Deadline::expired() const noexcept {
    return enabled_ && std::chrono::steady_clock::now() >= end_;
}

double Deadline::remainingMs() const noexcept {
    if (!enabled_) return 1e18;
    const auto now = std::chrono::steady_clock::now();
    if (now >= end_) return 0.0;
    return std::chrono::duration<double, std::milli>(end_ - now).count();
}

StopSequenceFilter::StopSequenceFilter(std::vector<std::string> stops) {
    for (auto& s : stops) {
        if (!s.empty()) stops_.push_back(std::move(s));
    }
}

std::string StopSequenceFilter::feed(const std::string& piece, bool& stopped) {
    stopped = false;
    if (stopped_) { stopped = true; return std::string(); }
    if (stops_.empty()) return piece;

    held_ += piece;

    // 1. Full stop sequence present? Emit text before the earliest match.
    size_t best = std::string::npos;
    for (const auto& s : stops_) {
        const size_t p = held_.find(s);
        if (p != std::string::npos && p < best) best = p;
    }
    if (best != std::string::npos) {
        std::string out = held_.substr(0, best);
        held_.clear();
        stopped_ = true;
        stopped = true;
        return out;
    }

    // 2. Hold back the longest suffix that is a proper prefix of any stop.
    size_t hold = 0;
    for (const auto& s : stops_) {
        const size_t maxk = std::min(s.size() - 1, held_.size());
        for (size_t k = maxk; k > hold; --k) {
            if (held_.compare(held_.size() - k, k, s, 0, k) == 0) {
                hold = k;
                break;
            }
        }
    }
    std::string out = held_.substr(0, held_.size() - hold);
    held_.erase(0, held_.size() - hold);
    return out;
}

std::string StopSequenceFilter::flush() {
    std::string out = std::move(held_);
    held_.clear();
    return out;
}

void trimIncompleteUtf8Tail(std::string& text) {
    // Walk back over at most 3 continuation bytes to the lead byte.
    size_t n = text.size();
    if (n == 0) return;
    size_t i = n;
    size_t back = 0;
    while (i > 0 && back < 4) {
        --i;
        ++back;
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if ((c & 0xC0) == 0x80) continue;  // continuation byte, keep walking
        size_t need = 1;
        if ((c & 0x80) == 0x00)      need = 1;
        else if ((c & 0xE0) == 0xC0) need = 2;
        else if ((c & 0xF0) == 0xE0) need = 3;
        else if ((c & 0xF8) == 0xF0) need = 4;
        else return;  // invalid lead byte: leave as-is, not an incomplete tail
        if (n - i < need) text.erase(i);  // lead present but sequence truncated
        return;
    }
}

} // namespace generation
} // namespace agr
