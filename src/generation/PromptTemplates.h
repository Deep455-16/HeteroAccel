// src/generation/PromptTemplates.h
// Phase 33: Versioned prompt template registry with request isolation.
//
// Every prompt sent to a neural engine is assembled here instead of at
// scattered call sites. Assembly is a pure function of its inputs (no shared
// mutable state), so concurrent requests cannot leak prompt content into one
// another. Untrusted text (conversation history, retrieved context) is
// sanitized of chat-control markers and wrapped as data.

#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace agr {
namespace generation {

/// Chat formatting family. PLAIN uses no control tokens (safe default for any
/// model); CHATML uses <|im_start|>/<|im_end|> (Qwen2 / Qwen2.5 instruct).
enum class TemplateFamily { PLAIN, CHATML };

const char* familyToString(TemplateFamily f) noexcept;

/// Chooses a template family from the GGUF architecture string. Unknown
/// architectures fall back to PLAIN rather than guessing control tokens.
TemplateFamily familyForArchitecture(const std::string& architecture);

struct PromptVariable {
    std::string name;
    bool        required  = false;
    bool        untrusted = false;  ///< Rendered inside a data block
};

struct PromptTemplate {
    std::string id;
    int         version = 1;
    std::string description;
    std::string system_text;               ///< Developer/system instruction
    std::vector<PromptVariable> variables;
};

struct ChatTurn {
    std::string role;     ///< "user" / "assistant" (others map to user)
    std::string content;
};

struct AssembleRequest {
    std::string template_id;
    TemplateFamily family = TemplateFamily::PLAIN;
    std::string query;                      ///< Required, never silently truncated
    std::vector<ChatTurn> history;          ///< Oldest first; trimmed oldest-first
    std::string untrusted_context;          ///< Optional retrieved/upstream data
    /// Extra template variables; keys must be declared by the template.
    std::map<std::string, std::string> extra_variables;
    /// Max prompt size in estimated tokens (0 = unlimited).
    int prompt_token_budget = 0;
};

struct RenderedPrompt {
    bool        ok = false;
    std::string error;
    std::string text;
    std::string template_id;
    int         template_version = 0;
    TemplateFamily family = TemplateFamily::PLAIN;
    bool        uses_control_tokens = false;  ///< True when text holds <|im_*|> markers
    int         estimated_tokens = 0;
    int         history_turns_dropped = 0;
    bool        context_truncated = false;
    std::vector<std::string> warnings;
    /// Stop sequences that guard against the model continuing the chat.
    std::vector<std::string> stop_sequences;
};

class PromptRegistry {
public:
    /// Immutable registry of built-in templates (thread-safe to share).
    static const PromptRegistry& builtin();

    const PromptTemplate* find(const std::string& id) const;
    std::vector<std::string> ids() const;

    /// Validates the request against the template and assembles the prompt.
    /// Fails (ok=false) on unknown template, empty query, undeclared or
    /// missing variables, or a query that cannot fit the token budget.
    RenderedPrompt assemble(const AssembleRequest& req) const;

private:
    std::map<std::string, PromptTemplate> templates_;
    PromptRegistry();
};

/// Strips chat-control markers (<|im_start|>, <|im_end|>, <|endoftext|>,
/// <|eot_id|>, <start_of_turn>, ...) from untrusted text so it cannot forge
/// role boundaries. Returns the number of markers removed via `removed`.
std::string sanitizeControlTokens(const std::string& text, int* removed = nullptr);

/// Conservative token estimate (UTF-8 bytes / 3, rounded up).
int estimateTokens(const std::string& text) noexcept;

} // namespace generation
} // namespace agr
