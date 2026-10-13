// src/hydroxl/HydroXLTokenizer.h
// Phase 17: BPE tokenizer for HydroXL (Qwen2.5 tiktoken-style).
//
// Qwen2.5 uses a BytePair Encoding (BPE) tokenizer stored in GGUF as:
//   tokenizer.ggml.model        = "gpt2"
//   tokenizer.ggml.tokens       = [array of strings]  (vocabulary)
//   tokenizer.ggml.token_type   = [array of int32]    (0=normal, 3=control/special)
//   tokenizer.ggml.merges       = [array of strings]  ("a b" → merge)
//   tokenizer.ggml.bos_token_id = 151643
//   tokenizer.ggml.eos_token_id = 151645
//
// This tokenizer:
//   1. Builds the vocabulary from the token array
//   2. Builds the merge priority map from the merges array
//   3. Encodes strings via BPE
//   4. Decodes token IDs back to UTF-8 strings
//
// Does NOT depend on llama.cpp tokenizer at any point.

#pragma once

#include "hydroxl/HydroXLTypes.h"
#include "hydroxl/GGUFLoader.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace agr {
namespace hydroxl {

class HydroXLTokenizer {
public:
    HydroXLTokenizer() = default;

    // Non-copyable (large vocab maps)
    HydroXLTokenizer(const HydroXLTokenizer&) = delete;
    HydroXLTokenizer& operator=(const HydroXLTokenizer&) = delete;
    HydroXLTokenizer(HydroXLTokenizer&&) = default;

    /// Build the tokenizer from a GGUF file.
    /// Returns false if required metadata is missing.
    bool loadFromGGUF(GGUFLoader& gguf);

    bool isLoaded() const { return loaded_; }

    uint32_t vocabSize() const { return static_cast<uint32_t>(vocab_.size()); }
    TokenID  bosToken()  const { return bos_id_; }
    TokenID  eosToken()  const { return eos_id_; }
    TokenID  unkToken()  const { return unk_id_; }

    /// Encode a UTF-8 string to a sequence of token IDs.
    /// Prepends BOS token if add_bos=true.
    std::vector<TokenID> encode(const std::string& text, bool add_bos = true) const;

    /// Decode a sequence of token IDs to a UTF-8 string.
    std::string decode(const std::vector<TokenID>& ids, bool skip_special = true) const;

    /// Decode a single token ID. Returns empty string for invalid IDs.
    std::string decodeToken(TokenID id) const;

    std::string lastError() const { return last_error_; }

private:
    // Encode single byte as the single-char token text used in BPE
    static std::string byteToStr(uint8_t b);

    // BPE encode a sequence of chars to merge into tokens
    std::vector<TokenID> bpe(const std::string& word) const;

    // Look up a string in the vocabulary → token ID (or TOKEN_INVALID)
    TokenID tokenToId(const std::string& s) const;

    bool loaded_ = false;

    std::vector<std::string> vocab_;            ///< vocab_[id] → token text
    std::unordered_map<std::string, TokenID> str_to_id_; ///< Reverse lookup

    // merge_rank_[a + " " + b] = rank (lower = higher priority)
    std::unordered_map<std::string, int> merge_rank_;

    std::vector<int> token_type_;              ///< 0=normal, 3=control

    TokenID bos_id_ = 151643;
    TokenID eos_id_ = 151645;
    TokenID unk_id_ = TOKEN_INVALID;

    mutable std::string last_error_;
};

} // namespace hydroxl
} // namespace agr
