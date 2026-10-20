// src/hydroxl/HydroXLTokenizer.h
// Phase 19: byte-level BPE tokenizer for HydroXL (GPT-2 / Qwen2 style).
//
// Qwen2.5 stores its tokenizer in GGUF as:
//   tokenizer.ggml.model        = "gpt2"
//   tokenizer.ggml.tokens       = [array of strings]  byte-level vocabulary
//   tokenizer.ggml.token_type   = [array of int32]    (1=normal 3=control 4=user)
//   tokenizer.ggml.scores       = [array of float32]
//   tokenizer.ggml.merges       = [array of strings]  ("a b" ordered merge ranks)
//   tokenizer.ggml.bos_token_id / eos_token_id / unknown_token_id
//
// Encoding follows the GPT-2 byte-level BPE algorithm:
//   1. Pre-tokenize text with the Qwen2 split rules (contractions,
//      letter runs with optional leading char, digit groups of up to three,
//      punctuation runs with optional leading space, newline runs,
//      whitespace splitting).
//   2. Map every byte of each piece through the GPT-2 byte→unicode table.
//   3. Repeatedly merge the adjacent pair with the lowest merge rank.
//   4. Look the resulting symbols up in the vocabulary.
// Decoding concatenates token strings and reverses the unicode mapping.
//
// Known approximation (documented limitation): the pre-tokenizer classifies
// every non-ASCII codepoint >= U+0080 as a letter. This matches the reference
// behaviour for ASCII/Latin and CJK text but may differ for non-ASCII
// punctuation or emoji adjacent to letters.
//
// Does NOT depend on the llama.cpp tokenizer at any point.

#pragma once

#include "hydroxl/HydroXLTypes.h"
#include "hydroxl/GGUFLoader.h"

#include <array>
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
    /// Returns false if required metadata is missing or the tokenizer
    /// family is unsupported (only "gpt2" byte-level BPE is supported).
    bool loadFromGGUF(GGUFLoader& gguf);

    bool isLoaded() const { return loaded_; }

    uint32_t vocabSize() const { return static_cast<uint32_t>(vocab_.size()); }
    TokenID  bosToken()  const { return bos_id_; }
    TokenID  eosToken()  const { return eos_id_; }
    TokenID  unkToken()  const { return unk_id_; }

    /// Whether this tokenizer prepends BOS on encode (Qwen2: false).
    bool addsBos() const { return add_bos_; }

    /// Encode a UTF-8 string to a sequence of token IDs.
    /// Prepends BOS token if add_bos=true.
    std::vector<TokenID> encode(const std::string& text, bool add_bos = true) const;

    /// Decode a sequence of token IDs to a UTF-8 string.
    std::string decode(const std::vector<TokenID>& ids, bool skip_special = true) const;

    /// Decode a single token ID. Returns empty string for invalid IDs.
    std::string decodeToken(TokenID id) const;

    std::string lastError() const { return last_error_; }

private:
    // Build the GPT-2 byte<->unicode mapping tables.
    void buildByteTables();

    // Split text into pre-tokenization pieces (Qwen2 rules, see header).
    std::vector<std::string> preTokenize(const std::string& text) const;

    // BPE encode one pre-token piece (already in raw byte form).
    std::vector<TokenID> bpe(const std::string& piece) const;

    // Look up a string in the vocabulary -> token ID (or TOKEN_INVALID)
    TokenID tokenToId(const std::string& s) const;

    bool loaded_ = false;
    bool add_bos_ = false;

    std::vector<std::string> vocab_;            ///< vocab_[id] -> token text (byte-level form)
    std::unordered_map<std::string, TokenID> str_to_id_; ///< Reverse lookup

    // merge_rank_["a b"] = rank (lower = merged earlier)
    std::unordered_map<std::string, int> merge_rank_;

    std::vector<int> token_type_;   ///< 1=normal 2=unknown 3=control 4=user 5=unused

    // GPT-2 byte-level mapping: byte -> unicode char (as UTF-8 string)
    std::array<std::string, 256> byte_to_unicode_;
    // Reverse: unicode char (UTF-8) -> byte
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;
    bool byte_tables_built_ = false;

    TokenID bos_id_ = 151643;
    TokenID eos_id_ = 151645;
    TokenID unk_id_ = TOKEN_INVALID;

    mutable std::string last_error_;
};

} // namespace hydroxl
} // namespace agr
