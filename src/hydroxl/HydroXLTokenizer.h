// src/hydroxl/HydroXLTokenizer.h
// Phase 19 / Phase 27: Universal BPE tokenizer for HydroXL.
//
// Supports both GGUF tokenizer families used across supported architectures:
//   1. "gpt2"  — Byte-level BPE with explicit merge ranks (Qwen2/Qwen2.5, Llama 3, Mistral v3)
//   2. "llama" — SentencePiece BPE with token scores, '▁' (U+2581) whitespace
//                normalization, and <0xXX> byte fallback (Llama 1/2, Mistral, Phi-3, Gemma 1/2)
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

enum class TokenizerType : uint32_t {
    GPT2  = 0, ///< Byte-level BPE (tokenizer.ggml.model = "gpt2")
    LLAMA = 1, ///< SentencePiece BPE (tokenizer.ggml.model = "llama")
};

class HydroXLTokenizer {
public:
    HydroXLTokenizer() = default;

    // Non-copyable (large vocab maps)
    HydroXLTokenizer(const HydroXLTokenizer&) = delete;
    HydroXLTokenizer& operator=(const HydroXLTokenizer&) = delete;
    HydroXLTokenizer(HydroXLTokenizer&&) = default;

    /// Build the tokenizer from a GGUF file.
    /// Returns false if required metadata is missing or the tokenizer
    /// family is unsupported.
    bool loadFromGGUF(GGUFLoader& gguf);

    bool isLoaded() const { return loaded_; }

    TokenizerType tokenizerType() const { return tok_type_; }
    uint32_t vocabSize() const { return static_cast<uint32_t>(vocab_.size()); }
    TokenID  bosToken()  const { return bos_id_; }
    TokenID  eosToken()  const { return eos_id_; }
    TokenID  unkToken()  const { return unk_id_; }

    /// Whether this tokenizer prepends BOS on encode.
    bool addsBos() const { return add_bos_; }

    /// Whether this SentencePiece tokenizer prepends a leading space.
    bool addsSpacePrefix() const { return add_space_prefix_; }

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

    // Split text into pre-tokenization pieces (GPT-2 / Qwen2 rules).
    std::vector<std::string> preTokenize(const std::string& text) const;

    // GPT-2 byte-level BPE on one pre-token piece.
    std::vector<TokenID> bpe(const std::string& piece) const;

    // SentencePiece BPE ("llama" model) on full text.
    std::vector<TokenID> encodeSentencePiece(const std::string& text) const;

    // Decode a single SentencePiece token string (<0xXX> or ▁ replacement).
    std::string decodeSentencePieceToken(const std::string& raw_tok) const;

    // Look up a string in the vocabulary -> token ID (or TOKEN_INVALID)
    TokenID tokenToId(const std::string& s) const;

    bool loaded_ = false;
    TokenizerType tok_type_ = TokenizerType::GPT2;
    bool add_bos_ = false;
    bool add_space_prefix_ = true;

    std::vector<std::string> vocab_;            ///< vocab_[id] -> token text
    std::unordered_map<std::string, TokenID> str_to_id_; ///< Reverse lookup

    // GPT-2: merge_rank_["a b"] = rank (lower = merged earlier)
    std::unordered_map<std::string, int> merge_rank_;

    // SentencePiece ("llama"): per-token merge priority score (higher = merged earlier)
    std::vector<float> scores_;
    std::array<TokenID, 256> byte_fallback_ids_{};

    std::vector<int> token_type_;   ///< 1=normal 2=unknown 3=control 4=user 5=unused 6=byte

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
