// src/hydroxl/HydroXLTokenizer.cpp
// Phase 19: byte-level BPE tokenizer (GPT-2 / Qwen2 style).
//
// See HydroXLTokenizer.h for the algorithm description and the documented
// pre-tokenizer approximation for non-ASCII input.

#include "hydroxl/HydroXLTokenizer.h"

#include <algorithm>
#include <limits>

namespace agr {
namespace hydroxl {

namespace {

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------

// Decode one UTF-8 codepoint at data[i]. Returns the codepoint and sets
// `len` to the number of bytes consumed. Invalid bytes decode as themselves
// with len=1 (lossy, matches typical byte-level tokenizer behaviour).
uint32_t decodeUtf8(const std::string& s, size_t i, size_t& len) {
    unsigned char c0 = static_cast<unsigned char>(s[i]);
    if (c0 < 0x80) { len = 1; return c0; }
    if ((c0 & 0xE0) == 0xC0 && i + 1 < s.size()) {
        len = 2;
        return ((c0 & 0x1Fu) << 6) | (static_cast<unsigned char>(s[i+1]) & 0x3Fu);
    }
    if ((c0 & 0xF0) == 0xE0 && i + 2 < s.size()) {
        len = 3;
        return ((c0 & 0x0Fu) << 12) | ((static_cast<unsigned char>(s[i+1]) & 0x3Fu) << 6)
             |  (static_cast<unsigned char>(s[i+2]) & 0x3Fu);
    }
    if ((c0 & 0xF8) == 0xF0 && i + 3 < s.size()) {
        len = 4;
        return ((c0 & 0x07u) << 18) | ((static_cast<unsigned char>(s[i+1]) & 0x3Fu) << 12)
             | ((static_cast<unsigned char>(s[i+2]) & 0x3Fu) << 6)
             |  (static_cast<unsigned char>(s[i+3]) & 0x3Fu);
    }
    len = 1;
    return c0; // invalid byte — treat as a single codepoint
}

// Encode a codepoint as UTF-8.
void encodeUtf8(uint32_t cp, std::string& out) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// Character classes for the pre-tokenizer (ASCII-precise, non-ASCII
// approximated — see header limitation note).
bool cpIsSpace(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' ||
           cp == '\v' || cp == '\f' || cp == '\r';
}
bool cpIsNewline(uint32_t cp) { return cp == '\r' || cp == '\n'; }
bool cpIsDigit(uint32_t cp)   { return cp >= '0' && cp <= '9'; }
bool cpIsLetter(uint32_t cp) {
    if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return true;
    return cp >= 0x80; // approximation: non-ASCII treated as letters
}
bool cpIsOther(uint32_t cp) { // [^\s\p{L}\p{N}]
    return !cpIsSpace(cp) && !cpIsDigit(cp) && !cpIsLetter(cp);
}
// The Qwen2 letter-run prefix class [^\r\n\p{L}\p{N}]: unlike cpIsOther this
// INCLUDES spaces and tabs (so " word" and "\tword" stay single pieces).
bool cpIsPrefixable(uint32_t cp) {
    return !cpIsNewline(cp) && !cpIsDigit(cp) && !cpIsLetter(cp);
}

// Lowercase ASCII letter (for case-insensitive contraction matching).
char asciiLower(char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

} // namespace

// ---------------------------------------------------------------------------
// GPT-2 byte<->unicode tables
// ---------------------------------------------------------------------------
void HydroXLTokenizer::buildByteTables() {
    if (byte_tables_built_) return;

    // Bytes that map to themselves: '!'..'~', U+00A1..U+00AC, U+00AE..U+00FF.
    // All other bytes map to codepoints 256, 257, ... in increasing order.
    uint32_t next = 256;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp;
        if ((b >= 33 && b <= 126) ||
            (b >= 161 && b <= 172) ||
            (b >= 174 && b <= 255)) {
            cp = static_cast<uint32_t>(b);
        } else {
            cp = next++;
        }
        std::string utf8;
        encodeUtf8(cp, utf8);
        byte_to_unicode_[static_cast<size_t>(b)] = utf8;
        unicode_to_byte_[utf8] = static_cast<uint8_t>(b);
    }
    byte_tables_built_ = true;
}

// ---------------------------------------------------------------------------
// Load from GGUF metadata
// ---------------------------------------------------------------------------
bool HydroXLTokenizer::loadFromGGUF(GGUFLoader& gguf) {
    loaded_ = false;
    buildByteTables();

    std::string model = gguf.kvString("tokenizer.ggml.model", "");
    if (model != "gpt2") {
        last_error_ = "Unsupported tokenizer model '" + model +
                      "' (only 'gpt2' byte-level BPE is supported)";
        return false;
    }

    std::vector<std::string> tokens = gguf.kvStringArray("tokenizer.ggml.tokens");
    if (tokens.empty()) {
        last_error_ = "tokenizer.ggml.tokens (string array) missing or empty";
        return false;
    }
    std::vector<std::string> merges = gguf.kvStringArray("tokenizer.ggml.merges");
    if (merges.empty()) {
        last_error_ = "tokenizer.ggml.merges (string array) missing or empty — "
                      "required for gpt2 BPE";
        return false;
    }

    vocab_ = std::move(tokens);
    str_to_id_.clear();
    str_to_id_.reserve(vocab_.size());
    for (size_t id = 0; id < vocab_.size(); ++id) {
        str_to_id_[vocab_[id]] = static_cast<TokenID>(id);
    }

    merge_rank_.clear();
    merge_rank_.reserve(merges.size());
    for (size_t i = 0; i < merges.size(); ++i) {
        // Each merge is "left right" with a single separating space.
        const std::string& m = merges[i];
        size_t sp = m.find(' ');
        if (sp == std::string::npos || m.find(' ', sp + 1) != std::string::npos) {
            last_error_ = "Malformed merge entry #" + std::to_string(i) + ": '" + m + "'";
            return false;
        }
        merge_rank_[m] = static_cast<int>(i);
    }

    // Optional token-type array (1=normal 2=unknown 3=control 4=user 5=unused)
    token_type_.assign(vocab_.size(), 1);
    if (auto* v = gguf.kv("tokenizer.ggml.token_type")) {
        if (auto* arr = std::get_if<std::vector<int32_t>>(v)) {
            for (size_t i = 0; i < arr->size() && i < token_type_.size(); ++i) {
                token_type_[i] = (*arr)[i];
            }
        }
    }

    bos_id_ = static_cast<TokenID>(gguf.kvUInt32("tokenizer.ggml.bos_token_id",
                                                 static_cast<uint32_t>(TOKEN_INVALID)));
    eos_id_ = static_cast<TokenID>(gguf.kvUInt32("tokenizer.ggml.eos_token_id",
                                                 static_cast<uint32_t>(TOKEN_INVALID)));
    unk_id_ = static_cast<TokenID>(gguf.kvUInt32("tokenizer.ggml.unknown_token_id",
                                                 static_cast<uint32_t>(TOKEN_INVALID)));
    if (auto* v = gguf.kv("tokenizer.ggml.add_bos_token")) {
        if (auto* b = std::get_if<bool>(v)) add_bos_ = *b;
    } else {
        add_bos_ = false; // Qwen2 does not prepend BOS
    }

    loaded_ = true;
    return true;
}

// ---------------------------------------------------------------------------
// Pre-tokenization (Qwen2 split rules)
// ---------------------------------------------------------------------------
std::vector<std::string> HydroXLTokenizer::preTokenize(const std::string& text) const {
    std::vector<std::string> pieces;
    size_t i = 0;
    const size_t n = text.size();

    while (i < n) {
        size_t l0 = 1, l1 = 1, l2 = 1;
        uint32_t c0 = decodeUtf8(text, i, l0);
        uint32_t c1 = (i + l0 < n) ? decodeUtf8(text, i + l0, l1) : 0;
        uint32_t c2 = (i + l0 + (c1 ? l1 : 0) < n)
                          ? decodeUtf8(text, i + l0 + (c1 ? l1 : 0), l2) : 0;

        // Rule 1: case-insensitive contractions 's 't 're 've 'm 'll 'd
        if (c0 == '\'' && c1 != 0) {
            char a = asciiLower(static_cast<char>(c1));
            size_t take = 0;
            if (a == 's' || a == 't' || a == 'm' || a == 'd') take = l0 + l1;
            else if (c2 != 0) {
                char b2 = asciiLower(static_cast<char>(c2));
                if ((a == 'r' && b2 == 'e') || (a == 'v' && b2 == 'e') ||
                    (a == 'l' && b2 == 'l')) take = l0 + l1 + l2;
            }
            if (take > 0) {
                pieces.push_back(text.substr(i, take));
                i += take;
                continue;
            }
        }

        // Rule 2: [^\r\n L N]? L+  — optional non-letter/digit/newline prefix
        if (cpIsLetter(c0)) {
            size_t j = i + l0;
            while (j < n) {
                size_t cl; uint32_t c = decodeUtf8(text, j, cl);
                if (!cpIsLetter(c)) break;
                j += cl;
            }
            pieces.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }
        if (cpIsPrefixable(c0) && c1 != 0 && cpIsLetter(c1)) {
            size_t j = i + l0;
            while (j < n) {
                size_t cl; uint32_t c = decodeUtf8(text, j, cl);
                if (!cpIsLetter(c)) break;
                j += cl;
            }
            pieces.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        // Rule 3: digits in groups of up to three
        if (cpIsDigit(c0)) {
            size_t j = i + l0;
            int cnt = 1;
            while (j < n && cnt < 3) {
                size_t cl; uint32_t c = decodeUtf8(text, j, cl);
                if (!cpIsDigit(c)) break;
                j += cl;
                ++cnt;
            }
            pieces.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        // Rule 4: " "? [^\s L N]+ [\r\n]*
        {
            size_t start = i;
            size_t j = i;
            if (c0 == ' ' && c1 != 0 && cpIsOther(c1) && !cpIsNewline(c1)) {
                j = i + l0; // consume the leading space
            }
            size_t k = j;
            bool any = false;
            while (k < n) {
                size_t cl; uint32_t c = decodeUtf8(text, k, cl);
                if (!(cpIsOther(c) && !cpIsNewline(c))) break;
                k += cl;
                any = true;
            }
            if (any) {
                while (k < n) {
                    size_t cl; uint32_t c = decodeUtf8(text, k, cl);
                    if (!cpIsNewline(c)) break;
                    k += cl;
                }
                pieces.push_back(text.substr(start, k - start));
                i = k;
                continue;
            }
            // space followed by end-of-text or a non-"other" char: handled below
        }

        // Rule 5: \s* [\r\n]+  (whitespace run containing a newline; the
        // match ends at the last newline of the run)
        if (cpIsSpace(c0)) {
            size_t j = i;
            size_t lastNl = std::string::npos;
            while (j < n) {
                size_t cl; uint32_t c = decodeUtf8(text, j, cl);
                if (!cpIsSpace(c)) break;
                if (cpIsNewline(c)) lastNl = j + cl;
                j += cl;
            }
            if (lastNl != std::string::npos) {
                pieces.push_back(text.substr(i, lastNl - i));
                i = lastNl;
                continue;
            }

            // Rule 6: \s+(?!\S) — trailing whitespace, or all but the final
            // space of a run whose last space attaches to the next piece.
            if (j == n) {
                pieces.push_back(text.substr(i, j - i)); // whitespace at end
                i = j;
                continue;
            }
            if (j - i >= 2) {
                size_t end = j - 1; // leave the last space for the next piece
                // The last space is one byte (' ')
                pieces.push_back(text.substr(i, end - i));
                i = end;
                continue;
            }
            // Single space before a non-letter/non-other char (a digit):
            // falls through to rule 7 semantics — emit the lone space.
            pieces.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        // Fallback: consume one codepoint (unexpected for well-formed input,
        // keeps the tokenizer total).
        pieces.push_back(text.substr(i, l0));
        i += l0;
    }

    return pieces;
}

// ---------------------------------------------------------------------------
// BPE on a single pre-token piece
// ---------------------------------------------------------------------------
std::vector<TokenID> HydroXLTokenizer::bpe(const std::string& piece) const {
    // Map raw bytes through the byte->unicode table; each mapped char is one
    // initial symbol.
    std::vector<std::string> symbols;
    symbols.reserve(piece.size());
    for (unsigned char b : piece) {
        symbols.push_back(byte_to_unicode_[b]);
    }

    // Repeatedly merge the lowest-rank adjacent pair (all occurrences,
    // left to right), exactly like the reference GPT-2 implementation.
    while (symbols.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        std::string best_pair;
        for (size_t s = 0; s + 1 < symbols.size(); ++s) {
            auto it = merge_rank_.find(symbols[s] + " " + symbols[s + 1]);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pair = symbols[s] + " " + symbols[s + 1];
            }
        }
        if (best_pair.empty()) break; // no more merges apply

        const std::string left  = best_pair.substr(0, best_pair.find(' '));
        const std::string right = best_pair.substr(best_pair.find(' ') + 1);
        std::vector<std::string> next;
        next.reserve(symbols.size() - 1);
        for (size_t s = 0; s < symbols.size(); ) {
            if (s + 1 < symbols.size() &&
                symbols[s] == left && symbols[s + 1] == right) {
                next.push_back(left + right);
                s += 2;
            } else {
                next.push_back(symbols[s]);
                s += 1;
            }
        }
        symbols = std::move(next);
    }

    std::vector<TokenID> ids;
    ids.reserve(symbols.size());
    for (const auto& sym : symbols) {
        TokenID id = tokenToId(sym);
        if (id == TOKEN_INVALID) {
            // A merged symbol not present in the vocabulary: fall back to
            // per-byte tokens so no input is silently dropped.
            for (char ch : sym) {
                TokenID bid = tokenToId(byte_to_unicode_[
                    static_cast<unsigned char>(ch)]);
                if (bid != TOKEN_INVALID) ids.push_back(bid);
            }
        } else {
            ids.push_back(id);
        }
    }
    return ids;
}

// ---------------------------------------------------------------------------
// Vocabulary lookup
// ---------------------------------------------------------------------------
TokenID HydroXLTokenizer::tokenToId(const std::string& s) const {
    auto it = str_to_id_.find(s);
    return (it == str_to_id_.end()) ? TOKEN_INVALID : it->second;
}

// ---------------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------------
std::vector<TokenID> HydroXLTokenizer::encode(const std::string& text,
                                               bool add_bos) const {
    std::vector<TokenID> result;
    if (!loaded_) return result;
    if (add_bos && bos_id_ != TOKEN_INVALID) result.push_back(bos_id_);

    for (const auto& piece : preTokenize(text)) {
        auto toks = bpe(piece);
        result.insert(result.end(), toks.begin(), toks.end());
    }
    return result;
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------
std::string HydroXLTokenizer::decodeToken(TokenID id) const {
    if (id < 0 || static_cast<size_t>(id) >= vocab_.size()) return "";
    return vocab_[static_cast<size_t>(id)];
}

std::string HydroXLTokenizer::decode(const std::vector<TokenID>& ids,
                                      bool skip_special) const {
    if (!loaded_) return "";

    // Concatenate the byte-level token strings.
    std::string mapped;
    for (TokenID id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= vocab_.size()) continue;
        size_t idx = static_cast<size_t>(id);
        if (skip_special && idx < token_type_.size()) {
            int t = token_type_[idx];
            if (t == 3 || t == 5) continue; // control / unused
        }
        mapped += vocab_[idx];
    }

    // Reverse the unicode mapping: each UTF-8 char back to its byte. Chars
    // not in the table (e.g. literal special-token text) pass through.
    std::string out;
    out.reserve(mapped.size());
    size_t i = 0;
    while (i < mapped.size()) {
        size_t len;
        uint32_t cp = decodeUtf8(mapped, i, len);
        (void)cp;
        std::string ch = mapped.substr(i, len);
        auto it = unicode_to_byte_.find(ch);
        if (it != unicode_to_byte_.end()) {
            out += static_cast<char>(it->second);
        } else {
            out += ch;
        }
        i += len;
    }
    return out;
}

} // namespace hydroxl
} // namespace agr
