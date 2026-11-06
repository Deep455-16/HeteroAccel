// src/hydroxl/HydroXLTokenizer.cpp
// Phase 19 / Phase 27: Universal BPE tokenizer (GPT-2 byte-level + SentencePiece llama).

#include "hydroxl/HydroXLTokenizer.h"

#include <algorithm>
#include <cstdio>
#include <limits>

namespace agr {
namespace hydroxl {

namespace {

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------
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
    return c0;
}

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

bool cpIsSpace(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' ||
           cp == '\v' || cp == '\f' || cp == '\r';
}
bool cpIsNewline(uint32_t cp) { return cp == '\r' || cp == '\n'; }
bool cpIsDigit(uint32_t cp)   { return cp >= '0' && cp <= '9'; }
bool cpIsLetter(uint32_t cp) {
    if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return true;
    return cp >= 0x80;
}
bool cpIsOther(uint32_t cp) {
    return !cpIsSpace(cp) && !cpIsDigit(cp) && !cpIsLetter(cp);
}
bool cpIsPrefixable(uint32_t cp) {
    return !cpIsNewline(cp) && !cpIsDigit(cp) && !cpIsLetter(cp);
}

char asciiLower(char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

int hexVal(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

} // namespace

// ---------------------------------------------------------------------------
// GPT-2 byte<->unicode tables
// ---------------------------------------------------------------------------
void HydroXLTokenizer::buildByteTables() {
    if (byte_tables_built_) return;

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
    if (model != "gpt2" && model != "llama") {
        last_error_ = "Unsupported tokenizer model '" + model +
                      "' (supported: 'gpt2', 'llama')";
        return false;
    }

    std::vector<std::string> tokens = gguf.kvStringArray("tokenizer.ggml.tokens");
    if (tokens.empty()) {
        last_error_ = "tokenizer.ggml.tokens (string array) missing or empty "
                      "(Unsupported tokenizer configuration)";
        return false;
    }

    vocab_ = std::move(tokens);
    str_to_id_.clear();
    str_to_id_.reserve(vocab_.size());
    for (size_t id = 0; id < vocab_.size(); ++id) {
        str_to_id_[vocab_[id]] = static_cast<TokenID>(id);
    }

    merge_rank_.clear();
    scores_.clear();
    byte_fallback_ids_.fill(TOKEN_INVALID);

    if (model == "gpt2") {
        tok_type_ = TokenizerType::GPT2;
        std::vector<std::string> merges = gguf.kvStringArray("tokenizer.ggml.merges");
        if (merges.empty()) {
            last_error_ = "tokenizer.ggml.merges (string array) missing or empty — "
                          "required for gpt2 BPE";
            return false;
        }
        merge_rank_.reserve(merges.size());
        for (size_t i = 0; i < merges.size(); ++i) {
            const std::string& m = merges[i];
            size_t sp = m.find(' ');
            if (sp == std::string::npos || m.find(' ', sp + 1) != std::string::npos) {
                last_error_ = "Malformed merge entry #" + std::to_string(i) + ": '" + m + "'";
                return false;
            }
            merge_rank_[m] = static_cast<int>(i);
        }
        add_bos_ = false;
        add_space_prefix_ = false;
    } else {
        tok_type_ = TokenizerType::LLAMA;
        scores_.resize(vocab_.size());
        bool has_scores = false;
        if (auto* v = gguf.kv("tokenizer.ggml.scores")) {
            if (auto* arr = std::get_if<std::vector<float>>(v)) {
                has_scores = !arr->empty();
                for (size_t i = 0; i < vocab_.size(); ++i) {
                    scores_[i] = (i < arr->size()) ? (*arr)[i] : -static_cast<float>(i);
                }
            }
        }
        if (!has_scores) {
            for (size_t i = 0; i < vocab_.size(); ++i) {
                scores_[i] = -static_cast<float>(i);
            }
        }

        for (int b = 0; b < 256; ++b) {
            char buf_upper[8];
            char buf_lower[8];
            std::snprintf(buf_upper, sizeof(buf_upper), "<0x%02X>", b);
            std::snprintf(buf_lower, sizeof(buf_lower), "<0x%02x>", b);
            TokenID tid = tokenToId(buf_upper);
            if (tid == TOKEN_INVALID) tid = tokenToId(buf_lower);
            byte_fallback_ids_[static_cast<size_t>(b)] = tid;
        }

        std::string arch = gguf.kvString("general.architecture", "");
        add_bos_ = true;
        add_space_prefix_ = (arch != "gemma" && arch != "gemma2");
        if (auto* v = gguf.kv("tokenizer.ggml.add_space_prefix")) {
            if (auto* b = std::get_if<bool>(v)) add_space_prefix_ = *b;
        }
    }

    // Optional token-type array (1=normal 2=unknown 3=control 4=user 5=unused 6=byte)
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
    }

    loaded_ = true;
    return true;
}

// ---------------------------------------------------------------------------
// Pre-tokenization (GPT-2 / Qwen2 split rules)
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

        // Rule 2: [^\r\n L N]? L+
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
                j = i + l0;
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
        }

        // Rule 5 & 6: whitespace runs
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
            if (j == n) {
                pieces.push_back(text.substr(i, j - i));
                i = j;
                continue;
            }
            if (j - i >= 2) {
                size_t end = j - 1;
                pieces.push_back(text.substr(i, end - i));
                i = end;
                continue;
            }
            pieces.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        pieces.push_back(text.substr(i, l0));
        i += l0;
    }

    return pieces;
}

// ---------------------------------------------------------------------------
// GPT-2 BPE on a single pre-token piece
// ---------------------------------------------------------------------------
std::vector<TokenID> HydroXLTokenizer::bpe(const std::string& piece) const {
    std::vector<std::string> symbols;
    symbols.reserve(piece.size());
    for (unsigned char b : piece) {
        symbols.push_back(byte_to_unicode_[b]);
    }

    std::string pair_buf;
    pair_buf.reserve(64);

    while (symbols.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        size_t best_idx = std::string::npos;
        std::string best_left, best_right;

        for (size_t s = 0; s + 1 < symbols.size(); ++s) {
            pair_buf.assign(symbols[s]);
            pair_buf.push_back(' ');
            pair_buf.append(symbols[s + 1]);

            auto it = merge_rank_.find(pair_buf);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank  = it->second;
                best_idx   = s;
                best_left  = symbols[s];
                best_right = symbols[s + 1];
            }
        }
        if (best_idx == std::string::npos) break;

        std::string merged = best_left + best_right;
        std::vector<std::string> next;
        next.reserve(symbols.size() - 1);
        for (size_t s = 0; s < symbols.size(); ) {
            if (s + 1 < symbols.size() &&
                symbols[s] == best_left && symbols[s + 1] == best_right) {
                next.push_back(merged);
                s += 2;
            } else {
                next.push_back(std::move(symbols[s]));
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
// SentencePiece BPE ("llama" model)
// ---------------------------------------------------------------------------
std::vector<TokenID> HydroXLTokenizer::encodeSentencePiece(const std::string& text) const {
    std::vector<TokenID> ids;
    if (text.empty()) return ids;

    // 1. Space prefix and ' ' -> '▁' (U+2581 = \xE2\x96\x81) replacement
    static const char* kSpSpace = "\xE2\x96\x81";
    std::string norm;
    norm.reserve(text.size() * 2 + 3);
    if (add_space_prefix_ && text[0] != ' ') {
        norm.append(kSpSpace, 3);
    }
    for (char c : text) {
        if (c == ' ') {
            norm.append(kSpSpace, 3);
        } else {
            norm.push_back(c);
        }
    }

    // 2. Split into initial UTF-8 codepoint symbols
    std::vector<std::string> symbols;
    size_t i = 0;
    while (i < norm.size()) {
        size_t len = 1;
        (void)decodeUtf8(norm, i, len);
        symbols.push_back(norm.substr(i, len));
        i += len;
    }

    // 3. Iteratively merge adjacent pair with highest score in vocab
    std::string candidate;
    candidate.reserve(64);
    while (symbols.size() > 1) {
        float best_score = -std::numeric_limits<float>::infinity();
        size_t best_idx = std::string::npos;

        for (size_t s = 0; s + 1 < symbols.size(); ++s) {
            candidate.assign(symbols[s]);
            candidate.append(symbols[s + 1]);
            TokenID tid = tokenToId(candidate);
            if (tid != TOKEN_INVALID) {
                float sc = scores_[static_cast<size_t>(tid)];
                if (sc > best_score) {
                    best_score = sc;
                    best_idx   = s;
                }
            }
        }
        if (best_idx == std::string::npos) break;

        symbols[best_idx] += symbols[best_idx + 1];
        symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_idx + 1));
    }

    // 4. Look up merged symbols or fall back to <0xXX> byte tokens / unk
    ids.reserve(symbols.size());
    for (const auto& sym : symbols) {
        TokenID tid = tokenToId(sym);
        if (tid != TOKEN_INVALID) {
            ids.push_back(tid);
        } else {
            for (unsigned char b : sym) {
                TokenID b_id = byte_fallback_ids_[b];
                if (b_id != TOKEN_INVALID) {
                    ids.push_back(b_id);
                } else if (unk_id_ != TOKEN_INVALID) {
                    ids.push_back(unk_id_);
                }
            }
        }
    }
    return ids;
}

std::string HydroXLTokenizer::decodeSentencePieceToken(const std::string& raw_tok) const {
    if (raw_tok.size() == 6 && raw_tok[0] == '<' && raw_tok[1] == '0' &&
        (raw_tok[2] == 'x' || raw_tok[2] == 'X') && raw_tok[5] == '>') {
        int hi = hexVal(raw_tok[3]);
        int lo = hexVal(raw_tok[4]);
        if (hi >= 0 && lo >= 0) {
            char b = static_cast<char>((hi << 4) | lo);
            return std::string(1, b);
        }
    }

    std::string out;
    out.reserve(raw_tok.size());
    for (size_t i = 0; i < raw_tok.size(); ) {
        if (i + 2 < raw_tok.size() &&
            static_cast<unsigned char>(raw_tok[i])     == 0xE2 &&
            static_cast<unsigned char>(raw_tok[i + 1]) == 0x96 &&
            static_cast<unsigned char>(raw_tok[i + 2]) == 0x81) {
            out.push_back(' ');
            i += 3;
        } else {
            out.push_back(raw_tok[i]);
            ++i;
        }
    }
    return out;
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

    if (tok_type_ == TokenizerType::LLAMA) {
        auto toks = encodeSentencePiece(text);
        result.insert(result.end(), toks.begin(), toks.end());
        return result;
    }

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
    if (tok_type_ == TokenizerType::LLAMA) {
        return decodeSentencePieceToken(vocab_[static_cast<size_t>(id)]);
    }
    return vocab_[static_cast<size_t>(id)];
}

std::string HydroXLTokenizer::decode(const std::vector<TokenID>& ids,
                                      bool skip_special) const {
    if (!loaded_) return "";

    if (tok_type_ == TokenizerType::LLAMA) {
        std::string out;
        for (TokenID id : ids) {
            if (id < 0 || static_cast<size_t>(id) >= vocab_.size()) continue;
            size_t idx = static_cast<size_t>(id);
            if (skip_special) {
                if (id == bos_id_ || id == eos_id_) continue;
                if (idx < token_type_.size()) {
                    int t = token_type_[idx];
                    if (t == 3 || t == 5) continue;
                }
            }
            out += decodeSentencePieceToken(vocab_[idx]);
        }
        if (add_space_prefix_ && !out.empty() && out[0] == ' ') {
            out.erase(out.begin());
        }
        return out;
    }

    std::string mapped;
    for (TokenID id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= vocab_.size()) continue;
        size_t idx = static_cast<size_t>(id);
        if (skip_special && idx < token_type_.size()) {
            int t = token_type_[idx];
            if (t == 3 || t == 5) continue;
        }
        mapped += vocab_[idx];
    }

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
