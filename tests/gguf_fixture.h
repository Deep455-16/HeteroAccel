// tests/gguf_fixture.h
// Header-only GGUF v3 writer + deterministic tiny-model generators used by tests.
//
// The generated files are REAL GGUF files that llama.cpp can load (the same
// files are used as the trusted reference implementation in numerical tests).
// Weights are deterministic pseudo-random values, so results are reproducible
// on every platform. They do not constitute a meaningful language model.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace fixture {

// GGUF metadata value types
enum : uint32_t {
    GT_U8 = 0, GT_I8 = 1, GT_U16 = 2, GT_I16 = 3, GT_U32 = 4, GT_I32 = 5,
    GT_F32 = 6, GT_BOOL = 7, GT_STR = 8, GT_ARR = 9, GT_U64 = 10, GT_I64 = 11, GT_F64 = 12
};

// ggml tensor types (subset the fixtures use)
enum : uint32_t { GGML_F32 = 0, GGML_F16 = 1 };

class GgufWriter {
public:
    void kvU32(const std::string& k, uint32_t v) { key(k, GT_U32); put(v); ++n_kv_; }
    void kvI32(const std::string& k, int32_t v)  { key(k, GT_I32); put(v); ++n_kv_; }
    void kvF32(const std::string& k, float v)    { key(k, GT_F32); put(v); ++n_kv_; }
    void kvBool(const std::string& k, bool v)    { key(k, GT_BOOL); uint8_t b = v ? 1 : 0; put(b); ++n_kv_; }
    void kvStr(const std::string& k, const std::string& v) { key(k, GT_STR); str(v); ++n_kv_; }
    void kvStrArr(const std::string& k, const std::vector<std::string>& v) {
        key(k, GT_ARR); put<uint32_t>(GT_STR); put<uint64_t>(v.size());
        for (const auto& s : v) str(s);
        ++n_kv_;
    }
    void kvF32Arr(const std::string& k, const std::vector<float>& v) {
        key(k, GT_ARR); put<uint32_t>(GT_F32); put<uint64_t>(v.size());
        for (float f : v) put(f);
        ++n_kv_;
    }
    void kvI32Arr(const std::string& k, const std::vector<int32_t>& v) {
        key(k, GT_ARR); put<uint32_t>(GT_I32); put<uint64_t>(v.size());
        for (int32_t i : v) put(i);
        ++n_kv_;
    }

    // dims in ggml order (ne[0] = fastest-varying). `data` holds the raw bytes.
    void tensor(const std::string& name, const std::vector<uint64_t>& dims,
                uint32_t ggml_type, const std::vector<uint8_t>& data) {
        T t{name, dims, ggml_type, data};
        tensors_.push_back(std::move(t));
    }

    void tensorF32(const std::string& name, const std::vector<uint64_t>& dims,
                   const std::vector<float>& values) {
        std::vector<uint8_t> raw(values.size() * 4);
        std::memcpy(raw.data(), values.data(), raw.size());
        tensor(name, dims, GGML_F32, raw);
    }

    bool write(const std::string& path, uint32_t alignment = 32) const {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        std::vector<uint8_t> hdr;
        auto w = [&](const void* p, size_t n) { hdr.insert(hdr.end(), (const uint8_t*)p, (const uint8_t*)p + n); };
        w("GGUF", 4);
        uint32_t ver = 3; w(&ver, 4);
        uint64_t nt = tensors_.size(); w(&nt, 8);
        uint64_t nk = n_kv_; w(&nk, 8);
        hdr.insert(hdr.end(), kv_.begin(), kv_.end());

        // tensor infos with aligned offsets
        uint64_t off = 0;
        std::vector<uint64_t> offsets;
        for (const auto& t : tensors_) {
            offsets.push_back(off);
            off += t.data.size();
            off = (off + alignment - 1) / alignment * alignment;
        }
        for (size_t i = 0; i < tensors_.size(); ++i) {
            const auto& t = tensors_[i];
            uint64_t len = t.name.size(); w(&len, 8); w(t.name.data(), t.name.size());
            uint32_t nd = (uint32_t)t.dims.size(); w(&nd, 4);
            for (uint64_t d : t.dims) w(&d, 8);
            w(&t.type, 4);
            w(&offsets[i], 8);
        }
        while (hdr.size() % alignment) hdr.push_back(0);
        f.write((const char*)hdr.data(), (std::streamsize)hdr.size());
        for (size_t i = 0; i < tensors_.size(); ++i) {
            f.write((const char*)tensors_[i].data.data(), (std::streamsize)tensors_[i].data.size());
            size_t pad = (alignment - (tensors_[i].data.size() % alignment)) % alignment;
            static const char zeros[64] = {0};
            f.write(zeros, (std::streamsize)pad);
        }
        return (bool)f;
    }

private:
    struct T { std::string name; std::vector<uint64_t> dims; uint32_t type; std::vector<uint8_t> data; };
    template <typename V> void put(V v) { const uint8_t* p = (const uint8_t*)&v; kv_.insert(kv_.end(), p, p + sizeof(V)); }
    void str(const std::string& s) { put<uint64_t>(s.size()); kv_.insert(kv_.end(), s.begin(), s.end()); }
    void key(const std::string& k, uint32_t type) { str(k); put<uint32_t>(type); }

    std::vector<uint8_t> kv_;
    uint64_t n_kv_ = 0;
    std::vector<T> tensors_;
};

// Deterministic generator (xorshift) -> roughly N(0,1)-ish via sum of uniforms.
class Rng {
public:
    explicit Rng(uint64_t seed) : s_(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    float uniform() { s_ ^= s_ << 13; s_ ^= s_ >> 7; s_ ^= s_ << 17; return (float)((s_ >> 11) * (1.0 / 9007199254740992.0)); }
    float normal() { float a = 0; for (int i = 0; i < 6; ++i) a += uniform(); return (a - 3.0f) * 1.4142135f; }
private:
    uint64_t s_;
};

inline std::vector<float> randomMatrix(Rng& rng, size_t n, float scale) {
    std::vector<float> v(n);
    for (auto& x : v) x = rng.normal() * scale;
    return v;
}

// SentencePiece-style vocabulary: <unk>, <s>, </s>, 256 byte tokens, then
// "▁" + 26 lowercase letters + a few common pieces. Total = 3 + 256 + 27 + extra.
struct TinyVocab {
    std::vector<std::string> tokens;
    std::vector<float> scores;
    std::vector<int32_t> types;   // 1 normal, 2 unknown, 3 control, 6 byte
};

inline TinyVocab makeSpmVocab() {
    TinyVocab v;
    auto add = [&](const std::string& t, float s, int32_t ty) { v.tokens.push_back(t); v.scores.push_back(s); v.types.push_back(ty); };
    add("<unk>", 0.0f, 2);
    add("<s>", 0.0f, 3);
    add("</s>", 0.0f, 3);
    for (int i = 0; i < 256; ++i) {
        char buf[16]; std::snprintf(buf, sizeof(buf), "<0x%02X>", i);
        add(buf, 0.0f, 6);
    }
    const std::string sp = "\xE2\x96\x81";   // U+2581 LOWER ONE EIGHTH BLOCK
    add(sp, -1.0f, 1);
    for (char c = 'a'; c <= 'z'; ++c) add(std::string(1, c), -2.0f, 1);
    const char* extra[] = {"he", "ll", "lo", "the", "in", "er", "an", "re"};
    for (const char* e : extra) add(e, -1.5f, 1);
    for (const char* e : {"hello", "world"}) add(sp + e, -0.5f, 1);
    return v;
}

// GPT-2 byte-level vocabulary helper
inline std::string mapGpt2Byte(unsigned char b) {
    uint32_t cp;
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
        cp = b;
    } else {
        uint32_t n = 0;
        cp = 0;
        for (int i = 0; i < 256; ++i) {
            if (!((i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255))) {
                if (i == b) { cp = 256 + n; break; }
                ++n;
            }
        }
    }
    std::string s;
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return s;
}

struct TinyGpt2Vocab {
    std::vector<std::string> tokens;
    std::vector<std::string> merges;
    std::vector<int32_t> types;
};

inline TinyGpt2Vocab makeGpt2Vocab() {
    TinyGpt2Vocab v;
    for (int b = 0; b < 256; ++b) {
        v.tokens.push_back(mapGpt2Byte(static_cast<unsigned char>(b)));
        v.types.push_back(1);
    }
    v.tokens.push_back("he");                  v.types.push_back(1);
    v.tokens.push_back("ll");                  v.types.push_back(1);
    v.tokens.push_back("llo");                 v.types.push_back(1);
    v.tokens.push_back("hello");               v.types.push_back(1);
    v.tokens.push_back(mapGpt2Byte(0x20) + "w"); v.types.push_back(1);
    v.tokens.push_back("or");                  v.types.push_back(1);
    v.tokens.push_back("<|endoftext|>");       v.types.push_back(3);

    v.merges = {
        "h e",
        "l l",
        "ll o",
        "he llo",
        mapGpt2Byte(0x20) + " w",
        "o r"
    };
    return v;
}

struct TinyLlamaSpec {
    uint32_t n_embd = 64;
    uint32_t n_head = 4;
    uint32_t n_kv_head = 2;
    uint32_t n_layer = 2;
    uint32_t n_ff = 128;
    uint32_t n_ctx = 128;
    uint64_t seed = 1234;
    float    weight_scale = 0.08f;
    std::string arch = "llama";
};

// Writes a minimal Llama-architecture GGUF (F32 weights). Returns vocabulary size.
inline uint32_t writeTinyLlama(const std::string& path, const TinyLlamaSpec& sp = {}) {
    GgufWriter w;
    TinyVocab vocab = makeSpmVocab();
    const uint32_t n_vocab = (uint32_t)vocab.tokens.size();
    const uint32_t head_dim = sp.n_embd / sp.n_head;

    w.kvStr("general.architecture", sp.arch);
    w.kvStr("general.name", "heteroaccel-tiny-" + sp.arch);
    w.kvU32(sp.arch + ".context_length", sp.n_ctx);
    w.kvU32(sp.arch + ".embedding_length", sp.n_embd);
    w.kvU32(sp.arch + ".block_count", sp.n_layer);
    w.kvU32(sp.arch + ".feed_forward_length", sp.n_ff);
    w.kvU32(sp.arch + ".attention.head_count", sp.n_head);
    w.kvU32(sp.arch + ".attention.head_count_kv", sp.n_kv_head);
    w.kvF32(sp.arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
    w.kvU32(sp.arch + ".rope.dimension_count", head_dim);
    w.kvF32(sp.arch + ".rope.freq_base", 10000.0f);
    w.kvStr("tokenizer.ggml.model", "llama");
    w.kvStrArr("tokenizer.ggml.tokens", vocab.tokens);
    w.kvF32Arr("tokenizer.ggml.scores", vocab.scores);
    w.kvI32Arr("tokenizer.ggml.token_type", vocab.types);
    w.kvU32("tokenizer.ggml.bos_token_id", 1);
    w.kvU32("tokenizer.ggml.eos_token_id", 2);
    w.kvU32("tokenizer.ggml.unknown_token_id", 0);
    w.kvBool("tokenizer.ggml.add_bos_token", true);

    Rng rng(sp.seed);
    const float s = sp.weight_scale;
    auto mat = [&](const std::string& name, uint64_t in, uint64_t out) {
        w.tensorF32(name, {in, out}, randomMatrix(rng, (size_t)(in * out), s));
    };
    auto ones = [&](const std::string& name, uint64_t n) {
        w.tensorF32(name, {n}, std::vector<float>(n, 1.0f));
    };

    mat("token_embd.weight", sp.n_embd, n_vocab);
    for (uint32_t l = 0; l < sp.n_layer; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        ones(b + "attn_norm.weight", sp.n_embd);
        mat(b + "attn_q.weight", sp.n_embd, (uint64_t)sp.n_head * head_dim);
        mat(b + "attn_k.weight", sp.n_embd, (uint64_t)sp.n_kv_head * head_dim);
        mat(b + "attn_v.weight", sp.n_embd, (uint64_t)sp.n_kv_head * head_dim);
        mat(b + "attn_output.weight", (uint64_t)sp.n_head * head_dim, sp.n_embd);
        ones(b + "ffn_norm.weight", sp.n_embd);
        mat(b + "ffn_gate.weight", sp.n_embd, sp.n_ff);
        mat(b + "ffn_up.weight", sp.n_embd, sp.n_ff);
        mat(b + "ffn_down.weight", sp.n_ff, sp.n_embd);
    }
    ones("output_norm.weight", sp.n_embd);
    mat("output.weight", sp.n_embd, n_vocab);

    w.write(path);
    return n_vocab;
}

// Universal architecture fixture generator (Phase 27)
struct TinyArchSpec {
    std::string arch       = "llama"; // "llama", "mistral", "mistral3", "qwen2", "phi3", "gemma", "gemma2"
    std::string tok_model  = "llama"; // "llama" or "gpt2"
    uint32_t n_embd        = 64;
    uint32_t n_head        = 4;
    uint32_t n_kv_head     = 2;
    uint32_t n_layer       = 2;
    uint32_t n_ff          = 128;
    uint32_t n_ctx         = 128;
    float    rms_eps       = 1e-5f;
    float    rope_base     = 10000.0f;
    bool     qkv_bias      = false;
    bool     fused_qkv     = false;
    bool     fused_ffn_up  = false;
    bool     tied_lm_head  = false;
    bool     post_norms    = false;
    float    attn_softcap  = 0.0f;
    float    final_softcap = 0.0f;
    uint32_t sliding_win   = 0;
    uint64_t seed          = 2027;
    float    weight_scale  = 0.08f;
};

inline uint32_t writeTinyArchModel(const std::string& path, const TinyArchSpec& sp) {
    GgufWriter w;
    uint32_t n_vocab = 0;

    if (sp.tok_model == "gpt2") {
        TinyGpt2Vocab gv = makeGpt2Vocab();
        n_vocab = static_cast<uint32_t>(gv.tokens.size());
        w.kvStr("tokenizer.ggml.model", "gpt2");
        w.kvStr("tokenizer.ggml.pre", "qwen2");
        w.kvStrArr("tokenizer.ggml.tokens", gv.tokens);
        w.kvStrArr("tokenizer.ggml.merges", gv.merges);
        w.kvI32Arr("tokenizer.ggml.token_type", gv.types);
        w.kvU32("tokenizer.ggml.bos_token_id", n_vocab - 1);
        w.kvU32("tokenizer.ggml.eos_token_id", n_vocab - 1);
        w.kvBool("tokenizer.ggml.add_bos_token", false);
    } else {
        TinyVocab sv = makeSpmVocab();
        for (const char* p : {"hell", "hello", "wo", "wor", "worl", "world"}) {
            sv.tokens.push_back(p);
            sv.scores.push_back(-1.0f);
            sv.types.push_back(1);
        }
        sv.tokens.push_back("<|endoftext|>");
        sv.scores.push_back(0.0f);
        sv.types.push_back(3);
        n_vocab = static_cast<uint32_t>(sv.tokens.size());
        w.kvStr("tokenizer.ggml.model", "llama");
        w.kvStrArr("tokenizer.ggml.tokens", sv.tokens);
        w.kvF32Arr("tokenizer.ggml.scores", sv.scores);
        w.kvI32Arr("tokenizer.ggml.token_type", sv.types);
        w.kvU32("tokenizer.ggml.bos_token_id", 1);
        w.kvU32("tokenizer.ggml.eos_token_id", 2);
        w.kvU32("tokenizer.ggml.unknown_token_id", 0);
        w.kvBool("tokenizer.ggml.add_bos_token", true);
        if (sp.arch == "gemma" || sp.arch == "gemma2") {
            w.kvBool("tokenizer.ggml.add_space_prefix", false);
        }
    }

    const uint32_t head_dim = sp.n_embd / sp.n_head;

    w.kvStr("general.architecture", sp.arch);
    w.kvStr("general.name", "heteroaccel-p27-" + sp.arch);
    w.kvU32(sp.arch + ".vocab_size", n_vocab);
    w.kvU32(sp.arch + ".context_length", sp.n_ctx);
    w.kvU32(sp.arch + ".embedding_length", sp.n_embd);
    w.kvU32(sp.arch + ".block_count", sp.n_layer);
    w.kvU32(sp.arch + ".feed_forward_length", sp.n_ff);
    w.kvU32(sp.arch + ".attention.head_count", sp.n_head);
    w.kvU32(sp.arch + ".attention.head_count_kv", sp.n_kv_head);
    w.kvU32(sp.arch + ".attention.key_length", head_dim);
    w.kvU32(sp.arch + ".attention.value_length", head_dim);
    w.kvF32(sp.arch + ".attention.layer_norm_rms_epsilon", sp.rms_eps);
    w.kvU32(sp.arch + ".rope.dimension_count", head_dim);
    w.kvF32(sp.arch + ".rope.freq_base", sp.rope_base);

    if (sp.sliding_win > 0) {
        w.kvU32(sp.arch + ".attention.sliding_window", sp.sliding_win);
    }
    if (sp.attn_softcap > 0.0f) {
        w.kvF32(sp.arch + ".attn_logit_softcapping", sp.attn_softcap);
    }
    if (sp.final_softcap > 0.0f) {
        w.kvF32(sp.arch + ".final_logit_softcapping", sp.final_softcap);
    }

    Rng rng(sp.seed);
    const float s = sp.weight_scale;
    auto mat = [&](const std::string& name, uint64_t in, uint64_t out) {
        w.tensorF32(name, {in, out}, randomMatrix(rng, static_cast<size_t>(in * out), s));
    };
    auto vec_rand = [&](const std::string& name, uint64_t n, float scale) {
        w.tensorF32(name, {n}, randomMatrix(rng, static_cast<size_t>(n), scale));
    };
    auto norm_vec = [&](const std::string& name, uint64_t n) {
        std::vector<float> nv(static_cast<size_t>(n));
        for (size_t i = 0; i < nv.size(); ++i) {
            nv[i] = 0.85f + 0.3f * rng.uniform();
        }
        w.tensorF32(name, {n}, nv);
    };

    mat("token_embd.weight", sp.n_embd, n_vocab);

    for (uint32_t l = 0; l < sp.n_layer; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        norm_vec(b + "attn_norm.weight", sp.n_embd);

        uint64_t q_rows  = static_cast<uint64_t>(sp.n_head)    * head_dim;
        uint64_t kv_rows = static_cast<uint64_t>(sp.n_kv_head) * head_dim;

        if (sp.fused_qkv) {
            mat(b + "attn_qkv.weight", sp.n_embd, q_rows + 2 * kv_rows);
            if (sp.qkv_bias) {
                vec_rand(b + "attn_qkv.bias", q_rows + 2 * kv_rows, 0.02f);
            }
        } else {
            mat(b + "attn_q.weight", sp.n_embd, q_rows);
            mat(b + "attn_k.weight", sp.n_embd, kv_rows);
            mat(b + "attn_v.weight", sp.n_embd, kv_rows);
            if (sp.qkv_bias) {
                vec_rand(b + "attn_q.bias", q_rows,  0.02f);
                vec_rand(b + "attn_k.bias", kv_rows, 0.02f);
                vec_rand(b + "attn_v.bias", kv_rows, 0.02f);
            }
        }

        mat(b + "attn_output.weight", q_rows, sp.n_embd);
        if (sp.post_norms) {
            norm_vec(b + "post_attention_norm.weight", sp.n_embd);
        }

        norm_vec(b + "ffn_norm.weight", sp.n_embd);
        if (sp.fused_ffn_up) {
            mat(b + "ffn_up.weight", sp.n_embd, 2 * static_cast<uint64_t>(sp.n_ff));
        } else {
            mat(b + "ffn_gate.weight", sp.n_embd, sp.n_ff);
            mat(b + "ffn_up.weight",   sp.n_embd, sp.n_ff);
        }
        mat(b + "ffn_down.weight", sp.n_ff, sp.n_embd);
        if (sp.post_norms) {
            norm_vec(b + "post_ffw_norm.weight", sp.n_embd);
        }
    }

    norm_vec("output_norm.weight", sp.n_embd);
    if (!sp.tied_lm_head) {
        mat("output.weight", sp.n_embd, n_vocab);
    }

    w.write(path);
    return n_vocab;
}

} // namespace fixture
