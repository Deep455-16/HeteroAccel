// tests/test_p19_gguf_tokenizer.cpp
// Phase 19: GGUF parser robustness + byte-level BPE tokenizer tests.
//
// A minimal synthetic GGUF v3 file is written by the test itself, containing:
//   - qwen2 architecture metadata and general.alignment
//   - a gpt2-style tokenizer section (tokens / merges / token_type arrays)
//   - a skipped-type array (UINT8) followed by another key, proving exact
//     skipping
//   - one Q4_K and one Q8_0 tensor, quantized with the REAL bundled GGML
//     quantize_row_*_ref() functions
//
// The file is parsed by HydroXL's GGUFLoader and used to verify metadata
// parsing, array skipping, tokenizer encode/decode with hand-computed
// expected token IDs, and full-stack tensor loading (readTensorData +
// dequant) against the GGML reference decode of the same blocks.

#include "mini_test.h"

#include "ggml-quants.h" // bundled GGML reference

#include "hydroxl/GGUFLoader.h"
#include "hydroxl/HydroXLDequant.h"
#include "hydroxl/HydroXLTokenizer.h"
#include "hydroxl/HydroXLTypes.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace agr::hydroxl;

// ---------------------------------------------------------------------------
// Tiny GGUF binary writer (little-endian) with byte-offset tracking
// ---------------------------------------------------------------------------
namespace {

class GgufWriter {
public:
    explicit GgufWriter(const std::string& path)
        : out_(path, std::ios::binary | std::ios::trunc) {}

    void u32(uint32_t v) { raw(&v, 4); }
    void u64(uint64_t v) { raw(&v, 8); }
    void f32(float v)    { raw(&v, 4); }
    void str(const std::string& s) {
        u64(s.size());
        raw(s.data(), s.size());
    }
    void raw(const void* p, size_t n) {
        out_.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
        pos_ += n;
    }
    void padTo(uint64_t mult) {
        uint64_t pad = (mult - (pos_ % mult)) % mult;
        for (uint64_t i = 0; i < pad; ++i) { char z = 0; raw(&z, 1); }
    }
    uint64_t pos() const { return pos_; }
    bool good() const { return out_.good(); }

private:
    std::ofstream out_;
    uint64_t pos_ = 0;
};

void writeStringKV(GgufWriter& w, const std::string& key, const std::string& val) {
    w.str(key); w.u32(8); w.str(val); // STRING
}
void writeU32KV(GgufWriter& w, const std::string& key, uint32_t val) {
    w.str(key); w.u32(4); w.u32(val); // UINT32
}

// GPT-2 byte->unicode mapping, written independently in this test from the
// published table definition.
std::string mapByte(unsigned char b) {
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

constexpr const char* kSynthPath = "test_p19_synth.gguf";

void writeSyntheticGguf(const char* path) {
    GgufWriter w(path);

    // ---- vocabulary: 256 byte tokens, merged tokens, one control token ----
    std::vector<std::string> tokens;
    for (int b = 0; b < 256; ++b) tokens.push_back(mapByte(static_cast<unsigned char>(b)));
    tokens.push_back("he");                    // 256
    tokens.push_back("llo");                   // 257
    tokens.push_back(mapByte(0x20) + "w");     // 258 "Ġw"
    tokens.push_back("or");                    // 259
    tokens.push_back("<|endoftext|>");         // 260 (CONTROL)

    std::vector<std::string> merges = {
        "l l", "ll o", "h e", "o r", mapByte(0x20) + " w"
    };
    std::vector<int32_t> types(tokens.size(), 1);
    types[260] = 3;

    // Deterministic tensor payloads quantized by the REAL ggml quantizers
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    const int q4_elems = 1024; // 4 blocks
    const int q8_elems = 512;  // 16 blocks
    std::vector<float> x4(q4_elems), x8(q8_elems);
    for (float& v : x4) v = dist(rng);
    for (float& v : x8) v = dist(rng);

    std::vector<block_q4_K> b4(q4_elems / 256);
    std::vector<block_q8_0> b8(q8_elems / 32);
    quantize_row_q4_K_ref(x4.data(), b4.data(), q4_elems);
    quantize_row_q8_0_ref(x8.data(), b8.data(), q8_elems);

    const uint64_t n_kv = 10;
    const uint64_t n_tensors = 2;

    w.u32(0x46554747); // 'GGUF'
    w.u32(3);          // version 3
    w.u64(n_tensors);
    w.u64(n_kv);

    writeStringKV(w, "general.architecture", "qwen2");          // 1
    writeU32KV(w, "general.alignment", 32);                     // 2
    writeStringKV(w, "tokenizer.ggml.model", "gpt2");           // 3
    w.str("tokenizer.ggml.tokens");                             // 4
    w.u32(9); w.u32(8); w.u64(tokens.size());
    for (const auto& t : tokens) w.str(t);
    w.str("tokenizer.ggml.merges");                             // 5
    w.u32(9); w.u32(8); w.u64(merges.size());
    for (const auto& m : merges) w.str(m);
    w.str("tokenizer.ggml.token_type");                         // 6
    w.u32(9); w.u32(5); w.u64(types.size());
    for (int32_t t : types) w.raw(&t, 4);
    w.str("debug.byte_array");                                  // 7 (skipped type)
    w.u32(9); w.u32(0); w.u64(7);
    {
        uint8_t payload[7] = {1, 2, 3, 4, 5, 6, 7};
        w.raw(payload, 7);
    }
    writeU32KV(w, "debug.after_array", 7777);                   // 8
    writeU32KV(w, "tokenizer.ggml.eos_token_id", 260);          // 9
    writeU32KV(w, "tokenizer.ggml.bos_token_id", 0xFFFFFFFFu);  // 10

    // ---- tensor infos ----
    w.str("test.q4k");
    w.u32(1); w.u64(1024); w.u32(12); w.u64(0);   // Q4_K @ 0

    w.str("test.q80");
    w.u32(1); w.u64(512);  w.u32(8);  w.u64(576); // Q8_0 @ 4*144

    // ---- aligned data section ----
    w.padTo(32);
    w.raw(b4.data(), b4.size() * sizeof(block_q4_K));
    w.raw(b8.data(), b8.size() * sizeof(block_q8_0));
}

} // namespace

int main() {
    // ==================================================================
    // 1. Synthetic file: header, metadata, arrays, tensors
    // ==================================================================
    writeSyntheticGguf(kSynthPath);

    GGUFLoader gguf;
    AGR_CHECK(gguf.open(kSynthPath));
    AGR_CHECK(gguf.kvString("general.architecture") == "qwen2");
    AGR_CHECK(gguf.kvUInt32("qwen2.embedding_length", 0) == 0); // not written
    AGR_CHECK(gguf.kvUInt32("general.alignment") == 32);

    // String-array retention
    std::vector<std::string> tokens = gguf.kvStringArray("tokenizer.ggml.tokens");
    AGR_CHECK(tokens.size() == 261);
    AGR_CHECK(tokens[256] == "he");
    AGR_CHECK(tokens[258] == mapByte(0x20) + "w");
    std::vector<std::string> merges = gguf.kvStringArray("tokenizer.ggml.merges");
    AGR_CHECK(merges.size() == 5);

    // Exact skipping: the UINT8 array must not desync the following key
    AGR_CHECK(gguf.kvUInt32("debug.after_array") == 7777);

    // Tokenizer metadata
    AGR_CHECK(gguf.kvString("tokenizer.ggml.model") == "gpt2");
    AGR_CHECK(gguf.kvUInt32("tokenizer.ggml.eos_token_id") == 260);

    // Tensor descriptors
    const GGUFTensorDesc* q4 = gguf.findTensor("test.q4k");
    const GGUFTensorDesc* q8 = gguf.findTensor("test.q80");
    AGR_CHECK(q4 != nullptr);
    AGR_CHECK(q8 != nullptr);
    if (q4 && q8) {
        AGR_CHECK(q4->type == GGMLType::Q4_K);
        AGR_CHECK(q4->n_bytes == 4 * 144);
        AGR_CHECK(q8->type == GGMLType::Q8_0);
        AGR_CHECK(q8->n_bytes == 16 * 34);
    }

    // ==================================================================
    // 2. Full-stack tensor read + dequant vs GGML reference decode
    // ==================================================================
    if (q4) {
        std::vector<uint8_t> raw(q4->n_bytes);
        AGR_CHECK(gguf.readTensorData(*q4, raw.data(), raw.size()));
        std::vector<float> got(1024), ref(1024);
        AGR_CHECK(dequant_q4k(raw.data(), got.data(), 1024));
        dequantize_row_q4_K(reinterpret_cast<const block_q4_K*>(raw.data()),
                            ref.data(), 1024);
        double maxdiff = 0;
        for (int i = 0; i < 1024; ++i)
            maxdiff = std::max(maxdiff, std::fabs((double)got[i] - ref[i]));
        std::cout << "  full-stack Q4_K maxdiff=" << maxdiff << "\n";
        AGR_CHECK(maxdiff <= 1e-6);
    }
    if (q8) {
        std::vector<uint8_t> raw(q8->n_bytes);
        AGR_CHECK(gguf.readTensorData(*q8, raw.data(), raw.size()));
        std::vector<float> got(512), ref(512);
        AGR_CHECK(dequant_q8_0(raw.data(), got.data(), 512));
        dequantize_row_q8_0(reinterpret_cast<const block_q8_0*>(raw.data()),
                            ref.data(), 512);
        double maxdiff = 0;
        for (int i = 0; i < 512; ++i)
            maxdiff = std::max(maxdiff, std::fabs((double)got[i] - ref[i]));
        AGR_CHECK(maxdiff <= 1e-6);
    }

    // ==================================================================
    // 3. Tokenizer on the synthetic vocabulary
    // ==================================================================
    HydroXLTokenizer tok;
    AGR_CHECK(tok.loadFromGGUF(gguf));
    AGR_CHECK(tok.vocabSize() == 261);
    AGR_CHECK(tok.eosToken() == 260);
    AGR_CHECK(tok.bosToken() == TOKEN_INVALID);
    AGR_CHECK(!tok.addsBos()); // qwen2-style: no BOS

    {
        // "hello" -> he(256) + llo(257); " world" -> Ġw(258) or(259) l(108) d(100)
        std::vector<TokenID> ids = tok.encode("hello world", false);
        std::vector<TokenID> expected = {256, 257, 258, 259, 108, 100};
        AGR_CHECK(ids == expected);
        AGR_CHECK(tok.decode(ids, false) == "hello world");
    }
    {
        // empty input
        AGR_CHECK(tok.encode("", false).empty());
        // whitespace handling and byte fallback round-trip
        std::vector<TokenID> ids = tok.encode("a  b", false);
        std::vector<TokenID> expected = {97, 32, 32, 98};
        AGR_CHECK(ids == expected);
        AGR_CHECK(tok.decode(ids, false) == "a  b");
    }
    {
        // contraction split + digits in groups of three
        std::vector<TokenID> ids = tok.encode("it's 12345", false);
        // pieces: "it" "'" ... -> "it" [105,116]; "'s" [39,115]; " " [32];
        // "123" [49,50,51]; "45" [52,53]
        std::vector<TokenID> expected = {105, 116, 39, 115, 32, 49, 50, 51, 52, 53};
        AGR_CHECK(ids == expected);
        AGR_CHECK(tok.decode(ids, false) == "it's 12345");
    }
    {
        // special token skipped (or not) on decode
        AGR_CHECK(tok.decode({260}, true) == "");
        AGR_CHECK(tok.decode({260}, false) == "<|endoftext|>");
        // invalid IDs decode to empty string
        AGR_CHECK(tok.decode({-5, 999999}, false) == "");
        // decodeToken bounds
        AGR_CHECK(tok.decodeToken(256) == "he");
        AGR_CHECK(tok.decodeToken(-1).empty());
        AGR_CHECK(tok.decodeToken(261).empty());
    }
    {
        // unicode round-trip through byte fallback (CJK treated as letters)
        std::string s = "GPU";
        std::vector<TokenID> ids = tok.encode(s, false);
        AGR_CHECK(ids.size() == 3); // no merges apply -> per-byte tokens
        AGR_CHECK(tok.decode(ids, false) == s);
    }

    // ==================================================================
    // 4. Unsupported tokenizer family is rejected clearly
    // ==================================================================
    {
        // Write the llamavocab GGUF file in a nested scope so the
        // GgufWriter (and its ofstream) is destroyed — and the file
        // therefore flushed and closed — before GGUFLoader reads it.
        {
            GgufWriter w("test_p19_llamavocab.gguf");
            w.u32(0x46554747); w.u32(3);
            w.u64(0); w.u64(2);
            writeStringKV(w, "general.architecture", "llama");
            writeStringKV(w, "tokenizer.ggml.model", "llama");
        } // w destroyed here: file flushed and closed

        GGUFLoader l2;
        AGR_CHECK(l2.open("test_p19_llamavocab.gguf"));
        HydroXLTokenizer t2;
        AGR_CHECK(!t2.loadFromGGUF(l2));
        AGR_CHECK(t2.lastError().find("Unsupported tokenizer") != std::string::npos);
        std::remove("test_p19_llamavocab.gguf");
    }

    // ==================================================================
    // 5. Malformed files are rejected
    // ==================================================================
    {   // bad magic
        { GgufWriter w("test_p19_badmagic.gguf"); w.u32(0xdeadbeef); w.u32(3); w.u64(0); w.u64(0); }
        GGUFLoader l;
        AGR_CHECK(!l.open("test_p19_badmagic.gguf"));
        std::remove("test_p19_badmagic.gguf");
    }
    {   // truncated real file
        {
            std::ifstream src(kSynthPath, std::ios::binary);
            std::vector<char> head(64);
            src.read(head.data(), 64);
            std::ofstream dst("test_p19_trunc.gguf", std::ios::binary);
            dst.write(head.data(), 64);
        } // dst closed here
        GGUFLoader l;
        AGR_CHECK(!l.open("test_p19_trunc.gguf"));
        std::remove("test_p19_trunc.gguf");
    }
    {   // unknown scalar value type in a KV entry
        {
            GgufWriter w("test_p19_badtype.gguf");
            w.u32(0x46554747); w.u32(3); w.u64(0); w.u64(1);
            w.str("weird.key");
            w.u32(99); // unknown value type
            w.u32(0);
        }
        GGUFLoader l;
        AGR_CHECK(!l.open("test_p19_badtype.gguf"));
        std::remove("test_p19_badtype.gguf");
    }
    {   // unknown ARRAY element type: rejected explicitly (cannot skip safely)
        {
            GgufWriter w("test_p19_badarr.gguf");
            w.u32(0x46554747); w.u32(3); w.u64(0); w.u64(1);
            w.str("weird.array");
            w.u32(9);   // ARRAY
            w.u32(99);  // unknown element type
            w.u64(1);
            w.u32(0);
        }
        GGUFLoader l;
        AGR_CHECK(!l.open("test_p19_badarr.gguf"));
        std::remove("test_p19_badarr.gguf");
    }
    {   // tensor data offset beyond EOF is refused on read
        // (uses the synthetic file: craft a descriptor we can abuse)
        GGUFTensorDesc bogus;
        bogus.name = "bogus"; bogus.type = GGMLType::Q8_0;
        bogus.shape = {512}; bogus.offset = 1ULL << 60; bogus.n_bytes = 544;
        std::vector<uint8_t> buf(544);
        AGR_CHECK(!gguf.readTensorData(bogus, buf.data(), buf.size()));
    }

    std::remove(kSynthPath);

    AGR_TEST_MAIN_END()
}
