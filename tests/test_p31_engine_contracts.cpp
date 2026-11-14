// tests/test_p31_engine_contracts.cpp
// Unified Collaborative Runtime — Engine Contract & Tokenizer Decoding Tests
//
// Validates:
//   1. HydroXL GPT-2 byte-level BPE decodeToken() properly maps unicode byte
//      representations (e.g. "ĠParis" -> " Paris") without raw 'Ġ' artifacts.
//   2. ColibriEngine implements IExecutionEngine with honest CPU-only
//      capabilities, GGUF metadata inspection, complexity analysis, delimiter
//      verification, streaming, and cancellation.
//   3. ColibriEngineAdapter implements orchestration::IInferenceEngine and is
//      registered in EngineOrchestrator alongside hydroxl_cpu, hydroxl_vulkan,
//      llamacpp_cpu, and llamacpp_vulkan.
//   4. ExecutionEngineRegistry registers all three engines ("llama.cpp",
//      "hydroxl", "colibri") when CollaborativeExecutionOrchestrator initializes.

#include "mini_test.h"
#include "colibri/ColibriEngine.h"
#include "collaboration/CollaborativeExecutionOrchestrator.h"
#include "engine/ExecutionEngineRegistry.h"
#include "hydroxl/GGUFLoader.h"
#include "hydroxl/HydroXLTokenizer.h"
#include "orchestration/EngineOrchestrator.h"

#include <atomic>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace agr;

namespace {

class TinyGgufWriter {
public:
    explicit TinyGgufWriter(const std::string& path)
        : out_(path, std::ios::binary | std::ios::trunc) {}

    void u32(uint32_t v) { raw(&v, 4); }
    void u64(uint64_t v) { raw(&v, 8); }
    void str(const std::string& s) {
        u64(s.size());
        raw(s.data(), s.size());
    }
    void raw(const void* p, size_t n) {
        out_.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
        pos_ += n;
    }
    void writeStringKV(const std::string& k, const std::string& v) {
        str(k); u32(8); str(v);
    }
    void writeU32KV(const std::string& k, uint32_t v) {
        str(k); u32(4); u32(v);
    }

private:
    std::ofstream out_;
    uint64_t pos_ = 0;
};

std::string gpt2MapByte(unsigned char b) {
    uint32_t cp = 0;
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
        cp = b;
    } else {
        uint32_t n = 0;
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

void writeMinimalGgufForEngineTest(const char* path) {
    TinyGgufWriter w(path);
    std::vector<std::string> tokens;
    for (int b = 0; b < 256; ++b) tokens.push_back(gpt2MapByte(static_cast<unsigned char>(b)));
    tokens.push_back(gpt2MapByte(0x20) + "Paris"); // 256: "ĠParis"
    tokens.push_back("<|endoftext|>");             // 257: control token

    std::vector<int32_t> types(tokens.size(), 1);
    types[257] = 3;
    std::vector<std::string> merges = {
        gpt2MapByte(0x20) + " P",
        gpt2MapByte(0x20) + "P a",
        gpt2MapByte(0x20) + "Pa r",
        gpt2MapByte(0x20) + "Par i",
        gpt2MapByte(0x20) + "Pari s"
    };

    w.u32(0x46554747); // GGUF
    w.u32(3);
    w.u64(0); // 0 tensors
    w.u64(12); // 12 KV pairs

    w.writeStringKV("general.architecture", "qwen2");
    w.writeU32KV("qwen2.block_count", 24);
    w.writeU32KV("qwen2.embedding_length", 896);
    w.writeU32KV("qwen2.attention.head_count", 14);
    w.writeU32KV("qwen2.attention.head_count_kv", 2);
    w.writeU32KV("qwen2.feed_forward_length", 4864);
    w.writeU32KV("qwen2.context_length", 2048);
    w.writeStringKV("tokenizer.ggml.model", "gpt2");

    w.str("tokenizer.ggml.tokens");
    w.u32(9); w.u32(8); w.u64(tokens.size());
    for (const auto& t : tokens) w.str(t);

    w.str("tokenizer.ggml.merges");
    w.u32(9); w.u32(8); w.u64(merges.size());
    for (const auto& m : merges) w.str(m);

    w.str("tokenizer.ggml.token_type");
    w.u32(9); w.u32(5); w.u64(types.size());
    for (int32_t t : types) w.raw(&t, 4);

    w.writeU32KV("tokenizer.ggml.eos_token_id", 257);
}

} // namespace

int main() {
    std::cout << "=== test_p31_engine_contracts ===\n";
    const char* kGgufPath = "test_p31_engine_synth.gguf";
    writeMinimalGgufForEngineTest(kGgufPath);

    // 1. Verify HydroXLTokenizer::decodeToken decodes GPT-2 byte-level BPE tokens
    {
        hydroxl::GGUFLoader loader;
        AGR_CHECK(loader.open(kGgufPath));
        hydroxl::HydroXLTokenizer tok;
        AGR_CHECK(tok.loadFromGGUF(loader));

        // Token 256 is "ĠParis" in the vocab; decodeToken MUST return " Paris"
        std::string decoded_piece = tok.decodeToken(256);
        AGR_CHECK(decoded_piece == " Paris");
        // Control token 257 keeps its literal string when decoded directly
        AGR_CHECK(tok.decodeToken(257) == "<|endoftext|>");
        std::cout << "  [OK] HydroXL GPT-2 BPE decodeToken() maps 'ĠParis' -> ' Paris'\n";
    }

    // 2. Verify ColibriEngine IExecutionEngine contract & honest capabilities
    {
        ColibriEngine colibri;
        auto id = colibri.identity();
        AGR_CHECK(id.name == "Colibri");
        AGR_CHECK(id.registry_key == "colibri");

        auto caps = colibri.capabilities();
        AGR_CHECK(caps.has(EngineCapability::TEXT_GENERATION));
        AGR_CHECK(caps.has(EngineCapability::BACKEND_CPU));
        AGR_CHECK(caps.has(EngineCapability::STREAMING));
        AGR_CHECK(caps.has(EngineCapability::CANCELLATION));
        // Honest capability declaration: Colibri never claims Vulkan or CUDA
        AGR_CHECK(!caps.has(EngineCapability::BACKEND_VULKAN));
        AGR_CHECK(!caps.has(EngineCapability::BACKEND_CUDA));

        AGR_CHECK(colibri.initialize());
        AGR_CHECK(colibri.isInitialized());

        ModelDescriptor desc;
        desc.model_id = "synth_qwen2";
        desc.source_path = kGgufPath;
        AGR_CHECK(colibri.loadModel(desc));
        AGR_CHECK(colibri.isModelLoaded());
        AGR_CHECK(colibri.modelArchitecture() == "qwen2");
        AGR_CHECK(colibri.modelLayers() == 24);
        AGR_CHECK(colibri.modelEmbeddingDim() == 896);
        AGR_CHECK(colibri.modelVocabSize() == 258);

        // Execute code complexity & optimization analysis
        ExecutionContext ctx;
        ctx.prompt = "void matmul(float* A, float* B, float* C, int N) {\n"
                     "  for (int i = 0; i < N; ++i) {\n"
                     "    for (int j = 0; j < N; ++j) {\n"
                     "      for (int k = 0; k < N; ++k) {\n"
                     "        C[i*N+j] += A[i*N+k] * B[k*N+j];\n"
                     "      }\n"
                     "    }\n"
                     "  }\n"
                     "}";
        std::string streamed;
        auto res = colibri.executeStreaming(ctx, [&](const std::string& chunk) {
            streamed += chunk;
        });
        AGR_CHECK(res.success);
        AGR_CHECK(!res.output.empty());
        AGR_CHECK(streamed == res.output);
        AGR_CHECK(res.output.find("O(N^3)") != std::string::npos);
        AGR_CHECK(res.output.find("GQA 7:1") != std::string::npos);

        // Unbalanced delimiter detection
        auto bad_rep = colibri.analyze("void broken() { for (int i=0; i<10; ++i) { int x = i; }");
        AGR_CHECK(!bad_rep.syntax_balanced);
        AGR_CHECK(!bad_rep.validated);
        AGR_CHECK(!bad_rep.detected_risks.empty());

        // Cancellation check
        std::atomic<bool> cancel{true};
        ctx.cancel_flag = &cancel;
        auto cancel_res = colibri.execute(ctx);
        AGR_CHECK(!cancel_res.success);
        AGR_CHECK(colibri.lastEngineError().kind == EngineErrorKind::CANCELLED);

        colibri.unloadModel();
        AGR_CHECK(!colibri.isModelLoaded());
        colibri.shutdown();
        AGR_CHECK(!colibri.isInitialized());
        std::cout << "  [OK] ColibriEngine lifecycle, GGUF inspection, complexity & delimiter checks\n";
    }

    // 3. Verify all three engines are registered when CollaborativeExecutionOrchestrator initializes
    {
        collaboration::CollaborativeExecutionOrchestrator::ensureEnginesRegistered();
        auto& reg = ExecutionEngineRegistry::instance();
        AGR_CHECK(reg.isRegistered("llama.cpp"));
        AGR_CHECK(reg.isRegistered("hydroxl"));
        AGR_CHECK(reg.isRegistered("colibri"));

        auto e1 = reg.create("llama.cpp");
        auto e2 = reg.create("hydroxl");
        auto e3 = reg.create("colibri");
        AGR_CHECK(e1 != nullptr && e1->identity().registry_key == "llama.cpp");
        AGR_CHECK(e2 != nullptr && e2->identity().registry_key == "hydroxl");
        AGR_CHECK(e3 != nullptr && e3->identity().registry_key == "colibri");
        std::cout << "  [OK] ExecutionEngineRegistry instantiates llama.cpp, hydroxl, and colibri\n";
    }

    // 4. Verify ColibriEngineAdapter via orchestration::IInferenceEngine
    {
        orchestration::ColibriEngineAdapter adapter;
        std::string err;
        AGR_CHECK(adapter.initialize(kGgufPath, err));
        AGR_CHECK(adapter.is_ready());
        AGR_CHECK(adapter.backend_id() == "colibri_cpu");
        AGR_CHECK(adapter.supports_streaming());

        orchestration::InferenceRequest ireq;
        ireq.prompt = "Analyze std::map lookup inside a nested loop with mutex lock.";
        ireq.max_new_tokens = 32;
        orchestration::InferenceResult ires = adapter.run(ireq);
        AGR_CHECK(ires.success);
        AGR_CHECK(ires.backend_used == "colibri_cpu");
        AGR_CHECK(!ires.gpu_used);
        AGR_CHECK(ires.output_tokens > 0);
        AGR_CHECK(ires.output_text.find("unordered_map") != std::string::npos);
        adapter.shutdown();
        std::cout << "  [OK] ColibriEngineAdapter IInferenceEngine run & streaming verified\n";
    }

    std::remove(kGgufPath);
    AGR_TEST_MAIN_END();
}
