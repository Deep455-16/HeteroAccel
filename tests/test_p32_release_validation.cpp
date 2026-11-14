// tests/test_p32_release_validation.cpp
// Phase 32: Final Execution, Cross-Validation, Stabilization & Release Suite
//
// Validates:
//   - Stage 3: 12-scenario Engine-Selection & Hardware-Scheduling Matrix
//              (selected engines, assigned tasks, selected backends, rejected
//               candidates & reasons, memory budget, execution mode, outcome).
//   - Stage 4: 20 UTF-8, Tokenizer & Streaming Validation Cases (including
//              split 2/3/4-byte UTF-8 sequences across tokens in
//              decodeTokenStreaming so incomplete multibyte sequences are
//              never emitted prematurely as replacement characters).
//   - Stage 5.4: Actual OS Process RSS / Peak RSS telemetry & repeated
//              collaborative inference stability.
//   - Stage 6.2: HydroXL Vulkan format validation (F32, F16, Q5_1, Q8_0)
//              and honest CPU routing for unsupported quants (Q4_K).
//   - Stage 8: All 11 ResultFusionEngine edge cases (empty output, malformed
//              output, repetitive output, duplicate findings, contradictory
//              complexity/concurrency findings, missing engine output, invalid
//              confidence NaN/<0/>1, unvalidated results, and anti-majority-
//              voting structural proof precedence).

#include "mini_test.h"
#include "colibri/ColibriEngine.h"
#include "collaboration/CollaborativeExecutionOrchestrator.h"
#include "collaboration/ResultFusionEngine.h"
#include "collaboration/WorkloadPartitioner.h"
#include "hydroxl/GGUFLoader.h"
#include "hydroxl/HydroXLTokenizer.h"
#include "orchestration/HardwareDetector.h"
#include "orchestration/HydroXLEngineAdapter.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace agr;
using namespace agr::collaboration;
using namespace agr::hydroxl;
using namespace agr::orchestration;

namespace {

// GPT-2 byte -> unicode string helper for constructing a complete 256-byte BPE vocab
std::string gpt2ByteToToken(unsigned char b) {
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

void writeFullByteLevelGpt2Gguf(const std::string& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    auto u32 = [&](uint32_t v) { out.write(reinterpret_cast<const char*>(&v), 4); };
    auto u64 = [&](uint64_t v) { out.write(reinterpret_cast<const char*>(&v), 8); };
    auto str = [&](const std::string& s) {
        u64(s.size());
        out.write(s.data(), static_cast<std::streamsize>(s.size()));
    };
    auto writeStrKV = [&](const std::string& k, const std::string& v) {
        str(k); u32(8); str(v);
    };
    auto writeU32KV = [&](const std::string& k, uint32_t v) {
        str(k); u32(4); u32(v);
    };

    std::vector<std::string> tokens;
    std::vector<int32_t> types;
    for (int b = 0; b < 256; ++b) {
        tokens.push_back(gpt2ByteToToken(static_cast<unsigned char>(b)));
        types.push_back(1); // NORMAL
    }
    tokens.push_back("in");                              types.push_back(1); // 256
    tokens.push_back("t" + gpt2ByteToToken(' '));        types.push_back(1); // 257
    tokens.push_back("<|endoftext|>");                   types.push_back(3); // 258 (CONTROL)
    tokens.push_back("<|im_start|>");                    types.push_back(3); // 259 (CONTROL)

    std::vector<std::string> merges = {
        "i n",
        "t " + gpt2ByteToToken(' ')
    };

    u32(0x46554747u); // GGUF
    u32(3u);          // version 3
    u64(0u);          // 0 tensors
    u64(7u);          // 7 KV pairs

    writeStrKV("general.architecture", "qwen2");
    writeStrKV("tokenizer.ggml.model", "gpt2");
    str("tokenizer.ggml.tokens"); u32(9); u32(8); u64(tokens.size());
    for (const auto& t : tokens) str(t);
    str("tokenizer.ggml.merges"); u32(9); u32(8); u64(merges.size());
    for (const auto& m : merges) str(m);
    str("tokenizer.ggml.token_type"); u32(9); u32(5); u64(types.size());
    for (int32_t t : types) out.write(reinterpret_cast<const char*>(&t), 4);
    writeU32KV("tokenizer.ggml.bos_token_id", 259u);
    writeU32KV("tokenizer.ggml.eos_token_id", 258u);
}

bool isValidUtf8(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = 0;
        if ((c & 0x80u) == 0x00u) len = 1;
        else if ((c & 0xE0u) == 0xC0u) len = 2;
        else if ((c & 0xF0u) == 0xE0u) len = 3;
        else if ((c & 0xF8u) == 0xF0u) len = 4;
        else return false;
        if (i + len > s.size()) return false;
        for (size_t k = 1; k < len; ++k) {
            if ((static_cast<unsigned char>(s[i + k]) & 0xC0u) != 0x80u) return false;
        }
        i += len;
    }
    return true;
}

} // namespace

int main() {
    std::cout << "=== test_p32_release_validation ===\n";

    // =========================================================================
    // STAGE 4.1 & 4.2: 20 UTF-8, Tokenizer & Streaming Validation Cases
    // =========================================================================
    {
        const char* kTokGgufPath = "test_p32_tok_vocab.gguf";
        writeFullByteLevelGpt2Gguf(kTokGgufPath);
        GGUFLoader gguf;
        AGR_CHECK(gguf.open(kTokGgufPath));
        HydroXLTokenizer tok;
        AGR_CHECK(tok.loadFromGGUF(gguf));

        std::string long_input;
        for (int i = 0; i < 64; ++i) {
            long_input += "HeteroAccel step " + std::to_string(i) + "; ";
        }
        std::string ctx_boundary_input(2048, 'A');

        struct TokenizerCase {
            int id;
            const char* name;
            std::string input;
        };

        std::vector<TokenizerCase> cases = {
            {1,  "Plain English sentence",          "HeteroAccel coordinates HydroXL, llama.cpp, and Colibri."},
            {2,  "Leading and trailing spaces",     "   leading and trailing spaces   "},
            {3,  "Multiple consecutive spaces",     "word1    word2        word3"},
            {4,  "Newlines",                        "line 1\nline 2\r\nline 3"},
            {5,  "Repeated newlines",               "para 1\n\n\n\npara 2"},
            {6,  "Tabs and indentation",            "\tif (x > 0) {\n\t\treturn x;\n\t}"},
            {7,  "Punctuation-heavy text",          "Wait... what?! Yes: (a, b; c - d) / [e + f]!"},
            {8,  "Quotes and brackets",             "\"Hello\" 'world' [array] {object} <template>"},
            {9,  "Mathematical symbols",            "∀x ∈ ℝ: ∑(i=1..n) x_i² ≥ √(α² + β²) ≈ π × 10⁻³"},
            {10, "C++ / Python / JSON snippets",    "void f(int* p) { if (!p) return; }\n{\"key\": [1, true, null]}"},
            {11, "Markdown formatting",             "# Heading\n- **bold** and *italic*\n```cpp\nint x = 42;\n```"},
            {12, "Accented Latin characters",       "Café résumé naïve façade Zürich São Paulo"},
            {13, "Hindi / Devanagari text",         "नमस्ते दुनिया, यह एक बहु-इंजन अनुमान प्रणाली है।"},
            {14, "Emoji characters",                "🚀🔥✨ GPU + CPU = ⚡🤖✅"},
            {15, "Mixed-language prompt",           "Explain भारत (India) and C++ templates 🚀 in 2 lines."},
            {16, "Empty input",                     ""},
            {17, "Special tokens / EOS text",       "Hello world"},
            {18, "Long input sequence",             long_input},
            {19, "Context-window boundary (2048B)", ctx_boundary_input},
            {20, "Split multibyte UTF-8 streaming", "Café नमस्ते 🚀🔥"}
        };

        for (const auto& tc : cases) {
            auto ids = tok.encode(tc.input, false);
            if (tc.input.empty()) {
                AGR_CHECK(ids.empty());
            } else {
                AGR_CHECK(!ids.empty());
            }

            // Round-trip full decode
            std::string decoded = tok.decode(ids, true);
            AGR_CHECK(decoded == tc.input);
            AGR_CHECK(isValidUtf8(decoded));
            AGR_CHECK(decoded.find("\xEF\xBF\xBD") == std::string::npos);

            // Token-by-token streaming decode using decodeTokenStreaming
            // Because our GPT-2 vocab encodes multibyte UTF-8 characters into
            // individual byte tokens (1 byte per token), every 2-byte (Café),
            // 3-byte (नमस्ते), and 4-byte (🚀🔥) UTF-8 character is split across
            // 2, 3, or 4 consecutive tokens!
            std::string pending_utf8;
            std::string streamed_accum;
            for (TokenID tid : ids) {
                std::string chunk = tok.decodeTokenStreaming(tid, pending_utf8, true);
                // Every emitted chunk MUST be valid complete UTF-8 and never contain U+FFFD
                AGR_CHECK(isValidUtf8(chunk));
                AGR_CHECK(chunk.find("\xEF\xBF\xBD") == std::string::npos);
                streamed_accum += chunk;
            }
            AGR_CHECK(pending_utf8.empty());
            AGR_CHECK(streamed_accum == tc.input);
        }

        // Case 17 extra check: BOS/EOS special token skipping vs keeping
        auto ids_with_bos = tok.encode("Hello", true);
        AGR_CHECK(ids_with_bos.size() == 6);
        AGR_CHECK(ids_with_bos.front() == tok.bosToken());
        ids_with_bos.push_back(tok.eosToken());
        AGR_CHECK(tok.decode(ids_with_bos, true) == "Hello");
        AGR_CHECK(tok.decode(ids_with_bos, false) == "<|im_start|>Hello<|endoftext|>");
        std::remove(kTokGgufPath);

        std::cout << "  [OK] Stage 4: All 20 UTF-8, Tokenizer & Split-Multibyte Streaming cases passed\n";
    }

    // =========================================================================
    // STAGE 3: 12-Scenario Engine-Selection & Hardware-Scheduling Test Matrix
    // =========================================================================
    {
        auto& hw = HardwareDetector::instance();
        CollaborativeResourceManager res_mgr;

        ModelMemoryProfile qwen_q8;
        qwen_q8.arch = "qwen2";
        qwen_q8.quant_type = 8; // Q8_0 (Vulkan-compatible for both HydroXL and llama.cpp)
        qwen_q8.hydroxl_estimated_bytes = 512ULL * 1024 * 1024;
        qwen_q8.llamacpp_estimated_bytes = 256ULL * 1024 * 1024;
        qwen_q8.colibri_estimated_bytes = 8ULL * 1024 * 1024;

        ModelMemoryProfile unsupported_arch = qwen_q8;
        unsupported_arch.arch = "mamba2_hybrid";

        // Configure mock Vulkan + CPU hardware for standard scenarios 1-9
        std::vector<DeviceInfo> vk_devs;
        DeviceInfo cpu_dev;
        cpu_dev.name = "Intel Core i5-1235U";
        cpu_dev.api_support = static_cast<uint32_t>(DeviceInfo::APISupport::CPU);
        vk_devs.push_back(cpu_dev);
        DeviceInfo vk_dev;
        vk_dev.name = "Intel Iris Xe Graphics";
        vk_dev.vendor = DeviceInfo::Vendor::Intel;
        vk_dev.api_support = static_cast<uint32_t>(DeviceInfo::APISupport::Vulkan);
        vk_dev.vulkan_ok = true;
        vk_dev.vram_bytes = 8054ULL * 1024 * 1024;
        vk_devs.push_back(vk_dev);

        hw.set_mock_environment(vk_devs, 16ULL * 1024 * 1024 * 1024, true, false);
        res_mgr.configureBudget(16ULL * 1024 * 1024 * 1024);

        // Scenario 1: General factual question (selective 1-engine plan)
        {
            CollaborativeRequest req;
            req.prompt = "What is the capital of France?";
            PartitionPlan p = WorkloadPartitioner::createPlan(req, qwen_q8, res_mgr, hw);
            AGR_CHECK(p.query_category == QueryCategory::SIMPLE_FACTUAL_QA);
            AGR_CHECK(p.tasks.size() == 1);
            AGR_CHECK(p.tasks[0].assigned_engine == EngineId::HYDROXL);
            AGR_CHECK(!p.tasks[0].rejected_candidates.empty());
        }

        // Scenario 2: Complex reasoning prompt (3-engine collaborative plan)
        {
            CollaborativeRequest req;
            req.prompt = "Explain how speculative decoding interacts with grouped-query attention KV caches.";
            PartitionPlan p = WorkloadPartitioner::createPlan(req, qwen_q8, res_mgr, hw);
            AGR_CHECK(p.query_category == QueryCategory::COMPLEX_TECHNICAL_QUESTION);
            AGR_CHECK(p.tasks.size() == 3);
            AGR_CHECK(p.waves.size() == 2);
        }

        // Scenario 3: C++ code analysis prompt (3-engine collaborative plan)
        {
            CollaborativeRequest req;
            req.prompt = "void process(int n) { for(int i=0;i<n;++i) for(int j=0;j<n;++j) for(int k=0;k<n;++k) {} }";
            PartitionPlan p = WorkloadPartitioner::createPlan(req, qwen_q8, res_mgr, hw);
            AGR_CHECK(p.query_category == QueryCategory::CODE_ANALYSIS_AND_OPTIMIZATION);
            AGR_CHECK(p.tasks.size() == 3);
            AGR_CHECK(p.tasks[0].assigned_engine == EngineId::HYDROXL);
            AGR_CHECK(p.tasks[1].assigned_engine == EngineId::LLAMA_CPP);
            AGR_CHECK(p.tasks[2].assigned_engine == EngineId::COLIBRI);
        }

        // Scenario 4: Loop complexity analysis only (selective 1-engine Colibri plan)
        {
            CollaborativeRequest req;
            req.prompt = "Static complexity only: for(int i=0;i<n;++i) for(int j=0;j<n;++j) sum += i*j;";
            PartitionPlan p = WorkloadPartitioner::createPlan(req, qwen_q8, res_mgr, hw);
            AGR_CHECK(p.query_category == QueryCategory::STATIC_COMPLEXITY_ONLY);
            AGR_CHECK(p.tasks.size() == 1);
            AGR_CHECK(p.tasks[0].assigned_engine == EngineId::COLIBRI);
            AGR_CHECK(p.tasks[0].assigned_backend == HardwareBackend::CPU);
        }

        // Scenario 5: Model metadata inspection (selective 1-engine Colibri plan)
        {
            CollaborativeRequest req;
            req.prompt = "Inspect GGUF metadata and tensor count of this model.";
            PartitionPlan p = WorkloadPartitioner::createPlan(req, qwen_q8, res_mgr, hw);
            AGR_CHECK(p.query_category == QueryCategory::MODEL_METADATA_INSPECTION);
            AGR_CHECK(p.tasks.size() == 1);
            AGR_CHECK(p.tasks[0].assigned_engine == EngineId::COLIBRI);
        }

        // Scenario 6: Comparative synthesis task (3-engine collaborative plan)
        {
            CollaborativeRequest req;
            req.prompt = "Compare renewable energy transition strategies and synthesize their economic trade-offs.";
            PartitionPlan p = WorkloadPartitioner::createPlan(req, qwen_q8, res_mgr, hw);
            AGR_CHECK(p.query_category == QueryCategory::GENERAL_REASONING_AND_SYNTHESIS);
            AGR_CHECK(p.tasks.size() == 3);
        }

        // Scenario 7: Unsupported model architecture for HydroXL -> reassigned to llama.cpp
        {
            CollaborativeRequest req;
            req.prompt = "Analyze this C++ function for memory bottlenecks.";
            PartitionPlan p = WorkloadPartitioner::createPlan(req, unsupported_arch, res_mgr, hw);
            AGR_CHECK(p.tasks.size() == 3);
            AGR_CHECK(p.tasks[0].assigned_engine == EngineId::LLAMA_CPP);
            AGR_CHECK(!p.tasks[0].rejected_candidates.empty());
        }

        // Scenario 8: Long-context request (n_ctx = 4096 > 2048) -> HydroXL rejected, routed to llama.cpp
        {
            CollaborativeRequest req;
            req.prompt = "Analyze this long C++ translation unit and verify thread safety.";
            req.n_ctx = 4096;
            PartitionPlan p = WorkloadPartitioner::createPlan(req, qwen_q8, res_mgr, hw);
            AGR_CHECK(p.tasks.size() == 3);
            AGR_CHECK(p.tasks[0].assigned_engine == EngineId::LLAMA_CPP);
            bool found_ctx_rejection = false;
            for (const auto& rc : p.tasks[0].rejected_candidates) {
                if (rc.find("4096") != std::string::npos) found_ctx_rejection = true;
            }
            AGR_CHECK(found_ctx_rejection);
        }

        // Scenario 9: Low-memory environment (512 MB budget) -> memory_constrained_mode = true
        {
            res_mgr.configureBudget(512ULL * 1024 * 1024);
            CollaborativeRequest req;
            req.prompt = "Optimize this C++ loop and verify bounds.";
            req.max_ram_budget_bytes = 512ULL * 1024 * 1024;
            PartitionPlan p = WorkloadPartitioner::createPlan(req, qwen_q8, res_mgr, hw);
            AGR_CHECK(p.memory_constrained_mode);
            res_mgr.configureBudget(16ULL * 1024 * 1024 * 1024);
        }

        // Scenario 10: CPU-only environment
        {
            std::vector<DeviceInfo> cpu_only_devs = {cpu_dev};
            hw.set_mock_environment(cpu_only_devs, 16ULL * 1024 * 1024 * 1024, false, false);
            CollaborativeRequest req;
            req.prompt = "Optimize this C++ loop and verify bounds.";
            PartitionPlan p = WorkloadPartitioner::createPlan(req, qwen_q8, res_mgr, hw);
            for (const auto& t : p.tasks) {
                AGR_CHECK(t.assigned_backend == HardwareBackend::CPU);
            }
        }

        // Scenario 11: Vulkan-enabled environment (Wave 0: HydroXL on Vulkan, llama.cpp on CPU for contention avoidance)
        {
            hw.set_mock_environment(vk_devs, 16ULL * 1024 * 1024 * 1024, true, false);
            CollaborativeRequest req;
            req.prompt = "Optimize this C++ loop and verify bounds.";
            PartitionPlan p = WorkloadPartitioner::createPlan(req, qwen_q8, res_mgr, hw);
            AGR_CHECK(p.tasks[0].assigned_backend == HardwareBackend::VULKAN);
            AGR_CHECK(p.tasks[0].orchestrator_backend_id == "hydroxl_vulkan");
            AGR_CHECK(p.tasks[1].assigned_backend == HardwareBackend::CPU);
            AGR_CHECK(p.tasks[1].orchestrator_backend_id == "llamacpp_cpu");
            AGR_CHECK(p.tasks[2].assigned_backend == HardwareBackend::CPU);
        }

        // Scenario 12: CUDA-enabled / disabled environment
        {
            std::vector<DeviceInfo> cuda_devs = {cpu_dev};
            DeviceInfo nv_dev;
            nv_dev.name = "Mock NVIDIA RTX";
            nv_dev.vendor = DeviceInfo::Vendor::NVIDIA;
            nv_dev.api_support = static_cast<uint32_t>(DeviceInfo::APISupport::CUDA);
            nv_dev.cuda_ok = true;
            nv_dev.vram_bytes = 12ULL * 1024 * 1024 * 1024;
            cuda_devs.push_back(nv_dev);
            hw.set_mock_environment(cuda_devs, 16ULL * 1024 * 1024 * 1024, false, true);

            CollaborativeRequest req;
            req.prompt = "Optimize this C++ loop and verify bounds.";
            req.preferred_gpu_backend = "cuda";
            PartitionPlan p = WorkloadPartitioner::createPlan(req, qwen_q8, res_mgr, hw);
            // HydroXL and Colibri never claim CUDA
            AGR_CHECK(p.tasks[0].assigned_backend == HardwareBackend::CPU);
            AGR_CHECK(p.tasks[2].assigned_backend == HardwareBackend::CPU);
            if (HardwareDetector::cuda_inference_compiled()) {
                AGR_CHECK(p.tasks[1].assigned_backend == HardwareBackend::CUDA);
            } else {
                AGR_CHECK(p.tasks[1].assigned_backend == HardwareBackend::CPU);
            }
        }

        hw.detect();
        std::cout << "  [OK] Stage 3: All 12 Engine-Selection & Hardware-Scheduling scenarios verified\n";
    }

    // =========================================================================
    // STAGE 6.2: HydroXL Vulkan Format Capability & Fallback Validation
    // =========================================================================
    {
        auto& hw = HardwareDetector::instance();
        std::vector<DeviceInfo> vk_devs;
        DeviceInfo cpu_dev;
        cpu_dev.name = "Intel Core i5-1235U";
        cpu_dev.api_support = static_cast<uint32_t>(DeviceInfo::APISupport::CPU);
        vk_devs.push_back(cpu_dev);
        DeviceInfo vk_dev;
        vk_dev.name = "Intel Iris Xe Graphics";
        vk_dev.vendor = DeviceInfo::Vendor::Intel;
        vk_dev.api_support = static_cast<uint32_t>(DeviceInfo::APISupport::Vulkan);
        vk_dev.vulkan_ok = true;
        vk_dev.vram_bytes = 8054ULL * 1024 * 1024;
        vk_devs.push_back(vk_dev);
        hw.set_mock_environment(vk_devs, 16ULL * 1024 * 1024 * 1024, true, false);

        // Supported formats on HydroXL Vulkan: F32 (0), F16 (1), Q5_1 (7), Q8_0 (8)
        const uint32_t kSupportedQuants[] = {0, 1, 7, 8};
        for (uint32_t q : kSupportedQuants) {
            ModelMemoryProfile prof;
            prof.arch = "qwen2";
            prof.quant_type = q;
            std::string bid, reason;
            HardwareBackend hb = WorkloadPartitioner::selectHardwareBackendForEngine(
                EngineId::HYDROXL, prof, hw, "", false, bid, reason);
            AGR_CHECK(hb == HardwareBackend::VULKAN);
            AGR_CHECK(bid == "hydroxl_vulkan");
        }

        // Q4_K (12) is not supported by HydroXL Vulkan shaders -> must route to CPU cleanly
        {
            ModelMemoryProfile prof_q4k;
            prof_q4k.arch = "qwen2";
            prof_q4k.quant_type = 12;
            std::string bid, reason;
            HardwareBackend hb = WorkloadPartitioner::selectHardwareBackendForEngine(
                EngineId::HYDROXL, prof_q4k, hw, "", false, bid, reason);
            AGR_CHECK(hb == HardwareBackend::CPU);
            AGR_CHECK(bid == "hydroxl_cpu");
        }

        hw.detect();
        std::cout << "  [OK] Stage 6.2: HydroXL Vulkan format support (F32, F16, Q5_1, Q8_0) & Q4_K CPU fallback verified\n";
    }

    // =========================================================================
    // STAGE 8: All 11 ResultFusionEngine Edge Cases & Anti-Majority-Voting
    // =========================================================================
    {
        // Edge cases 1, 2, 3, 8: Empty, Malformed, Repetitive, and Invalid Confidence in enrichIntermediateResult
        CollaborativeTask dummy_task;
        dummy_task.task_id = "t_edge";
        dummy_task.assigned_engine = EngineId::HYDROXL;
        dummy_task.assigned_backend = HardwareBackend::CPU;

        // 1) Empty output
        {
            IntermediateResult r;
            r.success = true;
            r.structured_result = "   \n\t  ";
            ResultFusionEngine::enrichIntermediateResult(dummy_task, r);
            AGR_CHECK(!r.validated);
            AGR_CHECK(r.confidence_score <= 0.1f);
        }

        // 2) Malformed output (contains UTF-8 replacement char U+FFFD or unbalanced braces)
        {
            IntermediateResult r;
            r.success = true;
            r.structured_result = "Corrupted token stream \xEF\xBF\xBD with broken syntax {{{";
            ResultFusionEngine::enrichIntermediateResult(dummy_task, r);
            AGR_CHECK(!r.validated);
            AGR_CHECK(r.confidence_score <= 0.35f);
        }

        // 3) Repetitive degenerate output
        {
            IntermediateResult r;
            r.success = true;
            r.structured_result = "loop loop loop loop loop loop loop loop";
            ResultFusionEngine::enrichIntermediateResult(dummy_task, r);
            AGR_CHECK(!r.validated);
            AGR_CHECK(r.confidence_score <= 0.20f);
            AGR_CHECK(r.validation_notes.find("Repetitive") != std::string::npos);
        }

        // 8) Invalid confidence values (NaN, < 0, > 1)
        {
            IntermediateResult r_nan;
            r_nan.success = true;
            r_nan.structured_result = "Valid text with NaN confidence.";
            r_nan.confidence_score = std::numeric_limits<float>::quiet_NaN();
            ResultFusionEngine::enrichIntermediateResult(dummy_task, r_nan);
            AGR_CHECK(!std::isnan(r_nan.confidence_score));
            AGR_CHECK(r_nan.confidence_score == 0.0f);
            AGR_CHECK(!r_nan.validated);

            IntermediateResult r_neg;
            r_neg.success = true;
            r_neg.structured_result = "Valid text with negative confidence.";
            r_neg.confidence_score = -2.5f;
            ResultFusionEngine::enrichIntermediateResult(dummy_task, r_neg);
            AGR_CHECK(r_neg.confidence_score == 0.0f);
            AGR_CHECK(!r_neg.validated);

            IntermediateResult r_high;
            r_high.success = true;
            r_high.structured_result = "Valid text with >1 confidence.";
            r_high.confidence_score = 9.9f;
            ResultFusionEngine::enrichIntermediateResult(dummy_task, r_high);
            AGR_CHECK(r_high.confidence_score == 1.0f);
        }

        // 4, 5, 6, 7, 9, 10: Full ResultFusionEngine::fuse test with:
        //   - Duplicate findings across hydroxl and llama.cpp
        //   - 2-vs-1 Majority Voting Attack: BOTH hydroxl and llama.cpp claim O(N) time complexity,
        //     while Colibri structurally proves O(N^3) time complexity!
        //   - Contradictory concurrency findings (thread-safe vs race condition)
        //   - Missing engine output (a 4th failed task)
        {
            CollaborativeTask t1;
            t1.task_id = "task_1_hydroxl";
            t1.assigned_engine = EngineId::HYDROXL;
            t1.assigned_backend = HardwareBackend::CPU;
            t1.state = TaskState::COMPLETED;
            t1.output.engine_id = "hydroxl";
            t1.output.success = true;
            t1.output.validated = true;
            t1.output.confidence_score = 0.89f;
            t1.output.structured_result = "Loop runs in O(N) time complexity. Cache blocking reduces L1 misses. Function is thread-safe.";
            t1.output.key_findings = {
                "Loop runs in O(N) time complexity.",
                "Cache blocking reduces L1 misses.",
                "Function is thread-safe."
            };

            CollaborativeTask t2;
            t2.task_id = "task_2_llamacpp";
            t2.assigned_engine = EngineId::LLAMA_CPP;
            t2.assigned_backend = HardwareBackend::CPU;
            t2.state = TaskState::COMPLETED;
            t2.output.engine_id = "llama.cpp";
            t2.output.success = true;
            t2.output.validated = true;
            t2.output.confidence_score = 0.88f;
            // Intentionally agrees with HydroXL's wrong O(N) claim (2-vs-1 majority!)
            t2.output.structured_result = "Loop runs in O(N) time complexity. Cache blocking reduces L1 misses.";
            t2.output.key_findings = {
                "Loop runs in O(N) time complexity.",
                "Cache blocking reduces L1 misses."
            };

            CollaborativeTask t3;
            t3.task_id = "task_3_colibri";
            t3.assigned_engine = EngineId::COLIBRI;
            t3.assigned_backend = HardwareBackend::CPU;
            t3.state = TaskState::COMPLETED;
            t3.output.engine_id = "colibri";
            t3.output.success = true;
            t3.output.validated = true;
            t3.output.confidence_score = 0.94f;
            t3.output.structured_result = "Complexity: O(N^3) time complexity from 3 nested loops. Risk: Potential race condition on shared accumulator.";
            t3.output.key_findings = {
                "Complexity: O(N^3) time complexity from 3 nested loops.",
                "Risk: Potential race condition on shared accumulator."
            };

            CollaborativeTask t4_failed;
            t4_failed.task_id = "task_4_missing";
            t4_failed.assigned_engine = EngineId::HYDROXL;
            t4_failed.state = TaskState::FAILED;
            t4_failed.output.success = false;
            t4_failed.output.error = "Simulated missing engine output";

            std::vector<CollaborativeTask> tasks = {t1, t2, t3, t4_failed};
            FusionReport rep = ResultFusionEngine::fuse(
                QueryCategory::CODE_ANALYSIS_AND_OPTIMIZATION,
                "void process(int n) { for(int i=0;i<n;++i) for(int j=0;j<n;++j) for(int k=0;k<n;++k) {} }",
                tasks);

            AGR_CHECK(rep.conflicts_detected >= 2);
            AGR_CHECK(!rep.resolved_conflicts.empty());
            AGR_CHECK(!rep.preserved_disagreements.empty());
            AGR_CHECK(rep.raw_findings_count > rep.deduplicated_findings_count);
            AGR_CHECK(rep.overall_confidence > 0.0f && rep.overall_confidence <= 1.0f);

            // Verify Colibri's O(N^3) structural proof appears BEFORE the disputed 2-engine O(N) claim!
            int idx_colibri_on3 = -1;
            int idx_disputed_on1 = -1;
            for (size_t i = 0; i < rep.deduplicated_findings.size(); ++i) {
                if (rep.deduplicated_findings[i].find("O(N^3)") != std::string::npos &&
                    rep.deduplicated_findings[i].find("[colibri]") != std::string::npos) {
                    idx_colibri_on3 = static_cast<int>(i);
                }
                if (rep.deduplicated_findings[i].find("Loop runs in O(N)") != std::string::npos) {
                    idx_disputed_on1 = static_cast<int>(i);
                }
            }
            AGR_CHECK(idx_colibri_on3 >= 0);
            AGR_CHECK(idx_disputed_on1 >= 0);
            AGR_CHECK(idx_colibri_on3 < idx_disputed_on1);
        }

        std::cout << "  [OK] Stage 8: All 11 ResultFusionEngine edge cases & anti-majority-voting verified\n";
    }

    // =========================================================================
    // STAGE 5.4: OS Process RSS / Peak RSS Telemetry & Repeated Execution Stability
    // =========================================================================
    {
        size_t rss_now = CollaborativeResourceManager::currentProcessResidentBytes();
        size_t rss_peak = CollaborativeResourceManager::peakProcessResidentBytes();
        AGR_CHECK(rss_now > 0);
        AGR_CHECK(rss_peak >= rss_now);

        CollaborativeExecutionOrchestrator orch;
        orch.setCustomExecutor(EngineId::HYDROXL, [](const CollaborativeTask& task, const CollaborativeRequest&) {
            IntermediateResult r;
            r.task_id = task.task_id;
            r.engine_id = "hydroxl";
            r.backend_used = "CPU";
            r.structured_result = "Structural loop analysis complete.";
            r.confidence_score = 0.87f;
            r.success = true;
            return r;
        });
        orch.setCustomExecutor(EngineId::LLAMA_CPP, [](const CollaborativeTask& task, const CollaborativeRequest&) {
            IntermediateResult r;
            r.task_id = task.task_id;
            r.engine_id = "llama.cpp";
            r.backend_used = "CPU";
            r.structured_result = "Implementation boundary check complete.";
            r.confidence_score = 0.86f;
            r.success = true;
            return r;
        });

        CollaborativeRequest req;
        req.prompt = "Optimize nested loop memory access in C++.";
        for (int iter = 0; iter < 5; ++iter) {
            CollaborativeResult res = orch.execute(req);
            AGR_CHECK(res.success);
            AGR_CHECK(res.resource_telemetry.actual_process_rss_bytes > 0);
            AGR_CHECK(res.resource_telemetry.actual_process_peak_rss_bytes >=
                      res.resource_telemetry.actual_process_rss_bytes);
        }
        std::cout << "  [OK] Stage 5.4: Actual OS RSS / Peak RSS telemetry ("
                  << (rss_now / (1024 * 1024)) << " MB / " << (rss_peak / (1024 * 1024))
                  << " MB) & 5x repeated execution stability verified\n";
    }

    // =========================================================================
    // STAGE 10: Regression Tests for Live-Discovered Defects
    //   1) Braceless single-line 3-nested-loop O(N^3) detection in ColibriEngine
    //   2) Dominant block weight quant detection (Q4_K vs Q8_0 token_embd)
    // =========================================================================
    {
        ColibriEngine col;
        AGR_CHECK(col.initialize());
        ColibriAnalysisReport rep = col.analyze(
            "void process(int n) { for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) for (int k = 0; k < n; ++k) {} }");
        AGR_CHECK(rep.max_loop_nesting_depth == 3);
        AGR_CHECK(rep.estimated_time_complexity == "O(N^3)");
        std::cout << "  [OK] Stage 10: Colibri single-line braceless 3-nested-loop O(N^3) regression verified\n";
    }

    // =========================================================================
    // STAGE 4.3 & STAGE 5.1: Live Real-Model UTF-8 Inference & Individual Engine
    // Benchmarks (Executed automatically when local GGUF model is present)
    // =========================================================================
    {
        const char* env_model = std::getenv("HETEROACCEL_MODEL_PATH");
        std::string real_model = env_model ? env_model : "D:\\HeteroAccel\\Models\\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf";
        std::error_code ec;
        if (std::filesystem::exists(real_model, ec) && !ec) {
            std::cout << "  [INFO] Running Stage 4.3 & Stage 5.1 live validation on: " << real_model << "\n";

            ModelMemoryProfile prof = CollaborativeResourceManager::estimateModelMemory(real_model, 512);
            AGR_CHECK(prof.quant_type == 12); // Q4_K dominant weight quantization

            // Verify real 151,936-token Qwen2.5 GGUF vocabulary across all 7 live UTF-8 categories
            GGUFLoader real_gguf;
            AGR_CHECK(real_gguf.open(real_model));
            HydroXLTokenizer real_tok;
            AGR_CHECK(real_tok.loadFromGGUF(real_gguf));
            AGR_CHECK(real_tok.vocabSize() == 151936);

            const char* kLivePrompts[] = {
                "Explain GPU parallelism in one sentence.",
                "\xE0\xA4\xAD\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\xA4 \xE0\xA4\x95\xE0\xA5\x80 \xE0\xA4\xB0\xE0\xA4\xBE\xE0\xA4\x9C\xE0\xA4\xA7\xE0\xA4\xBE\xE0\xA4\xA8\xE0\xA5\x80 \xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xAF\xE0\xA4\xBE \xE0\xA4\xB9\xE0\xA5\x88?",
                "Explain \xE0\xA4\xAD\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\xA4 (India) and C++ in 1 line.",
                "Explain \xF0\x9F\x9A\x80 GPU acceleration \xE2\x9A\xA1 in 1 line.",
                "int add(int a, int b) {\n    return a + b;\n}",
                "{\"engine\": \"HydroXL\", \"backend\": \"Vulkan\"}",
                "f(x) = \xE2\x88\x91 x_i\xC2\xB2 + \xE2\x88\x9A(\xCE\xB1\xC2\xB2 + \xCE\xB2\xC2\xB2)"
            };

            for (const char* p : kLivePrompts) {
                std::string sp(p);
                auto tids = real_tok.encode(sp, false);
                AGR_CHECK(!tids.empty());
                std::string full_dec = real_tok.decode(tids, true);
                AGR_CHECK(full_dec == sp);
                AGR_CHECK(isValidUtf8(full_dec));
                std::string pending;
                std::string stream_dec;
                for (TokenID tid : tids) {
                    std::string ch = real_tok.decodeTokenStreaming(tid, pending, true);
                    AGR_CHECK(isValidUtf8(ch));
                    AGR_CHECK(ch.find("\xEF\xBF\xBD") == std::string::npos);
                    stream_dec += ch;
                }
                AGR_CHECK(pending.empty());
                AGR_CHECK(stream_dec == sp);
            }

            // Live HydroXL streaming inference on Hindi/English/Emoji prompt
            HydroXLCpuEngineAdapter hx_cpu;
            std::string init_err;
            AGR_CHECK(hx_cpu.initialize(real_model, init_err));
            for (int pi = 0; pi < 4; ++pi) {
                agr::orchestration::InferenceRequest ireq;
                ireq.prompt = kLivePrompts[pi];
                ireq.max_new_tokens = 8;
                ireq.temperature = 0.0f;
                agr::orchestration::InferenceResult ires;
                bool ok = hx_cpu.run_stream(ireq, [&](const std::string& chunk, int) -> bool {
                    AGR_CHECK(isValidUtf8(chunk));
                    AGR_CHECK(chunk.find("\xEF\xBF\xBD") == std::string::npos);
                    return true;
                }, ires);
                AGR_CHECK(ok && ires.success);
                AGR_CHECK(isValidUtf8(ires.output_text));
                AGR_CHECK(ires.output_text.find("\xEF\xBF\xBD") == std::string::npos);
                std::cout << "    [Live HydroXL UTF-8 Prompt " << (pi + 1) << "] tokens=" << ires.output_tokens
                          << " speed=" << std::fixed << std::setprecision(2) << ires.tokens_per_second
                          << " tok/s out=\"" << ires.output_text << "\"\n";
            }
            hx_cpu.shutdown();
            std::cout << "  [OK] Stage 4.3: Live 151,936-vocab Qwen2.5 UTF-8 & multilingual streaming verified\n";
        }
    }

    AGR_TEST_MAIN_END();
}
