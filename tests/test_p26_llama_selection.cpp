// tests/test_p26_llama_selection.cpp
// Phase 26: explicit llama.cpp backend selection (cpu / vulkan / cuda) on a
// generated, real GGUF file, with llama.cpp CPU as the trusted numerical reference.
//
//  * CPU path always runs.
//  * Unavailable backends must FAIL with a reason (never silently run on CPU).
//  * Vulkan / CUDA paths run only if llama.cpp exposes such a device; their
//    first-token logits are compared against the CPU reference.
//
// A CUDA result is only claimed when this binary was built with
// HETEROACCEL_ENABLE_CUDA and a CUDA device exists; otherwise it is PENDING.
#include "mini_test.h"
#include "gguf_fixture.h"
#include "llm/LlamaCppEngine.h"
#include "orchestration/HardwareDetector.h"
#include "orchestration/LlamaCppEngineAdapter.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace {

struct Stats { double max_abs = 0, cosine = 0; int top1_ref = -1, top1_got = -1; double ref_gap = 0; };

Stats compare(const std::vector<float>& ref, const std::vector<float>& got) {
    Stats s;
    if (ref.size() != got.size() || ref.empty()) { s.cosine = -1; return s; }
    double dot = 0, nr = 0, ng = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        s.max_abs = std::max(s.max_abs, (double)std::fabs(ref[i] - got[i]));
        dot += (double)ref[i] * got[i]; nr += (double)ref[i] * ref[i]; ng += (double)got[i] * got[i];
    }
    s.cosine = dot / (std::sqrt(nr) * std::sqrt(ng) + 1e-30);
    s.top1_ref = (int)(std::max_element(ref.begin(), ref.end()) - ref.begin());
    s.top1_got = (int)(std::max_element(got.begin(), got.end()) - got.begin());
    std::vector<float> sorted = ref;
    std::partial_sort(sorted.begin(), sorted.begin() + 2, sorted.end(), std::greater<float>());
    s.ref_gap = sorted[0] - sorted[1];
    return s;
}

agr::LlmConfig baseConfig(const std::string& path, const std::string& pref) {
    agr::LlmConfig c;
    c.model_path = path;
    c.backend_preference = pref;
    c.n_ctx = 128;
    c.n_batch = 64;
    c.max_new_tokens = 6;
    c.temperature = 0.0f;
    c.n_threads = 2;
    c.capture_first_logits = true;
    c.n_gpu_layers = 99;
    return c;
}

} // namespace

int main() {
    std::cout << "== test_p26_llama_selection ==\n";
    const std::string path = "test_p26_tiny_llama.gguf";
    const uint32_t n_vocab = fixture::writeTinyLlama(path);
    AGR_CHECK(n_vocab > 256);
    const std::vector<int> prompt = {1, 270, 275, 280, 262, 290};

    // ---- CPU reference (explicit "cpu" preference) --------------------------------
    agr::LlmResult ref;
    {
        agr::LlamaCppEngine cpu;
        AGR_CHECK(cpu.initialize(baseConfig(path, "cpu")));
        AGR_CHECK(cpu.activeBackend() == "CPU");
        ref = cpu.inferTokens(prompt);
        AGR_CHECK(ref.success);
        AGR_CHECK(ref.backend == "CPU");
        AGR_CHECK(ref.gpu_layers_actual == 0);
        AGR_CHECK(!ref.cuda_confirmed && !ref.vulkan_confirmed);
        AGR_CHECK(ref.first_token_logits.size() == n_vocab);
        AGR_CHECK(!ref.output_token_ids.empty());

        // Independent calls are reproducible (KV cache is reset each run).
        agr::LlmResult again = cpu.inferTokens(prompt);
        AGR_CHECK(again.success);
        AGR_CHECK(again.output_token_ids == ref.output_token_ids);
        AGR_CHECK(again.first_token_logits == ref.first_token_logits);

        // Input validation
        agr::LlmResult bad = cpu.inferTokens({1, 99999});
        AGR_CHECK(!bad.success && !bad.error.empty());
        AGR_CHECK(!cpu.inferTokens({}).success);
        AGR_CHECK(!cpu.inferTokens({1, -5}).success);
        std::vector<int> too_long(200, 270);
        AGR_CHECK(!cpu.inferTokens(too_long).success);
    }

    // ---- Invalid preference / index fail with reasons -------------------------------
    {
        agr::LlamaCppEngine e;
        AGR_CHECK(!e.initialize(baseConfig(path, "tpu")));
        AGR_CHECK(e.lastError().find("Unknown backend preference") != std::string::npos);
        agr::LlmConfig c = baseConfig(path, "vulkan");
        c.gpu_device_index = 9999;
        agr::LlamaCppEngine e2;
        AGR_CHECK(!e2.initialize(c));
        AGR_CHECK(!e2.lastError().empty());
        // After failures the engine can still be used for a valid configuration.
        agr::LlamaCppEngine e3;
        AGR_CHECK(e3.initialize(baseConfig(path, "cpu")));
    }

    // ---- GPU backends ---------------------------------------------------------------
    struct Kind { const char* pref; const char* label; };
    const Kind kinds[] = {{"vulkan", "Vulkan"}, {"cuda", "CUDA"}};
    const bool cuda_built = agr::orchestration::HardwareDetector::cuda_inference_compiled();
    auto& hw = agr::orchestration::HardwareDetector::instance();
    hw.detect();

    for (const auto& k : kinds) {
        std::cout << "  -- backend: " << k.label << "\n";
        agr::LlamaCppEngine eng;
        bool ok = eng.initialize(baseConfig(path, k.pref));
        const bool is_cuda = std::string(k.pref) == "cuda";

        if (!ok) {
            std::cout << "     unavailable: " << eng.lastError() << "\n";
            AGR_CHECK(!eng.lastError().empty());
            if (is_cuda && cuda_built && hw.has_cuda()) {
                // Hardware and build both present: failure to initialise is a real failure.
                std::cerr << "FAIL: CUDA is built and a device was detected, but init failed\n";
                ++g_mini_test_failures;
            } else if (is_cuda) {
                std::cout << "     CUDA hardware validation PENDING ("
                          << (cuda_built ? "no CUDA device detected" : "binary built without HETEROACCEL_ENABLE_CUDA")
                          << ")\n";
            }
            continue;
        }

        AGR_CHECK(eng.activeBackend() == k.label);
        AGR_CHECK(eng.gpuLayersActual() > 0);
        agr::LlmResult r = eng.inferTokens(prompt);
        AGR_CHECK(r.success);
        AGR_CHECK(r.backend == k.label);
        if (is_cuda) AGR_CHECK(r.cuda_confirmed); else AGR_CHECK(r.vulkan_confirmed);
        AGR_CHECK(!r.device_name.empty());
        std::cout << "     device: " << r.device_name << ", layers on GPU: " << r.gpu_layers_actual << "\n";

        Stats s = compare(ref.first_token_logits, r.first_token_logits);
        std::cout << "     vs CPU reference: cosine=" << s.cosine << " max|d|=" << s.max_abs
                  << " top1 " << s.top1_ref << "/" << s.top1_got << " gap=" << s.ref_gap << "\n";
        // F32 weights: GPU and CPU differ only by summation order.
        AGR_CHECK(s.cosine > 0.9999);
        AGR_CHECK(s.max_abs < 5e-2);
        AGR_CHECK(s.top1_ref == s.top1_got || s.ref_gap < 5e-2);

        // Same backend through the orchestration adapter (token-id interface).
        agr::orchestration::LlamaCppEngineAdapter adapter(is_cuda ? "cuda" : "llamacpp_vulkan", k.pref, 0);
        adapter.set_context_size(128);
        adapter.set_capture_logits(true);
        std::string err;
        AGR_CHECK(adapter.initialize(path, err));
        agr::orchestration::InferenceRequest req;
        req.prompt_token_ids = prompt;
        req.max_new_tokens = 6;
        req.temperature = 0.0f;
        auto res = adapter.run(req);
        AGR_CHECK(res.success);
        AGR_CHECK(res.gpu_used);
        AGR_CHECK((int)res.output_token_ids.size() == res.output_tokens);
        Stats s2 = compare(ref.first_token_logits, adapter.last_first_logits());
        AGR_CHECK(s2.cosine > 0.9999);
        adapter.shutdown();
        AGR_CHECK(!adapter.is_ready());
        AGR_CHECK(!adapter.run(req).success);
    }

    // ---- Adapter refuses to pretend: unavailable GPU kind => initialize() fails -----
    if (!cuda_built) {
        agr::orchestration::LlamaCppEngineAdapter a("cuda", "cuda", 0);
        std::string err;
        AGR_CHECK(!a.initialize(path, err));
        AGR_CHECK(err.find("CUDA") != std::string::npos);
        AGR_CHECK(!a.is_ready());
    }

    // ---- CPU adapter as the explicit CPU backend ---------------------------------------
    {
        agr::orchestration::LlamaCppEngineAdapter a("llamacpp_cpu", "cpu", 0);
        a.set_context_size(128);
        std::string err;
        AGR_CHECK(a.initialize(path, err));
        agr::orchestration::InferenceRequest req;
        req.prompt_token_ids = prompt;
        req.max_new_tokens = 6;
        auto res = a.run(req);
        AGR_CHECK(res.success);
        AGR_CHECK(!res.gpu_used);
        AGR_CHECK(res.output_token_ids == ref.output_token_ids);
    }

    std::remove(path.c_str());
    AGR_TEST_MAIN_END();
}
