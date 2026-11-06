// tests/test_p26_cuda_real_model.cpp
// Phase 26 HARDWARE test: CUDA inference on an actual GGUF model, numerically
// compared against the llama.cpp CPU reference on the same file.
//
// SKIPs (exit 77, hardware validation PENDING) unless ALL hold:
//   * binary built with -DHETEROACCEL_ENABLE_CUDA=ON
//   * a CUDA device was detected at runtime
//   * HETEROACCEL_MODEL_PATH points to a GGUF model
#include "mini_test.h"
#include "llm/LlamaCppEngine.h"
#include "orchestration/HardwareDetector.h"
#include "cuda/CudaRuntime.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <vector>

int main() {
    std::cout << "== test_p26_cuda_real_model ==\n";
    using agr::orchestration::HardwareDetector;
    auto& hw = HardwareDetector::instance();
    hw.detect();

    if (!HardwareDetector::cuda_inference_compiled()) {
        std::cout << "  SKIP (PENDING): built without HETEROACCEL_ENABLE_CUDA\n";
        return 77;
    }
    if (!hw.has_cuda()) {
        std::cout << "  SKIP (PENDING): no usable CUDA device: " << hw.cuda_unavailable_reason() << "\n";
        return 77;
    }
    const char* env = std::getenv("HETEROACCEL_MODEL_PATH");
    if (!env || !std::filesystem::exists(env)) {
        std::cout << "  SKIP (PENDING): HETEROACCEL_MODEL_PATH not set or file missing\n";
        return 77;
    }

    auto cfgFor = [&](const char* pref) {
        agr::LlmConfig c;
        c.model_path = env;
        c.backend_preference = pref;
        c.n_ctx = 512;
        c.max_new_tokens = 16;
        c.temperature = 0.0f;
        c.seed = 42;
        c.capture_first_logits = true;
        c.n_gpu_layers = 999;
        return c;
    };
    const std::string prompt = "The capital of France is";

    agr::LlmResult ref, gpu;
    {
        agr::LlamaCppEngine cpu;
        if (!cpu.initialize(cfgFor("cpu"))) { std::cerr << cpu.lastError() << "\n"; return 1; }
        ref = cpu.infer(prompt);
    }
    AGR_CHECK(ref.success);
    AGR_CHECK(ref.backend == "CPU");

    {
        agr::LlamaCppEngine cuda;
        const bool ok = cuda.initialize(cfgFor("cuda"));
        if (!ok) std::cerr << "CUDA init failed: " << cuda.lastError() << "\n";
        AGR_CHECK(ok);
        if (!ok) return 1;
        gpu = cuda.infer(prompt);
        AGR_CHECK(cuda.activeBackend() == "CUDA");
    }
    AGR_CHECK(gpu.success);
    AGR_CHECK(gpu.backend == "CUDA");
    AGR_CHECK(gpu.cuda_confirmed);
    AGR_CHECK(gpu.gpu_layers_actual > 0);
    AGR_CHECK(!gpu.device_name.empty());
    AGR_CHECK(gpu.device_total_bytes > 0);
    std::cout << "  device: " << gpu.device_name << " layers=" << gpu.gpu_layers_actual
              << " vram_total=" << (gpu.device_total_bytes >> 20) << " MiB\n";
    std::cout << "  CPU : " << ref.output << "\n  CUDA: " << gpu.output << "\n";
    std::cout << "  tok/s cpu=" << ref.tokens_per_sec << " cuda=" << gpu.tokens_per_sec << "\n";

    AGR_CHECK(ref.first_token_logits.size() == gpu.first_token_logits.size());
    AGR_CHECK(!gpu.first_token_logits.empty());
    if (!gpu.first_token_logits.empty() && ref.first_token_logits.size() == gpu.first_token_logits.size()) {
        double dot = 0, nr = 0, ng = 0;
        for (size_t i = 0; i < ref.first_token_logits.size(); ++i) {
            dot += (double)ref.first_token_logits[i] * gpu.first_token_logits[i];
            nr  += (double)ref.first_token_logits[i] * ref.first_token_logits[i];
            ng  += (double)gpu.first_token_logits[i] * gpu.first_token_logits[i];
        }
        const double cosine = dot / (std::sqrt(nr) * std::sqrt(ng));
        const int top_ref = (int)(std::max_element(ref.first_token_logits.begin(), ref.first_token_logits.end()) - ref.first_token_logits.begin());
        const int top_gpu = (int)(std::max_element(gpu.first_token_logits.begin(), gpu.first_token_logits.end()) - gpu.first_token_logits.begin());
        std::vector<float> s = ref.first_token_logits;
        std::partial_sort(s.begin(), s.begin() + 2, s.end(), std::greater<float>());
        const double gap = s[0] - s[1];
        std::cout << "  logit cosine=" << cosine << " top1 cpu=" << top_ref << " cuda=" << top_gpu << " gap=" << gap << "\n";
        // Quantised kernels use different accumulation order and activation precision.
        AGR_CHECK(cosine > 0.999);
        AGR_CHECK(top_ref == top_gpu || gap < 0.1);
    }
    AGR_CHECK(!gpu.output.empty());
    AGR_TEST_MAIN_END();
}
