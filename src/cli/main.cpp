#include "hardware/HardwareDetector.h"
#include "json/HardwareJson.h"
#include "gpu/VulkanBackend.h"
#include "benchmark/VectorAddBenchmark.h"
#include "llm/LlamaCppEngine.h"
#include "llm/LlmConfig.h"
#include "llm/LlmResult.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace agr;

// =============================================================================
// Phase 1 — hardware display helpers
// =============================================================================
namespace {

void printHumanReadable(const HardwareInfo& info) {
    std::cout << "=== CPU ===\n";
    std::cout << "  Vendor:            " << info.cpu.vendor << "\n";
    std::cout << "  Model:             " << info.cpu.model_name << "\n";
    std::cout << "  Architecture:      " << info.cpu.architecture << "\n";
    std::cout << "  Logical processors:" << info.cpu.logical_processors << "\n";
    std::cout << "  Physical cores:    " << info.cpu.physical_cores << "\n\n";

    std::cout << "=== Memory ===\n";
    std::cout << "  Total RAM:     " << info.memory.total_physical_mb << " MB\n";
    std::cout << "  Available RAM: " << info.memory.available_physical_mb << " MB\n\n";

    std::cout << "=== GPU(s) ===\n";
    if (info.gpus.empty()) {
        std::cout << "  No GPUs detected on this system.\n\n";
    } else {
        for (const auto& g : info.gpus) {
            std::cout << "  Name:   " << g.name << "\n";
            std::cout << "  Vendor: " << toString(g.vendor) << "\n";
            std::cout << "  Kind:   " << toString(g.kind) << "\n";
            std::cout << "  Dedicated VRAM:   " << g.dedicated_vram_mb << " MB\n";
            std::cout << "  Shared System Mem:" << g.shared_system_memory_mb << " MB\n";
            std::cout << "  ---\n";
        }
        std::cout << "\n";
    }

    std::cout << "=== Vulkan ===\n";
    std::cout << "  Available: " << (info.vulkan.available ? "YES" : "NO") << "\n";
    if (!info.vulkan.available) {
        std::cout << "  Reason:    " << info.vulkan.unavailable_reason << "\n";
    }
    std::cout << "  Loader instance API: " << info.vulkan.instance_api_version_major << "."
               << info.vulkan.instance_api_version_minor << "."
               << info.vulkan.instance_api_version_patch << "\n";
    for (const auto& d : info.vulkan.devices) {
        std::cout << "  Device: " << d.name << "\n";
        std::cout << "    Type:            " << d.device_type << "\n";
        std::cout << "    API version:     " << d.api_version_major << "." << d.api_version_minor << "." << d.api_version_patch << "\n";
        std::cout << "    Device-local mem:" << d.device_local_memory_mb << " MB\n";
        std::cout << "    Compute queue:   " << (d.has_compute_capable_queue ? "Available" : "Not available") << "\n";
    }
    std::cout << "\n";

    std::cout << "=== CUDA ===\n";
    std::cout << "  Available: " << (info.cuda.available ? "YES" : "NO") << "\n";
    if (!info.cuda.available) {
        std::cout << "  Reason:    " << info.cuda.unavailable_reason << "\n";
    }
    for (const auto& d : info.cuda.devices) {
        std::cout << "  Device[" << d.index << "]: " << d.name << "\n";
    }
}

// =============================================================================
// Phase 2 — Vulkan vector-add benchmark
// =============================================================================
int runBenchmarkVulkan() {
    VulkanBackend backend;
    bool initialized = backend.initialize();

    if (initialized) {
        std::cout << "Vulkan backend initialized on: " << backend.deviceName() << "\n\n";
    } else {
        std::cout << "Vulkan backend NOT available: " << backend.lastError() << "\n";
        std::cout << "Continuing with CPU-only baseline (GPU columns will read N/A).\n\n";
    }

    std::vector<size_t> sizes = {
        1024,            // Small
        65536,           // Medium
        1048576,         // Large (1M)
        16777216,        // Very Large (16M) -- 64MB per buffer
    };

    BenchmarkReport report = VectorAddBenchmark::run(backend, sizes);
    std::cout << VectorAddBenchmark::formatText(report);

    backend.shutdown();
    return 0;
}

// =============================================================================
// Phase 3 helpers
// =============================================================================

void printLlmResult(const LlmResult& r, const LlmConfig& cfg) {
    std::cout << "\n--- Inference Results ---\n";
    std::cout << "Backend:           " << r.backend << "\n";

    if (r.vulkan_confirmed) {
        std::cout << "GPU device:        " << (r.device_name.empty() ? "Intel Iris Xe (Vulkan)" : r.device_name) << "\n";
        std::cout << "Vulkan confirmed:  YES (ggml_vulkan log intercepted)\n";
        std::cout << "GPU layers:        " << r.gpu_layers_actual
                  << " (requested " << r.gpu_layers_requested << ")\n";
    } else if (cfg.effective_gpu_layers() > 0) {
        std::cout << "GPU:               Vulkan requested but not confirmed in logs\n";
        std::cout << "                   (model may still be running on GPU)\n";
    } else {
        std::cout << "GPU layers:        0 (CPU-only mode)\n";
    }

    std::cout << "\n";
    std::cout << "Model load:        " << r.model_load_ms << " ms\n";
    std::cout << "Prompt eval:       " << r.prompt_eval_ms << " ms"
              << "  (" << r.prompt_tokens << " tokens)\n";
    std::cout << "Generation:        " << r.generation_ms << " ms"
              << "  (" << r.output_tokens << " tokens)\n";
    std::cout << "Tokens/sec:        ";
    if (r.tokens_per_sec > 0.0) {
        std::printf("%.2f\n", r.tokens_per_sec);
    } else {
        std::cout << "N/A\n";
    }
    std::cout << "Total time:        " << r.total_ms << " ms\n";
}

// Parse --flag value from args starting at index i.
// Returns true and advances i by 1 if found.
bool parseStringArg(const std::vector<std::string>& args, size_t& i,
                    const std::string& flag, std::string& out) {
    if (args[i] == flag && i + 1 < args.size()) {
        out = args[++i];
        return true;
    }
    return false;
}
bool parseIntArg(const std::vector<std::string>& args, size_t& i,
                 const std::string& flag, int& out) {
    if (args[i] == flag && i + 1 < args.size()) {
        try { out = std::stoi(args[++i]); return true; }
        catch (...) { return false; }
    }
    return false;
}

// =============================================================================
// adaptive-gpu llm ...
// =============================================================================
int runLlmInference(const std::vector<std::string>& args) {
    LlmConfig cfg;
    std::string prompt = "Tell me something interesting about GPUs.";

    for (size_t i = 0; i < args.size(); ++i) {
        if (parseStringArg(args, i, "--model",  cfg.model_path)) continue;
        if (parseStringArg(args, i, "--prompt", prompt))          continue;
        if (parseIntArg(args, i, "--gpu-layers", cfg.n_gpu_layers)) continue;
        if (parseIntArg(args, i, "--max-tokens", cfg.max_new_tokens)) continue;
        if (args[i] == "--cpu") { cfg.cpu_only = true; continue; }
    }

    // Allow env var override for model path
    if (cfg.model_path.empty()) {
        const char* env = std::getenv("HETEROACCEL_MODEL_PATH");
        if (env) cfg.model_path = env;
    }

    if (cfg.model_path.empty()) {
        std::cerr << "ERROR: No model path provided.\n";
        std::cerr << "  Usage: adaptive-gpu llm --model <path.gguf> --prompt \"...\"\n";
        std::cerr << "  Or set the HETEROACCEL_MODEL_PATH environment variable.\n";
        return 1;
    }

    std::cout << "=== HeteroAccel LLM Inference ===\n";
    std::cout << "Model:    " << cfg.model_path << "\n";
    std::cout << "Backend:  " << (cfg.cpu_only ? "CPU (forced)" : "Vulkan GPU") << "\n";
    std::cout << "GPU layers requested: " << cfg.effective_gpu_layers() << "\n";
    std::cout << "Prompt:   " << prompt << "\n\n";

    LlamaCppEngine engine;
    std::cout << "Loading model ...\n";
    if (!engine.initialize(cfg)) {
        std::cerr << "ERROR: " << engine.lastError() << "\n";
        return 1;
    }

    if (engine.vulkanConfirmed()) {
        std::cout << "Vulkan backend: ACTIVE\n";
        std::cout << "Device:         " << engine.detectedVulkanDevice() << "\n";
    } else if (!cfg.cpu_only) {
        std::cout << "Vulkan backend: not confirmed in logs (may still be active)\n";
    }
    std::cout << "\nGenerating ...\n";
    std::cout << "------------------------------------------------------\n";

    LlmResult result = engine.infer(prompt);

    if (!result.success) {
        std::cerr << "ERROR during inference: " << result.error << "\n";
        engine.shutdown();
        return 1;
    }

    std::cout << result.output << "\n";
    std::cout << "------------------------------------------------------\n";

    printLlmResult(result, cfg);
    engine.shutdown();
    return 0;
}

// =============================================================================
// adaptive-gpu benchmark llm ...
// =============================================================================
int runBenchmarkLlm(const std::vector<std::string>& args) {
    std::string model_path;
    int  n_gpu_layers  = 99;
    int  max_tokens    = 64;
    std::string prompt = "Explain what a GPU is in exactly one sentence.";

    for (size_t i = 0; i < args.size(); ++i) {
        if (parseStringArg(args, i, "--model",      model_path))    continue;
        if (parseIntArg(args, i,    "--gpu-layers",  n_gpu_layers))  continue;
        if (parseIntArg(args, i,    "--max-tokens",  max_tokens))    continue;
        if (parseStringArg(args, i, "--prompt",      prompt))        continue;
    }

    if (model_path.empty()) {
        const char* env = std::getenv("HETEROACCEL_MODEL_PATH");
        if (env) model_path = env;
    }

    if (model_path.empty()) {
        std::cerr << "ERROR: No model path provided.\n";
        std::cerr << "  Usage: adaptive-gpu benchmark llm --model <path.gguf>\n";
        return 1;
    }

    std::cout << "=== HeteroAccel LLM Benchmark: CPU vs Vulkan ===\n";
    std::cout << "Model:      " << model_path << "\n";
    std::cout << "Prompt:     " << prompt << "\n";
    std::cout << "Max tokens: " << max_tokens << "\n\n";

    // --- CPU run ---
    std::cout << "--- CPU-only run (n_gpu_layers=0) ---\n";
    LlmConfig cpuCfg;
    cpuCfg.model_path    = model_path;
    cpuCfg.cpu_only      = true;
    cpuCfg.max_new_tokens = max_tokens;
    cpuCfg.seed           = 42;

    LlmResult cpuResult;
    {
        LlamaCppEngine cpuEngine;
        if (!cpuEngine.initialize(cpuCfg)) {
            std::cerr << "ERROR (CPU): " << cpuEngine.lastError() << "\n";
            return 1;
        }
        cpuResult = cpuEngine.infer(prompt);
        cpuEngine.shutdown();
    }

    // --- Vulkan run ---
    std::cout << "--- Vulkan run (n_gpu_layers=" << n_gpu_layers << ") ---\n";
    LlmConfig gpuCfg;
    gpuCfg.model_path     = model_path;
    gpuCfg.n_gpu_layers   = n_gpu_layers;
    gpuCfg.max_new_tokens = max_tokens;
    gpuCfg.seed           = 42;

    LlmResult gpuResult;
    {
        LlamaCppEngine gpuEngine;
        if (!gpuEngine.initialize(gpuCfg)) {
            std::cerr << "ERROR (Vulkan): " << gpuEngine.lastError() << "\n";
            return 1;
        }
        gpuResult = gpuEngine.infer(prompt);
        gpuEngine.shutdown();
    }

    // --- Side-by-side report ---
    std::cout << "\n=== Benchmark Results ===\n";
    std::printf("%-22s %-12s %-12s\n", "Metric", "CPU", "Vulkan");
    std::printf("%-22s %-12s %-12s\n",
        "----------------------", "------------", "------------");
    std::printf("%-22s %-12s %-12s\n",
        "Backend", cpuResult.backend.c_str(), gpuResult.backend.c_str());
    std::printf("%-22s %-12d %-12d\n",
        "GPU layers",      cpuResult.gpu_layers_actual, gpuResult.gpu_layers_actual);
    std::printf("%-22s %-12d %-12d\n",
        "Prompt tokens",   cpuResult.prompt_tokens, gpuResult.prompt_tokens);
    std::printf("%-22s %-12d %-12d\n",
        "Output tokens",   cpuResult.output_tokens, gpuResult.output_tokens);
    std::printf("%-22s %-12.1f %-12.1f\n",
        "Model load (ms)", cpuResult.model_load_ms, gpuResult.model_load_ms);
    std::printf("%-22s %-12.1f %-12.1f\n",
        "Prompt eval (ms)", cpuResult.prompt_eval_ms, gpuResult.prompt_eval_ms);
    std::printf("%-22s %-12.1f %-12.1f\n",
        "Generation (ms)", cpuResult.generation_ms, gpuResult.generation_ms);
    std::printf("%-22s %-12.2f %-12.2f\n",
        "Tokens/sec",      cpuResult.tokens_per_sec, gpuResult.tokens_per_sec);

    if (cpuResult.tokens_per_sec > 0.0 && gpuResult.tokens_per_sec > 0.0) {
        double speedup = gpuResult.tokens_per_sec / cpuResult.tokens_per_sec;
        std::printf("\nSpeedup (Vulkan/CPU): %.2fx", speedup);
        if (speedup < 1.0) {
            std::cout << "  (GPU slower than CPU -- expected for integrated GPU + small model)\n";
        } else {
            std::cout << "  (GPU faster)\n";
        }
    }

    std::cout << "\n";
    if (gpuResult.vulkan_confirmed) {
        std::cout << "Vulkan confirmed: YES (Intel Iris Xe, ggml_vulkan log intercepted)\n";
        std::cout << "Device: " << gpuResult.device_name << "\n";
    } else {
        std::cout << "Vulkan confirmed: Not detected in logs\n";
    }

    return 0;
}

// =============================================================================
// Help
// =============================================================================
void printUsage() {
    std::cout << "adaptive-gpu - HeteroAccel Universal Adaptive GPU Acceleration Runtime\n\n";
    std::cout << "Phase 1 + 2 commands:\n";
    std::cout << "  adaptive-gpu hardware              Show detected hardware (human-readable)\n";
    std::cout << "  adaptive-gpu hardware --json       Show detected hardware (JSON)\n";
    std::cout << "  adaptive-gpu benchmark vulkan      Run CPU vs Vulkan vector-add benchmark\n";
    std::cout << "\nPhase 3 commands:\n";
    std::cout << "  adaptive-gpu llm --model <path.gguf> [options]\n";
    std::cout << "    --prompt  <text>    Prompt text (default: generic GPU question)\n";
    std::cout << "    --gpu-layers <N>    Layers to offload to GPU (default: 99 = all that fit)\n";
    std::cout << "    --max-tokens <N>    Max tokens to generate (default: 256)\n";
    std::cout << "    --cpu               Force CPU-only execution (no GPU offload)\n\n";
    std::cout << "  adaptive-gpu benchmark llm --model <path.gguf> [options]\n";
    std::cout << "    --gpu-layers <N>    GPU layers for Vulkan run (default: 99)\n";
    std::cout << "    --max-tokens <N>    Token limit (default: 64)\n\n";
    std::cout << "  Set HETEROACCEL_MODEL_PATH env var as an alternative to --model.\n\n";
    std::cout << "  adaptive-gpu --help                Show this message\n";
}

} // namespace

// =============================================================================
// main
// =============================================================================
int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);

    if (args.empty() || args[0] == "--help" || args[0] == "-h") {
        printUsage();
        return 0;
    }

    if (args[0] == "hardware") {
        bool json = false;
        for (size_t i = 1; i < args.size(); ++i) {
            if (args[i] == "--json") json = true;
        }
        HardwareInfo info = HardwareDetector::detectAll();
        if (json) {
            std::cout << toJson(info) << "\n";
        } else {
            printHumanReadable(info);
        }
        return 0;
    }

    if (args[0] == "benchmark") {
        if (args.size() >= 2 && args[1] == "vulkan") {
            return runBenchmarkVulkan();
        }
        if (args.size() >= 2 && args[1] == "llm") {
            std::vector<std::string> rest(args.begin() + 2, args.end());
            return runBenchmarkLlm(rest);
        }
        std::cerr << "Usage: adaptive-gpu benchmark [vulkan|llm] ...\n";
        return 1;
    }

    if (args[0] == "llm") {
        std::vector<std::string> rest(args.begin() + 1, args.end());
        return runLlmInference(rest);
    }

    std::cerr << "Unknown command: " << args[0] << "\n";
    printUsage();
    return 1;
}
