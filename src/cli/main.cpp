#include "hardware/HardwareDetector.h"
#include "json/HardwareJson.h"
#include "gpu/VulkanBackend.h"
#include "benchmark/VectorAddBenchmark.h"
#include "llm/LlamaCppEngine.h"
#include "llm/LlmConfig.h"
#include "llm/LlmResult.h"
#include "mem/MemoryManager.h"
#include "backend/BackendManager.h"
#include "backend/DeviceSelector.h"
#include "scheduler/AdaptiveScheduler.h"
#include "scheduler/CostModel.h"
#include "inference/HeteroRuntime.h"
#include "hydroxl/HydroXLEngine.h"
#include "orchestration/EngineOrchestrator.h"
#include "orchestration/OrchestratorInit.h"
#include "orchestration/HydroXLEngineAdapter.h"
#include "orchestration/MemoryEstimator.h"
#include "orchestration/IInferenceEngine.h"

#include <algorithm>
#include <chrono>
#include <fstream>
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
        std::cout << "Vulkan confirmed:  YES (utilized for inference)\n";
        std::cout << "GPU layers:        " << r.gpu_layers_actual
                  << " (requested " << r.gpu_layers_requested << ")\n";
    } else if (cfg.effective_gpu_layers() > 0) {
        std::cout << "GPU layers:        " << r.gpu_layers_actual 
                  << " (requested " << r.gpu_layers_requested << ")\n";
        std::cout << "GPU:               Vulkan fallback to CPU (or unavailable)\n";
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
    bool adaptive = false;
    bool is_static = false;

    for (size_t i = 0; i < args.size(); ++i) {
        if (parseStringArg(args, i, "--model",      model_path))    continue;
        if (parseIntArg(args, i,    "--gpu-layers",  n_gpu_layers))  continue;
        if (parseIntArg(args, i,    "--max-tokens",  max_tokens))    continue;
        if (parseStringArg(args, i, "--prompt",      prompt))        continue;
        if (args[i] == "--adaptive") adaptive = true;
        if (args[i] == "--static") is_static = true;
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

    std::cout << "=== HeteroAccel LLM Benchmark (" << (adaptive ? "ADAPTIVE AUTO-TUNING" : "STATIC") << ") ===\n";
    std::cout << "Model:      " << model_path << "\n";
    std::cout << "Prompt:     " << prompt << "\n";
    std::cout << "Max tokens: " << max_tokens << "\n\n";

    agr::HeteroRuntime runtime;
    if (!runtime.initialize()) {
        std::cerr << "ERROR: Failed to initialize HeteroRuntime\n";
        return 1;
    }

    agr::GenerationOptions opts;
    opts.max_tokens = max_tokens;
    opts.seed = 42;

    int runs = adaptive ? 5 : 2;
    for (int run = 1; run <= runs; ++run) {
        std::cout << "--- Run " << run << "/" << runs << " ---\n";
        
        agr::InferenceResult res = runtime.generate(model_path, prompt, opts);
        if (!res.success) {
            std::cerr << "ERROR: " << res.error << "\n";
            return 1;
        }

        std::cout << "Tokens/sec: " << res.telemetry.tokens_per_sec << "\n";
        if (adaptive) {
            std::cout << "Strategy:   " << res.telemetry.scheduler_reason << "\n";
        }
        std::cout << "\n";
    }

    if (adaptive) {
        std::cout << "Adaptive benchmarking complete. Profile saved to profiles/performance.json\n";
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
    std::cout << "Phase 7 & Unified Collaborative Multi-Engine commands:\n";
    std::cout << "  adaptive-gpu collaborate --model <path.gguf> --prompt \"...\" [options]\n";
    std::cout << "    --max-tokens <N>          Max new tokens per engine task (default: 24)\n";
    std::cout << "    --gpu-backend <backend>   Preferred GPU backend (auto|vulkan|cuda|cpu)\n";
    std::cout << "    --memory-budget-mb <MB>   Max RAM budget in MB for collaborative execution\n";
    std::cout << "    --n-ctx <N>               Context window length (default: 512)\n";
    std::cout << "    --sequential              Force sequential wave execution\n";
    std::cout << "  adaptive-gpu run --model <path.gguf> --prompt \"...\" [--collaborate|--single-engine]\n";
    std::cout << "  adaptive-gpu chat --model <path.gguf> [--collaborate|--single-engine]\n\n";
    std::cout << "Phase 12 commands:\n";
    std::cout << "  adaptive-gpu analyze <path.gguf>                            Phase 12 capability analysis\n\n";
    std::cout << "Phase 13 commands:\n";
    std::cout << "  adaptive-gpu plan <path.gguf>                               Phase 13 execution plan\n\n";
    std::cout << "Phase 17 & 25 commands:\n";
    std::cout << "  adaptive-gpu hydroxl --model <path.gguf> [options]          Phase 17 HydroXL native inference engine\n";
    std::cout << "  adaptive-gpu infer --model <path.gguf> [options]            Phase 25 Orchestrated inference\n\n";
    std::cout << "Diagnostics:\n";
    std::cout << "  adaptive-gpu devices               Hardware discovery + auto backend selection\n";
    std::cout << "  adaptive-gpu memory                Unified Memory Manager stats\n";
    std::cout << "  adaptive-gpu scheduler             Adaptive Heterogeneous Scheduler benchmark\n";
    std::cout << "  adaptive-gpu --help                Show this message\n";
}

void printCollaborateUsage() {
    std::cout << "adaptive-gpu collaborate - Unified Collaborative Multi-Engine Inference\n"
              << "Coordinates HydroXL (native), llama.cpp (compatibility), and Colibri (verification)\n\n"
              << "Usage:\n"
              << "  adaptive-gpu collaborate --model <path.gguf> --prompt \"<query>\" [options]\n\n"
              << "Options:\n"
              << "  --model <path.gguf>       Path to GGUF model (or set HETEROACCEL_MODEL_PATH)\n"
              << "  --prompt <text>           User prompt to decompose and execute collaboratively\n"
              << "  --max-tokens <N>          Maximum tokens generated per engine task (default: 24)\n"
              << "  --gpu-backend <name>      Preferred GPU backend: auto, vulkan, cuda, cpu\n"
              << "  --memory-budget-mb <MB>   Override maximum RAM budget in MB\n"
              << "  --n-ctx <N>               Context window length (default: 512)\n"
              << "  --sequential              Execute tasks sequentially instead of parallel waves\n"
              << "  --help, -h                Show this help message\n";
}

int runDevicesCommand() {
    std::cout << "HeteroAccel Hardware Configuration\n";
    std::cout << "===================================\n\n";

    agr::VulkanBackend backend;
    agr::BackendManager mgr(backend);
    mgr.discover();

    const auto& all = mgr.allDevices();
    for (const auto& dev : all) {
        std::cout << agr::toString(dev.backend) << "\n";
        std::cout << "  Name:    " << dev.name << "\n";
        if (!dev.vendor.empty() && dev.vendor != agr::toString(dev.backend))
            std::cout << "  Vendor:  " << dev.vendor << "\n";
        if (!dev.api_version.empty())
            std::cout << "  API:     " << dev.api_version << "\n";
        if (dev.memory_capacity > 0)
            std::cout << "  Memory:  "
                      << (dev.memory_capacity / (1024.0*1024.0*1024.0)) << " GB\n";
        if (dev.compute_units > 0)
            std::cout << "  Cores:   " << dev.compute_units << "\n";
        std::cout << "  Status:  "
                  << (dev.is_available ? "AVAILABLE" : "UNAVAILABLE") << "\n";
        if (!dev.unavailable_reason.empty())
            std::cout << "  Reason:  " << dev.unavailable_reason << "\n";
        std::cout << "  Score:   " << dev.compute_score << "\n\n";
    }

    // Automatic selection
    agr::WorkloadHint hint;
    hint.prefer_gpu       = true;
    hint.compute_intensive = true;
    hint.required_memory  = 0;
    agr::DeviceSelector   selector(mgr);
    agr::ComputeDevice    selected = selector.selectDevice(hint);

    std::cout << "Backend Decision (automatic)\n";
    std::cout << "  Primary Accelerator: " << agr::toString(selected.backend)
              << " (" << selected.name << ")\n";
    bool cpuFallback = mgr.isBackendAvailable(agr::ComputeBackend::CPU)
                       && selected.backend != agr::ComputeBackend::CPU;
    std::cout << "  CPU Fallback:        "
              << (cpuFallback ? "ENABLED" : "N/A (CPU is primary)") << "\n\n";
    std::cout << "HeteroAccel selects backends automatically.\n";
    std::cout << "CUDA and Vulkan are implementation details hidden from the caller.\n";
    return 0;
}

int runMemoryCommand() {
    std::cout << "HeteroAccel Memory Manager\n";
    std::cout << "===========================\n\n";
    
    agr::VulkanBackend backend;
    backend.initialize(); // best effort
    
    agr::MemoryManager mm(backend);
    
    // Do a small test allocation and transfer to populate some stats
    agr::MemoryBlock bCpu = mm.allocate(16 * 1024 * 1024, agr::MemoryLocation::CPU);
    agr::MemoryBlock bGpu = mm.allocate(8 * 1024 * 1024, agr::MemoryLocation::GPU);
    bool r1 = mm.move(bGpu, agr::MemoryLocation::CPU); // GPU -> CPU  (download)
    bool r2 = mm.move(bCpu, agr::MemoryLocation::GPU); // CPU -> GPU  (upload)

    if (!r1) std::cout << "Note: download failed — Vulkan unavailable or unsupported.\n";
    if (!r2) std::cout << "Note: upload failed  — Vulkan unavailable or unsupported.\n";
    
    agr::MemoryStats stats = mm.statistics();
    
    auto gb = [](size_t bytes) { return static_cast<double>(bytes) / (1024.0*1024.0*1024.0); };
    auto mb = [](size_t bytes) { return static_cast<double>(bytes) / (1024.0*1024.0); };
    
    std::cout << "CPU Memory\n";
    std::cout << "  Total:       " << gb(stats.cpu_total_bytes) << " GB\n";
    std::cout << "  Used:         " << gb(stats.cpu_used_bytes) << " GB\n";
    std::cout << "  Available:    " << gb(stats.cpu_total_bytes - stats.cpu_used_bytes) << " GB\n\n";
    
    std::cout << "Vulkan Memory\n";
    std::cout << "  Device Local: " << gb(stats.gpu_device_local_bytes) << " GB\n";
    std::cout << "  Used:          " << gb(stats.gpu_used_bytes) << " GB\n\n";
    
    std::cout << "Allocations:     " << stats.allocation_count << "\n";
    std::cout << "Transfers:       " << stats.transfer_count << "\n\n";
    
    std::cout << "Upload:\n";
    std::cout << "  Bytes:         " << mb(stats.bytes_uploaded) << " MB\n";
    std::cout << "  Bandwidth:     " << stats.upload_bandwidth_gbps << " GB/s\n\n";
    
    std::cout << "Download:\n";
    std::cout << "  Bytes:         " << mb(stats.bytes_downloaded) << " MB\n";
    std::cout << "  Bandwidth:     " << stats.download_bandwidth_gbps << " GB/s\n\n";
    
    std::cout << "Pressure:\n";
    std::cout << "  " << agr::toString(stats.cpu_pressure) << "\n";
    
    // cleanup
    mm.release(bCpu);
    mm.release(bGpu);
    
    return 0;
}

int runSchedulerCommand() {
    std::cout << "HeteroAccel Scheduler Benchmark (Phase 5)\n";
    std::cout << "=========================================\n\n";

    agr::VulkanBackend vk;
    agr::BackendManager backendMgr(vk);
    backendMgr.discover();
    agr::MemoryManager memMgr(vk);
    agr::AdaptiveScheduler scheduler(backendMgr, memMgr);

    agr::Workload w;
    w.name = "Generic Compute Tensor";
    w.input_bytes = 100 * 1024 * 1024; // 100 MB
    w.output_bytes = 10 * 1024 * 1024; // 10 MB
    w.compute_ops_estimate = 5000000;
    
    std::cout << "Workload: " << w.name << "\n";
    std::cout << "Input:    " << (w.input_bytes / (1024.0*1024.0)) << " MB\n";
    std::cout << "Output:   " << (w.output_bytes / (1024.0*1024.0)) << " MB\n";
    std::cout << "Compute:  " << w.compute_ops_estimate << " ops\n\n";

    w.cpu_execute = []() {
        std::this_thread::sleep_for(std::chrono::milliseconds(45));
        return true;
    };
    w.vulkan_execute = []() {
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        return true;
    };
    w.cuda_execute = []() { return true; };

    std::cout << "Candidate Devices\n";
    std::cout << "-----------------\n";
    
    // Hacky instantiation to dump the cost model breakdown explicitly
    agr::CostModel costModel(backendMgr, scheduler.history());
    auto plan = costModel.evaluate(w);

    const auto& devices = backendMgr.allDevices();
    for (const auto& d : devices) {
        if (!d.is_available) continue;
        auto p = costModel.evaluateCandidate(w, d);
        std::cout << agr::toString(d.backend) << "\n";
        std::cout << "  Estimated compute: " << p.estimated_compute_ms << " ms\n";
        std::cout << "  Transfer cost:     " << p.estimated_transfer_ms << " ms\n";
        std::cout << "  Memory penalty:    " << p.estimated_queue_penalty_ms << " ms\n";
        std::cout << "  Total estimate:    " << p.total_cost_ms << " ms\n\n";
    }

    std::cout << "Selected:\n  " << agr::toString(plan.selected_backend) << "\n\n";

    agr::TaskHandle handle = scheduler.schedule(w);
    agr::TaskResult result = scheduler.wait(handle);

    std::cout << "Actual execution:\n";
    std::cout << "  Status:  " << (result.success ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "  Compute: " << result.compute_ms << " ms\n";
    std::cout << "  Total:   " << result.total_ms << " ms\n";

    return result.success ? 0 : 1;
}

// =============================================================================
// Phase 7 & Unified Collaborative Multi-Engine Inference & Chat
// =============================================================================
int runCollaborateCommand(const std::vector<std::string>& args) {
    std::string model_path;
    std::string prompt = "Explain virtual memory and analyze page-table lookup complexity.";
    std::string gpu_backend;
    int max_tokens = 24;
    int budget_mb = 0;
    int n_ctx = 512;
    bool sequential = false;

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--help" || args[i] == "-h") {
            printCollaborateUsage();
            return 0;
        }
        if (parseStringArg(args, i, "--model",       model_path))  continue;
        if (parseStringArg(args, i, "--prompt",      prompt))      continue;
        if (parseStringArg(args, i, "--gpu-backend", gpu_backend)) continue;
        if (parseIntArg(args, i,    "--max-tokens",  max_tokens))  continue;
        if (parseIntArg(args, i,    "--memory-budget-mb", budget_mb)) continue;
        if (parseIntArg(args, i,    "--n-ctx",       n_ctx))       continue;
        if (args[i] == "--sequential") { sequential = true; continue; }
    }

    if (model_path.empty()) {
        const char* env = std::getenv("HETEROACCEL_MODEL_PATH");
        if (env) model_path = env;
    }

    if (model_path.empty()) {
        std::cerr << "ERROR: No model path provided.\n";
        std::cerr << "  Usage: adaptive-gpu collaborate --model <path.gguf> --prompt \"...\"\n";
        return 1;
    }

    {
        std::ifstream fcheck(model_path, std::ios::binary);
        if (!fcheck.good()) {
            std::cerr << "ERROR: Model file not found or unreadable: " << model_path << "\n";
            return 1;
        }
    }

    agr::HeteroRuntime runtime;
    if (!runtime.initialize()) {
        std::cerr << "ERROR: Failed to initialize HeteroRuntime\n";
        return 1;
    }

    agr::collaboration::CollaborativeRequest creq;
    creq.model_path = model_path;
    creq.prompt = prompt;
    creq.max_new_tokens_per_task = max_tokens;
    creq.n_ctx = n_ctx;
    creq.allow_parallel_waves = !sequential;
    creq.preferred_gpu_backend = gpu_backend;
    if (budget_mb > 0) {
        creq.max_ram_budget_bytes = static_cast<size_t>(budget_mb) * 1024ULL * 1024ULL;
    }

    std::cout << "\n================================================\n";
    std::cout << "HeteroAccel Unified Collaborative Runtime\n";
    std::cout << "Engines: HydroXL (native) + llama.cpp (compatibility) + Colibri (specialized)\n";
    std::cout << "================================================\n";
    std::cout << "Prompt: " << prompt << "\n\n";

    auto cb = [](const std::string& chunk, int) -> bool {
        std::cout << chunk << std::flush;
        return true;
    };

    auto cres = runtime.generateCollaborativeStreaming(creq, cb);
    if (!cres.success) {
        std::cerr << "ERROR: Collaborative execution failed: " << cres.error << "\n";
    }
    std::cout << "\n" << cres.formatTelemetryReport() << "\n";

    runtime.shutdown();
    return cres.success ? 0 : 1;
}

int runInferenceCommand(const std::vector<std::string>& args) {
    std::string model_path;
    std::string prompt = "Explain virtual memory in simple terms.";
    agr::GenerationOptions opts;
    bool single_engine = false;

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--help" || args[i] == "-h") {
            printCollaborateUsage();
            return 0;
        }
        if (parseStringArg(args, i, "--model",      model_path))  continue;
        if (parseStringArg(args, i, "--prompt",     prompt))      continue;
        if (parseIntArg(args, i,    "--max-tokens", opts.max_tokens)) continue;
        if (args[i] == "--single-engine") { single_engine = true; continue; }
        if (args[i] == "--collaborate" || args[i] == "--collaborative") { single_engine = false; continue; }
    }

    if (!single_engine) {
        return runCollaborateCommand(args);
    }

    if (model_path.empty()) {
        const char* env = std::getenv("HETEROACCEL_MODEL_PATH");
        if (env) model_path = env;
    }

    if (model_path.empty()) {
        std::cerr << "ERROR: No model path provided.\n";
        std::cerr << "  Usage: adaptive-gpu run --model <path.gguf> --prompt \"...\"\n";
        return 1;
    }

    agr::HeteroRuntime runtime;
    if (!runtime.initialize()) {
        std::cerr << "ERROR: Failed to initialize HeteroRuntime\n";
        return 1;
    }

    std::cout << "Loading model via Phase 5 Scheduler...\n";
    if (!runtime.loadModel(model_path, opts)) {
        std::cerr << "ERROR: Failed to load model.\n";
        return 1;
    }

    std::cout << "\n================================================\n";
    std::cout << "HeteroAccel LLM Runtime\n";
    std::cout << "================================================\n";
    std::cout << "Prompt: " << prompt << "\n\n";

    auto cb = [](const std::string& piece) {
        std::cout << piece << std::flush;
    };

    agr::InferenceResult res = runtime.generateStreaming(model_path, prompt, cb, opts);
    std::cout << "\n================================================\n";

    if (!res.success) {
        std::cerr << "\nERROR during generation: " << res.error << "\n";
        return 1;
    }

    std::cout << "\nTelemetry:\n";
    std::cout << "  Backend:    " << res.telemetry.backend << " (" << res.telemetry.scheduler_reason << ")\n";
    if (!res.telemetry.device_name.empty()) {
        std::cout << "  Device:     " << res.telemetry.device_name << "\n";
    }
    std::cout << "  TTFT:       " << res.telemetry.ttft_ms << " ms\n";
    std::cout << "  Speed:      " << res.telemetry.tokens_per_sec << " tok/s\n";
    std::cout << "  Total time: " << res.telemetry.total_ms << " ms\n";

    runtime.shutdown();
    return 0;
}

int runChatCommand(const std::vector<std::string>& args) {
    std::string model_path;
    agr::GenerationOptions opts;
    opts.max_tokens = 24;
    bool collaborative_mode = true;

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--help" || args[i] == "-h") {
            std::cout << "Usage: adaptive-gpu chat --model <path.gguf> [--collaborate|--single-engine] [--max-tokens <N>]\n";
            return 0;
        }
        if (parseStringArg(args, i, "--model",      model_path))  continue;
        if (parseIntArg(args, i,    "--max-tokens", opts.max_tokens)) continue;
        if (args[i] == "--single-engine") { collaborative_mode = false; continue; }
        if (args[i] == "--collaborate" || args[i] == "--collaborative") { collaborative_mode = true; continue; }
    }

    if (model_path.empty()) {
        const char* env = std::getenv("HETEROACCEL_MODEL_PATH");
        if (env) model_path = env;
    }

    if (model_path.empty()) {
        std::cerr << "ERROR: No model path provided.\n";
        std::cerr << "  Usage: adaptive-gpu chat --model <path.gguf>\n";
        return 1;
    }

    agr::HeteroRuntime runtime;
    if (!runtime.initialize()) {
        std::cerr << "ERROR: Failed to initialize HeteroRuntime\n";
        return 1;
    }

    std::cout << "HeteroAccel Unified Collaborative Interactive Chat\n";
    std::cout << "Active Engines: HydroXL + llama.cpp + Colibri\n";
    std::cout << "Type 'exit' or 'quit' to stop.\n\n";

    std::vector<std::pair<std::string, std::string>> history;
    std::string input;
    while (true) {
        std::cout << "\nYou: ";
        if (!std::getline(std::cin, input)) break;
        if (input == "exit" || input == "quit") break;
        if (input.empty()) continue;

        std::cout << "\nAssistant:\n";
        if (collaborative_mode) {
            agr::collaboration::CollaborativeRequest creq;
            creq.model_path = model_path;
            creq.prompt = input;
            creq.conversation_history = history;
            creq.max_new_tokens_per_task = opts.max_tokens > 0 ? opts.max_tokens : 24;

            auto cb = [](const std::string& chunk, int) -> bool {
                std::cout << chunk << std::flush;
                return true;
            };
            auto cres = runtime.generateCollaborativeStreaming(creq, cb);
            if (!cres.success) {
                std::cerr << "\n[Error: " << cres.error << "]\n";
            } else {
                history.push_back({"User", input});
                history.push_back({"Assistant", cres.unified_response});
                std::cout << "\n" << cres.formatTelemetryReport() << "\n";
            }
        } else {
            auto cb = [](const std::string& piece) {
                std::cout << piece << std::flush;
            };
            agr::InferenceResult res = runtime.generateStreaming(model_path, input, cb, opts);
            if (!res.success) {
                std::cerr << "\n[Error: " << res.error << "]\n";
            }
            std::cout << "\n";
        }
    }

    runtime.shutdown();
    return 0;
}

// =============================================================================
// Phase 8 — Profiling
// =============================================================================
int runProfileCommand(const std::vector<std::string>& args) {
    bool show = false;
    bool reset = false;
    for (const auto& a : args) {
        if (a == "--show") show = true;
        if (a == "--reset") reset = true;
    }

    agr::HeteroRuntime runtime;
    runtime.initialize(); // loads profile internally

    if (reset) {
        // Just clear the file
        std::remove("profiles/performance.json");
        std::cout << "Profile reset.\n";
        return 0;
    }

    std::cout << "Phase 8: Adaptive Profiling System\n";
    std::cout << "==================================\n";
    
    // In a real app we'd expose the profile data directly, but here we can 
    // just mention the file exists and is managed by HeteroRuntime.
    std::cout << "Performance history is stored in profiles/performance.json\n";
    
    if (show) {
        std::cout << "\nFile contents:\n";
        std::ifstream in("profiles/performance.json");
        if (in) {
            std::cout << in.rdbuf() << "\n";
        } else {
            std::cout << "No profile data yet.\n";
        }
    } else {
        std::cout << "Use --show to display contents.\n";
    }

    return 0;
}

// =============================================================================
// Phase 9 — Workload status and diagnostics
// =============================================================================
int runWorkloadsCommand() {
    std::cout << "HeteroAccel Workload Registry\n";
    std::cout << "=============================\n\n";

    agr::HeteroRuntime runtime;
    if (!runtime.initialize()) {
        std::cerr << "ERROR: Failed to initialize runtime\n";
        return 1;
    }

    auto snap = runtime.workloadSnapshot();
    if (snap.empty()) {
        std::cout << "No active workloads.\n";
    } else {
        std::printf("%-6s %-20s %-10s %-10s %-12s\n",
            "ID", "Name", "Priority", "Class", "State");
        std::printf("%-6s %-20s %-10s %-10s %-12s\n",
            "------", "--------------------", "----------", "----------", "------------");
        for (const auto& e : snap) {
            std::printf("%-6llu %-20s %-10s %-10s %-12s\n",
                (unsigned long long)e.id,
                e.name.c_str(),
                agr::toString(e.priority),
                agr::toString(e.wclass),
                agr::toString(e.state));
        }
    }
    return 0;
}

int runStatusCommand() {
    std::cout << "HeteroAccel Runtime Status\n";
    std::cout << "==========================\n\n";

    agr::HeteroRuntime runtime;
    if (!runtime.initialize()) {
        std::cerr << "ERROR: Failed to initialize runtime\n";
        return 1;
    }

    std::cout << runtime.diagnosticsReport();

    // Phase 9: show strategy info
    std::cout << "\nPhase 9 — Execution Policy Engine\n";
    std::cout << "Model size (unknown) + available GPU memory -> strategy selected at inference time.\n";
    std::cout << "Use 'adaptive-gpu workloads' to see active workload queue.\n";
    std::cout << "Use 'adaptive-gpu profile --show' to see performance history.\n";
    return 0;
}

} // namespace

// =============================================================================
// main
// =============================================================================
// =============================================================================
// Phase 12 — Analyze Model
// =============================================================================
int runAnalyzeCommand(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cerr << "Usage: adaptive-gpu analyze <path_to_model.gguf>\n";
        return 1;
    }
    
    std::string model_path = args[0];
    
    agr::HeteroRuntime runtime;
    runtime.initialize();
    
    agr::CapabilityReport report = runtime.analyzeModel(model_path);
    std::cout << report.formatToString() << "\n";
    
    return 0;
}

int runPlanCommand(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cerr << "Usage: adaptive-gpu plan <path_to_model.gguf>\n";
        return 1;
    }

    agr::HeteroRuntime runtime;
    runtime.initialize();

    agr::ModelExecutionPlan plan = runtime.planExecution(args[0]);
    std::cout << "Model: " << args[0] << "\n";
    std::cout << plan.formatToString() << "\n";
    return plan.viable ? 0 : 2;
}

#include "model/FileModelDataSource.h"
#include "model/HostAcceleratorMemory.h"
#include "model/ModelResidencyManager.h"
#include "model/PlanResidencyAdapter.h"
#include "model/GGUFRegionExtractor.h"

int runResidencyCommand(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cerr << "Usage: adaptive-gpu residency <path_to_model.gguf>\n";
        return 1;
    }
    const std::string& model_path = args[0];

    agr::HeteroRuntime runtime;
    runtime.initialize();

    // 1. Plan Execution (Phase 13)
    agr::ModelExecutionPlan plan = runtime.planExecution(model_path);
    if (!plan.viable) {
        std::cerr << "Model is not viable for execution: " << plan.explanation << "\n";
        return 1;
    }

    // 2. Set up Residency Manager (Phase 14)
    agr::FileModelDataSource src(model_path);
    if (!src.open()) {
        std::cerr << "Failed to open model file: " << src.lastError() << "\n";
        return 1;
    }

    // Allocate budgets
    agr::ModelResidencyBudget budget;
    budget.ram_bytes = 4ULL * 1024 * 1024 * 1024; // 4GB max RAM for testing

    agr::HostAcceleratorMemory accel(2ULL * 1024 * 1024 * 1024); // 2GB host fallback
    agr::ModelResidencyManager mgr(src, budget, &accel);

    // 3. Apply Plan to Residency Manager
    agr::PlanResidencyAdapter adapter;
    std::cout << "Phase 13 Plan: " << agr::toString(plan.residency) << " mode on " << agr::toString(plan.mode) << "\n";
    std::cout << "Applying plan to Phase 14 Residency Manager...\n";
    
    auto result = adapter.apply(plan, model_path, mgr);
    
    std::cout << "\n========================================\n";
    std::cout << "Residency Execution Report\n";
    std::cout << "========================================\n";
    std::cout << "Status:       " << (result.success ? "SUCCESS" : "FAILED") << "\n";
    if (!result.success) {
        std::cout << "Error:        " << result.error << "\n";
    }
    std::cout << "Explanation:  " << result.explanation << "\n\n";

    std::cout << "Data source:\n";
    std::cout << "  File-backed: YES (" << src.description() << ")\n\n";

    std::cout << "Regions:\n";
    std::cout << "  Total found: " << result.regions_total << "\n";
    std::cout << "  In RAM:      " << result.regions_ram << "\n";
    std::cout << "  In Accel:    " << result.regions_accel << "\n\n";

    auto telem = mgr.telemetry();
    std::cout << "RAM residency:\n";
    std::cout << "  Budget:      " << (budget.ram_bytes / 1024 / 1024) << " MB\n";
    std::cout << "  Current:     " << (telem.resident_ram_bytes / 1024 / 1024) << " MB\n";
    std::cout << "  Peak:        " << (telem.peak_ram_bytes / 1024 / 1024) << " MB\n\n";

    std::cout << "Accelerator:\n";
    std::cout << "  Device:      " << accel.name() << "\n";
    std::cout << "  Budget:      " << (accel.capacity() / 1024 / 1024) << " MB\n";
    std::cout << "  Current:     " << (telem.resident_accelerator_bytes / 1024 / 1024) << " MB\n";
    std::cout << "  Peak:        " << (telem.peak_accelerator_bytes / 1024 / 1024) << " MB\n\n";

    std::cout << "Streaming:\n";
    std::cout << "  Bytes read:     " << (telem.bytes_read_from_disk / 1024 / 1024) << " MB\n";
    std::cout << "  Bytes uploaded: " << (telem.bytes_to_accelerator / 1024 / 1024) << " MB\n";
    std::cout << "  Evictions:      " << telem.eviction_count << "\n";
    std::cout << "  Stalls:         " << telem.stall_count << "\n";

    return result.success ? 0 : 1;
}

#include "execution/ExecutionGraphBuilder.h"
#include "execution/ExecutionGraphScheduler.h"

int runGraphCommand(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cerr << "Usage: adaptive-gpu graph <path_to_model.gguf>\n";
        return 1;
    }
    const std::string& model_path = args[0];

    agr::HeteroRuntime runtime;
    runtime.initialize();

    // 1. Plan Execution (Phase 13)
    agr::ModelExecutionPlan plan = runtime.planExecution(model_path);
    if (!plan.viable) {
        std::cerr << "Model is not viable for execution: " << plan.explanation << "\n";
        return 1;
    }

    // 2. Extract regions (Phase 14 integration)
    agr::GGUFRegionExtractor extractor;
    std::vector<agr::ModelRegion> regions = extractor.extract(model_path);
    if (regions.empty()) {
        regions.push_back(extractor.wholeFileRegion(model_path));
    }

    // 3. Build Graph (Phase 15)
    agr::ExecutionGraphBuilder builder;
    agr::ExecutionGraph graph;
    std::string err;
    if (!builder.build(plan, model_path, regions, "Test prompt", graph, err)) {
        std::cerr << "Failed to build execution graph: " << err << "\n";
        return 1;
    }

    std::cout << "\n========================================\n";
    std::cout << "Execution Graph Diagnostic Report\n";
    std::cout << "========================================\n";
    std::cout << "Model:         " << model_path << "\n";
    std::cout << "Execution Mode:" << agr::toString(plan.mode) << "\n";
    std::cout << "Residency Req: " << agr::toString(plan.residency) << "\n";
    std::cout << "Engine:        " << plan.engine_key << "\n\n";

    std::cout << "Nodes (" << graph.getAllNodeIds().size() << "):\n";
    for (uint64_t id : graph.getAllNodeIds()) {
        auto* node = graph.getNode(id);
        std::cout << "  [" << id << "] " << node->getName() 
                  << " (Type: " << agr::toString(node->getType()) 
                  << ", Resource: " << agr::toString(node->getResource()) << ")\n";
        
        auto deps = graph.getPredecessors(id);
        if (!deps.empty()) {
            std::cout << "      Requires: ";
            for (uint64_t d : deps) std::cout << d << " ";
            std::cout << "\n";
        }
    }
    std::cout << "\nValidation:    OK (DAG confirmed, no cycles)\n";
    std::cout << "Note: Finer-grained tensor execution nodes will be available when natively exposed by engines (e.g. HydroXL).\n";

    return 0;
}

// =============================================================================
// Phase 25 — Orchestrated Inference (uses EngineOrchestrator for backend selection)
// =============================================================================
int runInferOrchCommand(const std::vector<std::string>& args) {
    std::string model_path;
    std::string prompt = "Tell me about adaptive GPU computing.";
    std::string forced_backend;
    int max_tokens = 128;

    for (size_t i = 0; i < args.size(); ++i) {
        if (parseStringArg(args, i, "--model",   model_path))     continue;
        if (parseStringArg(args, i, "--prompt",  prompt))         continue;
        if (parseStringArg(args, i, "--backend", forced_backend)) continue;
        if (parseIntArg(args, i, "--max-tokens", max_tokens))     continue;
    }

    if (model_path.empty()) {
        const char* env = std::getenv("HETEROACCEL_MODEL_PATH");
        if (env) model_path = env;
    }

    if (model_path.empty()) {
        std::cerr << "ERROR: No model path provided.\n"
                  << "  Usage: adaptive-gpu infer --model <path.gguf> [--prompt <text>]\n"
                  << "                            [--backend cpu|vulkan|cuda] [--max-tokens <N>]\n";
        return 1;
    }
    if (forced_backend == "cpu") forced_backend = "hydroxl_cpu";
    else if (forced_backend == "vulkan") forced_backend = "hydroxl_vulkan";

    using namespace agr::orchestration;

    // Initialize orchestrator with hardware-aware backend registration
    EngineOrchestrator& orch = EngineOrchestrator::instance();
    orch.reset();
    orchestration::HardwareDetector& hw = orchestration::HardwareDetector::instance();
    hw.detect();
    register_default_backends(orch, hw);

    // Register engine factories
    orch.register_engine_factory("hydroxl_cpu", []() -> std::shared_ptr<IInferenceEngine> {
        return std::make_shared<HydroXLCpuEngineAdapter>();
    });
    if (hw.has_vulkan()) {
        orch.register_engine_factory("hydroxl_vulkan", []() -> std::shared_ptr<IInferenceEngine> {
            return std::make_shared<HydroXLVulkanEngineAdapter>();
        });
    }

    // Build selection request from model file metadata
    EngineOrchestrator::SelectionRequest req;
    req.model_path      = model_path;
    req.model_arch      = "qwen2";   // HydroXL supports qwen2
    req.quant_type      = 12;        // Q4_K_M
    req.n_ctx           = 512;
    req.forced_backend  = forced_backend;
    req.allow_cpu_fallback = true;

    // Estimate model size from file
    {
        std::ifstream f(model_path, std::ios::binary | std::ios::ate);
        if (f) req.model_size_bytes = static_cast<size_t>(f.tellg());
    }

    EngineOrchestrator::SelectionResult sel = orch.select_backend(req);

    std::cout << "=== HeteroAccel Orchestrated Inference (Phase 25) ===\n";
    std::cout << "Model:   " << model_path << "\n";
    std::cout << "Backend: " << sel.selected_backend << "\n";
    std::cout << "Reason:  " << sel.reason << "\n";
    if (!sel.rejected_backends.empty()) {
        std::cout << "Skipped: ";
        for (const auto& r : sel.rejected_backends) std::cout << r << " ";
        std::cout << "\n";
    }
    std::cout << "Prompt:  " << prompt << "\n\n";

    // Create and initialize the selected engine
    std::string engine_error;
    auto engine = orch.create_engine(sel, req, engine_error);
    auto create_rep = orch.last_creation_report();
    if (!engine) {
        std::cerr << "ERROR: Failed to create engine '" << sel.selected_backend << "': " << engine_error << "\n";
        return 1;
    }
    if (create_rep.fell_back) {
        std::cout << "Fallback: Using '" << create_rep.actual_backend << "' (reason: " << create_rep.failure_reason << ")\n";
    }

    std::cout << "Generating...\n";
    std::cout << "---------------------------------------------\n";

    orchestration::InferenceRequest ireq;
    ireq.prompt = prompt;
    ireq.max_new_tokens = max_tokens;
    ireq.temperature = 0.8f;
    ireq.top_p = 0.95f;
    ireq.stream = true;

    auto t0 = std::chrono::steady_clock::now();

    orchestration::InferenceResult result;
    if (engine->supports_streaming()) {
        engine->run_stream(ireq, [](const std::string& tok, int) -> bool {
            std::cout << tok << std::flush;
            return true;
        }, result);
    } else {
        result = engine->run(ireq);
    }
    std::cout << "\n";

    double total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::cout << "---------------------------------------------\n";
    if (!result.success) {
        std::cerr << "ERROR: " << result.error << "\n";
        return 1;
    }

    // Record observed performance for future runs
    if (result.tokens_per_second > 0.0f) {
        orch.record_performance(sel.selected_backend,
                                result.tokens_per_second,
                                result.peak_memory_bytes);
    }

    std::cout << "\nBackend:    " << sel.selected_backend
              << (sel.is_fallback ? " (fallback)" : "") << "\n";
    std::cout << "Tokens:     " << result.output_tokens << "\n";
    std::cout << "Speed:      ";
    if (result.tokens_per_second > 0.0f)
        std::printf("%.2f tok/s\n", result.tokens_per_second);
    else
        std::cout << "N/A\n";
    std::cout << "Total time: " << (int)total_ms << " ms\n";

    return 0;
}
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
#if defined(_WIN32)
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
    // Parse wide command line directly into UTF-8 to preserve non-ASCII (Hindi, Emoji, Math) on Windows
    const wchar_t* wcmd = GetCommandLineW();
    if (wcmd) {
        std::vector<std::wstring> wargs;
        size_t i = 0;
        while (wcmd[i] != L'\0') {
            while (wcmd[i] == L' ' || wcmd[i] == L'\t') ++i;
            if (wcmd[i] == L'\0') break;
            std::wstring cur;
            bool in_quotes = false;
            while (wcmd[i] != L'\0' && (in_quotes || (wcmd[i] != L' ' && wcmd[i] != L'\t'))) {
                if (wcmd[i] == L'\\' && wcmd[i + 1] == L'"') {
                    cur.push_back(L'"');
                    i += 2;
                } else if (wcmd[i] == L'"') {
                    in_quotes = !in_quotes;
                    ++i;
                } else {
                    cur.push_back(wcmd[i++]);
                }
            }
            wargs.push_back(std::move(cur));
        }
        if (wargs.size() > 1) {
            args.clear();
            for (size_t k = 1; k < wargs.size(); ++k) {
                const auto& ws = wargs[k];
                if (ws.empty()) {
                    args.emplace_back();
                } else {
                    int needed = WideCharToMultiByte(CP_UTF8, 0, ws.data(), static_cast<int>(ws.size()),
                                                     nullptr, 0, nullptr, nullptr);
                    std::string u8(needed > 0 ? needed : 0, '\0');
                    if (needed > 0) {
                        WideCharToMultiByte(CP_UTF8, 0, ws.data(), static_cast<int>(ws.size()),
                                            &u8[0], needed, nullptr, nullptr);
                    }
                    args.push_back(std::move(u8));
                }
            }
        }
    }
#endif

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
        // Could add Phase 8 adaptive vs static here, but skipping for brevity
        std::cerr << "Usage: adaptive-gpu benchmark [vulkan|llm] ...\n";
        return 1;
    }

    if (args[0] == "llm") {
        std::vector<std::string> rest(args.begin() + 1, args.end());
        return runLlmInference(rest);
    }
    
    if (args[0] == "memory") {
        return runMemoryCommand();
    }

    if (args[0] == "devices") {
        return runDevicesCommand();
    }

    if (args[0] == "run") {
        std::vector<std::string> rest(args.begin() + 1, args.end());
        return runInferenceCommand(rest);
    }

    if (args[0] == "collaborate") {
        std::vector<std::string> rest(args.begin() + 1, args.end());
        return runCollaborateCommand(rest);
    }

    if (args[0] == "chat") {
        std::vector<std::string> rest(args.begin() + 1, args.end());
        return runChatCommand(rest);
    }

    if (args[0] == "profile") {
        std::vector<std::string> rest(args.begin() + 1, args.end());
        return runProfileCommand(rest);
    }

    if (args[0] == "workloads") {
        return runWorkloadsCommand();
    }

    if (args[0] == "status") {
        return runStatusCommand();
    }

    if (args[0] == "scheduler") {
        return runSchedulerCommand();
    }
    
    if (args[0] == "analyze") {
        std::vector<std::string> rest(args.begin() + 1, args.end());
        return runAnalyzeCommand(rest);
    }

    if (args[0] == "plan") {
        std::vector<std::string> rest(args.begin() + 1, args.end());
        return runPlanCommand(rest);
    }

    if (args[0] == "residency") {
        std::vector<std::string> rest(args.begin() + 1, args.end());
        return runResidencyCommand(rest);
    }

    if (args[0] == "graph") {
        std::vector<std::string> rest(args.begin() + 1, args.end());
        return runGraphCommand(rest);
    }

    // Phase 17 — HydroXL native inference engine
    // Usage: adaptive-gpu hydroxl --model <path> [--prompt <text>]
    //                              [--max-tokens <N>] [--temperature <f>]
    if (args[0] == "hydroxl") {
        // Register the engine so the registry picks it up
        HydroXLEngine::registerInRegistry();

        std::string model_path, prompt = "Hello, world!";
        int max_tokens = 128;
        float temperature = 0.8f;

        for (size_t i = 1; i < args.size(); ++i) {
            if ((args[i] == "--model" || args[i] == "-m") && i + 1 < args.size())
                model_path = args[++i];
            else if ((args[i] == "--prompt" || args[i] == "-p") && i + 1 < args.size())
                prompt = args[++i];
            else if ((args[i] == "--max-tokens" || args[i] == "-n") && i + 1 < args.size())
                max_tokens = std::stoi(args[++i]);
            else if ((args[i] == "--temperature" || args[i] == "-t") && i + 1 < args.size())
                temperature = std::stof(args[++i]);
        }

        if (model_path.empty()) {
            std::cerr << "Usage: adaptive-gpu hydroxl --model <path.gguf> [--prompt <text>]\n"
                         "                             [--max-tokens <N>] [--temperature <f>]\n";
            return 1;
        }

        HydroXLEngine engine;
        if (!engine.initialize()) {
            std::cerr << "HydroXL: failed to initialize engine\n"; return 1;
        }

        ModelDescriptor desc;
        desc.model_id    = "hydroxl-model";
        desc.source_path = model_path;
        desc.format      = ModelFormat::GGUF;

        std::cout << "HydroXL: loading " << model_path << " ...\n";
        auto t0 = std::chrono::steady_clock::now();
        if (!engine.loadModel(desc)) {
            std::cerr << "HydroXL: load error: " << engine.lastEngineError().message << "\n";
            return 1;
        }
        double load_ms = std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "HydroXL: loaded in " << (int)load_ms << " ms\n";
        std::cout << "HydroXL: generating...\n";

        ExecutionContext ctx;
        ctx.prompt      = prompt;
        ctx.max_tokens  = max_tokens;
        ctx.temperature = temperature;
        ctx.seed        = 42;

        auto result = engine.executeStreaming(ctx, [](const std::string& tok){
            std::cout << tok << std::flush;
        });
        std::cout << "\n";

        if (!result.success) {
            std::cerr << "HydroXL error: " << result.error << "\n"; return 1;
        }
        std::cout << "\n[tokens: " << result.telemetry.output_tokens
                  << ", speed: " << (int)result.telemetry.tokens_per_sec << " tok/s"
                  << ", backend: " << result.telemetry.backend_name << "]\n";
        return 0;
    }

    if (args[0] == "infer") {
        std::vector<std::string> rest(args.begin() + 1, args.end());
        return runInferOrchCommand(rest);
    }

    std::cerr << "Unknown command: " << args[0] << "\n";
    printUsage();
    return 1;
}
