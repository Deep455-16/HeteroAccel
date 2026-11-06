// src/orchestration/OrchestratorInit.cpp
// Phase 21 / Phase 27: Default Backend Registration

#include "orchestration/OrchestratorInit.h"

namespace agr {
namespace orchestration {

void register_default_backends(EngineOrchestrator& orch, const HardwareDetector& hw) {
    // HydroXL CPU backend
    BackendCapabilities cpu;
    cpu.backend_id = "hydroxl_cpu";
    cpu.supported_quant_types = {0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17};
    cpu.supported_arch = {
        "llama", "qwen2", "qwen2_vl",
        "mistral", "mistral3",
        "phi", "phi2", "phi3",
        "gemma", "gemma2"
    };
    cpu.requires_cuda = false;
    cpu.requires_vulkan = false;
    cpu.min_vram_bytes = 0;
    cpu.min_ram_bytes = 512 * 1024 * 1024;  // 512 MB minimum
    cpu.relative_throughput = 1.0f;
    cpu.init_cost_ms = 2000.0f;
    cpu.priority = 10;
    orch.register_backend(cpu);

    // HydroXL Vulkan backend (if Vulkan is available)
    if (hw.has_vulkan()) {
        BackendCapabilities vk;
        vk.backend_id = "hydroxl_vulkan";
        // Shaders implement ops for quants 0, 1, 7, 8 directly; quant 12 (Q4_K)
        // routes natively to CPU orchestrator target.
        vk.supported_quant_types = {0, 1, 7, 8};
        vk.supported_arch = {
            "llama", "qwen2", "qwen2_vl",
            "mistral", "mistral3",
            "phi", "phi2", "phi3",
            "gemma", "gemma2"
        };
        vk.requires_cuda = false;
        vk.requires_vulkan = true;
        vk.min_vram_bytes = 0;
        vk.min_ram_bytes = 512 * 1024 * 1024;
        vk.relative_throughput = 2.5f;
        vk.init_cost_ms = 500.0f;
        vk.priority = 20;
        orch.register_backend(vk);
    }

    // CUDA backend: real inference through llama.cpp's GGML_CUDA backend
    // (LlamaCppEngineAdapter). Registered only in binaries built with
    // HETEROACCEL_ENABLE_CUDA; whether it is *usable* is decided per run by
    // BackendCapabilities::is_available() against the detected CUDA devices.
#if defined(HETEROACCEL_CUDA_ENABLED) && HETEROACCEL_CUDA_ENABLED
    {
        BackendCapabilities cuda_b;
        cuda_b.backend_id = "cuda";
        cuda_b.requires_cuda = true;
        cuda_b.requires_vulkan = false;
        cuda_b.min_vram_bytes = 512ULL * 1024 * 1024;
        cuda_b.min_ram_bytes = 512 * 1024 * 1024;
        cuda_b.relative_throughput = 12.0f;
        cuda_b.init_cost_ms = 3000.0f;
        cuda_b.priority = 30;
        orch.register_backend(cuda_b);
    }
#endif
}
} // namespace orchestration
} // namespace agr
