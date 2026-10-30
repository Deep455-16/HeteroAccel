// src/orchestration/OrchestratorInit.cpp
// Phase 21: Default Backend Registration

#include "orchestration/OrchestratorInit.h"

namespace agr {
namespace orchestration {

void register_default_backends(EngineOrchestrator& orch, const HardwareDetector& hw) {
    // HydroXL CPU backend
    BackendCapabilities cpu;
    cpu.backend_id = "hydroxl_cpu";
    cpu.supported_quant_types = {0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17};  // all GGML types
    cpu.supported_arch = {"llama", "qwen2", "mistral", "phi", "gemma"};
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
        vk.supported_arch = {"llama", "qwen2", "mistral", "phi", "gemma"};
        vk.requires_cuda = false;
        vk.requires_vulkan = true;
        // Unified-memory devices (e.g. Intel Iris Xe) share host RAM; discrete GPUs
        // with dedicated VRAM still work — set to 0 so the orchestrator does not
        // incorrectly filter out unified-memory hardware.
        vk.min_vram_bytes = 0;
        vk.min_ram_bytes = 512 * 1024 * 1024;
        // Phase 22 measured: ~2.5x faster than CPU on Intel Iris Xe (5.2 vs 2.1 tok/s)
        vk.relative_throughput = 2.5f;
        vk.init_cost_ms = 500.0f;
        vk.priority = 20;
        orch.register_backend(vk);
    }

    // CUDA backend placeholder (compiles on all platforms, activates only when CUDA present)
#ifdef HETEROACCEL_CUDA_ENABLED
    if (hw.has_cuda()) {
        BackendCapabilities cuda_b;
        cuda_b.backend_id = "cuda";
        cuda_b.supported_quant_types = {0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17};
        cuda_b.supported_arch = {"llama", "qwen2", "mistral", "phi", "gemma"};
        cuda_b.requires_cuda = true;
        cuda_b.requires_vulkan = false;
        cuda_b.min_vram_bytes = 1024 * 1024 * 1024;
        cuda_b.min_ram_bytes = 512 * 1024 * 1024;
        cuda_b.relative_throughput = 5.0f;
        cuda_b.init_cost_ms = 3000.0f;
        cuda_b.priority = 30;
        orch.register_backend(cuda_b);
    }
#endif
}

} // namespace orchestration
} // namespace agr
