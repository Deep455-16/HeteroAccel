// src/orchestration/BackendCapabilities.h
// Phase 21: Backend Capability Descriptor

#pragma once

#include "orchestration/HardwareDetector.h"
#include <string>
#include <vector>
#include <algorithm>

namespace agr {
namespace orchestration {

struct BackendCapabilities {
    std::string backend_id;      // "hydroxl_cpu", "hydroxl_vulkan", "llamacpp", "plugin:<name>"

    // Supported quantization types (GGML type IDs)
    std::vector<int> supported_quant_types;

    // Supported model architectures
    std::vector<std::string> supported_arch;  // "llama", "qwen2", "phi", etc.

    // Hardware requirements
    bool requires_cuda = false;
    bool requires_vulkan = false;
    size_t min_vram_bytes = 0;    // 0 for CPU backends
    size_t min_ram_bytes = 0;

    // Performance characteristics (from benchmark history or heuristics)
    float relative_throughput = 1.0f;  // normalized, 1.0 = baseline CPU
    float init_cost_ms = 1000.0f;      // estimated cold-start cost

    // Priority for tie-breaking
    int priority = 0;

    bool is_available(const HardwareDetector& hw) const {
        if (requires_cuda && !hw.has_cuda()) {
            return false;
        }
        if (requires_vulkan && !hw.has_vulkan()) {
            return false;
        }
        if (min_vram_bytes > 0) {
            const auto* gpu = hw.best_gpu();
            if (!gpu || gpu->vram_bytes < min_vram_bytes) {
                return false;
            }
        }
        if (min_ram_bytes > 0) {
            if (hw.available_system_ram() < min_ram_bytes) {
                return false;
            }
        }
        return true;
    }

    bool supports_quant(int quant_type) const {
        if (supported_quant_types.empty()) return true; // empty = wildcard
        for (int q : supported_quant_types) {
            if (q == quant_type) return true;
        }
        return false;
    }

    bool supports_arch(const std::string& arch) const {
        if (arch.empty() || supported_arch.empty()) return true; // empty = wildcard
        std::string lower_arch = arch;
        std::transform(lower_arch.begin(), lower_arch.end(), lower_arch.begin(), ::tolower);
        for (const auto& a : supported_arch) {
            std::string lower_a = a;
            std::transform(lower_a.begin(), lower_a.end(), lower_a.begin(), ::tolower);
            if (lower_a == lower_arch) return true;
        }
        return false;
    }
};

} // namespace orchestration
} // namespace agr
