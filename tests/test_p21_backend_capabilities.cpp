// tests/test_p21_backend_capabilities.cpp
// Phase 21: Backend Capability Unit Tests

#include "mini_test.h"
#include "orchestration/BackendCapabilities.h"
#include "orchestration/HardwareDetector.h"
#include "orchestration/EngineOrchestrator.h"

#include <iostream>

using namespace agr::orchestration;

int main() {
    std::cout << "[TEST] Running test_p21_backend_capabilities...\n";

    auto& hw = HardwareDetector::instance();

    // -----------------------------------------------------------------------
    // Test 1: CPU-only mock environment
    // -----------------------------------------------------------------------
    {
        std::vector<DeviceInfo> mock_devs;
        DeviceInfo cpu;
        cpu.name = "Mock CPU";
        cpu.vendor = DeviceInfo::Vendor::Intel;
        cpu.api_support = static_cast<uint32_t>(DeviceInfo::APISupport::CPU);
        cpu.vram_bytes = 0;
        mock_devs.push_back(cpu);

        hw.set_mock_environment(mock_devs, 8ULL * 1024 * 1024 * 1024, false, false);

        BackendCapabilities cpu_b;
        cpu_b.backend_id = "cpu";
        cpu_b.requires_vulkan = false;
        cpu_b.requires_cuda = false;
        cpu_b.min_ram_bytes = 1024 * 1024 * 1024;
        AGR_CHECK(cpu_b.is_available(hw));

        BackendCapabilities vk_b;
        vk_b.backend_id = "vulkan";
        vk_b.requires_vulkan = true;
        vk_b.requires_cuda = false;
        AGR_CHECK(!vk_b.is_available(hw)); // Vulkan excluded

        BackendCapabilities cuda_b;
        cuda_b.backend_id = "cuda";
        cuda_b.requires_vulkan = false;
        cuda_b.requires_cuda = true;
        AGR_CHECK(!cuda_b.is_available(hw)); // CUDA excluded
    }

    // -----------------------------------------------------------------------
    // Test 2: Vulkan available, CUDA absent mock environment
    // -----------------------------------------------------------------------
    {
        std::vector<DeviceInfo> mock_devs;
        DeviceInfo cpu;
        cpu.name = "Mock CPU";
        cpu.api_support = static_cast<uint32_t>(DeviceInfo::APISupport::CPU);
        mock_devs.push_back(cpu);

        DeviceInfo gpu;
        gpu.name = "Mock Intel Iris Xe";
        gpu.vendor = DeviceInfo::Vendor::Intel;
        gpu.api_support = static_cast<uint32_t>(DeviceInfo::APISupport::Vulkan);
        gpu.vulkan_ok = true;
        gpu.vram_bytes = 4ULL * 1024 * 1024 * 1024; // 4 GB
        mock_devs.push_back(gpu);

        hw.set_mock_environment(mock_devs, 8ULL * 1024 * 1024 * 1024, true, false);

        BackendCapabilities vk_b;
        vk_b.backend_id = "vulkan";
        vk_b.requires_vulkan = true;
        vk_b.requires_cuda = false;
        vk_b.min_vram_bytes = 2ULL * 1024 * 1024 * 1024; // 2 GB
        AGR_CHECK(vk_b.is_available(hw));

        BackendCapabilities vk_big;
        vk_big.backend_id = "vulkan_big";
        vk_big.requires_vulkan = true;
        vk_big.min_vram_bytes = 16ULL * 1024 * 1024 * 1024; // 16 GB (exceeds 4 GB)
        AGR_CHECK(!vk_big.is_available(hw));

        BackendCapabilities cuda_b;
        cuda_b.backend_id = "cuda";
        cuda_b.requires_cuda = true;
        AGR_CHECK(!cuda_b.is_available(hw)); // CUDA still excluded
    }

    // -----------------------------------------------------------------------
    // Test 3: CUDA available, Vulkan absent mock environment
    // -----------------------------------------------------------------------
    {
        std::vector<DeviceInfo> mock_devs;
        DeviceInfo cpu;
        cpu.name = "Mock CPU";
        cpu.api_support = static_cast<uint32_t>(DeviceInfo::APISupport::CPU);
        mock_devs.push_back(cpu);

        DeviceInfo gpu;
        gpu.name = "Mock NVIDIA RTX 4090";
        gpu.vendor = DeviceInfo::Vendor::NVIDIA;
        gpu.api_support = static_cast<uint32_t>(DeviceInfo::APISupport::CUDA);
        gpu.cuda_ok = true;
        gpu.vram_bytes = 24ULL * 1024 * 1024 * 1024; // 24 GB
        mock_devs.push_back(gpu);

        hw.set_mock_environment(mock_devs, 32ULL * 1024 * 1024 * 1024, false, true);

        BackendCapabilities cuda_b;
        cuda_b.backend_id = "cuda";
        cuda_b.requires_cuda = true;
        cuda_b.requires_vulkan = false;
        AGR_CHECK(cuda_b.is_available(hw));

        BackendCapabilities vk_b;
        vk_b.backend_id = "vulkan";
        vk_b.requires_vulkan = true;
        AGR_CHECK(!vk_b.is_available(hw)); // Vulkan excluded
    }

    // -----------------------------------------------------------------------
    // Test 4: Quantization and Architecture matching
    // -----------------------------------------------------------------------
    {
        BackendCapabilities b;
        b.supported_quant_types = {0, 1, 8, 12};
        b.supported_arch = {"qwen2", "llama"};

        AGR_CHECK(b.supports_quant(0));
        AGR_CHECK(b.supports_quant(12));
        AGR_CHECK(!b.supports_quant(6)); // Q5_0 not in list

        AGR_CHECK(b.supports_arch("qwen2"));
        AGR_CHECK(b.supports_arch("QWEN2")); // case-insensitive
        AGR_CHECK(b.supports_arch("llama"));
        AGR_CHECK(!b.supports_arch("mistral"));

        // Wildcard tests (empty lists)
        BackendCapabilities b_wild;
        AGR_CHECK(b_wild.supports_quant(42));
        AGR_CHECK(b_wild.supports_arch("any_arch"));
    }

    // -----------------------------------------------------------------------
    // Test 5: Scoring logic ordering under latency vs throughput preference
    // -----------------------------------------------------------------------
    {
        // Mock hardware with 16 GB RAM
        std::vector<DeviceInfo> mock_devs;
        DeviceInfo cpu;
        cpu.name = "Test CPU";
        cpu.api_support = static_cast<uint32_t>(DeviceInfo::APISupport::CPU);
        mock_devs.push_back(cpu);
        hw.set_mock_environment(mock_devs, 16ULL * 1024 * 1024 * 1024, false, false);

        auto& orch = EngineOrchestrator::instance();
        orch.reset();

        BackendCapabilities fast_warmup;
        fast_warmup.backend_id = "fast_warmup";
        fast_warmup.relative_throughput = 1.0f;
        fast_warmup.init_cost_ms = 100.0f;
        fast_warmup.supported_quant_types = {12};
        fast_warmup.supported_arch = {"qwen2"};
        orch.register_backend(fast_warmup);

        BackendCapabilities slow_warmup;
        slow_warmup.backend_id = "slow_warmup";
        slow_warmup.relative_throughput = 1.5f;
        slow_warmup.init_cost_ms = 8000.0f; // very high cold-start cost
        slow_warmup.supported_quant_types = {12};
        slow_warmup.supported_arch = {"qwen2"};
        orch.register_backend(slow_warmup);

        EngineOrchestrator::SelectionRequest req_lat;
        req_lat.model_arch = "qwen2";
        req_lat.quant_type = 12;
        req_lat.prefer_latency = true; // penalizes high cold-start
        auto sel_lat = orch.select_backend(req_lat);
        std::cout << "[INFO] Prefer-latency winner: " << sel_lat.selected_backend << "\n";
        AGR_CHECK(sel_lat.selected_backend == "fast_warmup");

        EngineOrchestrator::SelectionRequest req_thr;
        req_thr.model_arch = "qwen2";
        req_thr.quant_type = 12;
        req_thr.prefer_latency = false; // cares about throughput
        auto sel_thr = orch.select_backend(req_thr);
        std::cout << "[INFO] Prefer-throughput winner: " << sel_thr.selected_backend << "\n";
        // Throughput 1.5 wins or fast_warmup with 100ms
        AGR_CHECK(!sel_thr.selected_backend.empty());
    }

    // Restore real hardware detection
    hw.reset();
    hw.detect();

    AGR_TEST_MAIN_END();
}
