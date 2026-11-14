// tests/test_p31_hardware_matrix.cpp
// Unified Collaborative Runtime — Engine vs. Hardware Backend Separation Matrix
//
// Validates (Sections 10 & 15.3):
//   1. Engine selection (HydroXL, llama.cpp, Colibri) is strictly separated from
//      Hardware Backend selection (CPU, Vulkan, CUDA).
//   2. HydroXL + CPU ("hydroxl_cpu"), HydroXL + Vulkan ("hydroxl_vulkan" for
//      supported quants {0,1,7,8}), and HydroXL + CUDA preference (honestly
//      selects CPU because HydroXL does not implement CUDA kernels).
//   3. llama.cpp + CPU ("llamacpp_cpu"), llama.cpp + Vulkan ("llamacpp_vulkan"),
//      and llama.cpp + CUDA ("cuda" when compiled and detected, or fallback).
//   4. Colibri + CPU ("colibri_cpu"), Colibri + Vulkan/CUDA preference (honestly
//      selects CPU because Colibri is a CPU specialized analyzer).
//   5. Cross-engine GPU contention avoidance within the same execution wave:
//      when HydroXL claims the Vulkan GPU in Wave 0, llama.cpp in the same
//      concurrent wave is assigned to CPU ("llamacpp_cpu") to prevent GPU queue
//      thrashing.

#include "mini_test.h"
#include "collaboration/WorkloadPartitioner.h"
#include "orchestration/HardwareDetector.h"

#include <iostream>
#include <string>
#include <vector>

using namespace agr::collaboration;
using namespace agr::orchestration;

int main() {
    std::cout << "=== test_p31_hardware_matrix ===\n";

    auto& hw = HardwareDetector::instance();

    ModelMemoryProfile prof_q8;
    prof_q8.arch = "qwen2";
    prof_q8.quant_type = 8; // Q8_0 (supported by HydroXL Vulkan shaders)
    prof_q8.hydroxl_estimated_bytes = 512ULL * 1024 * 1024;
    prof_q8.llamacpp_estimated_bytes = 256ULL * 1024 * 1024;
    prof_q8.colibri_estimated_bytes = 8ULL * 1024 * 1024;

    ModelMemoryProfile prof_q4k = prof_q8;
    prof_q4k.quant_type = 12; // Q4_K (routes HydroXL to CPU, llama.cpp to Vulkan/CPU)

    // -----------------------------------------------------------------------
    // 1. CPU-Only Hardware Environment
    // -----------------------------------------------------------------------
    {
        std::vector<DeviceInfo> devs;
        DeviceInfo cpu;
        cpu.name = "Mock CPU";
        cpu.api_support = static_cast<uint32_t>(DeviceInfo::APISupport::CPU);
        devs.push_back(cpu);
        hw.set_mock_environment(devs, 16ULL * 1024 * 1024 * 1024, false, false);

        std::string bid, reason;
        AGR_CHECK(WorkloadPartitioner::selectHardwareBackendForEngine(
            EngineId::HYDROXL, prof_q8, hw, "", false, bid, reason) == HardwareBackend::CPU);
        AGR_CHECK(bid == "hydroxl_cpu");

        AGR_CHECK(WorkloadPartitioner::selectHardwareBackendForEngine(
            EngineId::LLAMA_CPP, prof_q8, hw, "", false, bid, reason) == HardwareBackend::CPU);
        AGR_CHECK(bid == "llamacpp_cpu");

        AGR_CHECK(WorkloadPartitioner::selectHardwareBackendForEngine(
            EngineId::COLIBRI, prof_q8, hw, "", false, bid, reason) == HardwareBackend::CPU);
        AGR_CHECK(bid == "colibri_cpu");
        std::cout << "  [OK] CPU-only environment assigns all 3 engines to CPU backends\n";
    }

    // -----------------------------------------------------------------------
    // 2. Vulkan Hardware Environment (HydroXL + Vulkan, llama.cpp + Vulkan,
    //    Colibri + CPU, and GPU Contention Avoidance in Concurrent Wave)
    // -----------------------------------------------------------------------
    {
        std::vector<DeviceInfo> devs;
        DeviceInfo cpu;
        cpu.name = "Mock CPU";
        cpu.api_support = static_cast<uint32_t>(DeviceInfo::APISupport::CPU);
        devs.push_back(cpu);

        DeviceInfo vk_gpu;
        vk_gpu.name = "Mock Vulkan GPU";
        vk_gpu.vendor = DeviceInfo::Vendor::AMD;
        vk_gpu.api_support = static_cast<uint32_t>(DeviceInfo::APISupport::Vulkan);
        vk_gpu.vulkan_ok = true;
        vk_gpu.vram_bytes = 8ULL * 1024 * 1024 * 1024;
        devs.push_back(vk_gpu);

        hw.set_mock_environment(devs, 16ULL * 1024 * 1024 * 1024, true, false);

        std::string bid, reason;
        // HydroXL on Q8_0 -> Vulkan
        AGR_CHECK(WorkloadPartitioner::selectHardwareBackendForEngine(
            EngineId::HYDROXL, prof_q8, hw, "", false, bid, reason) == HardwareBackend::VULKAN);
        AGR_CHECK(bid == "hydroxl_vulkan");

        // HydroXL on Q4_K -> CPU (honest quant capability check)
        AGR_CHECK(WorkloadPartitioner::selectHardwareBackendForEngine(
            EngineId::HYDROXL, prof_q4k, hw, "", false, bid, reason) == HardwareBackend::CPU);
        AGR_CHECK(bid == "hydroxl_cpu");

        // llama.cpp when GPU is free -> Vulkan
        AGR_CHECK(WorkloadPartitioner::selectHardwareBackendForEngine(
            EngineId::LLAMA_CPP, prof_q4k, hw, "", false, bid, reason) == HardwareBackend::VULKAN);
        AGR_CHECK(bid == "llamacpp_vulkan");

        // llama.cpp when GPU is ALREADY claimed in the same wave -> CPU (contention avoidance!)
        AGR_CHECK(WorkloadPartitioner::selectHardwareBackendForEngine(
            EngineId::LLAMA_CPP, prof_q8, hw, "", true, bid, reason) == HardwareBackend::CPU);
        AGR_CHECK(bid == "llamacpp_cpu");
        AGR_CHECK(reason.find("contention") != std::string::npos);

        // Colibri even when Vulkan is preferred -> CPU (honest capability check)
        AGR_CHECK(WorkloadPartitioner::selectHardwareBackendForEngine(
            EngineId::COLIBRI, prof_q8, hw, "vulkan", false, bid, reason) == HardwareBackend::CPU);
        AGR_CHECK(bid == "colibri_cpu");
        std::cout << "  [OK] Vulkan hardware matrix & cross-engine GPU contention avoidance verified\n";
    }

    // -----------------------------------------------------------------------
    // 3. CUDA Preference & Honest Capability Negotiation
    // -----------------------------------------------------------------------
    {
        std::vector<DeviceInfo> devs;
        DeviceInfo cpu;
        cpu.name = "Mock CPU";
        cpu.api_support = static_cast<uint32_t>(DeviceInfo::APISupport::CPU);
        devs.push_back(cpu);

        DeviceInfo cuda_gpu;
        cuda_gpu.name = "Mock NVIDIA GPU";
        cuda_gpu.vendor = DeviceInfo::Vendor::NVIDIA;
        cuda_gpu.api_support = static_cast<uint32_t>(DeviceInfo::APISupport::CUDA);
        cuda_gpu.cuda_ok = true;
        cuda_gpu.vram_bytes = 12ULL * 1024 * 1024 * 1024;
        devs.push_back(cuda_gpu);

        hw.set_mock_environment(devs, 32ULL * 1024 * 1024 * 1024, false, true);

        std::string bid, reason;
        // HydroXL never claims CUDA
        AGR_CHECK(WorkloadPartitioner::selectHardwareBackendForEngine(
            EngineId::HYDROXL, prof_q8, hw, "cuda", false, bid, reason) == HardwareBackend::CPU);
        AGR_CHECK(bid == "hydroxl_cpu");
        AGR_CHECK(reason.find("does not implement CUDA") != std::string::npos);

        // Colibri never claims CUDA
        AGR_CHECK(WorkloadPartitioner::selectHardwareBackendForEngine(
            EngineId::COLIBRI, prof_q8, hw, "cuda", false, bid, reason) == HardwareBackend::CPU);
        AGR_CHECK(bid == "colibri_cpu");

        // llama.cpp selects CUDA only if HETEROACCEL_ENABLE_CUDA is compiled in
        HardwareBackend llama_hb = WorkloadPartitioner::selectHardwareBackendForEngine(
            EngineId::LLAMA_CPP, prof_q8, hw, "cuda", false, bid, reason);
        if (HardwareDetector::cuda_inference_compiled()) {
            AGR_CHECK(llama_hb == HardwareBackend::CUDA);
            AGR_CHECK(bid == "cuda");
        } else {
            AGR_CHECK(llama_hb == HardwareBackend::CPU);
            AGR_CHECK(bid == "llamacpp_cpu");
        }
        std::cout << "  [OK] CUDA capability negotiation & honest fallback verified\n";
    }

    // Restore real hardware detection state
    hw.detect();
    AGR_TEST_MAIN_END();
}
