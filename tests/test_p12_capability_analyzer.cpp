#include "analysis/CapabilityAnalyzer.h"
#include "backend/BackendManager.h"
#include "hardware/HardwareDetector.h"
#include "gpu/VulkanBackend.h"
#include <cassert>
#include <iostream>

using namespace agr;

int main() {
    // Setup dummy environment
    HardwareInfo hw = HardwareDetector::detectAll(); // Real hardware detector for the test
    
    VulkanBackend vulkan_backend;
    BackendManager backend_mgr(vulkan_backend);
    backend_mgr.discover();
    
    ExecutionEngineRegistry& registry = ExecutionEngineRegistry::instance();
    
    // Create requirements for a small model
    ModelRequirements req;
    req.format = ModelFormat::GGUF;
    req.required_capabilities.add(EngineCapability::TEXT_GENERATION);
    req.estimated_memory_bytes = 4ULL * 1024 * 1024 * 1024; // 4GB
    
    CapabilityAnalyzer analyzer;
    CapabilityReport report = analyzer.analyze(req, registry, &backend_mgr, hw);
    
    assert(report.model.format == ModelFormat::GGUF);
    
    // Check engines
    bool found_llama = false;
    for (const auto& eng : report.engines) {
        if (eng.engine_name == "llama.cpp") {
            found_llama = true;
            assert(eng.supported == true);
        }
    }
    assert(found_llama);
    
    // Check devices (CPU should be available)
    bool found_cpu = false;
    for (const auto& dev : report.devices) {
        if (dev.backend_type == "CPU") {
            found_cpu = true;
            assert(dev.available == true);
            assert(dev.supported == true);
            assert(dev.memory_capacity >= (hw.memory.available_physical_mb * 1024ULL * 1024ULL));
        }
    }
    assert(found_cpu);
    
    // Check feasibility
    assert(report.resources.full_residency_feasible == true);
    assert(report.resources.streaming_potentially_feasible == true);
    
    // Test memory bottleneck
    ModelRequirements giant_req = req;
    giant_req.estimated_memory_bytes = 200ULL * 1024 * 1024 * 1024; // 200 GB
    
    CapabilityReport giant_report = analyzer.analyze(giant_req, registry, &backend_mgr, hw);
    assert(giant_report.resources.full_residency_feasible == false);
    assert(!giant_report.resources.memory_bottleneck_reason.empty());
    
    std::cout << "test_p12_capability_analyzer passed\n";
    return 0;
}
