#include "analysis/CapabilityAnalyzer.h"
#include <algorithm>

namespace agr {

CapabilityReport CapabilityAnalyzer::analyze(const ModelRequirements& req,
                                             const ExecutionEngineRegistry& registry,
                                             BackendManager* backend_mgr,
                                             const HardwareInfo& hardware) {
    CapabilityReport report;
    report.model = req;
    
    // 1. Engine Compatibility Analysis
    auto keys = registry.registeredKeys();
    for (const auto& key : keys) {
        EngineCompatibility ec;
        ec.engine_name = key;
        
        auto engine = registry.create(key);
        if (engine) {
            bool supports_format = false;
            if (key == "llama.cpp" && req.format == ModelFormat::GGUF) {
                supports_format = true;
            } else if (req.format == ModelFormat::GGUF) {
                supports_format = false; 
            } else if (req.format == ModelFormat::UNKNOWN) {
                supports_format = false;
            }
            
            bool capabilities_match = true;
            EngineCapabilitySet ecaps = engine->capabilities();
            
            if (req.required_capabilities.has(EngineCapability::TEXT_GENERATION) && 
                !ecaps.has(EngineCapability::TEXT_GENERATION)) {
                capabilities_match = false;
            }
            if (req.required_capabilities.has(EngineCapability::VISION) && 
                !ecaps.has(EngineCapability::VISION)) {
                capabilities_match = false;
            }
            
            if (!supports_format) {
                ec.supported = false;
                ec.reason = "Engine does not support the model format.";
            } else if (!capabilities_match) {
                ec.supported = false;
                ec.reason = "Engine lacks required capabilities for this model.";
            } else {
                ec.supported = true;
            }
        } else {
            ec.supported = false;
            ec.reason = "Failed to instantiate engine for capability check.";
        }
        
        report.engines.push_back(ec);
    }
    
    // 2. Hardware / Device Compatibility Analysis
    if (backend_mgr) {
        // CPU
        DeviceCompatibility dev_cpu;
        dev_cpu.backend_type = "CPU";
        dev_cpu.device_name = hardware.cpu.model_name;
        const auto* cpu_backend = backend_mgr->getDevice(ComputeBackend::CPU);
        if (cpu_backend && cpu_backend->is_available) {
            dev_cpu.available = true;
            dev_cpu.supported = true;
            // Use available physical RAM for estimation
            dev_cpu.memory_capacity = hardware.memory.available_physical_mb * 1024ULL * 1024ULL;
        } else {
            dev_cpu.available = false;
            dev_cpu.reason = "CPU backend not initialized.";
        }
        report.devices.push_back(dev_cpu);
        
        // Vulkan
        DeviceCompatibility dev_vk;
        dev_vk.backend_type = "Vulkan";
        const auto* vk_backend = backend_mgr->getDevice(ComputeBackend::VULKAN);
        if (vk_backend && vk_backend->is_available) {
            dev_vk.device_name = vk_backend->name;
            dev_vk.available = true;
            dev_vk.supported = true;
            dev_vk.memory_capacity = vk_backend->memory_capacity;
        } else {
            dev_vk.device_name = "Unknown";
            dev_vk.available = false;
            dev_vk.reason = "Vulkan backend is unavailable on this machine.";
        }
        report.devices.push_back(dev_vk);
        
        // CUDA
        DeviceCompatibility dev_cu;
        dev_cu.backend_type = "CUDA";
        const auto* cu_backend = backend_mgr->getDevice(ComputeBackend::CUDA);
        if (cu_backend && cu_backend->is_available) {
            dev_cu.device_name = cu_backend->name;
            dev_cu.available = true;
            dev_cu.supported = true;
            dev_cu.memory_capacity = cu_backend->memory_capacity;
        } else {
            dev_cu.device_name = "Unknown";
            dev_cu.available = false;
            dev_cu.reason = "CUDA backend is unavailable on this machine.";
        }
        report.devices.push_back(dev_cu);
    }
    
    // 3. Resource Feasibility
    report.resources.full_residency_feasible = false;
    report.resources.streaming_potentially_feasible = false;
    
    if (req.estimated_memory_bytes) {
        uint64_t req_mem = *req.estimated_memory_bytes;
        uint64_t max_device_mem = 0;
        for (const auto& dev : report.devices) {
            if (dev.available && dev.supported) {
                if (dev.memory_capacity > max_device_mem) {
                    max_device_mem = dev.memory_capacity;
                }
            }
        }
        
        if (max_device_mem >= req_mem) {
            report.resources.full_residency_feasible = true;
        } else {
            report.resources.full_residency_feasible = false;
            report.resources.memory_bottleneck_reason = "Estimated required memory exceeds maximum available device memory.";
        }
        
        // Streaming feasibility heuristic: if there's enough memory for context + 1 layer, it's potentially feasible.
        // For Phase 12, just mark it potentially feasible if we have at least 1GB of memory.
        if (max_device_mem >= 1024ULL * 1024ULL * 1024ULL) {
            report.resources.streaming_potentially_feasible = true;
        } else {
            report.resources.streaming_potentially_feasible = false;
            report.resources.memory_bottleneck_reason = "Insufficient memory for even streaming configuration.";
        }
    } else {
        report.resources.memory_bottleneck_reason = "Cannot determine feasibility because model memory requirement is unknown.";
    }
    
    return report;
}

} // namespace agr
