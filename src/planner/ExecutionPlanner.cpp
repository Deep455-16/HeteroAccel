#include "planner/ExecutionPlanner.h"

#include <algorithm>
#include <sstream>

namespace agr {

namespace {

constexpr uint64_t kPartialResidentPercent = 15;

const DeviceCompatibility* findDevice(const CapabilityReport& report,
                                      const char* backend_type) {
    for (const auto& device : report.devices) {
        if (device.backend_type == backend_type) return &device;
    }
    return nullptr;
}

bool deviceUsable(const DeviceCompatibility* device) {
    return device && device->available && device->supported;
}

bool coversRequired(const EngineCapabilitySet& caps, const ModelRequirements& req) {
    const EngineCapability needed[] = {
        EngineCapability::TEXT_GENERATION,
        EngineCapability::EMBEDDINGS,
        EngineCapability::VISION,
        EngineCapability::AUDIO,
        EngineCapability::MULTIMODAL
    };
    for (EngineCapability bit : needed) {
        if (req.required_capabilities.has(bit) && !caps.has(bit)) return false;
    }
    return true;
}

bool hasBackend(const EngineCapabilitySet& caps, ExecutionMode mode) {
    switch (mode) {
        case ExecutionMode::CPU:    return caps.has(EngineCapability::BACKEND_CPU);
        case ExecutionMode::VULKAN: return caps.has(EngineCapability::BACKEND_VULKAN);
        case ExecutionMode::CUDA:   return caps.has(EngineCapability::BACKEND_CUDA);
        case ExecutionMode::HYBRID:
            return caps.has(EngineCapability::BACKEND_CPU) &&
                   (caps.has(EngineCapability::BACKEND_VULKAN) ||
                    caps.has(EngineCapability::BACKEND_CUDA));
        default:
            return false;
    }
}

struct Candidate {
    std::string key;
    EngineCapabilitySet caps;
    bool eligible = false;
    std::string rejection;
};

ModelExecutionPlan unsupportedPlan(PlanReason reason, const std::string& explanation) {
    ModelExecutionPlan plan;
    plan.viable = false;
    plan.executable = false;
    plan.mode = ExecutionMode::UNSUPPORTED;
    plan.residency = ExecutionStrategy::AUTO;
    plan.reasons.push_back(reason);
    plan.explanation = explanation;
    return plan;
}

void addReason(ModelExecutionPlan& plan, PlanReason reason) {
    if (!plan.hasReason(reason)) plan.reasons.push_back(reason);
}

const Candidate* selectEngine(const std::vector<Candidate>& candidates,
                              ExecutionMode mode) {
    const Candidate* best = nullptr;
    for (const auto& candidate : candidates) {
        if (!candidate.eligible) continue;
        if (!hasBackend(candidate.caps, mode)) continue;
        if (!best || candidate.key < best->key) best = &candidate;
    }
    return best;
}

uint32_t partialLayers(const ModelRequirements& req,
                       uint64_t accelerator_bytes,
                       uint64_t model_bytes) {
    if (!req.layer_count || *req.layer_count == 0 || model_bytes == 0) return 0;
    uint64_t layers = static_cast<uint64_t>(*req.layer_count) * accelerator_bytes / model_bytes;
    if (layers == 0) layers = 1;
    if (layers >= *req.layer_count) layers = static_cast<uint64_t>(*req.layer_count) - 1;
    if (layers == 0) layers = 1;
    return static_cast<uint32_t>(layers);
}

} // namespace

bool ModelExecutionPlan::hasReason(PlanReason reason) const {
    return std::find(reasons.begin(), reasons.end(), reason) != reasons.end();
}

std::string ModelExecutionPlan::formatToString() const {
    std::ostringstream out;
    out << "Execution Plan\n";
    out << "==============\n\n";
    out << "Engine:     " << (engine_key.empty() ? "(none)" : engine_key) << "\n";
    out << "Mode:       " << toString(mode) << "\n";
    out << "Residency:  " << toString(residency) << "\n";
    out << "Viable:     " << (viable ? "yes" : "no") << "\n";
    out << "Executable: " << (executable ? "yes" : "no") << "\n";
    out << "Memory budget: " << (memory_budget_bytes / (1024ULL * 1024ULL)) << " MB\n";
    out << "Model memory:  " << (estimated_model_memory_bytes / (1024ULL * 1024ULL)) << " MB\n";
    if (gpu_layer_budget) {
        out << "GPU layers: " << *gpu_layer_budget << "\n";
    }
    if (context_limit) {
        out << "Context limit: " << *context_limit << "\n";
    }
    out << "Fallback:   " << (fallback == ExecutionStrategy::AUTO ? "none" : toString(fallback)) << "\n";
    out << "Streaming required: " << (streaming_required ? "yes" : "no") << "\n";
    out << "Confidence: " << confidence << "\n";

    out << "\nReasons:\n";
    if (reasons.empty()) {
        out << "  (none)\n";
    } else {
        for (PlanReason reason : reasons) {
            out << "  " << toString(reason) << "\n";
        }
    }

    out << "\nExplanation:\n  " << explanation << "\n";

    out << "\nWarnings:\n";
    if (warnings.empty()) {
        out << "  (none)\n";
    } else {
        for (const auto& warning : warnings) {
            out << "  " << warning << "\n";
        }
    }

    if (!rejected_engines.empty()) {
        out << "\nRejected engines:\n";
        for (const auto& rejected : rejected_engines) {
            out << "  " << rejected << "\n";
        }
    }
    return out.str();
}

ModelExecutionPlan ExecutionPlanner::plan(const ModelRequirements& requirements,
                                     const CapabilityReport& report,
                                     const ExecutionEngineRegistry& registry,
                                     const PlannerConfig& config) const {
    if (!config.allow_accelerator &&
        !config.allow_partial_residency &&
        !config.allow_streaming_plan &&
        !config.allow_cpu_fallback) {
        return unsupportedPlan(
            PlanReason::INVALID_CONFIGURATION,
            "Planner configuration disables every execution strategy.");
    }

    if (requirements.format == ModelFormat::UNKNOWN) {
        ModelExecutionPlan plan = unsupportedPlan(
            PlanReason::UNSUPPORTED_MODEL,
            "Model format is missing or unsupported, so no engine can be selected.");
        plan.warnings.push_back("Missing model metadata: format is UNKNOWN.");
        return plan;
    }

    const bool memory_known = requirements.estimated_memory_bytes.has_value() &&
                              *requirements.estimated_memory_bytes > 0;
    const uint64_t model_bytes = memory_known ? *requirements.estimated_memory_bytes : 0;

    auto keys = registry.registeredKeys();
    std::sort(keys.begin(), keys.end());
    if (keys.empty()) {
        ModelExecutionPlan plan = unsupportedPlan(
            PlanReason::NO_REGISTERED_ENGINES,
            "No execution engines are registered.");
        plan.estimated_model_memory_bytes = model_bytes;
        return plan;
    }

    std::vector<Candidate> candidates;
    candidates.reserve(keys.size());
    for (const auto& key : keys) {
        Candidate candidate;
        candidate.key = key;
        auto engine = registry.create(key);
        if (!engine) {
            candidate.rejection = key + ": factory returned no engine";
            candidates.push_back(std::move(candidate));
            continue;
        }
        candidate.caps = engine->capabilities();

        const EngineCompatibility* listed = nullptr;
        for (const auto& entry : report.engines) {
            if (entry.engine_name == key) {
                listed = &entry;
                break;
            }
        }
        if (!report.engines.empty() && !listed) {
            candidate.rejection = key + ": absent from capability report";
        } else if (listed && !listed->supported) {
            candidate.rejection = key + ": " +
                (listed->reason.empty() ? "capability report rejected the engine" : listed->reason);
        } else if (!coversRequired(candidate.caps, requirements)) {
            candidate.rejection = key + ": missing a required engine capability";
        } else {
            candidate.eligible = true;
        }
        candidates.push_back(std::move(candidate));
    }

    auto failWithRejections = [&](PlanReason reason, const std::string& explanation) {
        ModelExecutionPlan plan = unsupportedPlan(reason, explanation);
        plan.estimated_model_memory_bytes = model_bytes;
        for (const auto& candidate : candidates) {
            if (!candidate.eligible) plan.rejected_engines.push_back(candidate.rejection);
        }
        if (!plan.rejected_engines.empty()) addReason(plan, PlanReason::ENGINE_REJECTED);
        return plan;
    };

    bool any_eligible = false;
    for (const auto& candidate : candidates) {
        if (candidate.eligible) any_eligible = true;
    }
    if (!any_eligible) {
        return failWithRejections(
            PlanReason::ENGINE_UNAVAILABLE,
            "Every registered engine was rejected for this workload.");
    }

    const DeviceCompatibility* cpu = findDevice(report, "CPU");
    const DeviceCompatibility* vulkan = findDevice(report, "Vulkan");
    const DeviceCompatibility* cuda = findDevice(report, "CUDA");

    auto fits = [&](const DeviceCompatibility* device) {
        return memory_known && deviceUsable(device) && device->memory_capacity >= model_bytes;
    };
    auto partial = [&](const DeviceCompatibility* device) {
        if (!config.allow_partial_residency || !memory_known || !deviceUsable(device)) return false;
        if (device->memory_capacity == 0 || device->memory_capacity >= model_bytes) return false;
        return device->memory_capacity * 100 >= model_bytes * kPartialResidentPercent;
    };

    ModelExecutionPlan plan;
    plan.estimated_model_memory_bytes = model_bytes;

    if (requirements.quantization == QuantizationType::UNKNOWN) {
        addReason(plan, PlanReason::QUANTIZATION_UNKNOWN);
        plan.warnings.push_back("Quantization is unknown. The plan does not assume a specific quant format.");
    }

    if (report.resources.full_residency_feasible) {
        const bool any_fit = fits(cpu) || fits(vulkan) || fits(cuda);
        if (!any_fit) {
            addReason(plan, PlanReason::CONTRADICTORY_CAPABILITY);
            plan.warnings.push_back(
                "Capability report says full residency is feasible, but no usable device can hold the model. Device memory is authoritative.");
        }
    }

    auto finishContext = [&]() {
        if (requirements.context_length) plan.context_limit = *requirements.context_length;
        if (config.context_ceiling > 0) {
            if (!plan.context_limit || *plan.context_limit > config.context_ceiling) {
                plan.context_limit = config.context_ceiling;
                plan.warnings.push_back("Context length was clamped to the planner configuration ceiling.");
            }
        }
    };

    auto noteRejections = [&]() {
        for (const auto& candidate : candidates) {
            if (!candidate.eligible) plan.rejected_engines.push_back(candidate.rejection);
        }
    };

    auto cpuFallbackAvailable = [&]() {
        return config.allow_cpu_fallback && fits(cpu) && selectEngine(candidates, ExecutionMode::CPU) != nullptr;
    };

    if (!memory_known) {
        plan.viable = false;
        plan.executable = false;
        plan.mode = ExecutionMode::UNSUPPORTED;
        addReason(plan, PlanReason::UNKNOWN_MEMORY);
        plan.explanation =
            "Model memory requirement is zero or unknown. The planner will not claim accelerator residency or CPU execution without a size.";
        plan.warnings.push_back("Missing or zero model memory requirement.");
        noteRejections();
        finishContext();
        return plan;
    }

    const bool streaming_possible = config.allow_streaming_plan &&
                                    report.resources.streaming_potentially_feasible;

    struct Choice {
        ExecutionMode mode = ExecutionMode::UNSUPPORTED;
        ExecutionStrategy residency = ExecutionStrategy::AUTO;
        const DeviceCompatibility* device = nullptr;
        bool executable = false;
        int confidence = 0;
    };

    Choice choice;
    bool chosen = false;

    auto tryFull = [&](ExecutionMode mode, const DeviceCompatibility* device) {
        if (chosen || !config.allow_accelerator || !fits(device)) return;
        if (!selectEngine(candidates, mode)) return;
        choice.mode = mode;
        choice.residency = ExecutionStrategy::FULL_RESIDENT;
        choice.device = device;
        choice.executable = true;
        choice.confidence = 90;
        chosen = true;
    };
    tryFull(ExecutionMode::CUDA, cuda);
    tryFull(ExecutionMode::VULKAN, vulkan);

    auto tryPartial = [&](ExecutionMode accel_mode, const DeviceCompatibility* device) {
        if (chosen || !config.allow_accelerator || !partial(device)) return;
        if (!selectEngine(candidates, accel_mode)) return;
        choice.mode = ExecutionMode::HYBRID;
        choice.residency = ExecutionStrategy::PARTIAL_RESIDENT;
        choice.device = device;
        choice.executable = true;
        choice.confidence = 70;
        chosen = true;
    };
    tryPartial(ExecutionMode::CUDA, cuda);
    tryPartial(ExecutionMode::VULKAN, vulkan);

    if (!chosen && streaming_possible && config.allow_accelerator &&
        (deviceUsable(vulkan) || deviceUsable(cuda))) {
        const Candidate* engine = selectEngine(candidates, ExecutionMode::HYBRID);
        if (!engine) engine = selectEngine(candidates, ExecutionMode::VULKAN);
        if (!engine) engine = selectEngine(candidates, ExecutionMode::CUDA);
        if (!engine) engine = selectEngine(candidates, ExecutionMode::CPU);
        if (engine) {
            choice.mode = ExecutionMode::HYBRID;
            choice.residency = ExecutionStrategy::STREAMING;
            choice.device = deviceUsable(vulkan) ? vulkan : (deviceUsable(cuda) ? cuda : cpu);
            choice.executable = false;
            choice.confidence = 40;
            chosen = true;
        }
    }

    if (!chosen && cpuFallbackAvailable()) {
        choice.mode = ExecutionMode::CPU;
        choice.residency = ExecutionStrategy::CPU_FALLBACK;
        choice.device = cpu;
        choice.executable = true;
        choice.confidence = 80;
        chosen = true;
    }

    noteRejections();
    finishContext();

    if (!chosen) {
        plan.viable = false;
        plan.executable = false;
        plan.mode = ExecutionMode::UNSUPPORTED;
        if (!fits(cpu) && deviceUsable(cpu)) addReason(plan, PlanReason::INSUFFICIENT_RAM);
        if (!deviceUsable(cpu) && !deviceUsable(vulkan) && !deviceUsable(cuda)) {
            addReason(plan, PlanReason::ENGINE_UNAVAILABLE);
        }
        if (plan.reasons.empty() || plan.reasons.front() == PlanReason::QUANTIZATION_UNKNOWN ||
            plan.reasons.front() == PlanReason::CONTRADICTORY_CAPABILITY) {
            addReason(plan, PlanReason::ENGINE_UNAVAILABLE);
        }
        plan.explanation = "No registered engine can execute this model on the reported devices.";
        plan.confidence = 0;
        return plan;
    }

    const ExecutionMode engine_mode =
        choice.residency == ExecutionStrategy::PARTIAL_RESIDENT ||
        choice.residency == ExecutionStrategy::STREAMING
            ? (choice.device == cuda ? ExecutionMode::CUDA :
               choice.device == vulkan ? ExecutionMode::VULKAN : ExecutionMode::CPU)
            : choice.mode;
    const Candidate* engine = selectEngine(candidates, engine_mode);
    if (!engine && choice.mode == ExecutionMode::HYBRID) {
        engine = selectEngine(candidates, ExecutionMode::HYBRID);
    }
    if (!engine) {
        return failWithRejections(
            PlanReason::ENGINE_UNAVAILABLE,
            "A placement was possible from device memory, but no eligible engine advertised that backend.");
    }

    plan.viable = true;
    plan.executable = choice.executable;
    plan.engine_key = engine->key;
    plan.mode = choice.mode;
    plan.residency = choice.residency;
    plan.confidence = choice.confidence;
    plan.memory_budget_bytes = choice.device ? choice.device->memory_capacity : 0;
    plan.streaming_required = choice.residency == ExecutionStrategy::STREAMING;

    if (choice.residency == ExecutionStrategy::FULL_RESIDENT) {
        if (choice.mode == ExecutionMode::CUDA) addReason(plan, PlanReason::CUDA_FEASIBLE);
        if (choice.mode == ExecutionMode::VULKAN) addReason(plan, PlanReason::VULKAN_FEASIBLE);
        addReason(plan, PlanReason::ACCELERATOR_MEMORY_SUFFICIENT);
        if (requirements.layer_count) plan.gpu_layer_budget = *requirements.layer_count;
        if (cpuFallbackAvailable()) plan.fallback = ExecutionStrategy::CPU_FALLBACK;
        std::ostringstream text;
        text << toString(choice.mode)
             << " full residency was selected because the model fits in the reported accelerator memory and engine '"
             << engine->key << "' advertises that backend.";
        plan.explanation = text.str();
    } else if (choice.residency == ExecutionStrategy::PARTIAL_RESIDENT) {
        addReason(plan, PlanReason::ACCELERATOR_MEMORY_INSUFFICIENT);
        addReason(plan, PlanReason::PARTIAL_RESIDENCY);
        if (choice.device == cuda) addReason(plan, PlanReason::CUDA_FEASIBLE);
        if (choice.device == vulkan) addReason(plan, PlanReason::VULKAN_FEASIBLE);
        uint32_t layers = partialLayers(requirements, plan.memory_budget_bytes, model_bytes);
        if (layers > 0) plan.gpu_layer_budget = layers;
        if (cpuFallbackAvailable()) plan.fallback = ExecutionStrategy::CPU_FALLBACK;
        std::ostringstream text;
        text << "Partial residency was selected because the model exceeds accelerator memory but at least "
             << kPartialResidentPercent
             << "% of it fits. Engine '" << engine->key
             << "' can offload a layer budget. This is existing layer offload, not Phase 14 tensor streaming.";
        plan.explanation = text.str();
    } else if (choice.residency == ExecutionStrategy::STREAMING) {
        addReason(plan, PlanReason::ACCELERATOR_MEMORY_INSUFFICIENT);
        addReason(plan, PlanReason::STREAMING_REQUIRED);
        addReason(plan, PlanReason::FUTURE_CAPABILITY);
        plan.executable = false;
        if (cpuFallbackAvailable()) plan.fallback = ExecutionStrategy::CPU_FALLBACK;
        plan.explanation =
            "Streaming plan requires Phase 14 execution support. The model does not fit a resident or partial accelerator placement, and the capability report says streaming may eventually be feasible. This plan is not executable.";
        plan.warnings.push_back("STREAMING is a plan only. Tensor streaming is not implemented.");
    } else if (choice.residency == ExecutionStrategy::CPU_FALLBACK) {
        addReason(plan, PlanReason::CPU_FALLBACK);
        plan.gpu_layer_budget = 0;
        if (!deviceUsable(vulkan)) {
            plan.warnings.push_back("Vulkan is not a usable device in the capability report.");
        }
        if (!deviceUsable(cuda)) {
            plan.warnings.push_back("CUDA is not a usable device in the capability report.");
        }
        std::ostringstream text;
        text << "CPU fallback was selected because no accelerator placement is currently possible and engine '"
             << engine->key << "' can execute on the reported CPU memory.";
        plan.explanation = text.str();
    }

    if (deviceUsable(cuda) == false && engine->caps.has(EngineCapability::BACKEND_CUDA)) {
        plan.warnings.push_back("The selected engine advertises CUDA, but CUDA was not selected because the capability report says it is unavailable.");
    }

    return plan;
}

} // namespace agr
