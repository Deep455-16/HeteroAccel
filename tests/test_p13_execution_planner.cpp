#include "planner/ExecutionPlanner.h"
#include "engine/ExecutionEngineRegistry.h"
#include "mini_test.h"

#include <string>

using namespace agr;

namespace {

constexpr uint64_t MB = 1024ULL * 1024ULL;
constexpr uint64_t GB = 1024ULL * MB;

class FakeEngine : public IExecutionEngine {
public:
    FakeEngine(std::string key, EngineCapabilitySet caps)
        : key_(std::move(key)), caps_(caps) {}

    EngineIdentity identity() const override {
        EngineIdentity id;
        id.name = key_;
        id.registry_key = key_;
        id.version = "test";
        id.type = "LLM";
        return id;
    }
    EngineCapabilitySet capabilities() const override { return caps_; }
    bool initialize() override { return true; }
    void shutdown() override {}
    bool isInitialized() const override { return true; }
    bool loadModel(const ModelDescriptor&) override { return false; }
    void unloadModel() override {}
    bool isModelLoaded() const override { return false; }
    ModelDescriptor currentModel() const override { return {}; }
    ExecutionResult execute(const ExecutionContext&) override { return {}; }
    ExecutionResult executeStreaming(const ExecutionContext&, TokenCallback) override { return {}; }
    EngineError lastEngineError() const override { return {}; }

private:
    std::string key_;
    EngineCapabilitySet caps_;
};

void resetRegistry() {
    ExecutionEngineRegistry::instance().clear();
}

void addEngine(const std::string& key, EngineCapabilitySet caps) {
    ExecutionEngineRegistry::instance().registerFactory(
        key, key, [key, caps]() { return std::make_shared<FakeEngine>(key, caps); });
}

EngineCapabilitySet llamaLikeCaps() {
    return EngineCapabilitySet{
        EngineCapability::TEXT_GENERATION,
        EngineCapability::BACKEND_CPU,
        EngineCapability::BACKEND_VULKAN,
        EngineCapability::BACKEND_CUDA,
        EngineCapability::STREAMING,
        EngineCapability::CANCELLATION,
        EngineCapability::DYNAMIC_LOADING
    };
}

DeviceCompatibility device(const char* type, bool available, uint64_t bytes) {
    DeviceCompatibility dev;
    dev.backend_type = type;
    dev.device_name = type;
    dev.available = available;
    dev.supported = available;
    dev.memory_capacity = available ? bytes : 0;
    if (!available) dev.reason = std::string(type) + " unavailable";
    return dev;
}

ModelRequirements model(uint64_t bytes) {
    ModelRequirements req;
    req.format = ModelFormat::GGUF;
    req.quantization = QuantizationType::Q4;
    req.modality = ModelModality::TEXT;
    req.required_capabilities.add(EngineCapability::TEXT_GENERATION);
    req.estimated_memory_bytes = bytes;
    req.layer_count = 32;
    req.context_length = 4096;
    req.architecture = "test";
    return req;
}

CapabilityReport reportFor(const ModelRequirements& req,
                           std::initializer_list<DeviceCompatibility> devices,
                           bool streaming = false) {
    CapabilityReport report;
    report.model = req;
    report.resources.streaming_potentially_feasible = streaming;
    bool full = false;
    for (const auto& dev : devices) {
        report.devices.push_back(dev);
        if (dev.available && req.estimated_memory_bytes &&
            dev.memory_capacity >= *req.estimated_memory_bytes) {
            full = true;
        }
    }
    report.resources.full_residency_feasible = full;
    for (const auto& key : ExecutionEngineRegistry::instance().registeredKeys()) {
        EngineCompatibility ec;
        ec.engine_name = key;
        ec.supported = true;
        report.engines.push_back(ec);
    }
    return report;
}

ExecutionPlanner planner;

} // namespace

int main() {
    // Test 1 — small model, sufficient Vulkan memory.
    {
        resetRegistry();
        addEngine("llama.cpp", llamaLikeCaps());
        ModelRequirements req = model(1 * GB);
        CapabilityReport report = reportFor(req, {
            device("CPU", true, 16 * GB),
            device("Vulkan", true, 8 * GB),
            device("CUDA", false, 0)
        });
        ModelExecutionPlan plan = planner.plan(req, report, ExecutionEngineRegistry::instance());
        AGR_CHECK(plan.viable);
        AGR_CHECK(plan.executable);
        AGR_CHECK(plan.mode == ExecutionMode::VULKAN);
        AGR_CHECK(plan.residency == ExecutionStrategy::FULL_RESIDENT);
        AGR_CHECK(plan.engine_key == "llama.cpp");
        AGR_CHECK(plan.hasReason(PlanReason::VULKAN_FEASIBLE));
        AGR_CHECK(plan.hasReason(PlanReason::ACCELERATOR_MEMORY_SUFFICIENT));
        AGR_CHECK(!plan.explanation.empty());

        // Test 7 — determinism. Test 8 — explainability.
        ModelExecutionPlan again = planner.plan(req, report, ExecutionEngineRegistry::instance());
        AGR_CHECK(plan.formatToString() == again.formatToString());
        AGR_CHECK(!plan.reasons.empty());
        AGR_CHECK(plan.confidence == again.confidence);
    }

    // Test 2 — model larger than accelerator memory, partial residency fits.
    {
        resetRegistry();
        addEngine("llama.cpp", llamaLikeCaps());
        ModelRequirements req = model(8 * GB);
        CapabilityReport report = reportFor(req, {
            device("CPU", true, 16 * GB),
            device("Vulkan", true, 4 * GB),
            device("CUDA", false, 0)
        }, true);
        ModelExecutionPlan plan = planner.plan(req, report, ExecutionEngineRegistry::instance());
        AGR_CHECK(plan.viable);
        AGR_CHECK(plan.executable);
        AGR_CHECK(plan.residency == ExecutionStrategy::PARTIAL_RESIDENT);
        AGR_CHECK(plan.mode == ExecutionMode::HYBRID);
        AGR_CHECK(plan.streaming_required == false);
        AGR_CHECK(plan.hasReason(PlanReason::ACCELERATOR_MEMORY_INSUFFICIENT));
        AGR_CHECK(plan.hasReason(PlanReason::PARTIAL_RESIDENCY));
        AGR_CHECK(plan.gpu_layer_budget.has_value());
        AGR_CHECK(*plan.gpu_layer_budget > 0);
        AGR_CHECK(*plan.gpu_layer_budget < 32);
    }

    // Streaming is planned only when partial residency cannot work, and it is not executable.
    {
        resetRegistry();
        addEngine("llama.cpp", llamaLikeCaps());
        ModelRequirements req = model(8 * GB);
        CapabilityReport report = reportFor(req, {
            device("CPU", true, 32 * GB),
            device("Vulkan", true, 64 * MB),
            device("CUDA", false, 0)
        }, true);
        ModelExecutionPlan plan = planner.plan(req, report, ExecutionEngineRegistry::instance());
        AGR_CHECK(plan.viable);
        AGR_CHECK(plan.executable == false);
        AGR_CHECK(plan.residency == ExecutionStrategy::STREAMING);
        AGR_CHECK(plan.streaming_required);
        AGR_CHECK(plan.hasReason(PlanReason::STREAMING_REQUIRED));
        AGR_CHECK(plan.hasReason(PlanReason::FUTURE_CAPABILITY));
        AGR_CHECK(plan.fallback == ExecutionStrategy::CPU_FALLBACK);
        AGR_CHECK(plan.explanation.find("Phase 14") != std::string::npos);
    }

    // Test 3 — Vulkan unavailable, CPU fallback.
    {
        resetRegistry();
        addEngine("llama.cpp", llamaLikeCaps());
        ModelRequirements req = model(1 * GB);
        CapabilityReport report = reportFor(req, {
            device("CPU", true, 16 * GB),
            device("Vulkan", false, 0),
            device("CUDA", false, 0)
        });
        ModelExecutionPlan plan = planner.plan(req, report, ExecutionEngineRegistry::instance());
        AGR_CHECK(plan.viable);
        AGR_CHECK(plan.executable);
        AGR_CHECK(plan.mode == ExecutionMode::CPU);
        AGR_CHECK(plan.residency == ExecutionStrategy::CPU_FALLBACK);
        AGR_CHECK(plan.hasReason(PlanReason::CPU_FALLBACK));
    }

    // Test 4 — CUDA advertised by the engine but unavailable on the device.
    {
        resetRegistry();
        addEngine("llama.cpp", llamaLikeCaps());
        ModelRequirements req = model(1 * GB);
        CapabilityReport report = reportFor(req, {
            device("CPU", true, 16 * GB),
            device("Vulkan", true, 8 * GB),
            device("CUDA", false, 0)
        });
        ModelExecutionPlan plan = planner.plan(req, report, ExecutionEngineRegistry::instance());
        AGR_CHECK(plan.mode != ExecutionMode::CUDA);
        AGR_CHECK(plan.mode == ExecutionMode::VULKAN);
        bool warned = false;
        for (const auto& warning : plan.warnings) {
            if (warning.find("CUDA") != std::string::npos) warned = true;
        }
        AGR_CHECK(warned);
    }

    // Test 5 — no registered engines.
    {
        resetRegistry();
        ModelRequirements req = model(1 * GB);
        CapabilityReport report;
        report.model = req;
        ModelExecutionPlan plan = planner.plan(req, report, ExecutionEngineRegistry::instance());
        AGR_CHECK(plan.viable == false);
        AGR_CHECK(plan.executable == false);
        AGR_CHECK(plan.mode == ExecutionMode::UNSUPPORTED);
        AGR_CHECK(plan.hasReason(PlanReason::NO_REGISTERED_ENGINES));
        AGR_CHECK(!plan.explanation.empty());
    }

    // Engine registered but incapable of the required workload.
    {
        resetRegistry();
        addEngine("embeddings-only", EngineCapabilitySet{EngineCapability::EMBEDDINGS, EngineCapability::BACKEND_CPU});
        ModelRequirements req = model(1 * GB);
        CapabilityReport report = reportFor(req, { device("CPU", true, 16 * GB) });
        ModelExecutionPlan plan = planner.plan(req, report, ExecutionEngineRegistry::instance());
        AGR_CHECK(plan.viable == false);
        AGR_CHECK(plan.hasReason(PlanReason::ENGINE_UNAVAILABLE));
        AGR_CHECK(!plan.rejected_engines.empty());
    }

    // Test 6 — capability filtering, not registration order.
    {
        resetRegistry();
        addEngine("zzz-vulkan", EngineCapabilitySet{
            EngineCapability::TEXT_GENERATION,
            EngineCapability::BACKEND_CPU,
            EngineCapability::BACKEND_VULKAN
        });
        addEngine("aaa-cpu-only", EngineCapabilitySet{
            EngineCapability::TEXT_GENERATION,
            EngineCapability::BACKEND_CPU
        });
        addEngine("mmm-vulkan", EngineCapabilitySet{
            EngineCapability::TEXT_GENERATION,
            EngineCapability::BACKEND_CPU,
            EngineCapability::BACKEND_VULKAN
        });
        ModelRequirements req = model(1 * GB);
        CapabilityReport report = reportFor(req, {
            device("CPU", true, 16 * GB),
            device("Vulkan", true, 8 * GB),
            device("CUDA", false, 0)
        });
        ModelExecutionPlan plan = planner.plan(req, report, ExecutionEngineRegistry::instance());
        AGR_CHECK(plan.engine_key == "mmm-vulkan");
        AGR_CHECK(plan.mode == ExecutionMode::VULKAN);
        AGR_CHECK(plan.residency == ExecutionStrategy::FULL_RESIDENT);
    }

    // Unknown memory, missing format, invalid config, insufficient RAM.
    {
        resetRegistry();
        addEngine("llama.cpp", llamaLikeCaps());
        ModelRequirements unknown = model(1 * GB);
        unknown.estimated_memory_bytes.reset();
        CapabilityReport report = reportFor(unknown, {
            device("CPU", true, 16 * GB),
            device("Vulkan", true, 8 * GB),
            device("CUDA", false, 0)
        });
        ModelExecutionPlan plan = planner.plan(unknown, report, ExecutionEngineRegistry::instance());
        AGR_CHECK(plan.viable == false);
        AGR_CHECK(plan.executable == false);
        AGR_CHECK(plan.mode != ExecutionMode::CUDA);
        AGR_CHECK(plan.mode != ExecutionMode::VULKAN);
        AGR_CHECK(plan.hasReason(PlanReason::UNKNOWN_MEMORY));

        ModelRequirements missing = model(1 * GB);
        missing.format = ModelFormat::UNKNOWN;
        ModelExecutionPlan bad_format = planner.plan(missing, report, ExecutionEngineRegistry::instance());
        AGR_CHECK(bad_format.viable == false);
        AGR_CHECK(bad_format.hasReason(PlanReason::UNSUPPORTED_MODEL));

        PlannerConfig invalid;
        invalid.allow_accelerator = false;
        invalid.allow_partial_residency = false;
        invalid.allow_streaming_plan = false;
        invalid.allow_cpu_fallback = false;
        ModelExecutionPlan bad_config = planner.plan(model(1 * GB), report, ExecutionEngineRegistry::instance(), invalid);
        AGR_CHECK(bad_config.hasReason(PlanReason::INVALID_CONFIGURATION));

        ModelRequirements huge = model(64 * GB);
        CapabilityReport tight = reportFor(huge, {
            device("CPU", true, 8 * GB),
            device("Vulkan", false, 0),
            device("CUDA", false, 0)
        }, false);
        tight.resources.full_residency_feasible = true;
        ModelExecutionPlan ram = planner.plan(huge, tight, ExecutionEngineRegistry::instance());
        AGR_CHECK(ram.viable == false);
        AGR_CHECK(ram.hasReason(PlanReason::INSUFFICIENT_RAM));
        AGR_CHECK(ram.hasReason(PlanReason::CONTRADICTORY_CAPABILITY));

        ModelRequirements unquant = model(1 * GB);
        unquant.quantization = QuantizationType::UNKNOWN;
        CapabilityReport qreport = reportFor(unquant, {
            device("CPU", true, 16 * GB),
            device("Vulkan", true, 8 * GB),
            device("CUDA", false, 0)
        });
        ModelExecutionPlan qplan = planner.plan(unquant, qreport, ExecutionEngineRegistry::instance());
        AGR_CHECK(qplan.viable);
        AGR_CHECK(qplan.hasReason(PlanReason::QUANTIZATION_UNKNOWN));
    }

    AGR_TEST_MAIN_END();
}
