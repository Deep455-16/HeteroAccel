// src/collaboration/WorkloadPartitioner.cpp
// Query Understanding, Task Decomposition, and Dynamic Workload Partitioning.

#include "collaboration/WorkloadPartitioner.h"
#include "execution/ExecutionGraph.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace agr {
namespace collaboration {

namespace {

std::string toLower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

bool hasAnyKeyword(const std::string& lower, std::initializer_list<const char*> keywords) {
    for (const char* kw : keywords) {
        if (lower.find(kw) != std::string::npos) return true;
    }
    return false;
}

bool hydroxlSupportsArch(const std::string& arch_raw) {
    if (arch_raw.empty()) return true;
    std::string a = toLower(arch_raw);
    static const char* kSupported[] = {
        "llama", "qwen2", "qwen2_vl", "mistral", "mistral3",
        "phi", "phi2", "phi3", "gemma", "gemma2"
    };
    for (const char* s : kSupported) {
        if (a == s) return true;
    }
    return false;
}

bool hydroxlSupportsVulkanQuant(int quant_type) {
    // HydroXL Vulkan shaders natively execute F32 (0), F16 (1), Q5_1 (7), Q8_0 (8)
    return quant_type == 0 || quant_type == 1 || quant_type == 7 || quant_type == 8;
}

} // namespace

QueryCategory WorkloadPartitioner::classifyQuery(const std::string& prompt) {
    std::string lower = toLower(prompt);

    if (hasAnyKeyword(lower, {"bug", "race", "deadlock", "crash", "segfault",
                              "verify", "validate", "leak", "hazard"})) {
        return QueryCategory::DEBUGGING_AND_VERIFICATION;
    }
    if (hasAnyKeyword(lower, {"code", "optimize", "complexity", "loop", "function",
                              "algorithm", "kernel", "c++", "python", "std::", "for(", "for ("})) {
        return QueryCategory::CODE_ANALYSIS_AND_OPTIMIZATION;
    }
    if (hasAnyKeyword(lower, {"architecture", "how does", "explain", "difference between",
                              "design", "gpu", "vulkan", "cuda", "attention", "transformer", "quantization"})) {
        return QueryCategory::COMPLEX_TECHNICAL_QUESTION;
    }
    return QueryCategory::GENERAL_REASONING_AND_SYNTHESIS;
}

EngineId WorkloadPartitioner::selectEngineForTask(
    TaskType task_type,
    const ModelMemoryProfile& model_profile,
    std::string& reason_out) {

    switch (task_type) {
        case TaskType::STRUCTURAL_AND_REASONING:
        case TaskType::CONCEPTUAL_EXPLANATION: {
            if (hydroxlSupportsArch(model_profile.arch)) {
                reason_out = "HydroXL selected for native structural/conceptual reasoning on supported architecture '" +
                             (model_profile.arch.empty() ? std::string("qwen2") : model_profile.arch) + "'";
                return EngineId::HYDROXL;
            }
            reason_out = "Architecture '" + model_profile.arch +
                         "' not in HydroXL native table; reassigned to llama.cpp compatibility engine";
            return EngineId::LLAMA_CPP;
        }
        case TaskType::IMPLEMENTATION_AND_EDGE_CASES:
        case TaskType::PRACTICAL_EXAMPLES: {
            reason_out = "llama.cpp selected for broad GGUF compatibility, implementation details, and edge-case generation";
            return EngineId::LLAMA_CPP;
        }
        case TaskType::COMPLEXITY_AND_OPTIMIZATION:
        case TaskType::VERIFICATION_AND_SUMMARY: {
            reason_out = "Colibri selected for specialized static complexity estimation, constraint checking, and cross-engine verification";
            return EngineId::COLIBRI;
        }
    }
    reason_out = "Default assignment to HydroXL";
    return EngineId::HYDROXL;
}

HardwareBackend WorkloadPartitioner::selectHardwareBackendForEngine(
    EngineId engine,
    const ModelMemoryProfile& model_profile,
    const orchestration::HardwareDetector& hw,
    const std::string& preferred_gpu_backend,
    bool gpu_already_claimed_in_wave,
    std::string& orchestrator_backend_id_out,
    std::string& reason_out) {

    std::string pref = toLower(preferred_gpu_backend);
    bool force_cpu = (pref == "cpu");
    bool prefer_vulkan = (pref == "vulkan");
    bool prefer_cuda = (pref == "cuda");

    // 1. Colibri is strictly CPU-only (never claims Vulkan or CUDA)
    if (engine == EngineId::COLIBRI) {
        orchestrator_backend_id_out = "colibri_cpu";
        reason_out = "Colibri specialized analyzer executes deterministically on CPU";
        return HardwareBackend::CPU;
    }

    // 2. HydroXL supports CPU and Vulkan (for quants {0, 1, 7, 8}); never CUDA
    if (engine == EngineId::HYDROXL) {
        if (!force_cpu && !prefer_cuda && hw.has_vulkan() &&
            hydroxlSupportsVulkanQuant(model_profile.quant_type) &&
            !gpu_already_claimed_in_wave) {
            const auto* gpu = hw.best_gpu();
            size_t vram = gpu ? gpu->vram_bytes : 0;
            if (vram == 0 || vram >= model_profile.hydroxl_estimated_bytes) {
                orchestrator_backend_id_out = "hydroxl_vulkan";
                reason_out = "HydroXL assigned to Vulkan compute backend (supported quant " +
                             std::to_string(model_profile.quant_type) + ")";
                return HardwareBackend::VULKAN;
            }
        }
        orchestrator_backend_id_out = "hydroxl_cpu";
        if (prefer_cuda) {
            reason_out = "HydroXL does not implement CUDA kernels; using HydroXL native CPU backend";
        } else if (hw.has_vulkan() && !hydroxlSupportsVulkanQuant(model_profile.quant_type)) {
            reason_out = "HydroXL assigned to CPU (quant type " + std::to_string(model_profile.quant_type) +
                         " routes to native CPU dequant kernels)";
        } else {
            reason_out = "HydroXL assigned to CPU backend";
        }
        return HardwareBackend::CPU;
    }

    // 3. llama.cpp supports CUDA (when compiled + available), Vulkan (when available), and CPU
    if (engine == EngineId::LLAMA_CPP) {
        if (!force_cpu && !prefer_vulkan &&
            orchestration::HardwareDetector::cuda_inference_compiled() &&
            hw.has_cuda()) {
            const auto* cgpu = hw.best_cuda_gpu();
            size_t vram = cgpu ? cgpu->vram_bytes : 0;
            if (vram == 0 || vram >= model_profile.llamacpp_estimated_bytes) {
                orchestrator_backend_id_out = "cuda";
                reason_out = "llama.cpp assigned to NVIDIA CUDA backend";
                return HardwareBackend::CUDA;
            }
        }

        if (!force_cpu && !prefer_cuda && hw.has_vulkan() && !gpu_already_claimed_in_wave) {
            const auto* vgpu = hw.best_gpu();
            size_t vram = vgpu ? vgpu->vram_bytes : 0;
            if (vram == 0 || vram >= model_profile.llamacpp_estimated_bytes) {
                orchestrator_backend_id_out = "llamacpp_vulkan";
                reason_out = "llama.cpp assigned to Vulkan backend";
                return HardwareBackend::VULKAN;
            }
        }

        orchestrator_backend_id_out = "llamacpp_cpu";
        if (gpu_already_claimed_in_wave) {
            reason_out = "llama.cpp assigned to CPU to avoid GPU contention with concurrent wave task";
        } else {
            reason_out = "llama.cpp assigned to CPU backend";
        }
        return HardwareBackend::CPU;
    }

    orchestrator_backend_id_out = "hydroxl_cpu";
    reason_out = "Fallback CPU backend";
    return HardwareBackend::CPU;
}

PartitionPlan WorkloadPartitioner::createPlan(
    const CollaborativeRequest& req,
    const ModelMemoryProfile& model_profile,
    const CollaborativeResourceManager& resource_mgr,
    const orchestration::HardwareDetector& hw) {

    PartitionPlan plan;
    plan.query_category = classifyQuery(req.prompt);
    plan.memory_constrained_mode = !resource_mgr.canKeepConcurrentNeuralEngines(model_profile);

    // Format conversation context prefix if history is present
    std::string conv_prefix;
    if (!req.conversation_history.empty()) {
        std::ostringstream oss;
        for (const auto& turn : req.conversation_history) {
            oss << turn.first << ": " << turn.second << "\n";
        }
        conv_prefix = oss.str();
    }
    std::string full_query = conv_prefix.empty() ? req.prompt : (conv_prefix + "User: " + req.prompt);

    // Define the 3 collaborative tasks tailored to the query category
    CollaborativeTask t1;
    t1.task_id = "task_1_hydroxl";
    t1.required_capabilities = EngineCapabilitySet{
        EngineCapability::TEXT_GENERATION,
        EngineCapability::BACKEND_CPU
    };
    t1.estimated_memory_bytes = model_profile.hydroxl_estimated_bytes;

    CollaborativeTask t2;
    t2.task_id = "task_2_llamacpp";
    t2.required_capabilities = EngineCapabilitySet{
        EngineCapability::TEXT_GENERATION,
        EngineCapability::BACKEND_CPU
    };
    t2.estimated_memory_bytes = model_profile.llamacpp_estimated_bytes;

    CollaborativeTask t3;
    t3.task_id = "task_3_colibri";
    t3.required_capabilities = EngineCapabilitySet{
        EngineCapability::TEXT_GENERATION,
        EngineCapability::BACKEND_CPU
    };
    t3.estimated_memory_bytes = model_profile.colibri_estimated_bytes;
    t3.dependencies = {"task_1_hydroxl", "task_2_llamacpp"};

    switch (plan.query_category) {
        case QueryCategory::CODE_ANALYSIS_AND_OPTIMIZATION:
        case QueryCategory::DEBUGGING_AND_VERIFICATION:
            t1.task_type = TaskType::STRUCTURAL_AND_REASONING;
            t1.description = "Structural control-flow and reasoning analysis";
            t1.input_context = full_query;

            t2.task_type = TaskType::IMPLEMENTATION_AND_EDGE_CASES;
            t2.description = "Implementation details, boundary conditions, and edge cases";
            t2.input_context = full_query;

            t3.task_type = TaskType::COMPLEXITY_AND_OPTIMIZATION;
            t3.description = "Algorithmic time/space complexity, static risk verification, and optimization synthesis";
            t3.input_context = full_query;
            break;

        case QueryCategory::COMPLEX_TECHNICAL_QUESTION:
        case QueryCategory::GENERAL_REASONING_AND_SYNTHESIS:
            t1.task_type = TaskType::CONCEPTUAL_EXPLANATION;
            t1.description = "Core conceptual architecture and theoretical explanation";
            t1.input_context = full_query;

            t2.task_type = TaskType::PRACTICAL_EXAMPLES;
            t2.description = "Concrete practical examples and runtime behavior";
            t2.input_context = full_query;

            t3.task_type = TaskType::VERIFICATION_AND_SUMMARY;
            t3.description = "Cross-engine consistency verification and structured technical summary";
            t3.input_context = full_query;
            break;
    }

    // Step 1: Engine selection (kept distinct from hardware backend selection)
    std::string eng_reason1, eng_reason2, eng_reason3;
    t1.assigned_engine = selectEngineForTask(t1.task_type, model_profile, eng_reason1);
    t2.assigned_engine = selectEngineForTask(t2.task_type, model_profile, eng_reason2);
    t3.assigned_engine = selectEngineForTask(t3.task_type, model_profile, eng_reason3);

    // Step 2: Hardware backend selection per engine
    bool gpu_claimed_wave0 = false;
    std::string hw_reason1, hw_reason2, hw_reason3;

    t1.assigned_backend = selectHardwareBackendForEngine(
        t1.assigned_engine, model_profile, hw, req.preferred_gpu_backend,
        gpu_claimed_wave0, t1.orchestrator_backend_id, hw_reason1);
    if (t1.assigned_backend != HardwareBackend::CPU) {
        gpu_claimed_wave0 = true;
    }
    t1.assignment_reason = eng_reason1 + " | " + hw_reason1;

    t2.assigned_backend = selectHardwareBackendForEngine(
        t2.assigned_engine, model_profile, hw, req.preferred_gpu_backend,
        gpu_claimed_wave0, t2.orchestrator_backend_id, hw_reason2);
    t2.assignment_reason = eng_reason2 + " | " + hw_reason2;

    t3.assigned_backend = selectHardwareBackendForEngine(
        t3.assigned_engine, model_profile, hw, req.preferred_gpu_backend,
        false, t3.orchestrator_backend_id, hw_reason3);
    t3.assignment_reason = eng_reason3 + " | " + hw_reason3;

    plan.tasks = {t1, t2, t3};

    // Validate DAG and compute topological execution waves using ExecutionGraph
    class CollaborativeGraphNode final : public IExecutionNode {
    public:
        CollaborativeGraphNode(uint64_t id, std::string name, ExecutionResource res)
            : id_(id), name_(std::move(name)), res_(res) {}
        uint64_t getId() const override { return id_; }
        std::string getName() const override { return name_; }
        ExecutionNodeType getType() const override { return ExecutionNodeType::COMPUTE; }
        ExecutionResource getResource() const override { return res_; }
        GraphNodeResult execute(ExecutionGraphContext&) override {
            GraphNodeResult r;
            r.success = true;
            return r;
        }
    private:
        uint64_t id_;
        std::string name_;
        ExecutionResource res_;
    };

    ExecutionGraph dag;
    std::map<std::string, uint64_t> id_to_num;
    std::map<uint64_t, std::string> num_to_id;

    for (size_t i = 0; i < plan.tasks.size(); ++i) {
        const auto& task = plan.tasks[i];
        uint64_t nid = static_cast<uint64_t>(i + 1);
        id_to_num[task.task_id] = nid;
        num_to_id[nid] = task.task_id;

        ExecutionResource res = ExecutionResource::CPU;
        switch (task.assigned_backend) {
            case HardwareBackend::CPU:    res = ExecutionResource::CPU; break;
            case HardwareBackend::VULKAN: res = ExecutionResource::VULKAN; break;
            case HardwareBackend::CUDA:   res = ExecutionResource::CUDA; break;
        }
        dag.addNode(std::make_unique<CollaborativeGraphNode>(nid, task.description, res));
    }

    for (const auto& task : plan.tasks) {
        uint64_t succ_id = id_to_num[task.task_id];
        for (const auto& dep : task.dependencies) {
            auto it = id_to_num.find(dep);
            if (it != id_to_num.end()) {
                dag.addDependency(it->second, succ_id);
            }
        }
    }

    std::string dag_err;
    if (dag.validate(dag_err)) {
        // Compute topological waves from ExecutionGraph predecessors/successors
        std::map<uint64_t, size_t> in_degree;
        for (uint64_t nid : dag.getAllNodeIds()) {
            in_degree[nid] = dag.getPredecessors(nid).size();
        }
        std::vector<uint64_t> current_wave;
        for (size_t i = 0; i < plan.tasks.size(); ++i) {
            uint64_t nid = static_cast<uint64_t>(i + 1);
            if (in_degree[nid] == 0) {
                current_wave.push_back(nid);
            }
        }
        while (!current_wave.empty()) {
            std::vector<std::string> wave_names;
            std::vector<uint64_t> next_wave;
            for (uint64_t u : current_wave) {
                wave_names.push_back(num_to_id[u]);
                for (uint64_t v : dag.getSuccessors(u)) {
                    if (--in_degree[v] == 0) {
                        next_wave.push_back(v);
                    }
                }
            }
            plan.waves.push_back(std::move(wave_names));
            current_wave = std::move(next_wave);
        }

        for (size_t w = 0; w < plan.waves.size(); ++w) {
            for (const auto& tid : plan.waves[w]) {
                for (auto& task : plan.tasks) {
                    if (task.task_id == tid) {
                        task.wave_index = w;
                        task.state = (w == 0) ? TaskState::READY : TaskState::PENDING;
                    }
                }
            }
        }
    } else {
        plan.waves = {{"task_1_hydroxl", "task_2_llamacpp"}, {"task_3_colibri"}};
    }

    std::ostringstream summary;
    summary << "Category=" << queryCategoryToString(plan.query_category)
            << ", Tasks=" << plan.tasks.size()
            << ", Waves=" << plan.waves.size()
            << ", MemoryMode=" << (plan.memory_constrained_mode ? "SequentialSwap(Constrained)" : "ConcurrentResident");
    plan.plan_summary = summary.str();

    return plan;
}

} // namespace collaboration
} // namespace agr
