// src/execution/ExecutionGraphBuilder.cpp

#include "execution/ExecutionGraphBuilder.h"
#include "execution/ExecutionNodes.h"
#include "model/ModelResidencyManager.h"
#include <memory>
#include <sstream>

namespace agr {

bool ExecutionGraphBuilder::build(const ModelExecutionPlan& plan,
                                  const std::string& model_path,
                                  const std::vector<ModelRegion>& regions,
                                  const std::string& prompt,
                                  ExecutionGraph& out_graph,
                                  std::string& out_error) {
    if (!plan.viable || plan.mode == ExecutionMode::UNSUPPORTED) {
        out_error = "Cannot build graph for unsupported or non-viable plan.";
        return false;
    }

    ExecutionResource compute_res = ExecutionResource::CPU;
    if (plan.mode == ExecutionMode::VULKAN) compute_res = ExecutionResource::VULKAN;
    else if (plan.mode == ExecutionMode::CUDA) compute_res = ExecutionResource::CUDA;

    ExecutionResource accel_res = compute_res;
    if (accel_res == ExecutionResource::CPU) accel_res = ExecutionResource::NONE; // No separate accel transfer needed

    // Phase 14 Integration: We model the residency strategy using graph nodes.
    // 
    // A more fine-grained graph would model FileRead -> RAM -> Upload.
    // For now, ensureResident handles RAM vs ACCEL targets directly in a single block.
    // We map that boundary explicitly to ResidencyNodes.
    
    std::vector<uint64_t> residency_node_ids;
    
    if (plan.residency == ExecutionStrategy::FULL_RESIDENT || plan.residency == ExecutionStrategy::AUTO) {
        ModelResidencyTarget target = (compute_res != ExecutionResource::CPU) ? ModelResidencyTarget::ACCELERATOR : ModelResidencyTarget::RAM;
        for (const auto& reg : regions) {
            uint64_t id = next_id_++;
            auto node = std::make_unique<ResidencyNode>(id, "Residency_" + reg.identifier, reg, target, accel_res != ExecutionResource::NONE ? accel_res : ExecutionResource::RAM);
            out_graph.addNode(std::move(node));
            residency_node_ids.push_back(id);
        }
    } else if (plan.residency == ExecutionStrategy::PARTIAL_RESIDENT || plan.residency == ExecutionStrategy::HYBRID) {
        // Load all to RAM, then upload what fits (modeled as RAM targets here for simplicity; 
        // the PlanResidencyAdapter has budget logic. We simplify for the graph builder test case).
        for (const auto& reg : regions) {
            uint64_t id = next_id_++;
            auto node = std::make_unique<ResidencyNode>(id, "Residency_" + reg.identifier, reg, ModelResidencyTarget::RAM, ExecutionResource::RAM);
            out_graph.addNode(std::move(node));
            residency_node_ids.push_back(id);
        }
    } else if (plan.residency == ExecutionStrategy::CPU_FALLBACK) {
        for (const auto& reg : regions) {
            uint64_t id = next_id_++;
            auto node = std::make_unique<ResidencyNode>(id, "Residency_" + reg.identifier, reg, ModelResidencyTarget::RAM, ExecutionResource::RAM);
            out_graph.addNode(std::move(node));
            residency_node_ids.push_back(id);
        }
    } else if (plan.residency == ExecutionStrategy::STREAMING) {
        // STREAMING: No pre-loading, so no residency nodes. Handled on demand.
    } else {
        // Unknown or MEMORY_PRESSURE
        out_error = "ExecutionStrategy not fully modeled in GraphBuilder.";
        return false;
    }

    // Sync node barrier
    uint64_t sync_id = next_id_++;
    out_graph.addNode(std::make_unique<SynchronizationNode>(sync_id, "Sync_Data_Ready"));
    for (uint64_t r_id : residency_node_ids) {
        out_graph.addDependency(r_id, sync_id);
    }

    // Compute node
    uint64_t compute_id = next_id_++;
    out_graph.addNode(std::make_unique<ComputeNode>(compute_id, "Engine_Compute", prompt, compute_res));
    out_graph.addDependency(sync_id, compute_id);

    // Validate generated graph
    return out_graph.validate(out_error);
}

} // namespace agr
