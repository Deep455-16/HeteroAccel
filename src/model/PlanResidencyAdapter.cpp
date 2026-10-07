// src/model/PlanResidencyAdapter.cpp
// Phase 14: translate ModelExecutionPlan → residency behaviour.

#include "model/PlanResidencyAdapter.h"

#include <sstream>

namespace agr {

PlanResidencyResult PlanResidencyAdapter::apply(
        const ModelExecutionPlan& plan,
        const std::string& model_path,
        ModelResidencyManager& mgr,
        std::atomic<bool>* cancel) {

    PlanResidencyResult result;

    // UNSUPPORTED — nothing to do.
    if (!plan.viable || plan.mode == ExecutionMode::UNSUPPORTED) {
        result.success = false;
        result.error   = "Plan is not viable: " + plan.explanation;
        result.explanation = "UNSUPPORTED plan — no residency operations performed.";
        return result;
    }

    // Extract regions.
    GGUFRegionExtractor extractor;
    std::vector<ModelRegion> regions = extractor.extract(model_path, /*id_base=*/1);
    if (regions.empty()) {
        // Try whole-file fallback.
        ModelRegion whole = extractor.wholeFileRegion(model_path, 1);
        if (!whole.valid()) {
            result.success = false;
            result.error   = "Cannot extract model regions: " + extractor.lastError();
            return result;
        }
        regions.push_back(whole);
    }

    result.regions_total = static_cast<uint32_t>(regions.size());

    const ExecutionStrategy strategy = plan.residency;

    // -----------------------------------------------------------------------
    // UNSUPPORTED strategy explicitly
    // -----------------------------------------------------------------------
    if (strategy == ExecutionStrategy::CPU_FALLBACK) {
        // Load all regions to RAM only — no accelerator.
        std::ostringstream oss;
        oss << "CPU_FALLBACK: loading " << regions.size() << " region(s) to RAM.";
        result.explanation = oss.str();

        for (auto& reg : regions) {
            if (cancel && cancel->load()) {
                result.error = "Cancelled";
                return result;
            }
            if (mgr.ensureResident(reg, ModelResidencyTarget::RAM, cancel))
                result.regions_ram++;
        }
        result.success = (result.regions_ram == result.regions_total);
        if (!result.success) result.error = mgr.lastError();
        return result;
    }

    // -----------------------------------------------------------------------
    // STREAMING — on-demand only. Adapter does not pre-load anything.
    // -----------------------------------------------------------------------
    if (strategy == ExecutionStrategy::STREAMING) {
        result.explanation = "STREAMING: on-demand file-backed residency. "
                             "No pre-loading performed. "
                             "Regions will be loaded individually as needed.";
        result.success = true;  // infrastructure ready
        return result;
    }

    // -----------------------------------------------------------------------
    // FULL_RESIDENT — load everything to RAM; upload all to accelerator
    //                 if the plan targets an accelerator.
    // -----------------------------------------------------------------------
    if (strategy == ExecutionStrategy::FULL_RESIDENT ||
        strategy == ExecutionStrategy::AUTO) {
        bool use_accel = (plan.mode == ExecutionMode::VULKAN ||
                          plan.mode == ExecutionMode::CUDA);

        std::ostringstream oss;
        oss << "FULL_RESIDENT: loading " << regions.size() << " region(s)"
            << (use_accel ? " + accelerator upload." : " to RAM only.");
        result.explanation = oss.str();

        ModelResidencyTarget target = use_accel
            ? ModelResidencyTarget::ACCELERATOR
            : ModelResidencyTarget::RAM;

        for (auto& reg : regions) {
            if (cancel && cancel->load()) {
                result.error = "Cancelled";
                return result;
            }
            bool ok = mgr.ensureResident(reg, target, cancel);
            if (ok) {
                result.regions_ram++;
                if (use_accel) result.regions_accel++;
            }
        }
        result.success = (result.regions_ram > 0);
        if (!result.success) result.error = mgr.lastError();
        return result;
    }

    // -----------------------------------------------------------------------
    // PARTIAL_RESIDENT — load all to RAM; upload to accelerator up to
    //                    memory_budget_bytes.
    // -----------------------------------------------------------------------
    if (strategy == ExecutionStrategy::PARTIAL_RESIDENT ||
        strategy == ExecutionStrategy::ACCELERATOR_OFFLOAD) {
        bool use_accel = (plan.mode == ExecutionMode::VULKAN ||
                          plan.mode == ExecutionMode::CUDA);
        uint64_t accel_budget = plan.memory_budget_bytes;

        std::ostringstream oss;
        oss << "PARTIAL_RESIDENT: loading " << regions.size()
            << " region(s) to RAM; "
            << (use_accel ? "uploading subset to accelerator within budget." : "no accelerator.");
        result.explanation = oss.str();

        uint64_t uploaded_bytes = 0;
        for (auto& reg : regions) {
            if (cancel && cancel->load()) {
                result.error = "Cancelled";
                return result;
            }
            if (!mgr.ensureResident(reg, ModelResidencyTarget::RAM, cancel)) continue;
            result.regions_ram++;

            if (use_accel && accel_budget > 0 &&
                uploaded_bytes + reg.size <= accel_budget) {
                if (mgr.ensureResident(reg, ModelResidencyTarget::ACCELERATOR, cancel)) {
                    result.regions_accel++;
                    uploaded_bytes += reg.size;
                }
            }
        }
        result.success = (result.regions_ram > 0);
        if (!result.success) result.error = mgr.lastError();
        return result;
    }

    // -----------------------------------------------------------------------
    // HYBRID — treat like PARTIAL_RESIDENT.
    // -----------------------------------------------------------------------
    if (strategy == ExecutionStrategy::HYBRID) {
        // Reuse PARTIAL_RESIDENT logic by adjusting strategy-like value.
        // For simplicity, load all to RAM and attempt accel upload.
        bool use_accel = (plan.mode == ExecutionMode::VULKAN ||
                          plan.mode == ExecutionMode::CUDA);

        for (auto& reg : regions) {
            if (cancel && cancel->load()) {
                result.error = "Cancelled";
                return result;
            }
            ModelResidencyTarget tgt = use_accel
                ? ModelResidencyTarget::ACCELERATOR
                : ModelResidencyTarget::RAM;
            if (mgr.ensureResident(reg, tgt, cancel)) {
                result.regions_ram++;
                if (use_accel) result.regions_accel++;
            }
        }
        result.explanation = "HYBRID: mixed CPU/accelerator residency.";
        result.success = (result.regions_ram > 0);
        if (!result.success) result.error = mgr.lastError();
        return result;
    }

    // Unknown strategy fallback — treat as RAM-only.
    result.explanation = "Unknown strategy — falling back to RAM-only residency.";
    for (auto& reg : regions) {
        if (mgr.ensureResident(reg, ModelResidencyTarget::RAM, cancel))
            result.regions_ram++;
    }
    result.success = (result.regions_ram > 0);
    if (!result.success) result.error = mgr.lastError();
    return result;
}

} // namespace agr
