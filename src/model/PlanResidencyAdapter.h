// src/model/PlanResidencyAdapter.h
// Phase 14: translates a Phase 13 ModelExecutionPlan into residency behaviour.
//
// Phase 13 decides *what* should happen.
// Phase 14 (via this adapter) *executes* the residency consequences.
//
// Design:
//   FULL_RESIDENT  → load every region to RAM; upload to accelerator if viable.
//   PARTIAL_RESIDENT → load first N regions up to accelerator budget.
//   STREAMING      → load one region at a time on demand (no pre-loading).
//   CPU_FALLBACK   → load all regions to RAM; no accelerator upload.
//   UNSUPPORTED    → do nothing; return immediately with an error.
//
// This class does NOT duplicate the planning logic.
// It only translates the already-made decision into residency calls.
#pragma once

#include "model/GGUFRegionExtractor.h"
#include "model/ModelResidencyManager.h"
#include "planner/ExecutionPlan.h"

#include <atomic>
#include <string>
#include <vector>

namespace agr {

struct PlanResidencyResult {
    bool        success       = false;
    uint32_t    regions_total = 0;
    uint32_t    regions_ram   = 0;
    uint32_t    regions_accel = 0;
    std::string explanation;
    std::string error;
};

/// Translates ModelExecutionPlan residency decisions into actual
/// ModelResidencyManager operations. Engine-independent: no llama.cpp /
/// Colibrì / HydroXL specific logic.
class PlanResidencyAdapter {
public:
    PlanResidencyAdapter() = default;

    /// Apply the plan's residency strategy.
    /// @param plan     The Phase 13 plan.
    /// @param model_path  Path to the model file.
    /// @param mgr      The residency manager to operate on.
    /// @param cancel   Optional cancellation flag.
    PlanResidencyResult apply(const ModelExecutionPlan& plan,
                              const std::string& model_path,
                              ModelResidencyManager& mgr,
                              std::atomic<bool>* cancel = nullptr);
};

} // namespace agr
