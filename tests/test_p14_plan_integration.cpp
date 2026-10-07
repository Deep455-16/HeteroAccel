// tests/test_p14_plan_integration.cpp
// Phase 14 Test 8: PlanResidencyAdapter — each strategy maps to the correct behaviour.
#include "model/FileModelDataSource.h"
#include "model/HostAcceleratorMemory.h"
#include "model/ModelResidencyManager.h"
#include "model/PlanResidencyAdapter.h"
#include "planner/ExecutionPlan.h"
#include "model/ModelTypes.h"

#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

static std::string tempPath() {
#ifdef _WIN32
    char buf[MAX_PATH];
    GetTempPathA(MAX_PATH, buf);
    return std::string(buf) + "agr_test_plan.bin";
#else
    return "/tmp/agr_test_plan.bin";
#endif
}

// Build a minimal viable plan with the given strategy.
static agr::ModelExecutionPlan makePlan(agr::ExecutionStrategy strategy,
                                         agr::ExecutionMode mode,
                                         uint64_t budget_bytes) {
    agr::ModelExecutionPlan p;
    p.viable   = true;
    p.executable = true;
    p.residency = strategy;
    p.mode      = mode;
    p.memory_budget_bytes = budget_bytes;
    return p;
}

int main() {
    const size_t FSIZ = 2048;
    const std::string path = tempPath();
    std::vector<uint8_t> data(FSIZ, 0x77);
    {
        FILE* f = nullptr;
#ifdef _WIN32
        fopen_s(&f, path.c_str(), "wb");
#else
        f = fopen(path.c_str(), "wb");
#endif
        assert(f); fwrite(data.data(), 1, FSIZ, f); fclose(f);
    }

    agr::PlanResidencyAdapter adapter;

    // --- UNSUPPORTED plan ---
    {
        agr::FileModelDataSource src(path);
        src.open();
        agr::ModelResidencyBudget bud; bud.ram_bytes = 8 * 1024 * 1024;
        agr::ModelResidencyManager mgr(src, bud);

        agr::ModelExecutionPlan bad;
        bad.viable = false;
        bad.mode   = agr::ExecutionMode::UNSUPPORTED;

        auto res = adapter.apply(bad, path, mgr);
        assert(!res.success);
        assert(res.regions_ram == 0);
    }

    // --- CPU_FALLBACK plan ---
    {
        agr::FileModelDataSource src(path);
        src.open();
        agr::ModelResidencyBudget bud; bud.ram_bytes = 8 * 1024 * 1024;
        agr::ModelResidencyManager mgr(src, bud);

        auto plan = makePlan(agr::ExecutionStrategy::CPU_FALLBACK,
                             agr::ExecutionMode::CPU, 8 * 1024 * 1024);
        auto res = adapter.apply(plan, path, mgr);
        assert(res.success);
        assert(res.regions_ram > 0);
        assert(res.regions_accel == 0);  // no accel for CPU fallback
    }

    // --- STREAMING plan ---
    {
        agr::FileModelDataSource src(path);
        src.open();
        agr::ModelResidencyBudget bud; bud.ram_bytes = 8 * 1024 * 1024;
        agr::ModelResidencyManager mgr(src, bud);

        auto plan = makePlan(agr::ExecutionStrategy::STREAMING,
                             agr::ExecutionMode::CPU, 0);
        auto res = adapter.apply(plan, path, mgr);
        // STREAMING: no pre-loading, but infrastructure is ready.
        assert(res.success);
        assert(res.regions_ram == 0);   // nothing pre-loaded
    }

    // --- FULL_RESIDENT plan (CPU mode) ---
    {
        agr::FileModelDataSource src(path);
        src.open();
        agr::ModelResidencyBudget bud; bud.ram_bytes = 8 * 1024 * 1024;
        agr::ModelResidencyManager mgr(src, bud);

        auto plan = makePlan(agr::ExecutionStrategy::FULL_RESIDENT,
                             agr::ExecutionMode::CPU, 8 * 1024 * 1024);
        auto res = adapter.apply(plan, path, mgr);
        assert(res.success);
        assert(res.regions_ram > 0);
    }

    // --- PARTIAL_RESIDENT plan with HostAcceleratorMemory ---
    {
        agr::FileModelDataSource src(path);
        src.open();
        agr::HostAcceleratorMemory accel(8 * 1024 * 1024);
        agr::ModelResidencyBudget bud;
        bud.ram_bytes         = 8 * 1024 * 1024;
        bud.accelerator_bytes = 8 * 1024 * 1024;
        agr::ModelResidencyManager mgr(src, bud, &accel);

        auto plan = makePlan(agr::ExecutionStrategy::PARTIAL_RESIDENT,
                             agr::ExecutionMode::VULKAN, FSIZ);
        auto res = adapter.apply(plan, path, mgr);
        assert(res.success);
        assert(res.regions_ram > 0);
    }

    std::remove(path.c_str());
    std::cout << "test_p14_plan_integration PASSED\n";
    return 0;
}
