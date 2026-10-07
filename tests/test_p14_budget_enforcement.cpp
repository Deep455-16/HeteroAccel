// tests/test_p14_budget_enforcement.cpp
// Phase 14 Test 4: RAM budget enforcement — system must not load more than budget.
#include "model/FileModelDataSource.h"
#include "model/ModelResidencyManager.h"
#include "model/ModelRegion.h"

#include <cassert>
#include <cstdio>
#include <cstring>
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
    return std::string(buf) + "agr_test_budget.bin";
#else
    return "/tmp/agr_test_budget.bin";
#endif
}

int main() {
    // File: 10 × 256-byte regions = 2560 bytes total.
    const size_t REGION_SIZE = 256;
    const int    N_REGIONS   = 10;
    const size_t FILE_SIZE   = REGION_SIZE * N_REGIONS;

    const std::string path = tempPath();
    std::vector<uint8_t> data(FILE_SIZE, 0xCC);
    {
        FILE* f = nullptr;
#ifdef _WIN32
        fopen_s(&f, path.c_str(), "wb");
#else
        f = fopen(path.c_str(), "wb");
#endif
        assert(f);
        fwrite(data.data(), 1, FILE_SIZE, f);
        fclose(f);
    }

    agr::FileModelDataSource src(path);
    assert(src.open());

    // Budget: only 3 regions can fit.
    agr::ModelResidencyBudget budget;
    budget.ram_bytes         = REGION_SIZE * 3;
    budget.accelerator_bytes = 0;

    agr::ModelResidencyManager mgr(src, budget);

    // Load 4 regions — 4th should trigger eviction of one older one.
    int success = 0;
    for (int i = 0; i < 4; ++i) {
        agr::ModelRegion r;
        r.id         = static_cast<uint64_t>(i + 1);
        r.offset     = static_cast<uint64_t>(i) * REGION_SIZE;
        r.size       = REGION_SIZE;
        r.identifier = "region_" + std::to_string(i);
        r.kind       = agr::ModelRegionKind::GENERIC;
        if (mgr.ensureResident(r, agr::ModelResidencyTarget::RAM))
            success++;
    }

    // All 4 should succeed (LRU evicts the oldest to make room for new).
    assert(success == 4);

    // Resident RAM must not exceed budget.
    assert(mgr.residentRamBytes() <= budget.ram_bytes);

    // Telemetry: at least one eviction should have happened.
    auto telem = mgr.telemetry();
    assert(telem.eviction_count >= 1);

    std::remove(path.c_str());
    std::cout << "test_p14_budget_enforcement PASSED\n";
    return 0;
}
