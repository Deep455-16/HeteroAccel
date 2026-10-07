// tests/test_p14_failure_handling.cpp
// Phase 14 Test 7: Failure handling — I/O errors and upload failures leave
// state consistent.
#include "model/FileModelDataSource.h"
#include "model/HostAcceleratorMemory.h"
#include "model/ModelResidencyManager.h"
#include "model/ModelRegion.h"

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
    return std::string(buf) + "agr_test_fail.bin";
#else
    return "/tmp/agr_test_fail.bin";
#endif
}

int main() {
    const std::string path = tempPath();
    const size_t FSIZ = 512;
    std::vector<uint8_t> data(FSIZ, 0x55);
    {
        FILE* f = nullptr;
#ifdef _WIN32
        fopen_s(&f, path.c_str(), "wb");
#else
        f = fopen(path.c_str(), "wb");
#endif
        assert(f); fwrite(data.data(), 1, FSIZ, f); fclose(f);
    }

    agr::FileModelDataSource src(path);
    assert(src.open());

    agr::HostAcceleratorMemory accel(1 * 1024 * 1024);
    agr::ModelResidencyBudget budget;
    budget.ram_bytes         = 4 * 1024 * 1024;
    budget.accelerator_bytes = 1 * 1024 * 1024;

    agr::ModelResidencyManager mgr(src, budget, &accel);

    agr::ModelRegion r;
    r.id = 1; r.offset = 0; r.size = 256; r.identifier = "r1";

    // Test 1: out-of-bounds region should fail, state must be NOT_RESIDENT or ERROR.
    agr::ModelRegion bad;
    bad.id = 99; bad.offset = FSIZ + 100; bad.size = 256; bad.identifier = "bad";
    assert(!mgr.ensureResident(bad, agr::ModelResidencyTarget::RAM));
    assert(!mgr.lastError().empty());
    // State must not be RESIDENT.
    auto s = mgr.state(99);
    assert(s != agr::ModelResidencyState::RESIDENT_IN_RAM &&
           s != agr::ModelResidencyState::RESIDENT_IN_ACCELERATOR);

    // Test 2: upload failure — host upload simulated to fail.
    accel.setFailNextUpload(true);
    assert(mgr.ensureResident(r, agr::ModelResidencyTarget::RAM));  // RAM OK
    bool upload_ok = mgr.ensureResident(r, agr::ModelResidencyTarget::ACCELERATOR);
    // Upload must have failed.
    assert(!upload_ok);
    // State must still be RESIDENT_IN_RAM (not promoted to ACCELERATOR).
    assert(mgr.state(1) == agr::ModelResidencyState::RESIDENT_IN_RAM);
    assert(!mgr.isResident(1, agr::ModelResidencyTarget::ACCELERATOR));
    // RAM pointer must still be valid.
    assert(mgr.ramPointer(1) != nullptr);

    // Telemetry: transfer_failures > 0.
    auto telem = mgr.telemetry();
    assert(telem.transfer_failures >= 1);

    std::remove(path.c_str());
    std::cout << "test_p14_failure_handling PASSED\n";
    return 0;
}
