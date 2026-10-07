// tests/test_p14_cancellation.cpp
// Phase 14 Test 10: Cancellation of streaming operations.
#include "model/FileModelDataSource.h"
#include "model/ModelResidencyManager.h"
#include "model/ModelRegion.h"

#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

static std::string tempPath() {
#ifdef _WIN32
    char buf[MAX_PATH];
    GetTempPathA(MAX_PATH, buf);
    return std::string(buf) + "agr_test_cancel.bin";
#else
    return "/tmp/agr_test_cancel.bin";
#endif
}

int main() {
    const size_t FSIZ = 1024;
    const std::string path = tempPath();
    std::vector<uint8_t> data(FSIZ, 0x99);
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
    agr::ModelResidencyBudget budget;
    budget.ram_bytes = 8 * 1024 * 1024;
    agr::ModelResidencyManager mgr(src, budget);

    // Test 1: cancel flag set BEFORE the call.
    {
        std::atomic<bool> cancel(true);
        agr::ModelRegion r;
        r.id = 1; r.offset = 0; r.size = 256; r.identifier = "r1";
        bool ok = mgr.ensureResident(r, agr::ModelResidencyTarget::RAM, &cancel);
        assert(!ok);
        // State must NOT be RESIDENT.
        auto s = mgr.state(1);
        assert(s != agr::ModelResidencyState::RESIDENT_IN_RAM);
        assert(!mgr.lastError().empty());
    }

    // Test 2: cancel flag NOT set — load should succeed.
    {
        std::atomic<bool> cancel(false);
        agr::ModelRegion r;
        r.id = 2; r.offset = 256; r.size = 256; r.identifier = "r2";
        bool ok = mgr.ensureResident(r, agr::ModelResidencyTarget::RAM, &cancel);
        assert(ok);
        assert(mgr.isResident(2, agr::ModelResidencyTarget::RAM));
    }

    // Test 3: cancel via FileModelDataSource directly.
    {
        std::atomic<bool> cancel(true);
        uint8_t buf[128] = {};
        bool ok = src.read(0, buf, 128, &cancel);
        assert(!ok);
    }

    std::remove(path.c_str());
    std::cout << "test_p14_cancellation PASSED\n";
    return 0;
}
