// tests/test_p14_prefetch.cpp
// Phase 14 Test 6: Prefetch — hit/miss accounting and actual byte availability.
#include "model/FileModelDataSource.h"
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
    return std::string(buf) + "agr_test_prefetch.bin";
#else
    return "/tmp/agr_test_prefetch.bin";
#endif
}

int main() {
    const size_t FSIZ = 1024;
    const std::string path = tempPath();
    std::vector<uint8_t> data(FSIZ);
    for (size_t i = 0; i < FSIZ; ++i) data[i] = static_cast<uint8_t>(i);
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
    budget.ram_bytes = 4 * 1024 * 1024;

    agr::ModelResidencyManager mgr(src, budget);

    agr::ModelRegion r1;
    r1.id = 1; r1.offset = 0; r1.size = 256; r1.identifier = "r1";

    // Prefetch miss: r1 not yet loaded.
    assert(mgr.prefetch(r1, agr::ModelResidencyTarget::RAM));
    auto t1 = mgr.telemetry();
    assert(t1.prefetch_misses == 1);
    assert(t1.prefetch_hits == 0);
    assert(t1.prefetch_count == 1);
    assert(mgr.isResident(1, agr::ModelResidencyTarget::RAM));

    // Prefetch hit: r1 already loaded.
    assert(mgr.prefetch(r1, agr::ModelResidencyTarget::RAM));
    auto t2 = mgr.telemetry();
    assert(t2.prefetch_hits == 1);
    assert(t2.prefetch_misses == 1);

    // Verify bytes correct.
    const uint8_t* ptr = mgr.ramPointer(1);
    assert(ptr);
    for (int i = 0; i < 256; ++i)
        assert(ptr[i] == data[i]);

    // Disable prefetch — prefetch should return false.
    mgr.setPrefetchEnabled(false);
    agr::ModelRegion r2;
    r2.id = 2; r2.offset = 256; r2.size = 256; r2.identifier = "r2";
    assert(!mgr.prefetch(r2, agr::ModelResidencyTarget::RAM));

    std::remove(path.c_str());
    std::cout << "test_p14_prefetch PASSED\n";
    return 0;
}
