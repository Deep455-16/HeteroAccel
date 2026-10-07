// tests/test_p14_eviction.cpp
// Phase 14 Test 5: LRU eviction — active/pinned regions cannot be evicted.
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
    return std::string(buf) + "agr_test_evict.bin";
#else
    return "/tmp/agr_test_evict.bin";
#endif
}

static agr::ModelRegion makeRegion(uint64_t id, uint64_t off, uint64_t sz) {
    agr::ModelRegion r;
    r.id = id; r.offset = off; r.size = sz;
    r.identifier = "r" + std::to_string(id);
    r.kind = agr::ModelRegionKind::GENERIC;
    return r;
}

int main() {
    const size_t RSIZ = 128;
    const size_t FSIZ = RSIZ * 4;

    const std::string path = tempPath();
    std::vector<uint8_t> data(FSIZ, 0xBB);
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

    // Budget fits exactly 2 regions.
    agr::ModelResidencyBudget budget;
    budget.ram_bytes = RSIZ * 2;

    agr::ModelResidencyManager mgr(src, budget);

    auto r1 = makeRegion(1, 0,        RSIZ);
    auto r2 = makeRegion(2, RSIZ,     RSIZ);
    auto r3 = makeRegion(3, RSIZ * 2, RSIZ);

    assert(mgr.ensureResident(r1, agr::ModelResidencyTarget::RAM));
    assert(mgr.ensureResident(r2, agr::ModelResidencyTarget::RAM));

    // Pin r1 so it cannot be evicted.
    mgr.pin(1);

    // Load r3 — r2 should be evicted (r1 is pinned).
    assert(mgr.ensureResident(r3, agr::ModelResidencyTarget::RAM));
    assert(!mgr.isResident(1, agr::ModelResidencyTarget::RAM) ||
           mgr.isResident(1, agr::ModelResidencyTarget::RAM)); // pinned stays

    // r2 should have been evicted.
    assert(!mgr.isResident(2, agr::ModelResidencyTarget::RAM));

    // r1 still resident (pinned).
    assert(mgr.isResident(1, agr::ModelResidencyTarget::RAM));

    // Explicit evict of pinned region must fail.
    assert(!mgr.evict(1));

    // Unpin, then evict should succeed.
    mgr.unpin(1);
    assert(mgr.evict(1));
    assert(!mgr.isResident(1, agr::ModelResidencyTarget::RAM));

    std::remove(path.c_str());
    std::cout << "test_p14_eviction PASSED\n";
    return 0;
}
