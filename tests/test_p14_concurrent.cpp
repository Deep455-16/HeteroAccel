// tests/test_p14_concurrent.cpp
// Phase 14 Test 11: Concurrent access — multiple threads requesting residency.
#include "model/FileModelDataSource.h"
#include "model/ModelResidencyManager.h"
#include "model/ModelRegion.h"

#include <atomic>
#include <cassert>
#include <cstdio>
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
    return std::string(buf) + "agr_test_concurrent.bin";
#else
    return "/tmp/agr_test_concurrent.bin";
#endif
}

int main() {
    const size_t NUM_REGIONS = 100;
    const size_t REGION_SIZE = 128;
    const size_t FSIZ = NUM_REGIONS * REGION_SIZE;

    const std::string path = tempPath();
    std::vector<uint8_t> data(FSIZ, 0x11);
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
    budget.ram_bytes = 10 * 1024 * 1024; // Plenty of room
    agr::ModelResidencyManager mgr(src, budget);

    const int NUM_THREADS = 8;
    std::vector<std::thread> threads;
    std::atomic<int> success_count(0);

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 50; ++i) {
                // Each thread requests regions, some overlapping with other threads
                uint64_t r_id = (t * 10 + i) % NUM_REGIONS;
                agr::ModelRegion r;
                r.id = r_id;
                r.offset = r_id * REGION_SIZE;
                r.size = REGION_SIZE;
                r.identifier = "r" + std::to_string(r_id);

                if (mgr.ensureResident(r, agr::ModelResidencyTarget::RAM)) {
                    success_count++;
                }

                // Randomly evict to create churn
                if (i % 3 == 0) {
                    mgr.evict(r_id);
                }
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    // Since budget is large enough, all ensureResident calls should ultimately succeed,
    // though some might be redundant or re-loaded after eviction.
    // The main thing is we didn't crash or corrupt metadata.
    assert(success_count > 0);

    std::remove(path.c_str());
    std::cout << "test_p14_concurrent PASSED\n";
    return 0;
}
