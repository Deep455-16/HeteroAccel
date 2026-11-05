// tests/test_p25_perf_persist.cpp
// Phase 25: PerfProfile save/load round-trip and corrupt-file resilience

#include "mini_test.h"
#include "orchestration/PerfProfile.h"
#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>

using namespace agr::orchestration;

int main() {
    std::cout << "--- Phase 25: PerfProfile Persistence ---\n";

    // Use a temp path that won't collide with Phase 24's test file
    const char* tmp_env = std::getenv("TEMP");
    std::string tmp_dir  = tmp_env ? tmp_env : ".";
    std::string path     = tmp_dir + "/ha_p25_perf_persist.txt";
    std::remove(path.c_str());

    // === Test 1: record + save ===
    {
        PerfProfile p;
        p.record("hydroxl_cpu",    "mhash_a", 12, 24, 896, 5.0f);
        p.record("hydroxl_vulkan", "mhash_a", 12, 24, 896, 12.0f);
        p.record("hydroxl_cpu",    "mhash_b", 12, 32, 1024, 3.5f);
        AGR_CHECK(p.record_count() == 3);

        // Second observation for hydroxl_cpu / mhash_a — should update in-place
        p.record("hydroxl_cpu", "mhash_a", 12, 24, 896, 7.0f);
        AGR_CHECK(p.record_count() == 3);

        bool saved = p.save(path);
        AGR_CHECK(saved);
        std::cout << "  Saved " << p.record_count() << " records to " << path << "\n";
    }

    // === Test 2: load back and verify ===
    {
        PerfProfile loaded = PerfProfile::load(path);
        AGR_CHECK(loaded.record_count() == 3);

        // Best backend for mhash_a should be vulkan (12.0 > avg(5+7)/2=6.0)
        std::string best = loaded.best_backend("mhash_a", 12, 24);
        AGR_CHECK(best == "hydroxl_vulkan");
        std::cout << "  Best backend for mhash_a: " << best << "\n";

        std::string best_b = loaded.best_backend("mhash_b", 12, 32);
        AGR_CHECK(best_b == "hydroxl_cpu");
        std::cout << "  Best backend for mhash_b: " << best_b << "\n";
    }

    // === Test 3: corrupt file — load must not crash ===
    {
        std::ofstream ofs(path);   // overwrite with garbage
        ofs << "{not valid at all\n";
        ofs << "backend=broken model_hash=x quant=zz\n";
        ofs << "   \n";
        // One valid trailing record
        ofs << "backend=hydroxl_cpu model_hash=mhash_c quant=12 layers=28 embd=1024"
               " toks_per_s=9.5 ts=1000 n=1\n";
    }
    {
        PerfProfile corrupt = PerfProfile::load(path);
        // Must not crash; should recover the one valid record
        std::cout << "  Corrupt load recovered " << corrupt.record_count() << " records\n";
        // Don't assert exact count — implementation may choose to drop all or recover partial
    }

    // === Test 4: reset ===
    {
        PerfProfile p = PerfProfile::load(path);
        p.reset();
        AGR_CHECK(p.record_count() == 0);
        AGR_CHECK(p.best_backend("mhash_a", 12, 24).empty());
        std::cout << "  Reset OK\n";
    }

    // === Test 5: default_profile_path is non-empty ===
    {
        std::string def = PerfProfile::default_profile_path();
        AGR_CHECK(!def.empty());
        std::cout << "  Default profile path: " << def << "\n";
    }

    // === Test 6: compute_model_hash on missing file returns empty ===
    {
        std::string h = PerfProfile::compute_model_hash("definitely_does_not_exist_xyz.bin");
        AGR_CHECK(h.empty());
        std::cout << "  Missing-file hash: empty (correct)\n";
    }

    std::remove(path.c_str());
    std::cout << "  PASS: test_p25_perf_persist\n";
    AGR_TEST_MAIN_END();
}
