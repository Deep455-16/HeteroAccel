// tests/test_p24_perf_profile.cpp
// Phase 24: Persistent Performance Profiles Tests

#include "mini_test.h"
#include "orchestration/PerfProfile.h"
#include <heteroaccel/heteroaccel.h>
#include <heteroaccel/HeteroAccel.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using namespace agr::orchestration;

int main() {
    std::cout << "--- Phase 24: PerfProfile Tests ---\n";

    // Test 1: Empty profile initialization
    PerfProfile prof;
    AGR_CHECK(prof.record_count() == 0);
    AGR_CHECK(prof.best_backend("model_a", 12, 24).empty());

    // Test 2: Record entries and query best backend
    prof.record("hydroxl_cpu", "model_a", 12, 24, 896, 14.5f);
    AGR_CHECK(prof.record_count() == 1);
    AGR_CHECK(prof.best_backend("model_a", 12, 24) == "hydroxl_cpu");

    // Add a faster backend for the same model
    prof.record("hydroxl_vulkan", "model_a", 12, 24, 896, 28.2f);
    AGR_CHECK(prof.record_count() == 2);
    AGR_CHECK(prof.best_backend("model_a", 12, 24) == "hydroxl_vulkan");

    // Record an entry for a different model
    prof.record("hydroxl_cpu", "model_b", 12, 32, 1024, 9.8f);
    AGR_CHECK(prof.record_count() == 3);
    AGR_CHECK(prof.best_backend("model_b", 12, 32) == "hydroxl_cpu");
    AGR_CHECK(prof.best_backend("model_a", 12, 24) == "hydroxl_vulkan");

    // Test 3: Running average update for existing record
    prof.record("hydroxl_cpu", "model_a", 12, 24, 896, 16.5f);
    AGR_CHECK(prof.record_count() == 3); // Not added as new entry, updated in-place
    const auto& recs = prof.records();
    bool found_updated = false;
    for (const auto& r : recs) {
        if (r.backend_id == "hydroxl_cpu" && r.model_hash == "model_a") {
            AGR_CHECK(r.sample_count == 2);
            // Average of 14.5 and 16.5 = 15.5
            AGR_CHECK(r.tokens_per_second > 15.4f && r.tokens_per_second < 15.6f);
            found_updated = true;
        }
    }
    AGR_CHECK(found_updated);

    // Test 4: Save and Load roundtrip
    std::string temp_path = "temp_p24_perf_profile.txt";
    AGR_CHECK(prof.save(temp_path));
    AGR_CHECK(std::filesystem::exists(temp_path));

    PerfProfile loaded = PerfProfile::load(temp_path);
    AGR_CHECK(loaded.record_count() == 3);
    AGR_CHECK(loaded.best_backend("model_a", 12, 24) == "hydroxl_vulkan");
    AGR_CHECK(loaded.best_backend("model_b", 12, 32) == "hydroxl_cpu");

    // Test 5: Corrupted line resilience
    {
        std::ofstream ofs(temp_path, std::ios::app);
        ofs << "\n";
        ofs << "not_key_value_line\n";
        ofs << "backend=bogus\n"; // Missing fields
        ofs << "backend=b3 model_hash=h3 quant=invalid_num layers=24 embd=896 toks_per_s=10 ts=1 n=1\n";
        ofs << "backend=b4 model_hash=h4 quant=12 layers=24 embd=896 toks_per_s=not_a_float ts=1 n=1\n";
        ofs << "   \n";
        // One valid line at the end
        ofs << "backend=hydroxl_cuda model_hash=model_c quant=12 layers=28 embd=1024 toks_per_s=45.0 ts=100 n=1\n";
    }

    PerfProfile resilient = PerfProfile::load(temp_path);
    // Should have original 3 + 1 new valid record = 4 records, corrupted lines ignored
    AGR_CHECK(resilient.record_count() == 4);
    AGR_CHECK(resilient.best_backend("model_c", 12, 28) == "hydroxl_cuda");

    // Test 6: Reset
    resilient.reset();
    AGR_CHECK(resilient.record_count() == 0);
    AGR_CHECK(resilient.best_backend("model_a", 12, 24).empty());

    // Clean up temporary file
    std::filesystem::remove(temp_path);

    // Test 7: Default profile path & model hash computation
    std::string def_path = PerfProfile::default_profile_path();
    AGR_CHECK(!def_path.empty());
    std::cout << "  Default profile path: " << def_path << "\n";

    // Hash computation on non-existent file
    std::string missing_hash = PerfProfile::compute_model_hash("nonexistent_file.bin");
    AGR_CHECK(missing_hash.empty());

    // Test 8: C SDK API integration
    HaRuntime rt = ha_runtime_create();
    AGR_CHECK(rt != nullptr);

    char path_buf[512] = {0};
    HaError err = ha_runtime_perf_path(rt, path_buf, sizeof(path_buf));
    AGR_CHECK(err == HA_OK);
    AGR_CHECK(std::strlen(path_buf) > 0);
    std::cout << "  SDK perf path: " << path_buf << "\n";

    // Invalid arguments
    AGR_CHECK(ha_runtime_perf_path(nullptr, path_buf, sizeof(path_buf)) == HA_ERR_INVALID_ARG);
    AGR_CHECK(ha_runtime_perf_path(rt, nullptr, sizeof(path_buf)) == HA_ERR_INVALID_ARG);
    AGR_CHECK(ha_runtime_perf_path(rt, path_buf, 0) == HA_ERR_INVALID_ARG);

    AGR_CHECK(ha_runtime_perf_reset(nullptr) == HA_ERR_INVALID_ARG);
    AGR_CHECK(ha_runtime_perf_reset(rt) == HA_OK);

    // Test 9: C++ SDK API
    {
        ha::Runtime cpp_rt;
        std::string cpp_path = cpp_rt.perf_path();
        AGR_CHECK(!cpp_path.empty());
        cpp_rt.perf_reset();
    }

    ha_runtime_destroy(rt);

    AGR_TEST_MAIN_END();
}
