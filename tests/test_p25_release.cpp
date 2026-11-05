// tests/test_p25_release.cpp
// Phase 25: Release API surface smoke test — verify all Phase 25 symbols are present
// and core lifecycle does not crash on edge inputs.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "mini_test.h"
#include <heteroaccel/heteroaccel.h>
#include <heteroaccel/HeteroAccel.hpp>
#include <iostream>
#include <cstring>

int main() {
    std::cout << "--- Phase 25: Release API Surface Smoke Test ---\n";

    // --- Version ---
    const char* v = ha_version_string();
    AGR_CHECK(v != nullptr);
    AGR_CHECK(std::strstr(v, "0.25") != nullptr);
    std::cout << "  Version: " << v << "\n";

    // --- Runtime lifecycle ---
    HaRuntime rt = ha_runtime_create();
    AGR_CHECK(rt != nullptr);

    // --- Model: null path must return null cleanly (no crash) ---
    HaModel null_model = ha_model_load(rt, nullptr, nullptr);
    AGR_CHECK(null_model == nullptr);

    // --- Memory estimate on null model must return an error (not crash) ---
    HaMemoryEstimate est = {};
    HaError merr = ha_model_memory_estimate(null_model, 512, &est);
    AGR_CHECK(merr != HA_OK);   // must report error for null model
    std::cout << "  Memory estimate on null model returned error (expected): " << merr << "\n";

    // --- ha_model_memory_estimate with null out pointer ---
    HaError merr2 = ha_model_memory_estimate(null_model, 512, nullptr);
    AGR_CHECK(merr2 != HA_OK);  // must not crash
    std::cout << "  Memory estimate with null out: error code " << merr2 << "\n";

    // --- Session on null model must return null cleanly ---
    HaSession null_sess = ha_session_create(null_model, nullptr);
    AGR_CHECK(null_sess == nullptr);

    // --- Perf profile API ---
    char path_buf[512] = {};
    HaError perr = ha_runtime_perf_path(rt, path_buf, sizeof(path_buf));
    AGR_CHECK(perr == HA_OK);
    AGR_CHECK(std::strlen(path_buf) > 0);
    std::cout << "  Perf profile path: " << path_buf << "\n";

    AGR_CHECK(ha_runtime_perf_reset(rt) == HA_OK);

    // --- Invalid args return HA_ERR_INVALID_ARG ---
    AGR_CHECK(ha_runtime_perf_reset(nullptr) == HA_ERR_INVALID_ARG);
    AGR_CHECK(ha_runtime_perf_path(nullptr, path_buf, sizeof(path_buf)) == HA_ERR_INVALID_ARG);

    ha_runtime_destroy(rt);

    // --- C++ RAII wrapper ---
    {
        ha::Runtime cpp_rt;
        std::string cpp_path = cpp_rt.perf_path();
        AGR_CHECK(!cpp_path.empty());
        cpp_rt.perf_reset();  // must not throw
    }

    std::cout << "  PASS: test_p25_release\n";
    AGR_TEST_MAIN_END();
}
