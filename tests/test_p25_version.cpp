// tests/test_p25_version.cpp
// Phase 25: Version string and compile-time macro verification
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
#include <cstring>
#include <iostream>

int main() {
    std::cout << "--- Phase 25: Version Verification ---\n";

    // Compile-time macros
    AGR_CHECK(HETEROACCEL_VERSION_MAJOR == 0);
    AGR_CHECK(HETEROACCEL_VERSION_MINOR == 25);
    AGR_CHECK(HETEROACCEL_VERSION_PATCH == 0);

    // Runtime version string
    HaRuntime rt = ha_runtime_create();
    AGR_CHECK(rt != nullptr);

    const char* ver = ha_version_string();
    AGR_CHECK(ver != nullptr);
    AGR_CHECK(std::strstr(ver, "0.25") != nullptr);
    std::cout << "  ha_version_string(): " << ver << "\n";

    ha_runtime_destroy(rt);
    AGR_TEST_MAIN_END();
}
