// tests/test_p23_sdk_init.cpp
// Phase 23: SDK Initialization, Versioning, and Error Strings Tests

#include "mini_test.h"
#include <heteroaccel/heteroaccel.h>
#include <heteroaccel/HeteroAccel.hpp>

#include <cstring>
#include <iostream>

int main() {
    std::cout << "--- Phase 23: SDK Init & Lifecycle Tests ---\n";

    // Test 1: Version string
    const char* ver = ha_version_string();
    AGR_CHECK(ver != nullptr);
    AGR_CHECK(std::strlen(ver) > 0);
    AGR_CHECK(std::strcmp(ver, "0.23.0") == 0);
    std::cout << "  ha_version_string: " << ver << "\n";

    // Test 2: Error string verification for all defined enum values
    HaError error_codes[] = {
        HA_OK,
        HA_ERR_INVALID_ARG,
        HA_ERR_MODEL_LOAD,
        HA_ERR_NO_BACKEND,
        HA_ERR_INFERENCE,
        HA_ERR_CANCELLED,
        HA_ERR_OOM,
        HA_ERR_UNSUPPORTED,
        HA_ERR_INTERNAL
    };

    for (HaError err : error_codes) {
        const char* s = ha_error_string(err);
        AGR_CHECK(s != nullptr);
        AGR_CHECK(std::strlen(s) > 0);
        AGR_CHECK(std::strcmp(s, "Unknown error") != 0 || err == (HaError)999);
    }
    // Test unknown error fallback
    const char* unk = ha_error_string((HaError)999);
    AGR_CHECK(unk != nullptr && std::strlen(unk) > 0);

    // Test 3: Runtime creation and destruction
    HaRuntime rt = ha_runtime_create();
    AGR_CHECK(rt != nullptr);

    int count = ha_backend_count(rt);
    AGR_CHECK(count >= 1); // At least hydroxl_cpu must be present

    ha_runtime_destroy(rt);

    // Test 4: Graceful handling of null pointer in destroy
    ha_runtime_destroy(nullptr);

    // Test 5: C++ RAII wrapper lifecycle
    {
        ha::Runtime cpp_rt;
        AGR_CHECK(cpp_rt.handle() != nullptr);
        AGR_CHECK(ha::Runtime::version() == "0.23.0");
        AGR_CHECK(cpp_rt.backend_count() >= 1);

        // Test move constructor
        ha::Runtime moved_rt(std::move(cpp_rt));
        AGR_CHECK(moved_rt.handle() != nullptr);
        AGR_CHECK(cpp_rt.handle() == nullptr);
    }

    AGR_TEST_MAIN_END();
}
