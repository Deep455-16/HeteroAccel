// tests/test_p23_sdk_backends.cpp
// Phase 23: Backend Enumeration, Introspection, and Fallback Tests

#include "mini_test.h"
#include <heteroaccel/heteroaccel.h>
#include <heteroaccel/HeteroAccel.hpp>

#include <cstring>
#include <iostream>

int main() {
    std::cout << "--- Phase 23: SDK Backend Introspection Tests ---\n";

    // Test 1: Null runtime handling
    AGR_CHECK(ha_backend_count(nullptr) == 0);
    HaBackendInfo dummy_info;
    AGR_CHECK(ha_backend_get(nullptr, 0, &dummy_info) == HA_ERR_INVALID_ARG);

    HaRuntime rt = ha_runtime_create();
    AGR_CHECK(rt != nullptr);

    // Test 2: Backend enumeration bounds checking
    int count = ha_backend_count(rt);
    std::cout << "  Registered backend count: " << count << "\n";
    AGR_CHECK(count >= 1);

    AGR_CHECK(ha_backend_get(rt, -1, &dummy_info) == HA_ERR_INVALID_ARG);
    AGR_CHECK(ha_backend_get(rt, count, &dummy_info) == HA_ERR_INVALID_ARG);
    AGR_CHECK(ha_backend_get(rt, count + 10, &dummy_info) == HA_ERR_INVALID_ARG);
    AGR_CHECK(ha_backend_get(rt, 0, nullptr) == HA_ERR_INVALID_ARG);

    // Test 3: Inspect each backend
    bool found_cpu = false;
    for (int i = 0; i < count; ++i) {
        HaBackendInfo info = {};
        HaError err = ha_backend_get(rt, i, &info);
        AGR_CHECK(err == HA_OK);
        AGR_CHECK(info.backend_id != nullptr);
        AGR_CHECK(std::strlen(info.backend_id) > 0);

        std::cout << "  Backend [" << i << "]: " << info.backend_id
                  << " (available=" << info.available
                  << ", vulkan=" << info.requires_vulkan
                  << ", cuda=" << info.requires_cuda << ")\n";

        if (std::strcmp(info.backend_id, "hydroxl_cpu") == 0) {
            found_cpu = true;
            AGR_CHECK(info.available == 1);
            AGR_CHECK(info.requires_vulkan == 0);
            AGR_CHECK(info.requires_cuda == 0);
        }
    }
    AGR_CHECK(found_cpu);

    // Test 4: C++ SDK Backend query
    {
        ha::Runtime cpp_rt;
        auto backends = cpp_rt.backends();
        AGR_CHECK(backends.size() == static_cast<size_t>(count));

        bool cpp_found_cpu = false;
        for (const auto& b : backends) {
            AGR_CHECK(!b.backend_id.empty());
            if (b.backend_id == "hydroxl_cpu") {
                cpp_found_cpu = true;
                AGR_CHECK(b.available);
                AGR_CHECK(!b.requires_vulkan);
                AGR_CHECK(!b.requires_cuda);
            }
        }
        AGR_CHECK(cpp_found_cpu);

        // Test exception on out of range
        bool threw = false;
        try {
            cpp_rt.backend_info(999);
        } catch (const ha::Exception& e) {
            threw = true;
            AGR_CHECK(e.code() == HA_ERR_INVALID_ARG);
        }
        AGR_CHECK(threw);
    }

    ha_runtime_destroy(rt);
    AGR_TEST_MAIN_END();
}
