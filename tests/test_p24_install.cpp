// tests/test_p24_install.cpp
// Phase 24: CMake Packaging, Installation, and Header Verification Tests

#include "mini_test.h"
#include <heteroaccel/heteroaccel.h>
#include <heteroaccel/HeteroAccel.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

static bool file_contains(const std::filesystem::path& path, const std::string& pattern) {
    if (!std::filesystem::exists(path)) return false;
    std::ifstream ifs(path);
    if (!ifs.is_open()) return false;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.find(pattern) != std::string::npos) {
            return true;
        }
    }
    return false;
}

int main() {
    std::cout << "--- Phase 24: CMake Packaging & Installation Tests ---\n";

    // Test 1: Header files presence and content
    std::filesystem::path inc_dir = "include/heteroaccel";
    std::filesystem::path c_header = inc_dir / "heteroaccel.h";
    std::filesystem::path cpp_header = inc_dir / "HeteroAccel.hpp";

    AGR_CHECK(std::filesystem::exists(c_header));
    AGR_CHECK(std::filesystem::exists(cpp_header));

    AGR_CHECK(file_contains(c_header, "HETEROACCEL_VERSION_MINOR 24"));
    AGR_CHECK(file_contains(c_header, "HaMemoryEstimate"));
    AGR_CHECK(file_contains(c_header, "ha_model_memory_estimate"));
    AGR_CHECK(file_contains(c_header, "ha_infer_stream"));
    AGR_CHECK(file_contains(c_header, "ha_session_cancel"));
    AGR_CHECK(file_contains(c_header, "ha_runtime_perf_reset"));
    AGR_CHECK(file_contains(c_header, "ha_runtime_perf_path"));

    AGR_CHECK(file_contains(cpp_header, "struct MemoryEstimate"));
    AGR_CHECK(file_contains(cpp_header, "infer_stream"));
    AGR_CHECK(file_contains(cpp_header, "perf_reset"));
    AGR_CHECK(file_contains(cpp_header, "perf_path"));

    // Test 2: Package templates
    std::filesystem::path config_in = "cmake/HeteroAccelConfig.cmake.in";
    std::filesystem::path pc_in = "cmake/heteroaccel.pc.in";

    AGR_CHECK(std::filesystem::exists(config_in));
    AGR_CHECK(std::filesystem::exists(pc_in));
    AGR_CHECK(file_contains(config_in, "HeteroAccelTargets.cmake"));
    AGR_CHECK(file_contains(pc_in, "-lheteroaccel"));

    // Test 3: Generated packaging artifacts (in build directory)
    std::vector<std::filesystem::path> search_dirs = {
        ".",
        "build",
        "../build"
    };

    bool found_pc = false;
    bool found_cmake_cfg = false;
    for (const auto& d : search_dirs) {
        if (file_contains(d / "heteroaccel.pc", "0.24.0")) {
            found_pc = true;
        }
        if (file_contains(d / "HeteroAccelConfig.cmake", "HeteroAccelTargets")) {
            found_cmake_cfg = true;
        }
    }

    AGR_CHECK(found_pc);
    AGR_CHECK(found_cmake_cfg);

    // Test 4: External consumer test files
    std::filesystem::path ext_dir = "tests/external_consumer";
    AGR_CHECK(std::filesystem::exists(ext_dir / "CMakeLists.txt"));
    AGR_CHECK(std::filesystem::exists(ext_dir / "main.c"));
    AGR_CHECK(file_contains(ext_dir / "CMakeLists.txt", "find_package(HeteroAccel"));
    AGR_CHECK(file_contains(ext_dir / "main.c", "ha_version_string"));

    AGR_TEST_MAIN_END();
}
