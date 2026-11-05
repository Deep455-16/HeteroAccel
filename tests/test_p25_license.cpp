// tests/test_p25_license.cpp
// Phase 25: Verify LICENSE file exists and contains MIT text

#include "mini_test.h"
#include <fstream>
#include <string>
#include <iostream>
#include <cstring>

// CMake passes the source directory at compile time
#ifndef HETEROACCEL_SOURCE_DIR
#  define HETEROACCEL_SOURCE_DIR "."
#endif

int main() {
    std::cout << "--- Phase 25: LICENSE File Check ---\n";

    std::string license_path = std::string(HETEROACCEL_SOURCE_DIR) + "/LICENSE";
    std::ifstream f(license_path);
    AGR_CHECK(f.good());   // file must exist and be readable

    if (!f.good()) {
        std::cerr << "  Looked for LICENSE at: " << license_path << "\n";
        AGR_TEST_MAIN_END();
    }

    std::string first_line;
    std::getline(f, first_line);
    AGR_CHECK(!first_line.empty());
    // Must be MIT
    AGR_CHECK(first_line.find("MIT") != std::string::npos);
    std::cout << "  First line: " << first_line << "\n";

    // Scan for "Permission is hereby granted" — canonical MIT body
    bool found_body = false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.find("Permission is hereby granted") != std::string::npos) {
            found_body = true;
            break;
        }
    }
    AGR_CHECK(found_body);

    std::cout << "  LICENSE path: " << license_path << "\n";
    AGR_TEST_MAIN_END();
}
