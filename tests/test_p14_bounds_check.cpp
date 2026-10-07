// tests/test_p14_bounds_check.cpp
// Phase 14 Test 2: Bounds-checking in FileModelDataSource.
#include "model/FileModelDataSource.h"

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
    return std::string(buf) + "agr_test_bounds.bin";
#else
    return "/tmp/agr_test_bounds.bin";
#endif
}

int main() {
    const std::string path = tempPath();
    const size_t FILE_SIZE = 64;
    std::vector<uint8_t> data(FILE_SIZE, 0xAB);
    {
        FILE* f = nullptr;
#ifdef _WIN32
        fopen_s(&f, path.c_str(), "wb");
#else
        f = fopen(path.c_str(), "wb");
#endif
        assert(f);
        fwrite(data.data(), 1, FILE_SIZE, f);
        fclose(f);
    }

    agr::FileModelDataSource src(path);
    assert(src.open());
    assert(src.size() == FILE_SIZE);

    uint8_t buf[128] = {};

    // Read exactly to the end — should succeed.
    assert(src.read(0, buf, FILE_SIZE));

    // Read 1 byte past end — must fail.
    assert(!src.read(FILE_SIZE, buf, 1));
    assert(!src.lastError().empty());

    // Read that straddles the end — must fail.
    assert(!src.read(60, buf, 10));

    // Read with zero size — must fail (invalid).
    assert(!src.read(0, buf, 0));

    // Read with null destination — must fail.
    assert(!src.read(0, nullptr, 16));

    // Cancellation flag set before read.
    std::atomic<bool> cancel(true);
    assert(!src.read(0, buf, 1, &cancel));
    assert(!src.lastError().empty());

    // Closed source.
    src.close();
    assert(!src.read(0, buf, 1));

    std::remove(path.c_str());

    std::cout << "test_p14_bounds_check PASSED\n";
    return 0;
}
