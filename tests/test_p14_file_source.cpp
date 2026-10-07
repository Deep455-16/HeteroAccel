// tests/test_p14_file_source.cpp
// Phase 14 Test 1: FileModelDataSource — real file-backed random-access reads.
#include "model/FileModelDataSource.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

static std::string tempPath() {
#ifdef _WIN32
    char buf[MAX_PATH];
    GetTempPathA(MAX_PATH, buf);
    return std::string(buf) + "agr_test_source.bin";
#else
    return "/tmp/agr_test_source.bin";
#endif
}

int main() {
    // Create a temporary test file with known content.
    const std::string path = tempPath();
    const size_t FILE_SIZE = 4096;
    std::vector<uint8_t> content(FILE_SIZE);
    for (size_t i = 0; i < FILE_SIZE; ++i)
        content[i] = static_cast<uint8_t>(i & 0xFF);

    {
        FILE* f = nullptr;
#ifdef _WIN32
        fopen_s(&f, path.c_str(), "wb");
#else
        f = fopen(path.c_str(), "wb");
#endif
        assert(f);
        fwrite(content.data(), 1, FILE_SIZE, f);
        fclose(f);
    }

    agr::FileModelDataSource src(path);
    assert(src.open());
    assert(src.isOpen());
    assert(src.size() == FILE_SIZE);

    // Read first 16 bytes.
    uint8_t buf[64] = {};
    assert(src.read(0, buf, 16));
    for (int i = 0; i < 16; ++i)
        assert(buf[i] == content[i]);

    // Read from middle.
    assert(src.read(256, buf, 32));
    for (int i = 0; i < 32; ++i)
        assert(buf[i] == content[256 + i]);

    // Read last byte.
    assert(src.read(FILE_SIZE - 1, buf, 1));
    assert(buf[0] == content[FILE_SIZE - 1]);

    // Async read.
    uint8_t abuf[8] = {};
    auto fut = src.readAsync(100, abuf, 8);
    assert(fut.get());
    for (int i = 0; i < 8; ++i)
        assert(abuf[i] == content[100 + i]);

    // description
    assert(!src.description().empty());

    src.close();
    std::remove(path.c_str());

    std::cout << "test_p14_file_source PASSED\n";
    return 0;
}
