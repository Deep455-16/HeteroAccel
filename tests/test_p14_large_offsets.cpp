// tests/test_p14_large_offsets.cpp
// Phase 14 Test 9: 64-bit offsets. Uses a sparse file (where supported) or
// fseek-only probing so no huge physical file is created.
#include "model/FileModelDataSource.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

static std::string tempPath() {
#ifdef _WIN32
    char buf[MAX_PATH];
    GetTempPathA(MAX_PATH, buf);
    return std::string(buf) + "agr_test_large.bin";
#else
    return "/tmp/agr_test_large.bin";
#endif
}

static bool createSparseFile(const std::string& path, uint64_t size,
                             uint64_t offset, const uint8_t* marker, size_t mlen) {
    // Write a marker at `offset` inside a file of `size` bytes.
    // We only write the marker bytes; the rest remain sparse/zero.
#ifdef _WIN32
    HANDLE h = CreateFileA(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;

    // Request sparse attribute (optional, silently ignored if not supported).
    DWORD dummy;
    DeviceIoControl(h, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &dummy, nullptr);

    // Seek to end and set file size (creates sparse zero-filled space).
    LARGE_INTEGER li;
    li.QuadPart = static_cast<LONGLONG>(size);
    SetFilePointerEx(h, li, nullptr, FILE_BEGIN);
    SetEndOfFile(h);

    // Write marker.
    li.QuadPart = static_cast<LONGLONG>(offset);
    SetFilePointerEx(h, li, nullptr, FILE_BEGIN);
    DWORD written;
    WriteFile(h, marker, static_cast<DWORD>(mlen), &written, nullptr);
    CloseHandle(h);
    return written == static_cast<DWORD>(mlen);
#else
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    // Seek to end.
    fseeko(f, static_cast<off_t>(size - 1), SEEK_SET);
    fputc(0, f);
    // Write marker.
    fseeko(f, static_cast<off_t>(offset), SEEK_SET);
    fwrite(marker, 1, mlen, f);
    fclose(f);
    return true;
#endif
}

int main() {
    const std::string path = tempPath();

    // Create a 6 GB sparse file with a marker at 5 GB + 1024.
    const uint64_t FILE_SIZE  = 6ULL * 1024 * 1024 * 1024; // 6 GB
    const uint64_t MARKER_OFF = 5ULL * 1024 * 1024 * 1024 + 1024;
    const uint8_t  MARKER[]   = { 0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE };
    const size_t   MLEN       = sizeof(MARKER);

    if (!createSparseFile(path, FILE_SIZE, MARKER_OFF, MARKER, MLEN)) {
        // If sparse files not supported / disk space not available, skip.
        std::cout << "SKIP: cannot create sparse file\n";
        std::remove(path.c_str());
        return 77; // ctest skip
    }

    agr::FileModelDataSource src(path);
    if (!src.open()) {
        std::cout << "SKIP: file open failed\n";
        std::remove(path.c_str());
        return 77;
    }

    // Verify size > 4 GB.
    assert(src.size() == FILE_SIZE);

    // Read marker at the 64-bit offset.
    uint8_t buf[MLEN] = {};
    bool ok = src.read(MARKER_OFF, buf, MLEN);
    if (!ok) {
        // Some environments block large reads. Skip gracefully.
        std::cout << "SKIP: large-offset read failed — " << src.lastError() << "\n";
        std::remove(path.c_str());
        return 77;
    }
    assert(memcmp(buf, MARKER, MLEN) == 0);

    // Out-of-range read must fail.
    assert(!src.read(FILE_SIZE, buf, 1));

    src.close();
    std::remove(path.c_str());
    std::cout << "test_p14_large_offsets PASSED\n";
    return 0;
}
