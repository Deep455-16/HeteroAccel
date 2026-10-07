// tests/test_p14_ram_residency.cpp
// Phase 14 Test 3: DISK → RAM residency — correct bytes copied.
#include "model/FileModelDataSource.h"
#include "model/HostAcceleratorMemory.h"
#include "model/ModelResidencyManager.h"
#include "model/ModelRegion.h"

#include <cassert>
#include <cstdio>
#include <cstring>
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
    return std::string(buf) + "agr_test_ram.bin";
#else
    return "/tmp/agr_test_ram.bin";
#endif
}

int main() {
    const std::string path = tempPath();
    const size_t FILE_SIZE = 1024;
    std::vector<uint8_t> data(FILE_SIZE);
    for (size_t i = 0; i < FILE_SIZE; ++i)
        data[i] = static_cast<uint8_t>((i * 7 + 3) & 0xFF);

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

    agr::ModelResidencyBudget budget;
    budget.ram_bytes = 8 * 1024 * 1024;  // 8 MB
    budget.accelerator_bytes = 1 * 1024 * 1024;

    agr::ModelResidencyManager mgr(src, budget);

    // Define a region covering bytes 128..255.
    agr::ModelRegion region;
    region.id         = 42;
    region.offset     = 128;
    region.size       = 128;
    region.identifier = "test_region_128";
    region.kind       = agr::ModelRegionKind::GENERIC;

    // Not resident yet.
    assert(mgr.state(42) == agr::ModelResidencyState::NOT_RESIDENT);

    // Load to RAM.
    assert(mgr.ensureResident(region, agr::ModelResidencyTarget::RAM));

    // Should be RESIDENT_IN_RAM.
    assert(mgr.state(42) == agr::ModelResidencyState::RESIDENT_IN_RAM);
    assert(mgr.isResident(42, agr::ModelResidencyTarget::RAM));
    assert(!mgr.isResident(42, agr::ModelResidencyTarget::ACCELERATOR));

    // Verify bytes match the original file.
    const uint8_t* ptr = mgr.ramPointer(42);
    assert(ptr != nullptr);
    for (size_t i = 0; i < 128; ++i)
        assert(ptr[i] == data[128 + i]);

    // Telemetry should reflect bytes loaded.
    auto telem = mgr.telemetry();
    assert(telem.bytes_read_from_disk >= 128);
    assert(telem.bytes_to_ram >= 128);
    assert(telem.resident_ram_bytes >= 128);

    std::remove(path.c_str());
    std::cout << "test_p14_ram_residency PASSED\n";
    return 0;
}
