// tests/test_p26_cuda_runtime_hw.cpp
// Phase 26 HARDWARE test: device enumeration, allocation/deallocation,
// host<->device transfers, memory cleanup and OOM handling on a real NVIDIA GPU.
// Exits 77 (CTest SKIP) when no CUDA device is present -- such a result means
// hardware validation is PENDING, not passed.
#include "mini_test.h"
#include "cuda/CudaRuntime.h"
#include "backend/CUDAAllocator.h"

#include <cstdint>
#include <numeric>
#include <vector>

int main() {
    std::cout << "== test_p26_cuda_runtime_hw ==\n";
    agr::cuda::Runtime& rt = agr::cuda::Runtime::shared();
    if (!rt.initialize() || !rt.available()) {
        std::cout << "  SKIP (hardware validation PENDING): " << rt.unavailableReason() << "\n";
        return 77;
    }

    // --- Enumeration -----------------------------------------------------------
    AGR_CHECK(rt.deviceCount() >= 1);
    AGR_CHECK(rt.driverVersion() > 0);
    for (const auto& d : rt.devices()) {
        std::cout << "  device " << d.index << ": " << d.name << " cc "
                  << d.compute_major << "." << d.compute_minor << " "
                  << (d.total_memory_bytes >> 20) << " MiB, " << d.multiprocessor_count << " SMs\n";
        AGR_CHECK(!d.name.empty());
        AGR_CHECK(d.compute_major >= 1);
        AGR_CHECK(d.total_memory_bytes > (64u << 20));
        AGR_CHECK(d.multiprocessor_count > 0);
    }

    const int dev = 0;
    const size_t base_allocs = rt.liveAllocations();
    const size_t base_bytes = rt.liveBytes();

    agr::cuda::MemoryInfo before;
    AGR_CHECK(rt.memoryInfo(dev, before));
    AGR_CHECK(before.total_bytes > 0 && before.free_bytes <= before.total_bytes);

    // --- Round-trip transfer with a non-trivial pattern -----------------------
    {
        const size_t n = 1 << 20;   // 1M floats = 4 MiB
        std::vector<float> src(n), dst(n, -1.0f);
        for (size_t i = 0; i < n; ++i) src[i] = static_cast<float>(i % 9973) * 0.25f - 17.5f;

        agr::cuda::DeviceBuffer buf(rt, dev);
        AGR_CHECK(buf.allocate(n * sizeof(float)));
        AGR_CHECK(buf.valid());
        AGR_CHECK(rt.liveAllocations() == base_allocs + 1);
        AGR_CHECK(rt.liveBytes() == base_bytes + n * sizeof(float));
        AGR_CHECK(buf.upload(src.data(), n * sizeof(float)));
        AGR_CHECK(buf.download(dst.data(), n * sizeof(float)));
        AGR_CHECK(std::memcmp(src.data(), dst.data(), n * sizeof(float)) == 0);

        // Partial transfer at an offset
        std::vector<float> patch(1000, 3.5f), back(1000, 0.0f);
        AGR_CHECK(buf.upload(patch.data(), patch.size() * 4, 512 * 4));
        AGR_CHECK(buf.download(back.data(), back.size() * 4, 512 * 4));
        AGR_CHECK(back == patch);

        // Bounds protection
        AGR_CHECK(!buf.upload(patch.data(), patch.size() * 4, n * 4 - 4));
        AGR_CHECK(!buf.download(back.data(), back.size() * 4, n * 4));
        AGR_CHECK(!buf.upload(patch.data(), SIZE_MAX, 8));          // overflow-safe

        // memset
        AGR_CHECK(buf.fill(0xAB));
        std::vector<uint8_t> bytes(256);
        AGR_CHECK(buf.download(bytes.data(), bytes.size()));
        bool all_ab = true;
        for (auto b : bytes) all_ab = all_ab && (b == 0xAB);
        AGR_CHECK(all_ab);
    }

    // --- Deallocation returns memory and clears accounting -------------------
    AGR_CHECK(rt.liveAllocations() == base_allocs);
    AGR_CHECK(rt.liveBytes() == base_bytes);

    // --- Many alloc/free cycles do not leak -------------------------------------
    for (int i = 0; i < 64; ++i) {
        agr::cuda::DeviceBuffer b(rt, dev);
        AGR_CHECK(b.allocate(1 << 20));
    }
    AGR_CHECK(rt.liveAllocations() == base_allocs);
    agr::cuda::MemoryInfo after;
    AGR_CHECK(rt.memoryInfo(dev, after));
    // Free memory must be within 64 MiB of the start (driver bookkeeping noise).
    const size_t diff = after.free_bytes > before.free_bytes ? after.free_bytes - before.free_bytes
                                                              : before.free_bytes - after.free_bytes;
    AGR_CHECK(diff < (64u << 20));

    // --- Out-of-memory is an error, not a crash --------------------------------
    {
        agr::cuda::DeviceBuffer huge(rt, dev);
        AGR_CHECK(!huge.allocate(before.total_bytes * 4));
        AGR_CHECK(!huge.lastError().empty());
        AGR_CHECK(!huge.valid());
        // Runtime still usable after the failure
        agr::cuda::DeviceBuffer ok(rt, dev);
        AGR_CHECK(ok.allocate(1 << 16));
    }

    // --- Invalid device index ------------------------------------------------
    {
        uint64_t p = 0; std::string err;
        AGR_CHECK(!rt.allocate(9999, 1024, p, &err));
        AGR_CHECK(!err.empty());
    }

    // --- Allocator interface ------------------------------------------------------
    {
        agr::CUDAAllocator alloc(dev);
        AGR_CHECK(alloc.cudaAvailable());
        AGR_CHECK(alloc.capacity() > 0);
        agr::MemoryBlock b = alloc.allocate(1 << 20);
        AGR_CHECK(b.isValid());
        AGR_CHECK(b.location == agr::MemoryLocation::GPU);
        AGR_CHECK(alloc.used() == (1u << 20));
        std::vector<uint8_t> src(4096), dst(4096, 0);
        std::iota(src.begin(), src.end(), uint8_t(0));
        AGR_CHECK(alloc.upload(b, src.data(), src.size()));
        AGR_CHECK(alloc.download(b, dst.data(), dst.size()));
        AGR_CHECK(src == dst);
        AGR_CHECK(!alloc.upload(b, src.data(), src.size(), b.size));   // OOB
        agr::MemoryBlock copy = b;
        alloc.free(b);
        AGR_CHECK(!b.isValid());
        AGR_CHECK(alloc.used() == 0);
        alloc.free(copy);                                              // double free is safe
        AGR_CHECK(alloc.used() == 0);
    }
    AGR_CHECK(rt.liveAllocations() == base_allocs);

    AGR_TEST_MAIN_END();
}
