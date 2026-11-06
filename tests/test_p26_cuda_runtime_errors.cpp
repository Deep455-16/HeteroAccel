// tests/test_p26_cuda_runtime_errors.cpp
// Phase 26: CUDA runtime layer behaviour that must hold on EVERY machine,
// including ones with no NVIDIA driver. Hardware-dependent checks live in
// test_p26_cuda_runtime_hw.
#include "mini_test.h"
#include "cuda/CudaRuntime.h"
#include "backend/CUDAAllocator.h"
#include "hardware/CUDADetector.h"
#include "platform/DynamicLibrary.h"

#include <vector>

int main() {
    std::cout << "== test_p26_cuda_runtime_errors ==\n";

    // --- Dynamic library loader: failure is reported, never thrown ---------
    {
        agr::platform::DynamicLibrary lib;
        std::string err;
        AGR_CHECK(!lib.open({"definitely_not_a_real_library_heteroaccel.dll",
                             "libdefinitely_not_a_real_library_heteroaccel.so"}, &err));
        AGR_CHECK(!err.empty());
        AGR_CHECK(!lib.isOpen());
        AGR_CHECK(lib.symbol("anything") == nullptr);
        AGR_CHECK(!lib.open({}, &err));
    }

    // --- Runtime with no driver ----------------------------------------------
    agr::cuda::Runtime rt({"definitely_not_nvcuda_heteroaccel.dll", "libdefinitely_not_cuda_heteroaccel.so"});
    AGR_CHECK(!rt.initialize());
    AGR_CHECK(!rt.available());
    AGR_CHECK(!rt.unavailableReason().empty());
    AGR_CHECK(rt.deviceCount() == 0);
    AGR_CHECK(rt.initialize() == false);   // idempotent

    uint64_t p = 123;
    std::string err;
    AGR_CHECK(!rt.allocate(0, 1024, p, &err));
    AGR_CHECK(p == 0);
    AGR_CHECK(!err.empty());
    AGR_CHECK(!rt.allocate(0, 0, p, &err));            // zero-size always rejected
    agr::cuda::MemoryInfo mi;
    AGR_CHECK(!rt.memoryInfo(0, mi, &err));
    char host[16] = {0};
    AGR_CHECK(!rt.copyHostToDevice(0, 0x1000, host, sizeof(host), &err));
    AGR_CHECK(!rt.copyDeviceToHost(0, host, 0x1000, sizeof(host), &err));
    AGR_CHECK(!rt.release(0, 0xdeadbeef, &err));       // unknown pointer
    AGR_CHECK(rt.release(0, 0, &err));                 // null free is a no-op
    AGR_CHECK(rt.liveAllocations() == 0 && rt.liveBytes() == 0);

    // --- DeviceBuffer error paths ---------------------------------------------
    {
        agr::cuda::DeviceBuffer buf(rt, 0);
        AGR_CHECK(!buf.allocate(4096));
        AGR_CHECK(!buf.valid());
        AGR_CHECK(!buf.lastError().empty());
        AGR_CHECK(!buf.upload(host, sizeof(host)));
        AGR_CHECK(!buf.download(host, sizeof(host)));
        agr::cuda::DeviceBuffer none;
        AGR_CHECK(!none.allocate(16));                 // no runtime bound
    }

    // --- CUDAAllocator over an unavailable runtime ---------------------------
    {
        agr::CUDAAllocator alloc(0, rt);
        AGR_CHECK(!alloc.cudaAvailable());
        agr::MemoryBlock b = alloc.allocate(1 << 20);
        AGR_CHECK(!b.isValid());
        AGR_CHECK(!alloc.lastError().empty());
        AGR_CHECK(alloc.used() == 0);
        AGR_CHECK(alloc.capacity() == 0);
        AGR_CHECK(alloc.available() == 0);
        alloc.free(b);                                 // freeing an invalid block is safe
        AGR_CHECK(alloc.name().find("unavailable") != std::string::npos);
    }

    // --- Detector on the real machine: must not crash, must explain ----------
    {
        agr::CUDAInfo info = agr::CUDADetector::detect();
        if (!info.available) {
            AGR_CHECK(!info.unavailable_reason.empty());
        } else {
            AGR_CHECK(!info.devices.empty());
            for (const auto& d : info.devices) {
                AGR_CHECK(!d.name.empty());
                AGR_CHECK(d.compute_major > 0);
                AGR_CHECK(d.total_memory_bytes > 0);
            }
        }
    }

    AGR_TEST_MAIN_END();
}
