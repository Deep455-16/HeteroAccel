// tests/test_mem_gpu_to_cpu_transfer.cpp
#include "mini_test.h"
#include "gpu/VulkanBackend.h"
#include "mem/VulkanAllocator.h"
#include "mem/TransferManager.h"
#include <vector>

int main() {
    std::cout << "== test_mem_gpu_to_cpu_transfer ==\n";
    
    agr::VulkanBackend backend;
    if (!backend.initialize()) return 77;
    
    agr::VulkanAllocator alloc(backend);
    agr::TransferManager tm(backend);
    
    size_t count = 1024;
    size_t bytes = count * sizeof(float);
    std::vector<float> dataOut(count);
    for (size_t i = 0; i < count; ++i) dataOut[i] = static_cast<float>(i);
    
    agr::MemoryBlock gpuBlock = alloc.allocate(bytes);
    tm.upload(dataOut.data(), gpuBlock, bytes);
    
    std::vector<float> dataIn(count, 0.0f);
    agr::TransferHandle th = tm.download(gpuBlock, dataIn.data(), bytes);
    
    AGR_CHECK(th.valid());
    AGR_CHECK(dataIn[123] == 123.0f);
    AGR_CHECK(tm.totalBytesDownloaded() == bytes);
    
    alloc.free(gpuBlock);
    AGR_TEST_MAIN_END();
}
