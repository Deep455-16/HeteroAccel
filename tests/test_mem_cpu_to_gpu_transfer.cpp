// tests/test_mem_cpu_to_gpu_transfer.cpp
#include "mini_test.h"
#include "gpu/VulkanBackend.h"
#include "mem/VulkanAllocator.h"
#include "mem/TransferManager.h"
#include <vector>

int main() {
    std::cout << "== test_mem_cpu_to_gpu_transfer ==\n";
    
    agr::VulkanBackend backend;
    if (!backend.initialize()) {
        std::cout << "  SKIP: Vulkan not available.\n";
        return 77;
    }
    
    agr::VulkanAllocator alloc(backend);
    agr::TransferManager tm(backend);
    
    size_t count = 1024;
    size_t bytes = count * sizeof(float);
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i) data[i] = static_cast<float>(i);
    
    agr::MemoryBlock gpuBlock = alloc.allocate(bytes);
    AGR_CHECK(gpuBlock.isValid());
    
    agr::TransferHandle th = tm.upload(data.data(), gpuBlock, bytes);
    AGR_CHECK(th.valid());
    AGR_CHECK(tm.isComplete(th));
    AGR_CHECK(tm.records().size() == 1);
    AGR_CHECK(tm.records()[0].success == true);
    AGR_CHECK(tm.totalBytesUploaded() == bytes);
    
    alloc.free(gpuBlock);
    
    AGR_TEST_MAIN_END();
}
