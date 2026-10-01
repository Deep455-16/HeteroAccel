#include "mini_test.h"
#include "model/ModelManager.h"
#include "model/FileStorageBackend.h"

int main() {
    std::cout << "== test_model_manager ==\n";
    agr::VulkanBackend vk;
    auto backendMgr = std::make_shared<agr::BackendManager>(vk);
    backendMgr->discover();
    auto memMgr = std::make_shared<agr::MemoryManager>(vk);
    auto scheduler = std::make_shared<agr::AdaptiveScheduler>(*backendMgr, *memMgr);

    auto storage = std::make_shared<agr::FileStorageBackend>();
    auto residencyMgr = std::make_shared<agr::ResourceResidencyManager>();
    auto streamingEngine = std::make_shared<agr::StreamingEngine>(storage, residencyMgr, memMgr, scheduler);
    auto layerMgr = std::make_shared<agr::LayerManager>();
    auto prefetchEngine = std::make_shared<agr::PrefetchEngine>(layerMgr, streamingEngine, residencyMgr, memMgr);

    agr::ModelManager modelMgr(scheduler, streamingEngine, prefetchEngine, layerMgr, residencyMgr);

    uint64_t modelId = modelMgr.registerModel("synthetic-test-model");
    AGR_CHECK(modelId > 0);

    auto res = std::make_shared<agr::ModelResource>();
    res->id = 101;
    res->name = "weights.bin";
    res->size_bytes = 1024 * 1024; // 1 MB
    modelMgr.registerResource(res);

    auto retrieved = modelMgr.getResource(101);
    AGR_CHECK(retrieved != nullptr);
    AGR_CHECK(retrieved->name == "weights.bin");

    AGR_CHECK(residencyMgr->getResidency(101) == agr::ResourceResidency::DISK);

    AGR_TEST_MAIN_END();
}
