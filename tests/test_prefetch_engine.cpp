#include "mini_test.h"
#include "model/PrefetchEngine.h"
#include "model/FileStorageBackend.h"
#include <thread>

int main() {
    std::cout << "== test_prefetch_engine ==\n";
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

    // Set up a simple 2-layer sequence
    auto l1 = std::make_shared<agr::ModelLayer>();
    l1->id = 1; l1->model_id = 10;
    
    auto l2 = std::make_shared<agr::ModelLayer>();
    l2->id = 2; l2->model_id = 10;
    l2->depends_on_layers.push_back(1);
    l2->input_resources.push_back(1002);
    
    layerMgr->registerLayer(l1);
    layerMgr->registerLayer(l2);

    auto r2 = std::make_shared<agr::ModelResource>();
    r2->id = 1002;
    r2->size_bytes = 1024; // Need non-zero size for allocation to succeed
    residencyMgr->setResidency(1002, agr::ResourceResidency::DISK);
    
    auto lookup = [&](uint64_t id) -> std::shared_ptr<agr::ModelResource> {
        if (id == 1002) return r2;
        return nullptr;
    };

    prefetchEngine->setPrefetchDistance(1);
    int requested = prefetchEngine->onLayerRequested(1, lookup);
    
    AGR_CHECK(requested == 1); // Should prefetch r2 for Layer 2
    
    // Wait slightly for async to begin transition
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Check if state is now LOADING (since streamingEngine is async, it might be LOADING or WARM already)
    auto state = residencyMgr->getResidency(1002);
    AGR_CHECK(state == agr::ResourceResidency::LOADING || state == agr::ResourceResidency::WARM);

    AGR_TEST_MAIN_END();
}
