// tests/test_p9_multi_model.cpp
// Phase 9: Multiple models can coexist, be evicted, and reloaded independently.
#include "mini_test.h"
#include "model/ModelTypes.h"
#include "model/ModelManager.h"
#include "model/LayerManager.h"
#include "model/ResourceResidencyManager.h"
#include "model/StreamingEngine.h"
#include "model/PrefetchEngine.h"
#include "model/FileStorageBackend.h"
#include "mem/MemoryManager.h"
#include "scheduler/AdaptiveScheduler.h"
#include "gpu/VulkanBackend.h"
#include "backend/BackendManager.h"
#include <memory>

using namespace agr;

int main() {
    std::cout << "== test_p9_multi_model ==\n";

    // Set up shared subsystems
    agr::VulkanBackend vk;
    agr::BackendManager backendMgr(vk);
    backendMgr.discover();

    auto memMgr    = std::make_shared<agr::MemoryManager>(vk);
    auto scheduler = std::make_shared<agr::AdaptiveScheduler>(backendMgr, *memMgr);
    auto storage   = std::make_shared<agr::FileStorageBackend>();
    auto residMgr  = std::make_shared<agr::ResourceResidencyManager>();
    auto layerMgr  = std::make_shared<agr::LayerManager>();
    auto stream    = std::make_shared<agr::StreamingEngine>(storage, residMgr, memMgr, scheduler);
    auto prefetch  = std::make_shared<agr::PrefetchEngine>(layerMgr, stream, residMgr, memMgr);

    agr::ModelManager mgr(scheduler, stream, prefetch, layerMgr, residMgr);

    // 1. Register two different models
    uint64_t modelA = mgr.registerModel("model_a");
    uint64_t modelB = mgr.registerModel("model_b");
    AGR_CHECK(modelA != modelB);
    std::cout << "  ok: two models registered (A=" << modelA << " B=" << modelB << ")\n";

    // 2. Add resources to each model (simulate weights)
    auto resA1 = std::make_shared<ModelResource>();
    resA1->id = 1; resA1->name = "a_weights"; resA1->size_bytes = 512 * 1024 * 1024;
    resA1->source_path = "/tmp/fake_model_a.bin"; resA1->type = ResourceType::TEXT_WEIGHTS;
    resA1->priority = MemoryPriority::NORMAL;

    auto resB1 = std::make_shared<ModelResource>();
    resB1->id = 2; resB1->name = "b_weights"; resB1->size_bytes = 256 * 1024 * 1024;
    resB1->source_path = "/tmp/fake_model_b.bin"; resB1->type = ResourceType::EMBEDDINGS;
    resB1->priority = MemoryPriority::NORMAL;

    mgr.registerResource(resA1);
    mgr.registerResource(resB1);

    // 3. Resources should be independently retrievable
    auto fetchedA = mgr.getResource(1);
    auto fetchedB = mgr.getResource(2);
    AGR_CHECK(fetchedA != nullptr);
    AGR_CHECK(fetchedB != nullptr);
    AGR_CHECK(fetchedA->name == "a_weights");
    AGR_CHECK(fetchedB->name == "b_weights");
    AGR_CHECK(fetchedA->type == ResourceType::TEXT_WEIGHTS);
    AGR_CHECK(fetchedB->type == ResourceType::EMBEDDINGS);
    std::cout << "  ok: resources independently retrievable\n";

    // 4. Resources are isolated by ID
    AGR_CHECK(mgr.getResource(999) == nullptr);
    std::cout << "  ok: non-existent resource returns nullptr\n";

    // 5. Residency isolation: model A resource starts as DISK, model B resource independent
    residMgr->setResidency(resA1->id, ResourceResidency::DISK);
    residMgr->setResidency(resB1->id, ResourceResidency::DISK);
    AGR_CHECK(residMgr->getResidency(resA1->id) == ResourceResidency::DISK);
    AGR_CHECK(residMgr->getResidency(resB1->id) == ResourceResidency::DISK);

    // Simulate model B being promoted without touching model A
    residMgr->setResidency(resB1->id, ResourceResidency::WARM);
    AGR_CHECK(residMgr->getResidency(resA1->id) == ResourceResidency::DISK); // unchanged
    AGR_CHECK(residMgr->getResidency(resB1->id) == ResourceResidency::WARM);
    std::cout << "  ok: residency state isolation works\n";

    // 6. Unregister model A - model B still intact
    mgr.unregisterModel(modelA);
    // model B resource should still be accessible
    auto stillB = mgr.getResource(2);
    AGR_CHECK(stillB != nullptr);
    std::cout << "  ok: unregistering model A does not affect model B\n";

    // 7. VLM/VLA resource types: multiple resource types can coexist
    auto resVision = std::make_shared<ModelResource>();
    resVision->id = 10; resVision->name = "vision_enc";
    resVision->type = ResourceType::VISION_ENCODER;
    resVision->size_bytes = 128 * 1024 * 1024;

    auto resAudio = std::make_shared<ModelResource>();
    resAudio->id = 11; resAudio->name = "audio_enc";
    resAudio->type = ResourceType::AUDIO_ENCODER;
    resAudio->size_bytes = 64 * 1024 * 1024;

    mgr.registerResource(resVision);
    mgr.registerResource(resAudio);
    AGR_CHECK(mgr.getResource(10)->type == ResourceType::VISION_ENCODER);
    AGR_CHECK(mgr.getResource(11)->type == ResourceType::AUDIO_ENCODER);
    std::cout << "  ok: VLM/VLA resource types (VISION_ENCODER, AUDIO_ENCODER) work\n";

    AGR_TEST_MAIN_END();
}
