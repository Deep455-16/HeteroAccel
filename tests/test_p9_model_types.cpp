// tests/test_p9_model_types.cpp
// Phase 9: Verify all new ModelTypes enumerations, ResourceType, WorkloadClass, ExecutionStrategy.
#include "mini_test.h"
#include "model/ModelTypes.h"
#include <string>

using namespace agr;

int main() {
    std::cout << "== test_p9_model_types ==\n";

    // 1. ResourceType round-trip
    AGR_CHECK(std::string(toString(ResourceType::GENERIC))        == "GENERIC");
    AGR_CHECK(std::string(toString(ResourceType::TEXT_WEIGHTS))   == "TEXT_WEIGHTS");
    AGR_CHECK(std::string(toString(ResourceType::VISION_ENCODER)) == "VISION_ENCODER");
    AGR_CHECK(std::string(toString(ResourceType::AUDIO_ENCODER))  == "AUDIO_ENCODER");
    AGR_CHECK(std::string(toString(ResourceType::PROJECTION))     == "PROJECTION");
    AGR_CHECK(std::string(toString(ResourceType::DECODER))        == "DECODER");
    AGR_CHECK(std::string(toString(ResourceType::EMBEDDINGS))     == "EMBEDDINGS");
    AGR_CHECK(std::string(toString(ResourceType::KV_CACHE))       == "KV_CACHE");
    std::cout << "  ok: ResourceType all 8 values\n";

    // 2. WorkloadClass round-trip
    AGR_CHECK(std::string(toString(WorkloadClass::DEFAULT))       == "DEFAULT");
    AGR_CHECK(std::string(toString(WorkloadClass::LOW_LATENCY))   == "LOW_LATENCY");
    AGR_CHECK(std::string(toString(WorkloadClass::THROUGHPUT))    == "THROUGHPUT");
    AGR_CHECK(std::string(toString(WorkloadClass::MEMORY_BOUND))  == "MEMORY_BOUND");
    AGR_CHECK(std::string(toString(WorkloadClass::COMPUTE_BOUND)) == "COMPUTE_BOUND");
    AGR_CHECK(std::string(toString(WorkloadClass::STREAMING))     == "STREAMING");
    AGR_CHECK(std::string(toString(WorkloadClass::BACKGROUND))    == "BACKGROUND");
    AGR_CHECK(std::string(toString(WorkloadClass::INTERACTIVE))   == "INTERACTIVE");
    AGR_CHECK(std::string(toString(WorkloadClass::BATCH))         == "BATCH");
    std::cout << "  ok: WorkloadClass all 9 values\n";

    // 3. ExecutionStrategy round-trip
    AGR_CHECK(std::string(toString(ExecutionStrategy::AUTO))                == "AUTO");
    AGR_CHECK(std::string(toString(ExecutionStrategy::FULL_RESIDENT))       == "FULL_RESIDENT");
    AGR_CHECK(std::string(toString(ExecutionStrategy::PARTIAL_RESIDENT))    == "PARTIAL_RESIDENT");
    AGR_CHECK(std::string(toString(ExecutionStrategy::STREAMING))           == "STREAMING");
    AGR_CHECK(std::string(toString(ExecutionStrategy::MEMORY_PRESSURE))     == "MEMORY_PRESSURE");
    AGR_CHECK(std::string(toString(ExecutionStrategy::CPU_FALLBACK))        == "CPU_FALLBACK");
    AGR_CHECK(std::string(toString(ExecutionStrategy::ACCELERATOR_OFFLOAD)) == "ACCELERATOR_OFFLOAD");
    AGR_CHECK(std::string(toString(ExecutionStrategy::HYBRID))              == "HYBRID");
    std::cout << "  ok: ExecutionStrategy all 8 values\n";

    // 4. ModelResource has type field with default GENERIC
    ModelResource res;
    AGR_CHECK(res.type == ResourceType::GENERIC);
    res.type = ResourceType::VISION_ENCODER;
    AGR_CHECK(res.type == ResourceType::VISION_ENCODER);
    std::cout << "  ok: ModelResource.type field works\n";

    // 5. Residency hierarchy completeness
    AGR_CHECK(std::string(toString(ResourceResidency::DISK))     == "DISK");
    AGR_CHECK(std::string(toString(ResourceResidency::COLD))     == "COLD");
    AGR_CHECK(std::string(toString(ResourceResidency::WARM))     == "WARM");
    AGR_CHECK(std::string(toString(ResourceResidency::HOT))      == "HOT");
    AGR_CHECK(std::string(toString(ResourceResidency::LOADING))  == "LOADING");
    AGR_CHECK(std::string(toString(ResourceResidency::EVICTING)) == "EVICTING");
    std::cout << "  ok: ResourceResidency hierarchy complete\n";

    AGR_TEST_MAIN_END();
}
