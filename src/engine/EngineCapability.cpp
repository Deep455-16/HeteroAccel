// src/engine/EngineCapability.cpp
#include "engine/EngineCapability.h"

namespace agr {

std::string EngineCapabilitySet::toString() const {
    std::string s;
    auto check = [&](EngineCapability c, const char* name) {
        if (has(c)) {
            if (!s.empty()) s += '|';
            s += name;
        }
    };
    check(EngineCapability::TEXT_GENERATION, "TEXT_GENERATION");
    check(EngineCapability::EMBEDDINGS,      "EMBEDDINGS");
    check(EngineCapability::VISION,          "VISION");
    check(EngineCapability::AUDIO,           "AUDIO");
    check(EngineCapability::MULTIMODAL,      "MULTIMODAL");
    check(EngineCapability::BACKEND_CPU,     "BACKEND_CPU");
    check(EngineCapability::BACKEND_VULKAN,  "BACKEND_VULKAN");
    check(EngineCapability::BACKEND_CUDA,    "BACKEND_CUDA");
    check(EngineCapability::BACKEND_NPU,     "BACKEND_NPU");
    check(EngineCapability::STREAMING,       "STREAMING");
    check(EngineCapability::BATCHING,        "BATCHING");
    check(EngineCapability::CANCELLATION,    "CANCELLATION");
    check(EngineCapability::DYNAMIC_LOADING, "DYNAMIC_LOADING");
    if (s.empty()) s = "(none)";
    return s;
}

} // namespace agr
