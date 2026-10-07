// tests/test_p14_real_gguf.cpp
// Phase 14 Test 12: Real GGUF file parsing for tensor regions using GGUFRegionExtractor.
#include "model/GGUFRegionExtractor.h"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main() {
    const char* env_path = std::getenv("HETEROACCEL_MODEL_PATH");
    if (!env_path) {
        std::cout << "SKIP: HETEROACCEL_MODEL_PATH not set.\n";
        return 77; // ctest skip code
    }

    std::string path(env_path);
    if (path.empty()) {
        std::cout << "SKIP: HETEROACCEL_MODEL_PATH is empty.\n";
        return 77;
    }

    agr::GGUFRegionExtractor extractor;
    auto regions = extractor.extract(path, 100);

    if (regions.empty()) {
        std::cerr << "FAIL: extract() returned empty for: " << path << "\n";
        std::cerr << "Error: " << extractor.lastError() << "\n";
        return 1;
    }

    std::cout << "Extracted " << regions.size() << " regions from GGUF.\n";

    // Verify properties of the regions
    bool found_tensor = false;
    for (const auto& r : regions) {
        assert(r.id >= 100);
        assert(r.size > 0);
        assert(!r.identifier.empty());
        if (r.kind == agr::ModelRegionKind::TENSOR) {
            found_tensor = true;
        }
    }

    // Depending on llama.cpp version, we either get TENSORs or one GENERIC fallback.
    if (found_tensor) {
        std::cout << "LLAMA_HAS_TENSOR_METADATA is active (found TENSOR regions).\n";
    } else {
        std::cout << "LLAMA_HAS_TENSOR_METADATA is inactive (used fallback whole-file region).\n";
    }

    std::cout << "test_p14_real_gguf PASSED\n";
    return 0;
}
