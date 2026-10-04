#include "analysis/GGUFInspector.h"
#include <cassert>
#include <iostream>
#include <cstdlib>

using namespace agr;

int main() {
    const char* model_path_env = std::getenv("HETEROACCEL_MODEL_PATH");
    if (!model_path_env) {
        std::cout << "SKIP: HETEROACCEL_MODEL_PATH not set.\n";
        return 77;
    }
    
    std::string model_path(model_path_env);
    
    GGUFInspector inspector;
    ModelRequirements req = inspector.inspect(model_path);
    
    assert(req.format == ModelFormat::GGUF);
    assert(req.modality == ModelModality::TEXT);
    assert(req.required_capabilities.has(EngineCapability::TEXT_GENERATION));
    
    assert(req.file_size_bytes.has_value());
    assert(req.file_size_bytes.value() > 0);
    
    assert(req.estimated_memory_bytes.has_value());
    assert(req.estimated_memory_bytes.value() > req.file_size_bytes.value()); // working memory adds overhead
    
    // Qwen2.5-0.5B-Instruct-Q4_K_M.gguf specifics
    if (model_path.find("Qwen2") != std::string::npos || model_path.find("qwen2") != std::string::npos) {
        assert(req.architecture.has_value());
        assert(req.architecture.value() == "qwen2");
        assert(req.quantization == QuantizationType::Q4);
        assert(req.parameter_count.has_value());
        assert(req.layer_count.has_value());
        assert(req.context_length.has_value());
    }
    
    std::cout << "test_p12_gguf_inspector passed\n";
    return 0;
}
