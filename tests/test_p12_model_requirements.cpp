#include "analysis/ModelRequirements.h"
#include <cassert>
#include <iostream>

using namespace agr;

int main() {
    ModelRequirements req;
    
    assert(req.format == ModelFormat::UNKNOWN);
    assert(req.quantization == QuantizationType::UNKNOWN);
    assert(req.modality == ModelModality::UNKNOWN);
    
    req.format = ModelFormat::GGUF;
    req.architecture = "llama";
    req.parameter_count = 7000000000ULL;
    req.quantization = QuantizationType::Q4;
    req.modality = ModelModality::TEXT;
    req.required_capabilities.add(EngineCapability::TEXT_GENERATION);
    
    assert(req.format == ModelFormat::GGUF);
    assert(req.architecture.value() == "llama");
    assert(req.parameter_count.value() == 7000000000ULL);
    assert(req.quantization == QuantizationType::Q4);
    assert(req.required_capabilities.has(EngineCapability::TEXT_GENERATION));
    
    std::cout << "test_p12_model_requirements passed\n";
    return 0;
}
