#include "analysis/GGUFInspector.h"
#include "llama.h"
#include <filesystem>
#include <algorithm>
#include <iostream>

namespace agr {

QuantizationType GGUFInspector::inferQuantization(const std::string& path) {
    std::string upper_path = path;
    std::transform(upper_path.begin(), upper_path.end(), upper_path.begin(), ::toupper);

    if (upper_path.find("Q4_K_M") != std::string::npos || upper_path.find("Q4_K_S") != std::string::npos || upper_path.find("Q4_0") != std::string::npos || upper_path.find("Q4_1") != std::string::npos || upper_path.find("Q4") != std::string::npos) {
        return QuantizationType::Q4;
    }
    if (upper_path.find("Q8_0") != std::string::npos || upper_path.find("Q8") != std::string::npos) return QuantizationType::Q8;
    if (upper_path.find("Q6_K") != std::string::npos || upper_path.find("Q6") != std::string::npos) return QuantizationType::Q6;
    if (upper_path.find("Q5_K") != std::string::npos || upper_path.find("Q5") != std::string::npos) return QuantizationType::Q5;
    if (upper_path.find("Q3_K") != std::string::npos || upper_path.find("Q3") != std::string::npos) return QuantizationType::Q3;
    if (upper_path.find("Q2_K") != std::string::npos || upper_path.find("Q2") != std::string::npos) return QuantizationType::Q2;
    if (upper_path.find("F16") != std::string::npos || upper_path.find("FP16") != std::string::npos) return QuantizationType::F16;
    if (upper_path.find("F32") != std::string::npos || upper_path.find("FP32") != std::string::npos) return QuantizationType::F32;
    if (upper_path.find("BF16") != std::string::npos) return QuantizationType::BF16;

    return QuantizationType::UNKNOWN;
}

ModelRequirements GGUFInspector::inspect(const std::string& model_path) {
    ModelRequirements req;
    req.source_path = model_path;
    req.format = ModelFormat::GGUF;
    req.modality = ModelModality::TEXT; // Currently assume text for GGUFs unless we parse vision architecture
    req.required_capabilities.add(EngineCapability::TEXT_GENERATION);
    
    // File size
    std::error_code ec;
    uint64_t file_size = std::filesystem::file_size(model_path, ec);
    if (!ec) {
        req.file_size_bytes = file_size;
    }

    req.quantization = inferQuantization(model_path);

    // Use llama.cpp vocab_only to read metadata without allocating large tensors
    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;
    mparams.use_mmap = false;

    // Supress llama.cpp stderr if possible, or let it print.
    // For Phase 12, we can just load it.
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model) {
        uint64_t params = llama_model_n_params(model);
        if (params > 0) req.parameter_count = params;
        
        uint32_t layers = llama_model_n_layer(model);
        if (layers > 0) req.layer_count = layers;
        
        uint32_t ctx = llama_model_n_ctx_train(model);
        if (ctx > 0) req.context_length = ctx;

        char buf[256];
        int32_t len = llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf));
        if (len > 0) {
            req.architecture = std::string(buf);
            
            // Adjust modality if it's a known vision architecture like llava
            if (req.architecture == "llava" || req.architecture == "clip") {
                req.modality = ModelModality::VISION;
                req.required_capabilities.add(EngineCapability::VISION);
            }
        }
        
        llama_model_free(model);
    }

    // Memory estimation:
    // model_size + conservative context estimate + temporary buffer
    // context: n_ctx * n_layer * 2 * n_embd * sizeof(f16) (approx KV cache)
    // For a generic inspector, if file size is known, we can estimate resident memory as:
    // Resident memory = file size + roughly 250MB for workspace + some context scaling
    // Since we don't have n_embd easily without parsing deeply, we use a conservative heuristic:
    if (req.file_size_bytes) {
        uint64_t base_mem = *req.file_size_bytes;
        uint64_t workspace = 256 * 1024 * 1024; // 256 MB overhead
        // KV cache estimate based on parameters (rough heuristic: 200MB per billion params per 1k context)
        uint64_t kv_cache_est = 0;
        if (req.parameter_count && req.context_length) {
            double billions = *req.parameter_count / 1e9;
            double context_k = *req.context_length / 1024.0;
            kv_cache_est = static_cast<uint64_t>(billions * context_k * 150 * 1024 * 1024);
        } else {
            // fallback: 20% of file size
            kv_cache_est = static_cast<uint64_t>(base_mem * 0.2);
        }
        req.estimated_memory_bytes = base_mem + workspace + kv_cache_est;
    }

    return req;
}

} // namespace agr
