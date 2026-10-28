// src/orchestration/HydroXLEngineAdapter.h
// Phase 21: Native HydroXL adapters implementing IInferenceEngine

#pragma once

#include "orchestration/IInferenceEngine.h"
#include "hydroxl/HydroXLModel.h"
#include "hydroxl/HydroXLTokenizer.h"
#include "hydroxl/HydroXLKVCache.h"
#include "hydroxl/HydroXLForwardPass.h"
#include "hydroxl/HydroXLSampler.h"
#include "hydroxl/VulkanTensorOps.h"

#include <memory>
#include <mutex>

namespace agr {
namespace orchestration {

class HydroXLCpuEngineAdapter : public IInferenceEngine {
public:
    HydroXLCpuEngineAdapter();
    ~HydroXLCpuEngineAdapter() override;

    bool initialize(const std::string& model_path, std::string& error) override;
    InferenceResult run(const InferenceRequest& req) override;
    void shutdown() override;
    std::string backend_id() const override;
    bool is_ready() const override;

private:
    std::unique_ptr<hydroxl::HydroXLModel>       model_;
    std::unique_ptr<hydroxl::HydroXLTokenizer>   tokenizer_;
    std::unique_ptr<hydroxl::HydroXLKVCache>     kv_cache_;
    std::unique_ptr<hydroxl::HydroXLForwardPass> forward_pass_;

    mutable std::mutex mu_;
    bool is_ready_ = false;
    std::string model_path_;
};

class HydroXLVulkanEngineAdapter : public IInferenceEngine {
public:
    HydroXLVulkanEngineAdapter();
    ~HydroXLVulkanEngineAdapter() override;

    bool initialize(const std::string& model_path, std::string& error) override;
    InferenceResult run(const InferenceRequest& req) override;
    void shutdown() override;
    std::string backend_id() const override;
    bool is_ready() const override;

private:
    bool forward_token(uint32_t token_id, uint32_t pos, std::vector<float>& logits, std::string& err);

    std::unique_ptr<hydroxl::HydroXLModel>       model_;
    std::unique_ptr<hydroxl::HydroXLTokenizer>   tokenizer_;
    std::unique_ptr<hydroxl::HydroXLKVCache>     kv_cache_;
    hydroxl::VulkanTensorOps                     vk_ops_;

    mutable std::mutex mu_;
    bool is_ready_ = false;
    std::string model_path_;

    std::vector<float> hidden_;
    std::vector<float> normed_;
    std::vector<float> q_buf_;
    std::vector<float> k_buf_;
    std::vector<float> v_buf_;
    std::vector<float> attn_out_;
    std::vector<float> proj_out_;
    std::vector<float> gate_buf_;
    std::vector<float> up_buf_;
    std::vector<float> ffn_buf_;
    std::vector<float> ffn_out_;
    std::vector<float> attn_scores_;
};

} // namespace orchestration
} // namespace agr
