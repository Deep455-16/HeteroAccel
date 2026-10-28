// src/orchestration/HydroXLEngineAdapter.h
// Phase 21: Native HydroXL adapters implementing IInferenceEngine

#pragma once

#include "orchestration/IInferenceEngine.h"
#include "hydroxl/HydroXLModel.h"
#include "hydroxl/HydroXLTokenizer.h"
#include "hydroxl/HydroXLKVCache.h"
#include "hydroxl/HydroXLForwardPass.h"
#include "hydroxl/HydroXLSampler.h"

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
    HydroXLVulkanEngineAdapter() = default;
    ~HydroXLVulkanEngineAdapter() override = default;

    bool initialize(const std::string& model_path, std::string& error) override;
    InferenceResult run(const InferenceRequest& req) override;
    void shutdown() override;
    std::string backend_id() const override;
    bool is_ready() const override;

private:
    bool is_ready_ = false;
};

} // namespace orchestration
} // namespace agr
