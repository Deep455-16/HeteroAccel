// src/hydroxl/VulkanTensorOps.h
// Phase 22: Vulkan compute operations for HydroXL tensor inference.
// Wraps VulkanBackend to provide HydroXL-specific tensor operations.
#pragma once

#include "gpu/VulkanBackend.h"
#include "gpu/GPUBackend.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#if AGR_HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace agr {
namespace hydroxl {

#if AGR_HAVE_VULKAN

// Manages Vulkan device buffers for HydroXL tensors.
// Each named tensor can be uploaded once and reused across forward passes.
struct VkTensorBuffer {
    GPUBackend::Buffer buf;
    uint32_t n_elements = 0;
    bool valid = false;
};

// Provides Vulkan compute pipelines for HydroXL tensor operations.
// All operations validate inputs and return false on error.
class VulkanTensorOps {
public:
    VulkanTensorOps();
    ~VulkanTensorOps();

    // Initialize Vulkan backend and load all HydroXL shaders.
    // Returns false if Vulkan is unavailable or any shader fails to load.
    bool initialize(std::string& error_out);

    bool isReady() const { return ready_; }
    std::string lastError() const { return last_error_; }
    const std::string& deviceName() const { return device_name_; }

    // Upload a float32 tensor to a named device buffer (reuse if same size).
    bool uploadTensor(const std::string& name, const float* data, uint32_t n_elements);

    // Download a named device buffer back to host float32.
    bool downloadTensor(const std::string& name, float* dst, uint32_t n_elements);

    // Allocate a transient device buffer (not named, caller manages lifetime).
    // Returns buffer id (handle), 0 on failure.
    GPUBackend::Buffer allocTransient(uint32_t n_elements);
    void freeTransient(GPUBackend::Buffer& buf);

    // --- Tensor operations ---

    // Matrix-vector multiply: out[m] = A[m,k] * x[k]
    // A_name: named buffer [m_rows * k_cols], x: transient buffer [k_cols]
    bool matvec(const std::string& A_name, const GPUBackend::Buffer& x_buf,
                GPUBackend::Buffer& out_buf, uint32_t m_rows, uint32_t k_cols);

    // Matvec with host input (uploads x, runs matvec, downloads result)
    bool matvecHost(const std::string& A_name,
                    const float* x_host, float* out_host,
                    uint32_t m_rows, uint32_t k_cols);

    // RMS normalization: out[i] = x[i]/rms(x) * weight[i]
    bool rmsnorm(const float* x_host, const std::string& weight_name,
                 float* out_host, uint32_t n, float eps);

    // Element-wise: residual add (op=0), silu (op=1), swiglu (op=2)
    bool elemwise(const float* a_host, const float* b_host,
                  float* out_host, uint32_t n, uint32_t op_type);

    // Softmax in-place on host array (uses GPU if n large enough, else CPU)
    bool softmax(float* x_host, uint32_t n);

    void shutdown();

private:
    bool loadPipeline(const std::string& spv_name,
                      uint32_t n_bindings, uint32_t push_constant_bytes,
                      VkPipeline& pipeline_out,
                      VkDescriptorSetLayout& dsl_out,
                      VkPipelineLayout& pll_out);

    bool runPipeline(VkPipeline pipeline,
                     VkPipelineLayout pll,
                     VkDescriptorSetLayout dsl,
                     const std::vector<VkBuffer>& bindings,
                     const void* push_data, uint32_t push_bytes,
                     uint32_t group_x, uint32_t group_y = 1, uint32_t group_z = 1);

    VkBuffer getVkBuffer(const GPUBackend::Buffer& buf);

    VulkanBackend vk_;
    bool ready_ = false;
    std::string last_error_;
    std::string device_name_;

    std::unordered_map<std::string, VkTensorBuffer> tensors_;

    // Per-operation pipelines
    struct Pipeline {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
        VkPipelineLayout pll = VK_NULL_HANDLE;
        bool valid = false;
    };
    Pipeline p_matvec_;
    Pipeline p_rmsnorm_;
    Pipeline p_elemwise_;
    Pipeline p_softmax_;

    // Shared descriptor pool (sized for all operations)
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    VkCommandPool   cmd_pool_  = VK_NULL_HANDLE;
    VkDevice        device_    = VK_NULL_HANDLE;
    VkQueue         queue_     = VK_NULL_HANDLE;

    // Transient buffers for host-roundtrip operations
    GPUBackend::Buffer tmp_x_;
    GPUBackend::Buffer tmp_out_;
    GPUBackend::Buffer tmp_b_;
    uint32_t tmp_x_elems_   = 0;
    uint32_t tmp_out_elems_ = 0;
    uint32_t tmp_b_elems_   = 0;
};

#else // !AGR_HAVE_VULKAN

// No-op stub for CPU-only builds
class VulkanTensorOps {
public:
    bool initialize(std::string& e) { e = "Built without Vulkan"; return false; }
    bool isReady() const { return false; }
    std::string lastError() const { return "No Vulkan"; }
    const std::string& deviceName() const { static std::string s; return s; }
    bool uploadTensor(const std::string&, const float*, uint32_t) { return false; }
    bool downloadTensor(const std::string&, float*, uint32_t) { return false; }
    GPUBackend::Buffer allocTransient(uint32_t) { return GPUBackend::Buffer{}; }
    void freeTransient(GPUBackend::Buffer&) {}
    bool matvec(const std::string&, const GPUBackend::Buffer&, GPUBackend::Buffer&, uint32_t, uint32_t) { return false; }
    bool matvecHost(const std::string&, const float*, float*, uint32_t, uint32_t) { return false; }
    bool rmsnorm(const float*, const std::string&, float*, uint32_t, float) { return false; }
    bool elemwise(const float*, const float*, float*, uint32_t, uint32_t) { return false; }
    bool softmax(float*, uint32_t) { return false; }
    void shutdown() {}
};

#endif // AGR_HAVE_VULKAN

} // namespace hydroxl
} // namespace agr
