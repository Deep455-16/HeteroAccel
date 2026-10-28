// src/hydroxl/VulkanTensorOps.cpp
// Phase 22: Vulkan compute operations for HydroXL tensor inference.

#include "hydroxl/VulkanTensorOps.h"

#if AGR_HAVE_VULKAN

#include "gpu/ShaderLocator.h"

#include <cmath>
#include <fstream>
#include <iostream>

namespace agr {
namespace hydroxl {

VulkanTensorOps::VulkanTensorOps() = default;

VulkanTensorOps::~VulkanTensorOps() {
    shutdown();
}

bool VulkanTensorOps::initialize(std::string& error_out) {
    if (ready_) return true;

    if (!vk_.initialize()) {
        error_out = "VulkanBackend::initialize failed: " + vk_.lastError();
        last_error_ = error_out;
        ready_ = false;
        return false;
    }

    device_ = vk_.device();
    queue_ = vk_.computeQueue();
    cmd_pool_ = vk_.commandPool();
    device_name_ = vk_.deviceName();

    if (device_ == VK_NULL_HANDLE || queue_ == VK_NULL_HANDLE || cmd_pool_ == VK_NULL_HANDLE) {
        error_out = "Invalid Vulkan handles from VulkanBackend";
        last_error_ = error_out;
        ready_ = false;
        return false;
    }

    // Create shared descriptor pool
    VkDescriptorPoolSize poolSizes[1]{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 128;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 32;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;

    if (vkCreateDescriptorPool(device_, &poolInfo, nullptr, &desc_pool_) != VK_SUCCESS) {
        error_out = "Failed to create VkDescriptorPool";
        last_error_ = error_out;
        ready_ = false;
        return false;
    }

    // Load 4 pipelines
    if (!loadPipeline("hydroxl_matvec.comp.spv", 3, 2 * sizeof(uint32_t),
                      p_matvec_.pipeline, p_matvec_.dsl, p_matvec_.pll)) {
        error_out = "Failed to load hydroxl_matvec pipeline: " + last_error_;
        shutdown();
        return false;
    }
    p_matvec_.valid = true;

    if (!loadPipeline("hydroxl_rmsnorm.comp.spv", 3, sizeof(uint32_t) + sizeof(float),
                      p_rmsnorm_.pipeline, p_rmsnorm_.dsl, p_rmsnorm_.pll)) {
        error_out = "Failed to load hydroxl_rmsnorm pipeline: " + last_error_;
        shutdown();
        return false;
    }
    p_rmsnorm_.valid = true;

    if (!loadPipeline("hydroxl_elemwise.comp.spv", 3, 2 * sizeof(uint32_t),
                      p_elemwise_.pipeline, p_elemwise_.dsl, p_elemwise_.pll)) {
        error_out = "Failed to load hydroxl_elemwise pipeline: " + last_error_;
        shutdown();
        return false;
    }
    p_elemwise_.valid = true;

    if (!loadPipeline("hydroxl_softmax.comp.spv", 1, sizeof(uint32_t),
                      p_softmax_.pipeline, p_softmax_.dsl, p_softmax_.pll)) {
        error_out = "Failed to load hydroxl_softmax pipeline: " + last_error_;
        shutdown();
        return false;
    }
    p_softmax_.valid = true;

    ready_ = true;
    return true;
}

bool VulkanTensorOps::loadPipeline(const std::string& spv_name,
                                   uint32_t n_bindings, uint32_t push_constant_bytes,
                                   VkPipeline& pipeline_out,
                                   VkDescriptorSetLayout& dsl_out,
                                   VkPipelineLayout& pll_out) {
    std::string spv_path = ShaderLocator::find(spv_name);
    if (spv_path.empty()) {
        last_error_ = "Could not find shader file: " + spv_name;
        return false;
    }

    std::ifstream file(spv_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        last_error_ = "Could not open shader file: " + spv_path;
        return false;
    }
    size_t fileSize = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<uint32_t> code(fileSize / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(fileSize));
    file.close();

    VkShaderModuleCreateInfo moduleInfo{};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = fileSize;
    moduleInfo.pCode = code.data();

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device_, &moduleInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        last_error_ = "vkCreateShaderModule failed for " + spv_name;
        return false;
    }

    std::vector<VkDescriptorSetLayoutBinding> bindings(n_bindings);
    for (uint32_t i = 0; i < n_bindings; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = n_bindings;
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &dsl_out) != VK_SUCCESS) {
        vkDestroyShaderModule(device_, shaderModule, nullptr);
        last_error_ = "vkCreateDescriptorSetLayout failed for " + spv_name;
        return false;
    }

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = push_constant_bytes;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &dsl_out;
    if (push_constant_bytes > 0) {
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pcRange;
    }

    if (vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pll_out) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device_, dsl_out, nullptr);
        vkDestroyShaderModule(device_, shaderModule, nullptr);
        last_error_ = "vkCreatePipelineLayout failed for " + spv_name;
        return false;
    }

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pll_out;

    VkResult res = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_out);
    vkDestroyShaderModule(device_, shaderModule, nullptr);

    if (res != VK_SUCCESS) {
        vkDestroyPipelineLayout(device_, pll_out, nullptr);
        vkDestroyDescriptorSetLayout(device_, dsl_out, nullptr);
        last_error_ = "vkCreateComputePipelines failed for " + spv_name + " (VkResult=" + std::to_string(res) + ")";
        return false;
    }

    return true;
}

bool VulkanTensorOps::runPipeline(VkPipeline pipeline,
                                  VkPipelineLayout pll,
                                  VkDescriptorSetLayout dsl,
                                  const std::vector<VkBuffer>& bindings,
                                  const void* push_data, uint32_t push_bytes,
                                  uint32_t group_x, uint32_t group_y, uint32_t group_z) {
    if (!ready_ || device_ == VK_NULL_HANDLE) {
        last_error_ = "VulkanTensorOps not ready";
        return false;
    }
    if (pipeline == VK_NULL_HANDLE || pll == VK_NULL_HANDLE || dsl == VK_NULL_HANDLE) {
        last_error_ = "Invalid pipeline handle in runPipeline";
        return false;
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = desc_pool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &dsl;

    VkDescriptorSet desc_set = VK_NULL_HANDLE;
    VkResult res = vkAllocateDescriptorSets(device_, &allocInfo, &desc_set);
    if (res != VK_SUCCESS) {
        last_error_ = "vkAllocateDescriptorSets failed (VkResult=" + std::to_string(res) + ")";
        return false;
    }

    std::vector<VkDescriptorBufferInfo> buf_infos(bindings.size());
    std::vector<VkWriteDescriptorSet> writes(bindings.size());
    for (size_t i = 0; i < bindings.size(); ++i) {
        buf_infos[i].buffer = bindings[i];
        buf_infos[i].offset = 0;
        buf_infos[i].range = VK_WHOLE_SIZE;

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = desc_set;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].dstArrayElement = 0;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].descriptorCount = 1;
        writes[i].pBufferInfo = &buf_infos[i];
    }
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = cmd_pool_;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    res = vkAllocateCommandBuffers(device_, &cmdAllocInfo, &cmd);
    if (res != VK_SUCCESS) {
        vkResetDescriptorPool(device_, desc_pool_, 0);
        last_error_ = "vkAllocateCommandBuffers failed (VkResult=" + std::to_string(res) + ")";
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pll, 0, 1, &desc_set, 0, nullptr);
    if (push_data && push_bytes > 0) {
        vkCmdPushConstants(cmd, pll, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push_data);
    }
    vkCmdDispatch(cmd, group_x, group_y, group_z);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    res = vkQueueSubmit(queue_, 1, &submitInfo, VK_NULL_HANDLE);
    if (res != VK_SUCCESS) {
        vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd);
        vkResetDescriptorPool(device_, desc_pool_, 0);
        last_error_ = "vkQueueSubmit failed (VkResult=" + std::to_string(res) + ")";
        return false;
    }

    res = vkQueueWaitIdle(queue_);
    vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd);
    vkResetDescriptorPool(device_, desc_pool_, 0);

    if (res != VK_SUCCESS) {
        last_error_ = "vkQueueWaitIdle failed (VkResult=" + std::to_string(res) + ")";
        return false;
    }

    return true;
}

VkBuffer VulkanTensorOps::getVkBuffer(const GPUBackend::Buffer& buf) {
    return vk_.vkBufferOf(buf);
}

bool VulkanTensorOps::uploadTensor(const std::string& name, const float* data, uint32_t n_elements) {
    if (!ready_ || !data || n_elements == 0) return false;

    auto it = tensors_.find(name);
    if (it != tensors_.end() && it->second.valid && it->second.n_elements == n_elements) {
        return vk_.upload(it->second.buf, data, n_elements, nullptr);
    }

    if (it != tensors_.end() && it->second.valid) {
        vk_.destroyBuffer(it->second.buf);
        tensors_.erase(it);
    }

    GPUBackend::Buffer buf = vk_.createBuffer(n_elements * sizeof(float));
    if (buf.id == 0) {
        last_error_ = "Failed to create buffer for tensor: " + name;
        return false;
    }

    if (!vk_.upload(buf, data, n_elements, nullptr)) {
        vk_.destroyBuffer(buf);
        last_error_ = "Failed to upload data for tensor: " + name;
        return false;
    }

    tensors_[name] = {buf, n_elements, true};
    return true;
}

bool VulkanTensorOps::downloadTensor(const std::string& name, float* dst, uint32_t n_elements) {
    if (!ready_ || !dst || n_elements == 0) return false;

    auto it = tensors_.find(name);
    if (it == tensors_.end() || !it->second.valid || it->second.n_elements < n_elements) {
        last_error_ = "Tensor not found or insufficient size: " + name;
        return false;
    }

    return vk_.download(it->second.buf, dst, n_elements, nullptr);
}

GPUBackend::Buffer VulkanTensorOps::allocTransient(uint32_t n_elements) {
    if (!ready_ || n_elements == 0) return {};
    return vk_.createBuffer(n_elements * sizeof(float));
}

void VulkanTensorOps::freeTransient(GPUBackend::Buffer& buf) {
    if (buf.id != 0) {
        vk_.destroyBuffer(buf);
    }
}

bool VulkanTensorOps::matvec(const std::string& A_name, const GPUBackend::Buffer& x_buf,
                             GPUBackend::Buffer& out_buf, uint32_t m_rows, uint32_t k_cols) {
    if (!ready_ || m_rows == 0 || k_cols == 0) return false;

    auto it = tensors_.find(A_name);
    if (it == tensors_.end() || !it->second.valid) {
        last_error_ = "Matrix tensor not found: " + A_name;
        return false;
    }

    VkBuffer A_vk = getVkBuffer(it->second.buf);
    VkBuffer x_vk = getVkBuffer(x_buf);
    VkBuffer out_vk = getVkBuffer(out_buf);

    if (A_vk == VK_NULL_HANDLE || x_vk == VK_NULL_HANDLE || out_vk == VK_NULL_HANDLE) {
        last_error_ = "Invalid VkBuffer handle for matvec";
        return false;
    }

    struct PC {
        uint32_t m_rows;
        uint32_t k_cols;
    } pc{m_rows, k_cols};

    return runPipeline(p_matvec_.pipeline, p_matvec_.pll, p_matvec_.dsl,
                       {A_vk, x_vk, out_vk},
                       &pc, sizeof(pc),
                       m_rows, 1, 1);
}

bool VulkanTensorOps::matvecHost(const std::string& A_name,
                                 const float* x_host, float* out_host,
                                 uint32_t m_rows, uint32_t k_cols) {
    if (!ready_ || !x_host || !out_host || m_rows == 0 || k_cols == 0) return false;

    if (tmp_x_elems_ < k_cols || tmp_x_.id == 0) {
        if (tmp_x_.id != 0) vk_.destroyBuffer(tmp_x_);
        tmp_x_ = vk_.createBuffer(k_cols * sizeof(float));
        tmp_x_elems_ = k_cols;
    }
    if (tmp_out_elems_ < m_rows || tmp_out_.id == 0) {
        if (tmp_out_.id != 0) vk_.destroyBuffer(tmp_out_);
        tmp_out_ = vk_.createBuffer(m_rows * sizeof(float));
        tmp_out_elems_ = m_rows;
    }

    if (!vk_.upload(tmp_x_, x_host, k_cols, nullptr)) return false;

    if (!matvec(A_name, tmp_x_, tmp_out_, m_rows, k_cols)) return false;

    return vk_.download(tmp_out_, out_host, m_rows, nullptr);
}

bool VulkanTensorOps::rmsnorm(const float* x_host, const std::string& weight_name,
                              float* out_host, uint32_t n, float eps) {
    if (!ready_ || !x_host || !out_host || n == 0) return false;

    auto it = tensors_.find(weight_name);
    if (it == tensors_.end() || !it->second.valid) {
        last_error_ = "Weight tensor not found: " + weight_name;
        return false;
    }

    if (tmp_x_elems_ < n || tmp_x_.id == 0) {
        if (tmp_x_.id != 0) vk_.destroyBuffer(tmp_x_);
        tmp_x_ = vk_.createBuffer(n * sizeof(float));
        tmp_x_elems_ = n;
    }
    if (tmp_out_elems_ < n || tmp_out_.id == 0) {
        if (tmp_out_.id != 0) vk_.destroyBuffer(tmp_out_);
        tmp_out_ = vk_.createBuffer(n * sizeof(float));
        tmp_out_elems_ = n;
    }

    if (!vk_.upload(tmp_x_, x_host, n, nullptr)) return false;

    VkBuffer x_vk = getVkBuffer(tmp_x_);
    VkBuffer w_vk = getVkBuffer(it->second.buf);
    VkBuffer out_vk = getVkBuffer(tmp_out_);

    struct PC {
        uint32_t n;
        float eps;
    } pc{n, eps};

    if (!runPipeline(p_rmsnorm_.pipeline, p_rmsnorm_.pll, p_rmsnorm_.dsl,
                     {x_vk, w_vk, out_vk},
                     &pc, sizeof(pc),
                     1, 1, 1)) {
        return false;
    }

    return vk_.download(tmp_out_, out_host, n, nullptr);
}

bool VulkanTensorOps::elemwise(const float* a_host, const float* b_host,
                               float* out_host, uint32_t n, uint32_t op_type) {
    if (!ready_ || !a_host || !out_host || n == 0) return false;

    if (tmp_x_elems_ < n || tmp_x_.id == 0) {
        if (tmp_x_.id != 0) vk_.destroyBuffer(tmp_x_);
        tmp_x_ = vk_.createBuffer(n * sizeof(float));
        tmp_x_elems_ = n;
    }
    if (tmp_b_elems_ < n || tmp_b_.id == 0) {
        if (tmp_b_.id != 0) vk_.destroyBuffer(tmp_b_);
        tmp_b_ = vk_.createBuffer(n * sizeof(float));
        tmp_b_elems_ = n;
    }
    if (tmp_out_elems_ < n || tmp_out_.id == 0) {
        if (tmp_out_.id != 0) vk_.destroyBuffer(tmp_out_);
        tmp_out_ = vk_.createBuffer(n * sizeof(float));
        tmp_out_elems_ = n;
    }

    if (!vk_.upload(tmp_x_, a_host, n, nullptr)) return false;
    if (b_host) {
        if (!vk_.upload(tmp_b_, b_host, n, nullptr)) return false;
    }

    VkBuffer a_vk = getVkBuffer(tmp_x_);
    VkBuffer b_vk = getVkBuffer(tmp_b_);
    VkBuffer out_vk = getVkBuffer(tmp_out_);

    struct PC {
        uint32_t n;
        uint32_t op_type;
    } pc{n, op_type};

    uint32_t group_x = (n + 255) / 256;
    if (!runPipeline(p_elemwise_.pipeline, p_elemwise_.pll, p_elemwise_.dsl,
                     {a_vk, b_vk, out_vk},
                     &pc, sizeof(pc),
                     group_x, 1, 1)) {
        return false;
    }

    return vk_.download(tmp_out_, out_host, n, nullptr);
}

bool VulkanTensorOps::softmax(float* x_host, uint32_t n) {
    if (!ready_ || !x_host || n == 0) return false;

    if (tmp_x_elems_ < n || tmp_x_.id == 0) {
        if (tmp_x_.id != 0) vk_.destroyBuffer(tmp_x_);
        tmp_x_ = vk_.createBuffer(n * sizeof(float));
        tmp_x_elems_ = n;
    }

    if (!vk_.upload(tmp_x_, x_host, n, nullptr)) return false;

    VkBuffer x_vk = getVkBuffer(tmp_x_);

    struct PC {
        uint32_t n;
    } pc{n};

    if (!runPipeline(p_softmax_.pipeline, p_softmax_.pll, p_softmax_.dsl,
                     {x_vk},
                     &pc, sizeof(pc),
                     1, 1, 1)) {
        return false;
    }

    return vk_.download(tmp_x_, x_host, n, nullptr);
}

void VulkanTensorOps::shutdown() {
    if (device_ != VK_NULL_HANDLE) {
        vkQueueWaitIdle(queue_);

        auto destroyPipeline = [this](Pipeline& p) {
            if (p.pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device_, p.pipeline, nullptr);
            if (p.pll != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, p.pll, nullptr);
            if (p.dsl != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, p.dsl, nullptr);
            p = Pipeline{};
        };

        destroyPipeline(p_matvec_);
        destroyPipeline(p_rmsnorm_);
        destroyPipeline(p_elemwise_);
        destroyPipeline(p_softmax_);

        if (desc_pool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device_, desc_pool_, nullptr);
            desc_pool_ = VK_NULL_HANDLE;
        }

        if (tmp_x_.id != 0) vk_.destroyBuffer(tmp_x_);
        if (tmp_out_.id != 0) vk_.destroyBuffer(tmp_out_);
        if (tmp_b_.id != 0) vk_.destroyBuffer(tmp_b_);
        tmp_x_elems_ = 0;
        tmp_out_elems_ = 0;
        tmp_b_elems_ = 0;

        for (auto& kv : tensors_) {
            if (kv.second.valid && kv.second.buf.id != 0) {
                vk_.destroyBuffer(kv.second.buf);
            }
        }
        tensors_.clear();
    }

    vk_.shutdown();

    device_ = VK_NULL_HANDLE;
    queue_ = VK_NULL_HANDLE;
    cmd_pool_ = VK_NULL_HANDLE;
    ready_ = false;
}

} // namespace hydroxl
} // namespace agr

#endif // AGR_HAVE_VULKAN
