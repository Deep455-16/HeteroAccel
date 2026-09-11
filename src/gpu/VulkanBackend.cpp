#include "gpu/VulkanBackend.h"

#if AGR_HAVE_VULKAN

#include "gpu/ShaderLocator.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>

namespace agr {

namespace {

int rankDeviceType(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 3;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 2;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 1;
        default: return 0;
    }
}

double nowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

} // namespace

VulkanBackend::~VulkanBackend() {
    shutdown();
}

void VulkanBackend::fail(const std::string& what) {
    lastError_ = what;
    available_ = false;
}

bool VulkanBackend::initialize() {
    if (!createInstance()) return false;
    if (!selectPhysicalDevice()) return false;
    if (!createLogicalDeviceAndQueue()) return false;
    if (!createCommandPoolAndBuffer()) return false;
    if (!loadShaderAndCreatePipeline()) return false;
    if (!createDescriptorInfrastructure()) return false;

    available_ = true;
    return true;
}

bool VulkanBackend::createInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "adaptive-gpu-compute";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "AdaptiveGPURuntime";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance_);
    if (result != VK_SUCCESS) {
        fail("vkCreateInstance failed (VkResult=" + std::to_string(static_cast<int>(result)) + ")");
        return false;
    }
    return true;
}

bool VulkanBackend::selectPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
    if (deviceCount == 0) {
        fail("No Vulkan physical devices enumerated");
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

    int bestRank = -1;
    VkPhysicalDevice bestDevice = VK_NULL_HANDLE;
    uint32_t bestQueueFamily = 0;
    std::string bestName;

    for (VkPhysicalDevice dev : devices) {
        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &familyCount, families.data());

        int32_t computeOnlyFamily = -1;
        int32_t anyComputeFamily = -1;
        for (uint32_t i = 0; i < familyCount; ++i) {
            bool hasCompute = (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
            bool hasGraphics = (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
            if (hasCompute && anyComputeFamily < 0) anyComputeFamily = static_cast<int32_t>(i);
            if (hasCompute && !hasGraphics && computeOnlyFamily < 0) computeOnlyFamily = static_cast<int32_t>(i);
        }

        int32_t chosenFamily = computeOnlyFamily >= 0 ? computeOnlyFamily : anyComputeFamily;
        if (chosenFamily < 0) continue; // this device has no compute queue at all

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);
        int rank = rankDeviceType(props.deviceType);

        if (rank > bestRank) {
            bestRank = rank;
            bestDevice = dev;
            bestQueueFamily = static_cast<uint32_t>(chosenFamily);
            bestName = props.deviceName;
        }
    }

    if (bestDevice == VK_NULL_HANDLE) {
        fail("No Vulkan device with a compute-capable queue family was found");
        return false;
    }

    physicalDevice_ = bestDevice;
    computeQueueFamilyIndex_ = bestQueueFamily;
    deviceName_ = bestName;
    return true;
}

bool VulkanBackend::createLogicalDeviceAndQueue() {
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = computeQueueFamilyIndex_;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;

    VkResult result = vkCreateDevice(physicalDevice_, &deviceInfo, nullptr, &device_);
    if (result != VK_SUCCESS) {
        fail("vkCreateDevice failed (VkResult=" + std::to_string(static_cast<int>(result)) + ")");
        return false;
    }

    vkGetDeviceQueue(device_, computeQueueFamilyIndex_, 0, &computeQueue_);
    return true;
}

bool VulkanBackend::createCommandPoolAndBuffer() {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = computeQueueFamilyIndex_;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_) != VK_SUCCESS) {
        fail("vkCreateCommandPool failed");
        return false;
    }
    return true;
}

bool VulkanBackend::loadShaderAndCreatePipeline() {
    shaderPath_ = ShaderLocator::find("vector_add.comp.spv");
    if (shaderPath_.empty()) {
        fail("Could not locate compiled shader 'vector_add.comp.spv' next to the executable or in the build tree. "
             "Was the shader compiled (glslangValidator) and copied during the build?");
        return false;
    }

    std::ifstream file(shaderPath_, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        fail("Could not open shader file: " + shaderPath_);
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
        fail("vkCreateShaderModule failed for " + shaderPath_);
        return false;
    }

    // 3 storage buffers: A (in), B (in), C (out).
    VkDescriptorSetLayoutBinding bindings[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorSetLayout_) != VK_SUCCESS) {
        fail("vkCreateDescriptorSetLayout failed");
        vkDestroyShaderModule(device_, shaderModule, nullptr);
        return false;
    }

    VkPushConstantRange pushConstant{};
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(uint32_t); // element_count

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstant;

    if (vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout_) != VK_SUCCESS) {
        fail("vkCreatePipelineLayout failed");
        vkDestroyShaderModule(device_, shaderModule, nullptr);
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
    pipelineInfo.layout = pipelineLayout_;

    VkResult result = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline_);

    // The shader module is only needed during pipeline creation.
    vkDestroyShaderModule(device_, shaderModule, nullptr);

    if (result != VK_SUCCESS) {
        fail("vkCreateComputePipelines failed (VkResult=" + std::to_string(static_cast<int>(result)) + ")");
        return false;
    }

    return true;
}

bool VulkanBackend::createDescriptorInfrastructure() {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 3; // 3 storage-buffer bindings per set (A, B, C)

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    // VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT is required so that
    // vkFreeDescriptorSets() (called at the end of every executeVectorAdd)
    // actually returns the set back to the pool.  Without this flag the free
    // call is a no-op per the Vulkan spec, the single slot stays consumed
    // permanently, and every subsequent executeVectorAdd fails with
    // VK_ERROR_OUT_OF_POOL_MEMORY ("vkAllocateDescriptorSets failed").
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1; // We only ever hold one descriptor set at a time

    if (vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_) != VK_SUCCESS) {
        fail("vkCreateDescriptorPool failed");
        return false;
    }
    return true;
}

int32_t VulkanBackend::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags desired, bool* exactMatch) const {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        bool typeAllowed = (typeBits & (1u << i)) != 0;
        bool hasDesired = (memProps.memoryTypes[i].propertyFlags & desired) == desired;
        if (typeAllowed && hasDesired) {
            if (exactMatch) *exactMatch = true;
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

Buffer VulkanBackend::createBuffer(size_t sizeBytes) {
    Buffer result;
    if (sizeBytes == 0) {
        fail("createBuffer called with sizeBytes == 0");
        return result;
    }

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = sizeBytes;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        fail("vkCreateBuffer failed");
        return result;
    }

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device_, buffer, &memReqs);

    // Preference order (never assumes dedicated VRAM exists):
    //   1) DEVICE_LOCAL + HOST_VISIBLE  -- unified memory, typical of
    //      integrated GPUs (e.g. Intel Iris Xe): no staging copy needed.
    //   2) HOST_VISIBLE + HOST_COHERENT -- always guaranteed to exist per
    //      the Vulkan spec; works everywhere, still no staging copy.
    //   3) DEVICE_LOCAL only -- typical discrete-GPU VRAM; requires a
    //      staging buffer for every upload/download (handled below).
    bool unified = false;
    VkMemoryPropertyFlags chosenFlags = 0;
    int32_t memTypeIndex = findMemoryType(memReqs.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                           nullptr);
    if (memTypeIndex >= 0) {
        unified = true;
        chosenFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    } else {
        memTypeIndex = findMemoryType(memReqs.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                       nullptr);
        if (memTypeIndex >= 0) {
            unified = true;
            chosenFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        } else {
            memTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, nullptr);
            unified = false;
            chosenFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        }
    }

    if (memTypeIndex < 0) {
        fail("Could not find a suitable memory type for this buffer");
        vkDestroyBuffer(device_, buffer, nullptr);
        return result;
    }

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = static_cast<uint32_t>(memTypeIndex);

    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (vkAllocateMemory(device_, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        fail("vkAllocateMemory failed (requested " + std::to_string(memReqs.size) + " bytes)");
        vkDestroyBuffer(device_, buffer, nullptr);
        return result;
    }

    vkBindBufferMemory(device_, buffer, memory, 0);

    BufferRecord record;
    record.buffer = buffer;
    record.memory = memory;
    record.size = memReqs.size;
    record.unifiedMemory = unified;
    record.memoryFlags = chosenFlags;

    uint64_t id = nextBufferId_++;
    buffers_[id] = record;

    result.id = id;
    result.size_bytes = sizeBytes;
    return result;
}

void VulkanBackend::destroyBuffer(Buffer& buffer) {
    auto it = buffers_.find(buffer.id);
    if (it == buffers_.end()) return;

    if (it->second.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device_, it->second.buffer, nullptr);
    if (it->second.memory != VK_NULL_HANDLE) vkFreeMemory(device_, it->second.memory, nullptr);

    buffers_.erase(it);
    buffer = Buffer{};
}

bool VulkanBackend::copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device_, &allocInfo, &cmd) != VK_SUCCESS) {
        fail("vkAllocateCommandBuffers failed (copyBuffer)");
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(cmd, src, dst, 1, &copyRegion);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    vkQueueSubmit(computeQueue_, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(computeQueue_);

    vkFreeCommandBuffers(device_, commandPool_, 1, &cmd);
    return true;
}

bool VulkanBackend::upload(const Buffer& buffer, const float* data, size_t count, double* outMs) {
    double start = nowMs();

    auto it = buffers_.find(buffer.id);
    if (it == buffers_.end()) {
        fail("upload() called with unknown buffer id");
        return false;
    }
    BufferRecord& record = it->second;
    VkDeviceSize sizeBytes = count * sizeof(float);

    if (record.unifiedMemory) {
        void* mapped = nullptr;
        if (vkMapMemory(device_, record.memory, 0, sizeBytes, 0, &mapped) != VK_SUCCESS) {
            fail("vkMapMemory failed during upload");
            return false;
        }
        std::memcpy(mapped, data, sizeBytes);
        if (!(record.memoryFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            VkMappedMemoryRange range{};
            range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            range.memory = record.memory;
            range.offset = 0;
            range.size = sizeBytes;
            vkFlushMappedMemoryRanges(device_, 1, &range);
        }
        vkUnmapMemory(device_, record.memory);
    } else {
        // Discrete-GPU path: stage through a host-visible buffer, then
        // copy staging -> device-local via the queue.
        VkBufferCreateInfo stagingInfo{};
        stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingInfo.size = sizeBytes;
        stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer staging = VK_NULL_HANDLE;
        if (vkCreateBuffer(device_, &stagingInfo, nullptr, &staging) != VK_SUCCESS) {
            fail("vkCreateBuffer (staging, upload) failed");
            return false;
        }
        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device_, staging, &memReqs);
        int32_t memTypeIndex = findMemoryType(memReqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, nullptr);
        if (memTypeIndex < 0) {
            fail("No host-visible+coherent memory type for staging buffer (upload)");
            vkDestroyBuffer(device_, staging, nullptr);
            return false;
        }
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = static_cast<uint32_t>(memTypeIndex);
        VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
        if (vkAllocateMemory(device_, &allocInfo, nullptr, &stagingMemory) != VK_SUCCESS) {
            fail("vkAllocateMemory (staging, upload) failed");
            vkDestroyBuffer(device_, staging, nullptr);
            return false;
        }
        vkBindBufferMemory(device_, staging, stagingMemory, 0);

        void* mapped = nullptr;
        vkMapMemory(device_, stagingMemory, 0, sizeBytes, 0, &mapped);
        std::memcpy(mapped, data, sizeBytes);
        vkUnmapMemory(device_, stagingMemory);

        bool ok = copyBuffer(staging, record.buffer, sizeBytes);

        vkDestroyBuffer(device_, staging, nullptr);
        vkFreeMemory(device_, stagingMemory, nullptr);

        if (!ok) return false;
    }

    if (outMs) *outMs = nowMs() - start;
    return true;
}

bool VulkanBackend::download(const Buffer& buffer, float* data, size_t count, double* outMs) {
    double start = nowMs();

    auto it = buffers_.find(buffer.id);
    if (it == buffers_.end()) {
        fail("download() called with unknown buffer id");
        return false;
    }
    BufferRecord& record = it->second;
    VkDeviceSize sizeBytes = count * sizeof(float);

    if (record.unifiedMemory) {
        void* mapped = nullptr;
        if (vkMapMemory(device_, record.memory, 0, sizeBytes, 0, &mapped) != VK_SUCCESS) {
            fail("vkMapMemory failed during download");
            return false;
        }
        if (!(record.memoryFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            VkMappedMemoryRange range{};
            range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            range.memory = record.memory;
            range.offset = 0;
            range.size = sizeBytes;
            vkInvalidateMappedMemoryRanges(device_, 1, &range);
        }
        std::memcpy(data, mapped, sizeBytes);
        vkUnmapMemory(device_, record.memory);
    } else {
        VkBufferCreateInfo stagingInfo{};
        stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingInfo.size = sizeBytes;
        stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer staging = VK_NULL_HANDLE;
        if (vkCreateBuffer(device_, &stagingInfo, nullptr, &staging) != VK_SUCCESS) {
            fail("vkCreateBuffer (staging, download) failed");
            return false;
        }
        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device_, staging, &memReqs);
        int32_t memTypeIndex = findMemoryType(memReqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, nullptr);
        if (memTypeIndex < 0) {
            fail("No host-visible+coherent memory type for staging buffer (download)");
            vkDestroyBuffer(device_, staging, nullptr);
            return false;
        }
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = static_cast<uint32_t>(memTypeIndex);
        VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
        if (vkAllocateMemory(device_, &allocInfo, nullptr, &stagingMemory) != VK_SUCCESS) {
            fail("vkAllocateMemory (staging, download) failed");
            vkDestroyBuffer(device_, staging, nullptr);
            return false;
        }
        vkBindBufferMemory(device_, staging, stagingMemory, 0);

        bool ok = copyBuffer(record.buffer, staging, sizeBytes);
        if (ok) {
            void* mapped = nullptr;
            vkMapMemory(device_, stagingMemory, 0, sizeBytes, 0, &mapped);
            std::memcpy(data, mapped, sizeBytes);
            vkUnmapMemory(device_, stagingMemory);
        }

        vkDestroyBuffer(device_, staging, nullptr);
        vkFreeMemory(device_, stagingMemory, nullptr);

        if (!ok) return false;
    }

    if (outMs) *outMs = nowMs() - start;
    return true;
}

bool VulkanBackend::executeVectorAdd(const Buffer& a, const Buffer& b, Buffer& c,
                                      uint32_t elementCount, double* outExecuteMs) {
    auto itA = buffers_.find(a.id);
    auto itB = buffers_.find(b.id);
    auto itC = buffers_.find(c.id);
    if (itA == buffers_.end() || itB == buffers_.end() || itC == buffers_.end()) {
        fail("executeVectorAdd called with an unknown buffer id");
        return false;
    }

    VkDescriptorSetAllocateInfo dsAllocInfo{};
    dsAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsAllocInfo.descriptorPool = descriptorPool_;
    dsAllocInfo.descriptorSetCount = 1;
    dsAllocInfo.pSetLayouts = &descriptorSetLayout_;

    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(device_, &dsAllocInfo, &descriptorSet) != VK_SUCCESS) {
        fail("vkAllocateDescriptorSets failed");
        return false;
    }

    VkDescriptorBufferInfo bufInfos[3] = {
        {itA->second.buffer, 0, VK_WHOLE_SIZE},
        {itB->second.buffer, 0, VK_WHOLE_SIZE},
        {itC->second.buffer, 0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptorSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = commandPool_;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device_, &cmdAllocInfo, &cmd) != VK_SUCCESS) {
        fail("vkAllocateCommandBuffers failed (executeVectorAdd)");
        vkFreeDescriptorSets(device_, descriptorPool_, 1, &descriptorSet);
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &elementCount);

    constexpr uint32_t kLocalSizeX = 256;
    uint32_t groupCountX = (elementCount + kLocalSizeX - 1) / kLocalSizeX;
    vkCmdDispatch(cmd, groupCountX, 1, 1);

    vkEndCommandBuffer(cmd);

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(device_, &fenceInfo, nullptr, &fence);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    // GPU execution time: measured strictly from submit to fence-signaled
    // (i.e. actual device work), deliberately excluding descriptor/command
    // buffer setup above and excluding upload/download, which are timed
    // separately by upload()/download().
    double execStart = nowMs();
    VkResult submitResult = vkQueueSubmit(computeQueue_, 1, &submitInfo, fence);
    if (submitResult == VK_SUCCESS) {
        vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);
    }
    double execEnd = nowMs();

    vkDestroyFence(device_, fence, nullptr);
    vkFreeCommandBuffers(device_, commandPool_, 1, &cmd);
    vkFreeDescriptorSets(device_, descriptorPool_, 1, &descriptorSet);

    if (submitResult != VK_SUCCESS) {
        fail("vkQueueSubmit failed (VkResult=" + std::to_string(static_cast<int>(submitResult)) + ")");
        return false;
    }

    if (outExecuteMs) *outExecuteMs = execEnd - execStart;
    return true;
}

void VulkanBackend::shutdown() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
    }

    for (auto& kv : buffers_) {
        if (kv.second.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device_, kv.second.buffer, nullptr);
        if (kv.second.memory != VK_NULL_HANDLE) vkFreeMemory(device_, kv.second.memory, nullptr);
    }
    buffers_.clear();

    if (descriptorPool_ != VK_NULL_HANDLE) { vkDestroyDescriptorPool(device_, descriptorPool_, nullptr); descriptorPool_ = VK_NULL_HANDLE; }
    if (computePipeline_ != VK_NULL_HANDLE) { vkDestroyPipeline(device_, computePipeline_, nullptr); computePipeline_ = VK_NULL_HANDLE; }
    if (pipelineLayout_ != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr); pipelineLayout_ = VK_NULL_HANDLE; }
    if (descriptorSetLayout_ != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr); descriptorSetLayout_ = VK_NULL_HANDLE; }
    if (commandPool_ != VK_NULL_HANDLE) { vkDestroyCommandPool(device_, commandPool_, nullptr); commandPool_ = VK_NULL_HANDLE; }
    if (device_ != VK_NULL_HANDLE) { vkDestroyDevice(device_, nullptr); device_ = VK_NULL_HANDLE; }
    if (instance_ != VK_NULL_HANDLE) { vkDestroyInstance(instance_, nullptr); instance_ = VK_NULL_HANDLE; }

    physicalDevice_ = VK_NULL_HANDLE;
    computeQueue_ = VK_NULL_HANDLE;
    available_ = false;
}

} // namespace agr

#endif // AGR_HAVE_VULKAN
