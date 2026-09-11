#include "hardware/VulkanDetector.h"

#if AGR_HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif

#include <cstring>

namespace agr {

#if AGR_HAVE_VULKAN

namespace {

const char* deviceTypeToString(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "Discrete GPU";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "Integrated GPU";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "Virtual GPU";
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU";
        default: return "Other";
    }
}

} // namespace

VulkanInfo VulkanDetector::detect() {
    VulkanInfo info;

    // Instance-level API version, if the loader supports querying it.
    uint32_t instanceVersion = VK_API_VERSION_1_0;
    PFN_vkEnumerateInstanceVersion enumerateInstanceVersion =
        reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
            vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));
    if (enumerateInstanceVersion != nullptr) {
        enumerateInstanceVersion(&instanceVersion);
    }
    info.instance_api_version_major = VK_API_VERSION_MAJOR(instanceVersion);
    info.instance_api_version_minor = VK_API_VERSION_MINOR(instanceVersion);
    info.instance_api_version_patch = VK_API_VERSION_PATCH(instanceVersion);

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "adaptive-gpu-hardware-probe";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "AdaptiveGPURuntime";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0; // widest compatibility for detection

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkInstance instance = VK_NULL_HANDLE;
    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

    if (result != VK_SUCCESS) {
        info.available = false;
        switch (result) {
            case VK_ERROR_INCOMPATIBLE_DRIVER:
                info.unavailable_reason = "No compatible Vulkan driver/ICD found (VK_ERROR_INCOMPATIBLE_DRIVER)";
                break;
            case VK_ERROR_OUT_OF_HOST_MEMORY:
                info.unavailable_reason = "Out of host memory during vkCreateInstance";
                break;
            default:
                info.unavailable_reason = "vkCreateInstance failed (VkResult=" + std::to_string(static_cast<int>(result)) + ")";
                break;
        }
        return info;
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
        info.available = false;
        info.unavailable_reason = "Vulkan loader initialized successfully, but zero physical devices were enumerated (no GPU / no ICD registered on this machine)";
        vkDestroyInstance(instance, nullptr);
        return info;
    }

    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data());

    for (VkPhysicalDevice device : physicalDevices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);

        VulkanDeviceInfo devInfo;
        devInfo.name = props.deviceName;
        devInfo.vendor_id = props.vendorID;
        devInfo.device_id = props.deviceID;
        devInfo.device_type = deviceTypeToString(props.deviceType);
        devInfo.api_version_major = VK_API_VERSION_MAJOR(props.apiVersion);
        devInfo.api_version_minor = VK_API_VERSION_MINOR(props.apiVersion);
        devInfo.api_version_patch = VK_API_VERSION_PATCH(props.apiVersion);
        devInfo.driver_version_raw = props.driverVersion;

        // Memory properties: sum device-local heaps and host-visible heaps.
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(device, &memProps);

        uint64_t deviceLocalBytes = 0;
        for (uint32_t h = 0; h < memProps.memoryHeapCount; ++h) {
            if (memProps.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                deviceLocalBytes += memProps.memoryHeaps[h].size;
            }
        }
        devInfo.device_local_memory_mb = deviceLocalBytes / (1024ULL * 1024ULL);

        // Approximate host-visible capacity by inspecting memory *types*
        // that are host-visible and summing their parent heap sizes once each.
        uint64_t hostVisibleBytes = 0;
        std::vector<bool> heapCounted(memProps.memoryHeapCount, false);
        for (uint32_t t = 0; t < memProps.memoryTypeCount; ++t) {
            if (memProps.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
                uint32_t heapIndex = memProps.memoryTypes[t].heapIndex;
                if (!heapCounted[heapIndex]) {
                    hostVisibleBytes += memProps.memoryHeaps[heapIndex].size;
                    heapCounted[heapIndex] = true;
                }
            }
        }
        devInfo.host_visible_memory_mb = hostVisibleBytes / (1024ULL * 1024ULL);

        // Queue families.
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        for (uint32_t q = 0; q < queueFamilyCount; ++q) {
            VulkanQueueFamilyInfo qInfo;
            qInfo.index = q;
            qInfo.queue_count = queueFamilies[q].queueCount;
            qInfo.supports_graphics = (queueFamilies[q].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
            qInfo.supports_compute = (queueFamilies[q].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
            qInfo.supports_transfer = (queueFamilies[q].queueFlags & VK_QUEUE_TRANSFER_BIT) != 0;
            if (qInfo.supports_compute) {
                devInfo.has_compute_capable_queue = true;
            }
            devInfo.queue_families.push_back(qInfo);
        }

        info.devices.push_back(devInfo);
    }

    vkDestroyInstance(instance, nullptr);

    // "Available" means: instance created AND at least one device AND at
    // least one device with a usable compute queue.
    bool anyComputeCapable = false;
    for (const auto& d : info.devices) {
        if (d.has_compute_capable_queue) { anyComputeCapable = true; break; }
    }

    if (!anyComputeCapable) {
        info.available = false;
        info.unavailable_reason = "Vulkan devices were enumerated, but none exposed a compute-capable queue family";
    } else {
        info.available = true;
    }

    return info;
}

#else // AGR_HAVE_VULKAN == 0 : Vulkan SDK/loader not present at build time

VulkanInfo VulkanDetector::detect() {
    VulkanInfo info;
    info.available = false;
    info.unavailable_reason = "This binary was built without the Vulkan loader/headers available (AGR_HAVE_VULKAN=0)";
    return info;
}

#endif

} // namespace agr
