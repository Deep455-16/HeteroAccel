#pragma once
#include "core/HardwareInfo.h"

namespace agr {

// Real Vulkan detection only -- no compute workloads are run here
// (that is Phase 2). Creates a throwaway VkInstance, enumerates
// physical devices, and reads their properties/memory/queue families.
// Gracefully reports unavailable (with a reason) on machines with no
// Vulkan loader, no ICD, or no compute-capable device.
class VulkanDetector {
public:
    static VulkanInfo detect();
};

} // namespace agr
