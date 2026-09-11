#pragma once
#include "core/HardwareInfo.h"
#include <vector>

namespace agr {

// Enumerates GPUs via the OS's native adapter-enumeration API.
//
// Windows (the Phase-1 target platform): DXGI (IDXGIFactory1::EnumAdapters1).
// This works for Intel/AMD/NVIDIA, integrated or dedicated, without
// requiring any vendor SDK.
//
// Non-Windows (dev-only fallback used to keep this repo buildable/testable
// outside Windows): best-effort read of /sys/class/drm. On a machine with
// no GPU (e.g. a headless CI/VM) this correctly returns an empty list --
// that is the honest answer, not a bug.
class GPUDetector {
public:
    static std::vector<GPUInfo> detect();
};

} // namespace agr
