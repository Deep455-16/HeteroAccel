#include "hardware/CUDADetector.h"

#if defined(_WIN32)
    #include <windows.h>
    using LibHandle = HMODULE;
#else
    #include <dlfcn.h>
    using LibHandle = void*;
#endif

namespace agr {

namespace {

// Minimal subset of the CUDA Driver API declared by hand so this file
// compiles with zero CUDA Toolkit dependency. Types mirror cuda.h.
using CUresult = int;
using CUdevice = int;

using PFN_cuInit = CUresult (*)(unsigned int);
using PFN_cuDriverGetVersion = CUresult (*)(int*);
using PFN_cuDeviceGetCount = CUresult (*)(int*);
using PFN_cuDeviceGet = CUresult (*)(CUdevice*, int);
using PFN_cuDeviceGetName = CUresult (*)(char*, int, CUdevice);

constexpr CUresult CUDA_SUCCESS_VALUE = 0;

LibHandle openCudaDriverLibrary() {
#if defined(_WIN32)
    return LoadLibraryA("nvcuda.dll");
#else
    LibHandle h = dlopen("libcuda.so.1", RTLD_NOW);
    if (h == nullptr) {
        h = dlopen("libcuda.so", RTLD_NOW);
    }
    return h;
#endif
}

void* resolveSymbol(LibHandle lib, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(lib, name));
#else
    return dlsym(lib, name);
#endif
}

void closeLibrary(LibHandle lib) {
#if defined(_WIN32)
    if (lib) FreeLibrary(lib);
#else
    if (lib) dlclose(lib);
#endif
}

} // namespace

CUDAInfo CUDADetector::detect() {
    CUDAInfo info;

    LibHandle lib = openCudaDriverLibrary();
    if (lib == nullptr) {
        info.available = false;
        info.unavailable_reason = "CUDA driver library not found (nvcuda.dll / libcuda.so absent) -- no NVIDIA driver installed";
        return info;
    }

    auto cuInitFn = reinterpret_cast<PFN_cuInit>(resolveSymbol(lib, "cuInit"));
    auto cuDriverGetVersionFn = reinterpret_cast<PFN_cuDriverGetVersion>(resolveSymbol(lib, "cuDriverGetVersion"));
    auto cuDeviceGetCountFn = reinterpret_cast<PFN_cuDeviceGetCount>(resolveSymbol(lib, "cuDeviceGetCount"));
    auto cuDeviceGetFn = reinterpret_cast<PFN_cuDeviceGet>(resolveSymbol(lib, "cuDeviceGet"));
    auto cuDeviceGetNameFn = reinterpret_cast<PFN_cuDeviceGetName>(resolveSymbol(lib, "cuDeviceGetName"));

    if (!cuInitFn || !cuDriverGetVersionFn || !cuDeviceGetCountFn || !cuDeviceGetFn || !cuDeviceGetNameFn) {
        info.available = false;
        info.unavailable_reason = "CUDA driver library found but required symbols were missing";
        closeLibrary(lib);
        return info;
    }

    CUresult initResult = cuInitFn(0);
    if (initResult != CUDA_SUCCESS_VALUE) {
        info.available = false;
        info.unavailable_reason = "cuInit() failed (CUresult=" + std::to_string(initResult) + ") -- driver present but no usable NVIDIA device/context";
        closeLibrary(lib);
        return info;
    }

    int driverVersion = 0;
    cuDriverGetVersionFn(&driverVersion);
    info.driver_version = driverVersion;

    int deviceCount = 0;
    cuDeviceGetCountFn(&deviceCount);

    if (deviceCount == 0) {
        info.available = false;
        info.unavailable_reason = "CUDA driver initialized successfully, but zero CUDA-capable devices were found";
        closeLibrary(lib);
        return info;
    }

    for (int i = 0; i < deviceCount; ++i) {
        CUdevice device = 0;
        if (cuDeviceGetFn(&device, i) != CUDA_SUCCESS_VALUE) {
            continue;
        }
        char name[256] = {0};
        cuDeviceGetNameFn(name, sizeof(name), device);

        CUDADeviceInfo devInfo;
        devInfo.index = i;
        devInfo.name = name;
        info.devices.push_back(devInfo);
    }

    info.available = !info.devices.empty();
    if (!info.available) {
        info.unavailable_reason = "Devices were reported but none could be queried successfully";
    }

    closeLibrary(lib);
    return info;
}

} // namespace agr
