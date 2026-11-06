// src/platform/DynamicLibrary.cpp
#include "platform/DynamicLibrary.h"

#if defined(_WIN32)
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
    #define NOMINMAX
    #endif
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

namespace agr {
namespace platform {

bool DynamicLibrary::open(const std::vector<std::string>& candidates, std::string* error_out) {
    close();
    std::string last_error = "no candidate library names supplied";
    for (const auto& name : candidates) {
#if defined(_WIN32)
        HMODULE h = LoadLibraryA(name.c_str());
        if (h) { handle_ = reinterpret_cast<void*>(h); return true; }
        last_error = "LoadLibrary(" + name + ") failed, error " + std::to_string(GetLastError());
#else
        void* h = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (h) { handle_ = h; return true; }
        const char* e = dlerror();
        last_error = std::string("dlopen(") + name + ") failed: " + (e ? e : "unknown error");
#endif
    }
    if (error_out) *error_out = last_error;
    return false;
}

void* DynamicLibrary::symbol(const char* name) const {
    if (!handle_ || !name) return nullptr;
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(handle_), name));
#else
    return dlsym(handle_, name);
#endif
}

void DynamicLibrary::close() {
    if (!handle_) return;
#if defined(_WIN32)
    FreeLibrary(reinterpret_cast<HMODULE>(handle_));
#else
    dlclose(handle_);
#endif
    handle_ = nullptr;
}

} // namespace platform
} // namespace agr
