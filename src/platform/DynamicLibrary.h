// src/platform/DynamicLibrary.h
// Portable runtime library loading (LoadLibrary / dlopen) used for optional
// vendor driver libraries such as the NVIDIA CUDA driver. No vendor SDK
// headers are required to build against this.
#pragma once

#include <string>
#include <vector>

namespace agr {
namespace platform {

class DynamicLibrary {
public:
    DynamicLibrary() = default;
    ~DynamicLibrary() { close(); }

    DynamicLibrary(const DynamicLibrary&)            = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;
    DynamicLibrary(DynamicLibrary&& o) noexcept : handle_(o.handle_) { o.handle_ = nullptr; }
    DynamicLibrary& operator=(DynamicLibrary&& o) noexcept {
        if (this != &o) { close(); handle_ = o.handle_; o.handle_ = nullptr; }
        return *this;
    }

    /// Try each candidate name in order. Returns true on the first that loads.
    /// On failure `error_out` (if non-null) receives a description.
    bool open(const std::vector<std::string>& candidates, std::string* error_out = nullptr);

    /// Resolve a symbol; nullptr when not found or library not open.
    void* symbol(const char* name) const;

    template <typename Fn>
    Fn symbolAs(const char* name) const { return reinterpret_cast<Fn>(symbol(name)); }

    void close();
    bool isOpen() const { return handle_ != nullptr; }

private:
    void* handle_ = nullptr;
};

} // namespace platform
} // namespace agr
