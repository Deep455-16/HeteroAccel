// src/model/IStorageBackend.h
#pragma once

#include <string>
#include <cstddef>
#include <vector>

namespace agr {

/// Generic abstraction for storage I/O.
class IStorageBackend {
public:
    virtual ~IStorageBackend() = default;

    /// Read data synchronously from storage into a pre-allocated buffer.
    /// Returns true if successful.
    virtual bool read(const std::string& path, size_t offset, size_t size, void* destination) = 0;

    /// Returns the total size of the resource if it exists, otherwise 0.
    virtual size_t size(const std::string& path) = 0;

    /// Checks if the resource exists.
    virtual bool exists(const std::string& path) = 0;
};

} // namespace agr
