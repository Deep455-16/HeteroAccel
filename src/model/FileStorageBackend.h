// src/model/FileStorageBackend.h
#pragma once

#include "model/IStorageBackend.h"

namespace agr {

/// Standard local file system implementation of storage backend.
class FileStorageBackend : public IStorageBackend {
public:
    bool read(const std::string& path, size_t offset, size_t size, void* destination) override;
    size_t size(const std::string& path) override;
    bool exists(const std::string& path) override;
};

} // namespace agr
