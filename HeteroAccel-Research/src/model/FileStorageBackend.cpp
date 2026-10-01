// src/model/FileStorageBackend.cpp
#include "model/FileStorageBackend.h"
#include <fstream>
#include <sys/stat.h>

namespace agr {

bool FileStorageBackend::read(const std::string& path, size_t offset, size_t size, void* destination) {
    if (!destination) return false;

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    file.seekg(offset, std::ios::beg);
    if (!file) return false;

    file.read(static_cast<char*>(destination), size);
    return !!file;
}

size_t FileStorageBackend::size(const std::string& path) {
    struct stat stat_buf;
    int rc = stat(path.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : 0;
}

bool FileStorageBackend::exists(const std::string& path) {
    struct stat stat_buf;
    return (stat(path.c_str(), &stat_buf) == 0);
}

} // namespace agr
