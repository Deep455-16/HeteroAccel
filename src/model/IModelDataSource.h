// src/model/IModelDataSource.h
// Phase 14: random-access model bytes without loading the whole file.
#pragma once

#include <atomic>
#include <cstdint>
#include <future>
#include <string>

namespace agr {

class IModelDataSource {
public:
    virtual ~IModelDataSource() = default;

    virtual uint64_t size() const = 0;
    virtual std::string description() const = 0;
    virtual std::string lastError() const = 0;

    /// Read `size` bytes at `offset` into `destination`. Never loads the
    /// whole file. Returns false on bounds errors, I/O errors, or cancel.
    virtual bool read(uint64_t offset, void* destination, size_t size,
                      std::atomic<bool>* cancel = nullptr) = 0;

    /// Real background read (std::async). Wait on the returned future.
    virtual std::future<bool> readAsync(uint64_t offset, void* destination, size_t size,
                                        std::atomic<bool>* cancel = nullptr) = 0;
};

} // namespace agr
