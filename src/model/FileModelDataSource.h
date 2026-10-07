#pragma once

#include "model/IModelDataSource.h"

#include <mutex>
#include <string>

namespace agr {

/// File-backed random-access source. Uses 64-bit offsets. Thread-safe reads.
class FileModelDataSource : public IModelDataSource {
public:
    explicit FileModelDataSource(std::string path);
    ~FileModelDataSource() override;

    FileModelDataSource(const FileModelDataSource&) = delete;
    FileModelDataSource& operator=(const FileModelDataSource&) = delete;

    bool open();
    void close();
    bool isOpen() const { return file_ != nullptr; }

    uint64_t size() const override { return size_; }
    std::string description() const override;
    std::string lastError() const override;

    bool read(uint64_t offset, void* destination, size_t size,
              std::atomic<bool>* cancel = nullptr) override;
    std::future<bool> readAsync(uint64_t offset, void* destination, size_t size,
                                std::atomic<bool>* cancel = nullptr) override;

private:
    std::string path_;
    FILE* file_ = nullptr;
    uint64_t size_ = 0;
    mutable std::mutex mutex_;
    std::string last_error_;
};

} // namespace agr
