#include "model/FileModelDataSource.h"

#include <cstdio>
#include <cstring>

#ifdef _WIN32
#include <io.h>
#else
#include <sys/types.h>
#endif

namespace agr {

namespace {

bool seek64(FILE* f, uint64_t offset) {
#ifdef _WIN32
    return _fseeki64(f, static_cast<__int64>(offset), SEEK_SET) == 0;
#else
    return fseeko(f, static_cast<off_t>(offset), SEEK_SET) == 0;
#endif
}

uint64_t tell64(FILE* f) {
#ifdef _WIN32
    __int64 p = _ftelli64(f);
    return p < 0 ? 0 : static_cast<uint64_t>(p);
#else
    off_t p = ftello(f);
    return p < 0 ? 0 : static_cast<uint64_t>(p);
#endif
}

} // namespace

FileModelDataSource::FileModelDataSource(std::string path)
    : path_(std::move(path)) {}

FileModelDataSource::~FileModelDataSource() {
    close();
}

bool FileModelDataSource::open() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_) return true;
#ifdef _WIN32
    file_ = nullptr;
    fopen_s(&file_, path_.c_str(), "rb");
#else
    file_ = fopen(path_.c_str(), "rb");
#endif
    if (!file_) {
        last_error_ = "Failed to open model file: " + path_;
        return false;
    }
    if (!seek64(file_, 0)) {
        last_error_ = "Failed to seek to start of " + path_;
        fclose(file_);
        file_ = nullptr;
        return false;
    }
#ifdef _WIN32
    if (_fseeki64(file_, 0, SEEK_END) != 0) {
#else
    if (fseeko(file_, 0, SEEK_END) != 0) {
#endif
        last_error_ = "Failed to measure size of " + path_;
        fclose(file_);
        file_ = nullptr;
        return false;
    }
    size_ = tell64(file_);
    last_error_.clear();
    return true;
}

void FileModelDataSource::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_) {
        fclose(file_);
        file_ = nullptr;
    }
}

std::string FileModelDataSource::description() const {
    return "file:" + path_;
}

std::string FileModelDataSource::lastError() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_error_;
}

bool FileModelDataSource::read(uint64_t offset, void* destination, size_t nbytes,
                               std::atomic<bool>* cancel) {
    if (!destination || nbytes == 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_error_ = "Invalid destination or zero-length read";
        return false;
    }
    if (cancel && cancel->load()) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_error_ = "Read cancelled";
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (!file_) {
        last_error_ = "File is not open";
        return false;
    }
    if (offset > size_ || nbytes > size_ - offset) {
        last_error_ = "Read exceeds file bounds";
        return false;
    }
    if (!seek64(file_, offset)) {
        last_error_ = "Seek failed";
        return false;
    }

    auto* out = static_cast<uint8_t*>(destination);
    size_t remaining = nbytes;
    while (remaining > 0) {
        if (cancel && cancel->load()) {
            last_error_ = "Read cancelled";
            return false;
        }
        size_t chunk = remaining > 1024 * 1024 ? 1024 * 1024 : remaining;
        size_t got = fread(out, 1, chunk, file_);
        if (got == 0) {
            last_error_ = ferror(file_) ? "I/O error during read" : "Unexpected EOF (truncated file)";
            return false;
        }
        out += got;
        remaining -= got;
    }
    last_error_.clear();
    return true;
}

std::future<bool> FileModelDataSource::readAsync(uint64_t offset, void* destination, size_t nbytes,
                                                 std::atomic<bool>* cancel) {
    return std::async(std::launch::async, [this, offset, destination, nbytes, cancel]() {
        return this->read(offset, destination, nbytes, cancel);
    });
}

} // namespace agr
