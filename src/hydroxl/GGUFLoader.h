// src/hydroxl/GGUFLoader.h
// Phase 17: HydroXL native GGUF file parser.
//
// Parses GGUF v2/v3 file headers and tensor descriptors WITHOUT using
// any llama.cpp API. All parsing is done against the raw binary file.
// Tensor weights are loaded on demand by the model loader.
//
// GGUF binary format (little-endian):
//   magic         : uint32  = 0x46554747 ('GGUF')
//   version       : uint32  (2 or 3)
//   n_tensors     : uint64
//   n_kv          : uint64
//   kv entries    : key-value metadata pairs
//   tensor infos  : tensor name, ndim, dims, type, offset
//   alignment pad
//   tensor data   : raw quantized bytes
//
// See: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

#pragma once

#include "hydroxl/HydroXLTypes.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace agr {
namespace hydroxl {

// ---------------------------------------------------------------------------
// GGUF KV value union — only types HydroXL needs
// ---------------------------------------------------------------------------
using GGUFValue = std::variant<
    uint32_t,    // UINT32
    uint64_t,    // UINT64
    int32_t,     // INT32
    int64_t,     // INT64
    float,       // FLOAT32
    double,      // FLOAT64
    bool,        // BOOL
    std::string, // STRING
    std::vector<uint32_t>,
    std::vector<int32_t>,
    std::vector<float>,
    std::vector<std::string>  // STRING arrays (tokenizer vocab / merges)
>;

// ---------------------------------------------------------------------------
// GGUFLoader: parses the header and builds metadata + tensor descriptors
// ---------------------------------------------------------------------------
class GGUFLoader {
public:
    GGUFLoader() = default;
    ~GGUFLoader() = default;

    // Non-copyable, movable
    GGUFLoader(const GGUFLoader&) = delete;
    GGUFLoader& operator=(const GGUFLoader&) = delete;
    GGUFLoader(GGUFLoader&&) = default;

    /// Open and parse a GGUF file.
    /// Returns true on success. On failure, lastError() contains the reason.
    bool open(const std::string& path);

    /// Whether open() has succeeded.
    bool isOpen() const { return open_; }

    // -----------------------------------------------------------------------
    // Metadata accessors
    // -----------------------------------------------------------------------

    /// Lookup a metadata key. Returns nullptr if not found.
    const GGUFValue* kv(const std::string& key) const;

    /// Convenience: extract uint32 metadata value. Returns 0 if not found.
    uint32_t kvUInt32(const std::string& key, uint32_t def = 0) const;
    uint64_t kvUInt64(const std::string& key, uint64_t def = 0) const;
    int32_t  kvInt32 (const std::string& key, int32_t  def = 0) const;
    float    kvFloat (const std::string& key, float    def = 0.0f) const;
    std::string kvString(const std::string& key, const std::string& def = "") const;

    /// Convenience: extract a STRING-array metadata value (e.g. tokenizer
    /// vocabulary or merges). Returns an empty vector if the key is absent
    /// or is not a retained string array.
    std::vector<std::string> kvStringArray(const std::string& key) const;

    /// All key-value entries.
    const std::unordered_map<std::string, GGUFValue>& metadata() const { return kv_; }

    // -----------------------------------------------------------------------
    // Tensor descriptors
    // -----------------------------------------------------------------------
    const std::vector<GGUFTensorDesc>& tensors() const { return tensors_; }

    /// Find tensor by name. Returns nullptr if not found.
    const GGUFTensorDesc* findTensor(const std::string& name) const;

    /// Byte offset from the START of the file to the tensor data section.
    uint64_t dataOffset() const { return data_offset_; }

    /// Total size of the opened file in bytes (0 until open() succeeds).
    uint64_t fileSize() const { return file_size_; }

    /// File path.
    const std::string& path() const { return path_; }

    std::string lastError() const { return last_error_; }

    // -----------------------------------------------------------------------
    // Raw tensor data reading
    // -----------------------------------------------------------------------

    /// Read raw bytes for a tensor (from its offset in the data section).
    /// Caller allocates dst of at least desc.n_bytes bytes.
    /// Returns false on IO error or range violation.
    bool readTensorData(const GGUFTensorDesc& desc, void* dst, uint64_t max_bytes) const;

private:
    bool parseHeader();
    bool parseKV();
    bool parseTensors();
    bool readString(std::string& out);
    bool readValue(GGUFValueType vtype, GGUFValue& out);
    bool readArray(GGUFValue& out);

    template<typename T>
    bool readRaw(T& out) {
        if (!file_.read(reinterpret_cast<char*>(&out), sizeof(T))) {
            last_error_ = "Unexpected EOF reading " + std::to_string(sizeof(T)) + " bytes";
            return false;
        }
        return true;
    }

    std::string path_;
    mutable std::ifstream file_;
    bool open_   = false;

    uint32_t version_    = 0;
    uint64_t n_tensors_  = 0;
    uint64_t n_kv_       = 0;
    uint64_t data_offset_ = 0;  ///< Byte offset from start of file to first tensor data
    uint64_t file_size_   = 0;  ///< Total file size, used for bounds checks

    std::unordered_map<std::string, GGUFValue> kv_;
    std::vector<GGUFTensorDesc> tensors_;

    std::string last_error_;
};

} // namespace hydroxl
} // namespace agr
