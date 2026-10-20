// src/hydroxl/GGUFLoader.cpp
// Phase 17: GGUF binary parser — independent of llama.cpp.
//
// Implements the GGUF v2/v3 format as documented at:
// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

#include "hydroxl/GGUFLoader.h"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace agr {
namespace hydroxl {

static constexpr uint32_t GGUF_MAGIC   = 0x46554747; // 'GGUF'
static constexpr uint32_t GGUF_VER_MIN = 2;
static constexpr uint32_t GGUF_VER_MAX = 3;
static constexpr uint32_t GGUF_DEFAULT_ALIGN = 32;

bool GGUFLoader::open(const std::string& path) {
    path_ = path;
    file_.open(path, std::ios::binary);
    if (!file_.is_open()) {
        last_error_ = "Cannot open file: " + path;
        return false;
    }

    // Record total file size for bounds checking during tensor reads.
    file_.seekg(0, std::ios::end);
    if (!file_) {
        last_error_ = "Cannot seek file: " + path;
        return false;
    }
    file_size_ = static_cast<uint64_t>(file_.tellg());
    file_.seekg(0, std::ios::beg);
    if (file_size_ < 12) { // magic + version + two counts is the bare minimum
        last_error_ = "File too small to be GGUF: " + path;
        return false;
    }

    if (!parseHeader()) return false;
    if (!parseKV())     return false;
    if (!parseTensors()) return false;

    // Compute data section offset: current file position (after tensor infos)
    // padded to the alignment in metadata (general.alignment, default 32).
    uint32_t align = kvUInt32("general.alignment", GGUF_DEFAULT_ALIGN);
    if (align == 0) align = GGUF_DEFAULT_ALIGN;

    uint64_t cur_pos = static_cast<uint64_t>(file_.tellg());
    uint64_t pad = (align - (cur_pos % align)) % align;
    if (cur_pos > UINT64_MAX - pad) {
        last_error_ = "Data offset overflow while padding to alignment";
        return false;
    }
    data_offset_ = cur_pos + pad;

    // When there are no tensors (metadata-only file) the data section can
    // be aligned to a position that equals or exceeds the file size — that
    // is valid because there is nothing to read past the KV metadata.
    // Only reject if tensors are declared but the data section is unreachable.
    if (n_tensors_ > 0 && data_offset_ >= file_size_) {
        last_error_ = "Data section offset " + std::to_string(data_offset_) +
                      " is at or beyond file size " + std::to_string(file_size_) +
                      " with " + std::to_string(n_tensors_) + " declared tensors";
        return false;
    }

    open_ = true;
    return true;
}

bool GGUFLoader::parseHeader() {
    uint32_t magic = 0;
    if (!readRaw(magic)) return false;
    if (magic != GGUF_MAGIC) {
        last_error_ = "Not a GGUF file (bad magic)";
        return false;
    }

    if (!readRaw(version_)) return false;
    if (version_ < GGUF_VER_MIN || version_ > GGUF_VER_MAX) {
        last_error_ = "Unsupported GGUF version: " + std::to_string(version_);
        return false;
    }

    if (!readRaw(n_tensors_)) return false;
    if (!readRaw(n_kv_))      return false;

    // Sanity checks
    if (n_tensors_ > 65536) {
        last_error_ = "n_tensors exceeds 65536 — corrupt file?";
        return false;
    }
    if (n_kv_ > 65536) {
        last_error_ = "n_kv exceeds 65536 — corrupt file?";
        return false;
    }

    return true;
}

bool GGUFLoader::readString(std::string& out) {
    uint64_t len = 0;
    if (!readRaw(len)) return false;
    if (len > (1 << 20)) { // 1 MiB sanity limit per string
        last_error_ = "String length " + std::to_string(len) + " exceeds 1 MiB limit";
        return false;
    }
    out.resize(static_cast<size_t>(len));
    if (len > 0 && !file_.read(out.data(), static_cast<std::streamsize>(len))) {
        last_error_ = "EOF reading string of length " + std::to_string(len);
        return false;
    }
    return true;
}

bool GGUFLoader::readArray(GGUFValue& out) {
    uint32_t elem_type_raw = 0;
    uint64_t count = 0;
    if (!readRaw(elem_type_raw)) return false;
    if (!readRaw(count))         return false;
    if (count > (1u << 24))  {
        last_error_ = "Array count " + std::to_string(count) + " exceeds 16M limit";
        return false;
    }

    auto elem_type = static_cast<GGUFValueType>(elem_type_raw);

    // Retained array types (tokenizer vocab, merges, token types, scores).
    switch (elem_type) {
    case GGUFValueType::UINT32: {
        std::vector<uint32_t> v(static_cast<size_t>(count));
        if (!file_.read(reinterpret_cast<char*>(v.data()),
                        static_cast<std::streamsize>(count * 4))) {
            last_error_ = "EOF reading UINT32 array";
            return false;
        }
        out = std::move(v);
        return true;
    }
    case GGUFValueType::INT32: {
        std::vector<int32_t> v(static_cast<size_t>(count));
        if (!file_.read(reinterpret_cast<char*>(v.data()),
                        static_cast<std::streamsize>(count * 4))) {
            last_error_ = "EOF reading INT32 array";
            return false;
        }
        out = std::move(v);
        return true;
    }
    case GGUFValueType::FLOAT32: {
        std::vector<float> v(static_cast<size_t>(count));
        if (!file_.read(reinterpret_cast<char*>(v.data()),
                        static_cast<std::streamsize>(count * 4))) {
            last_error_ = "EOF reading FLOAT32 array";
            return false;
        }
        out = std::move(v);
        return true;
    }
    case GGUFValueType::STRING: {
        std::vector<std::string> v;
        v.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i) {
            std::string s;
            if (!readString(s)) return false;
            v.push_back(std::move(s));
        }
        out = std::move(v);
        return true;
    }
    default:
        break; // not retained — skip exactly, below
    }

    // Skipped-but-known element types: consume exactly count * elem_size
    // bytes (or recursing for nested arrays) so the stream stays in sync.
    uint64_t elem_size = 0;
    switch (elem_type) {
    case GGUFValueType::UINT8:
    case GGUFValueType::INT8:
    case GGUFValueType::BOOL:   elem_size = 1; break;
    case GGUFValueType::UINT16:
    case GGUFValueType::INT16:  elem_size = 2; break;
    case GGUFValueType::UINT64:
    case GGUFValueType::INT64:
    case GGUFValueType::FLOAT64: elem_size = 8; break;
    case GGUFValueType::ARRAY: {
        // Nested arrays: recurse so variable-size elements are consumed
        // exactly. Discard the values.
        for (uint64_t i = 0; i < count; ++i) {
            GGUFValue discarded;
            if (!readArray(discarded)) return false;
        }
        out = std::string("<array_type_skipped>");
        return true;
    }
    default:
        // Unknown element type: the element size is undefined, so the
        // stream position after this value cannot be known. Reject the
        // file rather than silently mis-parsing everything that follows.
        last_error_ = "Unknown array element type: " +
                      std::to_string(elem_type_raw);
        return false;
    }

    if (count > 0) {
        uint64_t skip = count * elem_size; // count <= 16M, elem_size <= 8: no overflow
        if (!file_.seekg(static_cast<std::streamoff>(
                static_cast<uint64_t>(file_.tellg()) + skip))) {
            last_error_ = "EOF skipping array of type " +
                          std::to_string(elem_type_raw);
            return false;
        }
    }
    out = std::string("<array_type_skipped>");
    return true;
}

bool GGUFLoader::readValue(GGUFValueType vtype, GGUFValue& out) {
    switch (vtype) {
    case GGUFValueType::UINT8:  { uint8_t  v; if (!readRaw(v)) return false; out = static_cast<uint32_t>(v); break; }
    case GGUFValueType::INT8:   { int8_t   v; if (!readRaw(v)) return false; out = static_cast<int32_t>(v);  break; }
    case GGUFValueType::UINT16: { uint16_t v; if (!readRaw(v)) return false; out = static_cast<uint32_t>(v); break; }
    case GGUFValueType::INT16:  { int16_t  v; if (!readRaw(v)) return false; out = static_cast<int32_t>(v);  break; }
    case GGUFValueType::UINT32: { uint32_t v; if (!readRaw(v)) return false; out = v; break; }
    case GGUFValueType::INT32:  { int32_t  v; if (!readRaw(v)) return false; out = v; break; }
    case GGUFValueType::FLOAT32:{ float    v; if (!readRaw(v)) return false; out = v; break; }
    case GGUFValueType::BOOL:   { uint8_t  v; if (!readRaw(v)) return false; out = (v != 0); break; }
    case GGUFValueType::STRING: { std::string s; if (!readString(s)) return false; out = std::move(s); break; }
    case GGUFValueType::ARRAY:  { return readArray(out); }
    case GGUFValueType::UINT64: { uint64_t v; if (!readRaw(v)) return false; out = v; break; }
    case GGUFValueType::INT64:  { int64_t  v; if (!readRaw(v)) return false; out = static_cast<int64_t>(v); break; }
    case GGUFValueType::FLOAT64:{ double   v; if (!readRaw(v)) return false; out = v; break; }
    default:
        last_error_ = "Unknown GGUF value type: " + std::to_string(static_cast<uint32_t>(vtype));
        return false;
    }
    return true;
}

bool GGUFLoader::parseKV() {
    for (uint64_t i = 0; i < n_kv_; ++i) {
        std::string key;
        if (!readString(key)) return false;

        uint32_t vtype_raw = 0;
        if (!readRaw(vtype_raw)) return false;
        auto vtype = static_cast<GGUFValueType>(vtype_raw);

        GGUFValue val;
        if (!readValue(vtype, val)) return false;

        kv_[key] = std::move(val);
    }
    return true;
}

bool GGUFLoader::parseTensors() {
    tensors_.reserve(static_cast<size_t>(n_tensors_));

    for (uint64_t i = 0; i < n_tensors_; ++i) {
        GGUFTensorDesc desc;

        // Name
        if (!readString(desc.name)) return false;

        // n_dims
        uint32_t n_dims = 0;
        if (!readRaw(n_dims)) return false;
        if (n_dims > 4) {
            last_error_ = "Tensor '" + desc.name + "' has " +
                          std::to_string(n_dims) + " dims (max 4)";
            return false;
        }

        // Dimensions (innermost = dim[0])
        desc.shape.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            if (!readRaw(desc.shape[d])) return false;
            if (desc.shape[d] > (1ull << 32)) {
                last_error_ = "Tensor '" + desc.name + "' dim[" + std::to_string(d) +
                              "] = " + std::to_string(desc.shape[d]) + " is too large";
                return false;
            }
        }

        // Type
        uint32_t type_raw = 0;
        if (!readRaw(type_raw)) return false;
        desc.type = static_cast<GGMLType>(type_raw);

        // Offset (relative to start of data section)
        if (!readRaw(desc.offset)) return false;

        // Compute n_bytes based on type and shape
        uint64_t n_elems = 1;
        for (auto d : desc.shape) n_elems *= d;

        switch (desc.type) {
        case GGMLType::F32:
            desc.n_bytes = n_elems * 4;
            break;
        case GGMLType::F16:
            desc.n_bytes = n_elems * 2;
            break;
        case GGMLType::Q4_0:
            // 32 elements per block, 18 bytes per block
            desc.n_bytes = ((n_elems + Q4_0_BLOCK_SIZE - 1) / Q4_0_BLOCK_SIZE) * Q4_0_BLOCK_BYTES;
            break;
        case GGMLType::Q4_1:
            // 32 elements per block, 20 bytes per block
            desc.n_bytes = ((n_elems + Q4_1_BLOCK_SIZE - 1) / Q4_1_BLOCK_SIZE) * Q4_1_BLOCK_BYTES;
            break;
        case GGMLType::Q4_K:
            // 256 elements per block, 144 bytes per block
            desc.n_bytes = ((n_elems + Q4K_BLOCK_SIZE - 1) / Q4K_BLOCK_SIZE) * Q4K_RAW_BLOCK_BYTES;
            break;
        case GGMLType::Q3_K:
            // 256 elements per block, 110 bytes per block
            desc.n_bytes = ((n_elems + Q3K_BLOCK_SIZE - 1) / Q3K_BLOCK_SIZE) * Q3K_RAW_BLOCK_BYTES;
            break;
        case GGMLType::Q5_0:
            // 32 elements per block, 22 bytes per block
            desc.n_bytes = ((n_elems + 31) / 32) * 22;
            break;
        case GGMLType::Q5_1:
            // 32 elements per block, 24 bytes per block
            desc.n_bytes = ((n_elems + 31) / 32) * 24;
            break;
        case GGMLType::Q6_K:
            // 256 elements per block, 210 bytes per block
            desc.n_bytes = ((n_elems + 255) / 256) * 210;
            break;
        case GGMLType::Q8_0:
            desc.n_bytes = ((n_elems + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;
            break;
        case GGMLType::Q8_1:
            // 32 elements per block, 36 bytes per block (d: fp16, s: fp16, qs: 32 x int8)
            desc.n_bytes = ((n_elems + 31) / 32) * 36;
            break;
        case GGMLType::Q2_K:
            // 256 elements per block, 84 bytes per block
            desc.n_bytes = ((n_elems + 255) / 256) * 84;
            break;
        case GGMLType::Q5_K:
            // 256 elements per block, 176 bytes per block
            desc.n_bytes = ((n_elems + 255) / 256) * 176;
            break;
        case GGMLType::Q8_K:
            // 256 elements per block, 292 bytes per block
            desc.n_bytes = ((n_elems + 255) / 256) * 292;
            break;
        default:
            // Unknown — store 0; loader will skip/reject
            desc.n_bytes = 0;
            break;
        }

        tensors_.push_back(std::move(desc));
    }
    return true;
}

const GGUFValue* GGUFLoader::kv(const std::string& key) const {
    auto it = kv_.find(key);
    return (it == kv_.end()) ? nullptr : &it->second;
}

uint32_t GGUFLoader::kvUInt32(const std::string& key, uint32_t def) const {
    auto* v = kv(key);
    if (!v) return def;
    if (auto* p = std::get_if<uint32_t>(v)) return *p;
    if (auto* p = std::get_if<int32_t>(v))  return static_cast<uint32_t>(*p);
    if (auto* p = std::get_if<uint64_t>(v)) return static_cast<uint32_t>(*p);
    return def;
}

uint64_t GGUFLoader::kvUInt64(const std::string& key, uint64_t def) const {
    auto* v = kv(key);
    if (!v) return def;
    if (auto* p = std::get_if<uint64_t>(v)) return *p;
    if (auto* p = std::get_if<uint32_t>(v)) return static_cast<uint64_t>(*p);
    return def;
}

int32_t GGUFLoader::kvInt32(const std::string& key, int32_t def) const {
    auto* v = kv(key);
    if (!v) return def;
    if (auto* p = std::get_if<int32_t>(v))  return *p;
    if (auto* p = std::get_if<uint32_t>(v)) return static_cast<int32_t>(*p);
    return def;
}

float GGUFLoader::kvFloat(const std::string& key, float def) const {
    auto* v = kv(key);
    if (!v) return def;
    if (auto* p = std::get_if<float>(v))  return *p;
    if (auto* p = std::get_if<double>(v)) return static_cast<float>(*p);
    return def;
}

std::string GGUFLoader::kvString(const std::string& key, const std::string& def) const {
    auto* v = kv(key);
    if (!v) return def;
    if (auto* p = std::get_if<std::string>(v)) return *p;
    return def;
}

std::vector<std::string> GGUFLoader::kvStringArray(const std::string& key) const {
    auto* v = kv(key);
    if (!v) return {};
    if (auto* p = std::get_if<std::vector<std::string>>(v)) return *p;
    return {};
}

const GGUFTensorDesc* GGUFLoader::findTensor(const std::string& name) const {
    for (const auto& t : tensors_) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

bool GGUFLoader::readTensorData(const GGUFTensorDesc& desc,
                                 void* dst, uint64_t max_bytes) const {
    if (!open_) { return false; }
    if (desc.n_bytes == 0 || !dst) return false;
    if (desc.n_bytes > max_bytes) return false;

    // Overflow-safe bounds check against the real file size.
    if (desc.offset > file_size_ || desc.n_bytes > file_size_ - desc.offset) {
        return false;
    }
    uint64_t file_offset = data_offset_ + desc.offset;
    if (file_offset > file_size_ || desc.n_bytes > file_size_ - file_offset) {
        return false;
    }

    file_.seekg(static_cast<std::streamoff>(file_offset));
    if (!file_) return false;

    file_.read(static_cast<char*>(dst), static_cast<std::streamsize>(desc.n_bytes));
    return !!file_;
}

} // namespace hydroxl
} // namespace agr
