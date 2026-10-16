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

    if (!parseHeader()) return false;
    if (!parseKV())     return false;
    if (!parseTensors()) return false;

    // Compute data section offset: current file position (after tensor infos)
    // padded to the alignment in metadata (general.alignment, default 32).
    uint32_t align = kvUInt32("general.alignment", GGUF_DEFAULT_ALIGN);
    if (align == 0) align = GGUF_DEFAULT_ALIGN;

    uint64_t cur_pos = static_cast<uint64_t>(file_.tellg());
    uint64_t pad = (align - (cur_pos % align)) % align;
    data_offset_ = cur_pos + pad;

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
    if (count > (1u << 20))  {
        last_error_ = "Array count " + std::to_string(count) + " exceeds 1M limit";
        return false;
    }

    auto elem_type = static_cast<GGUFValueType>(elem_type_raw);
    switch (elem_type) {
    case GGUFValueType::UINT32: {
        std::vector<uint32_t> v(static_cast<size_t>(count));
        if (!file_.read(reinterpret_cast<char*>(v.data()),
                        static_cast<std::streamsize>(count * 4))) {
            last_error_ = "EOF reading UINT32 array";
            return false;
        }
        out = std::move(v);
        break;
    }
    case GGUFValueType::INT32: {
        std::vector<int32_t> v(static_cast<size_t>(count));
        if (!file_.read(reinterpret_cast<char*>(v.data()),
                        static_cast<std::streamsize>(count * 4))) {
            last_error_ = "EOF reading INT32 array";
            return false;
        }
        out = std::move(v);
        break;
    }
    case GGUFValueType::FLOAT32: {
        std::vector<float> v(static_cast<size_t>(count));
        if (!file_.read(reinterpret_cast<char*>(v.data()),
                        static_cast<std::streamsize>(count * 4))) {
            last_error_ = "EOF reading FLOAT32 array";
            return false;
        }
        out = std::move(v);
        break;
    }
    case GGUFValueType::STRING: {
        // Array of strings: skip all strings (too large for in-memory storage here)
        // Store as empty string (we do not need string arrays for hparams/vocab)
        std::string skipped;
        for (uint64_t i = 0; i < count; ++i) {
            if (!readString(skipped)) return false;
        }
        out = std::string("<string_array_skipped>");
        break;
    }
    default: {
        // Skip unknown array element types
        // We do not need them for hparams
        out = std::string("<array_type_skipped>");
        // We cannot safely skip without knowing element size.
        // Mark as best-effort parse; caller ignores these keys.
        break;
    }
    }
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
        case GGMLType::Q4_K:
            // 256 elements per block, 144 bytes per block
            desc.n_bytes = ((n_elems + Q4K_BLOCK_SIZE - 1) / Q4K_BLOCK_SIZE) * Q4K_RAW_BLOCK_BYTES;
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

    uint64_t file_offset = data_offset_ + desc.offset;

    file_.seekg(static_cast<std::streamoff>(file_offset));
    if (!file_) return false;

    file_.read(static_cast<char*>(dst), static_cast<std::streamsize>(desc.n_bytes));
    return !!file_;
}

} // namespace hydroxl
} // namespace agr
