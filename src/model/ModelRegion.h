// src/model/ModelRegion.h
// Phase 14: engine-neutral slice of a model file.
#pragma once

#include <cstdint>
#include <string>

namespace agr {

/// Kind of payload a region represents. Not limited to transformer layers.
enum class ModelRegionKind {
    GENERIC,
    TENSOR,
    LAYER,
    WEIGHT_BLOCK,
    ATTENTION_BLOCK,
    EMBEDDING_BLOCK,
    EXPERT,
    CUSTOM
};

inline const char* toString(ModelRegionKind k) {
    switch (k) {
        case ModelRegionKind::GENERIC:          return "GENERIC";
        case ModelRegionKind::TENSOR:           return "TENSOR";
        case ModelRegionKind::LAYER:            return "LAYER";
        case ModelRegionKind::WEIGHT_BLOCK:     return "WEIGHT_BLOCK";
        case ModelRegionKind::ATTENTION_BLOCK:  return "ATTENTION_BLOCK";
        case ModelRegionKind::EMBEDDING_BLOCK:  return "EMBEDDING_BLOCK";
        case ModelRegionKind::EXPERT:           return "EXPERT";
        case ModelRegionKind::CUSTOM:           return "CUSTOM";
        default:                                return "GENERIC";
    }
}

/// Byte range inside a model file. Offset/size are 64-bit so files >4 GB work.
struct ModelRegion {
    uint64_t        id         = 0;
    uint64_t        offset     = 0;
    uint64_t        size       = 0;
    std::string     identifier;
    ModelRegionKind kind       = ModelRegionKind::GENERIC;
    uint32_t        layer      = 0;
    uint32_t        tensor     = 0;

    bool valid() const { return size > 0; }
};

} // namespace agr
