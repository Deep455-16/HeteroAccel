// src/model/GGUFRegionExtractor.h
// Phase 14: extract real tensor byte-ranges from a GGUF file without loading weights.
//
// Uses llama.cpp's vocab_only load to obtain metadata, then probes the file
// for the data section via a lightweight header-only parse. The regions
// produced are fed into ModelResidencyManager as model data sources.
#pragma once

#include "model/ModelRegion.h"

#include <string>
#include <vector>

namespace agr {

/// Lightweight extractor that produces ModelRegions from a GGUF file.
/// It does NOT load tensor weights. It returns byte-range descriptors
/// that callers can hand to FileModelDataSource / ModelResidencyManager.
class GGUFRegionExtractor {
public:
    GGUFRegionExtractor() = default;

    /// Extract all tensor regions from the GGUF file.
    /// Returns an empty vector and sets lastError() on failure.
    /// Assigns sequential region IDs starting at id_base.
    std::vector<ModelRegion> extract(const std::string& path,
                                     uint64_t id_base = 1);

    /// A single synthetic region representing the whole file.
    /// Useful when the full file should be resident (FULL_RESIDENT plan).
    ModelRegion wholeFileRegion(const std::string& path, uint64_t id = 0);

    std::string lastError() const { return last_error_; }

private:
    std::string last_error_;
};

} // namespace agr
