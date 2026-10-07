// src/model/GGUFRegionExtractor.cpp
// Phase 14: extract tensor byte-ranges from a GGUF file.
//
// Strategy: use llama_model_load_from_file with vocab_only=true to read
// metadata.  The llama.cpp API exposes n_tensors and per-tensor metadata
// via llama_model_tensor_* (available in llama.cpp ≥ b3000).  Where that
// API is available we enumerate tensors directly.  Where it is not yet
// available (or where the API differs) we fall back to a single synthetic
// "whole-file" region so the residency manager still works end-to-end.
//
// Callers that need precise per-tensor streaming should use the whole-file
// region as a single-chunk fallback; future engine integration (Colibrì,
// HydroXL) can refine this when those APIs are available.

#include "model/GGUFRegionExtractor.h"
#include "llama.h"

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace agr {

// ---------------------------------------------------------------------------
// wholeFileRegion — always works, even without per-tensor API
// ---------------------------------------------------------------------------
ModelRegion GGUFRegionExtractor::wholeFileRegion(const std::string& path,
                                                  uint64_t id) {
    ModelRegion r;
    r.id         = id;
    r.offset     = 0;
    r.identifier = path;
    r.kind       = ModelRegionKind::GENERIC;
    r.layer      = 0;
    r.tensor     = 0;

    std::error_code ec;
    auto sz = std::filesystem::file_size(path, ec);
    r.size = ec ? 0 : static_cast<uint64_t>(sz);
    if (r.size == 0) last_error_ = "Cannot stat file: " + path;
    return r;
}

// ---------------------------------------------------------------------------
// extract — enumerate tensors via llama.cpp metadata API.
//
// llama.cpp exposes model-level tensor count and per-tensor name/offset/size
// through:
//   llama_model_n_tensors(model)   → int32_t
//   llama_model_tensor_name(model, i)        → const char*
//   llama_model_tensor_data_offset(model, i) → uint64_t (byte offset in file)
//   llama_model_tensor_data_size(model, i)   → uint64_t
//
// These were added incrementally; we probe at compile-time with
// LLAMA_HAS_TENSOR_METADATA. If the build-time llama.cpp does not expose
// them, we fall back to a single whole-file region.
// ---------------------------------------------------------------------------

std::vector<ModelRegion> GGUFRegionExtractor::extract(const std::string& path,
                                                       uint64_t id_base) {
    last_error_.clear();

    // Probe file size first.
    std::error_code ec;
    uint64_t file_size = 0;
    {
        auto sz = std::filesystem::file_size(path, ec);
        if (ec) {
            last_error_ = "File not found or unreadable: " + path;
            return {};
        }
        file_size = static_cast<uint64_t>(sz);
    }

    // Load with vocab_only=true — no weights are allocated.
    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;
    mparams.use_mmap   = false;

    llama_model* model = llama_model_load_from_file(path.c_str(), mparams);
    if (!model) {
        last_error_ = "llama_model_load_from_file failed for: " + path;
        return {};
    }

    std::vector<ModelRegion> regions;

#if defined(LLAMA_HAS_TENSOR_METADATA)
    // Full per-tensor enumeration — only compiled when the API is present.
    int32_t n = llama_model_n_tensors(model);
    regions.reserve(static_cast<size_t>(n > 0 ? n : 0));

    for (int32_t i = 0; i < n; ++i) {
        const char* tname = llama_model_tensor_name(model, i);
        uint64_t    toff  = llama_model_tensor_data_offset(model, i);
        uint64_t    tsz   = llama_model_tensor_data_size(model, i);

        if (!tname || tsz == 0) continue;
        if (toff + tsz > file_size) continue;  // guard against corrupt metadata

        ModelRegion r;
        r.id         = id_base + static_cast<uint64_t>(i);
        r.offset     = toff;
        r.size       = tsz;
        r.identifier = tname;
        r.kind       = ModelRegionKind::TENSOR;
        r.layer      = 0;   // layer index not exposed by this API directly
        r.tensor     = static_cast<uint32_t>(i);
        regions.push_back(r);
    }
#else
    // Fallback: one whole-file region.
    // The residency manager handles this correctly; per-tensor granularity
    // will be added when llama.cpp exposes the required metadata API.
    (void)file_size; // used above already
    ModelRegion whole;
    whole.id         = id_base;
    whole.offset     = 0;
    whole.size       = file_size;
    whole.identifier = path;
    whole.kind       = ModelRegionKind::GENERIC;
    regions.push_back(whole);
#endif

    llama_model_free(model);

    if (regions.empty()) {
        // Even if no tensors found, return whole-file fallback.
        ModelRegion whole;
        whole.id         = id_base;
        whole.offset     = 0;
        whole.size       = file_size;
        whole.identifier = path;
        whole.kind       = ModelRegionKind::GENERIC;
        regions.push_back(whole);
    }

    return regions;
}

} // namespace agr
