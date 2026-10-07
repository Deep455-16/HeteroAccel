#pragma once

#include "model/ModelRegion.h"

#include <chrono>
#include <cstdint>
#include <string>

namespace agr {

enum class ModelResidencyState {
    NOT_RESIDENT,
    LOADING_TO_RAM,
    RESIDENT_IN_RAM,
    LOADING_TO_ACCELERATOR,
    RESIDENT_IN_ACCELERATOR,
    EVICTING,
    ERROR
};

inline const char* toString(ModelResidencyState s) {
    switch (s) {
        case ModelResidencyState::NOT_RESIDENT:            return "NOT_RESIDENT";
        case ModelResidencyState::LOADING_TO_RAM:          return "LOADING_TO_RAM";
        case ModelResidencyState::RESIDENT_IN_RAM:         return "RESIDENT_IN_RAM";
        case ModelResidencyState::LOADING_TO_ACCELERATOR:  return "LOADING_TO_ACCELERATOR";
        case ModelResidencyState::RESIDENT_IN_ACCELERATOR: return "RESIDENT_IN_ACCELERATOR";
        case ModelResidencyState::EVICTING:                return "EVICTING";
        case ModelResidencyState::ERROR:                   return "ERROR";
        default:                                           return "UNKNOWN";
    }
}

enum class ModelResidencyTarget {
    RAM,
    ACCELERATOR
};

struct ModelResidencyBudget {
    uint64_t ram_bytes = 64ull * 1024 * 1024;
    uint64_t accelerator_bytes = 64ull * 1024 * 1024;
    uint64_t prefetch_bytes = 16ull * 1024 * 1024;
};

struct StreamingTelemetry {
    uint64_t bytes_read_from_disk = 0;
    uint64_t bytes_to_ram = 0;
    uint64_t bytes_to_accelerator = 0;
    double   read_ms = 0.0;
    double   upload_ms = 0.0;
    uint64_t eviction_count = 0;
    uint64_t prefetch_count = 0;
    uint64_t prefetch_hits = 0;
    uint64_t prefetch_misses = 0;
    uint64_t resident_ram_bytes = 0;
    uint64_t resident_accelerator_bytes = 0;
    uint64_t peak_ram_bytes = 0;
    uint64_t peak_accelerator_bytes = 0;
    uint64_t stall_count = 0;
    uint64_t transfer_failures = 0;
    std::string last_error;
};

} // namespace agr
