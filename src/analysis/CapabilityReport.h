#pragma once
#include "analysis/ModelRequirements.h"
#include <vector>
#include <string>

namespace agr {

struct EngineCompatibility {
    std::string engine_name;
    bool supported = false;
    std::string reason;
};

struct DeviceCompatibility {
    std::string device_name;
    std::string backend_type;
    bool available = false;
    bool supported = false;
    uint64_t memory_capacity = 0;
    std::string reason;
};

struct ResourceFeasibility {
    bool full_residency_feasible = false;
    bool streaming_potentially_feasible = false;
    std::string memory_bottleneck_reason;
};

struct CapabilityReport {
    ModelRequirements model;
    std::vector<EngineCompatibility> engines;
    std::vector<DeviceCompatibility> devices;
    ResourceFeasibility resources;
    std::vector<std::string> warnings;
    bool historical_observations_available = false;

    std::string formatToString() const;
};

} // namespace agr
