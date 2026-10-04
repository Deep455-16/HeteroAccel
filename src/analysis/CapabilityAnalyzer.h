#pragma once
#include "analysis/CapabilityReport.h"
#include "engine/ExecutionEngineRegistry.h"
#include "backend/BackendManager.h"
#include "hardware/HardwareDetector.h"

namespace agr {

class CapabilityAnalyzer {
public:
    CapabilityAnalyzer() = default;
    
    CapabilityReport analyze(const ModelRequirements& req,
                             const ExecutionEngineRegistry& registry,
                             BackendManager* backend_mgr,
                             const HardwareInfo& hardware);
};

} // namespace agr
