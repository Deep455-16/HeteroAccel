// src/orchestration/OrchestratorInit.h
// Phase 21: Default Backend Registration

#pragma once

#include "orchestration/EngineOrchestrator.h"
#include "orchestration/HardwareDetector.h"

namespace agr {
namespace orchestration {

void register_default_backends(EngineOrchestrator& orch, const HardwareDetector& hw);

} // namespace orchestration
} // namespace agr
