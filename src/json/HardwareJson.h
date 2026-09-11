#pragma once
#include "core/HardwareInfo.h"
#include <string>

namespace agr {

// Serializes a HardwareInfo into machine-readable JSON. Intentionally
// hand-rolled (no external JSON dependency) since the required shape
// is small and fixed -- keeps the dependency footprint minimal per
// project ground rules.
std::string toJson(const HardwareInfo& info, bool pretty = true);

} // namespace agr
