#include "analysis/CapabilityReport.h"
#include <sstream>
#include <iomanip>

namespace agr {

std::string CapabilityReport::formatToString() const {
    std::ostringstream oss;
    oss << "Model Analysis\n";
    oss << "==============\n\n";

    oss << "Model:\n";
    oss << "  Format: " << toString(model.format) << "\n";
    oss << "  Architecture: " << model.architecture.value_or("unknown") << "\n";
    
    if (model.parameter_count) {
        double params_b = *model.parameter_count / 1e9;
        oss << "  Parameters: " << std::fixed << std::setprecision(2) << params_b << "B\n";
    } else {
        oss << "  Parameters: unknown\n";
    }
    
    oss << "  Quantization: " << toString(model.quantization) << "\n";
    oss << "  Modality: " << toString(model.modality) << "\n";
    
    if (model.context_length) {
        oss << "  Context Length: " << *model.context_length << "\n";
    }
    
    oss << "\nMemory:\n";
    if (model.file_size_bytes) {
        oss << "  File size: " << (*model.file_size_bytes / (1024 * 1024)) << " MB\n";
    }
    if (model.estimated_memory_bytes) {
        oss << "  Estimated required memory: " << (*model.estimated_memory_bytes / (1024 * 1024)) << " MB\n";
    } else {
        oss << "  Estimated required memory: unknown\n";
    }

    oss << "\nEngines:\n";
    if (engines.empty()) {
        oss << "  None registered.\n";
    }
    for (const auto& eng : engines) {
        oss << "  " << eng.engine_name << ": " << (eng.supported ? "COMPATIBLE" : "INCOMPATIBLE") << "\n";
        if (!eng.supported && !eng.reason.empty()) {
            oss << "    Reason: " << eng.reason << "\n";
        }
    }

    oss << "\nDevices:\n";
    for (const auto& dev : devices) {
        oss << "  " << dev.backend_type << " (" << dev.device_name << "): ";
        if (!dev.available) {
            oss << "UNAVAILABLE\n";
            if (!dev.reason.empty()) oss << "    Reason: " << dev.reason << "\n";
        } else {
            oss << (dev.supported ? "FEASIBLE" : "INFEASIBLE") << "\n";
            if (!dev.supported && !dev.reason.empty()) {
                oss << "    Reason: " << dev.reason << "\n";
            }
        }
    }

    oss << "\nResource Feasibility:\n";
    if (resources.full_residency_feasible) {
        oss << "  Full Residency: FEASIBLE\n";
    } else {
        oss << "  Full Residency: INFEASIBLE\n";
        if (!resources.memory_bottleneck_reason.empty()) {
            oss << "    Reason: " << resources.memory_bottleneck_reason << "\n";
        }
    }
    oss << "  Streaming: " << (resources.streaming_potentially_feasible ? "POTENTIALLY FEASIBLE" : "INFEASIBLE") << "\n";

    if (!warnings.empty()) {
        oss << "\nWarnings / Limitations:\n";
        for (const auto& w : warnings) {
            oss << "  * " << w << "\n";
        }
    }

    if (historical_observations_available) {
        oss << "\n[Historical observations available for this model]\n";
    }

    oss << "\nExecution planning:\n";
    oss << "  NOT PERFORMED\n";

    return oss.str();
}

} // namespace agr
