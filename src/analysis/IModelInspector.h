#pragma once
#include "analysis/ModelRequirements.h"
#include <string>
#include <memory>

namespace agr {

class IModelInspector {
public:
    virtual ~IModelInspector() = default;

    // Analyzes a model file and returns its requirements/metadata
    virtual ModelRequirements inspect(const std::string& model_path) = 0;
};

} // namespace agr
