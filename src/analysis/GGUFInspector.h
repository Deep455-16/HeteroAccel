#pragma once
#include "analysis/IModelInspector.h"

namespace agr {

class GGUFInspector : public IModelInspector {
public:
    GGUFInspector() = default;
    virtual ~GGUFInspector() = default;

    ModelRequirements inspect(const std::string& model_path) override;
    
private:
    QuantizationType inferQuantization(const std::string& path);
};

} // namespace agr
