// src/engine/ModelDescriptor.cpp
#include "engine/ModelDescriptor.h"
#include <sstream>

namespace agr {

std::string ModelDescriptor::summary() const {
    std::ostringstream ss;
    ss << "[" << model_id << "] " << toString(format);
    if (!architecture.empty()) ss << " " << architecture;
    if (!quantization.empty()) ss << "/" << quantization;
    if (size_bytes.has_value())
        ss << " " << (size_bytes.value() / (1024*1024)) << " MiB";
    return ss.str();
}

} // namespace agr
