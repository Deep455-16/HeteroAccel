// src/primitives/OpResult.h
// Phase 16: Operation result type for CPU primitives.
//
// All CPU operations return OpResult. Callers must check success before
// using output tensors. Error messages are human-readable strings suitable
// for logging or test assertions.

#pragma once

#include <string>

namespace agr {
namespace primitives {

struct OpResult {
    bool        success = false;
    std::string error;

    static OpResult ok()                         { return {true, ""}; }
    static OpResult fail(std::string msg)        { return {false, std::move(msg)}; }
};

} // namespace primitives
} // namespace agr
