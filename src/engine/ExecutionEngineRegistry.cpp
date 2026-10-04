// src/engine/ExecutionEngineRegistry.cpp
#include "engine/ExecutionEngineRegistry.h"
#include "engine/LlamaCppExecutionEngine.h"

namespace agr {

ExecutionEngineRegistry::ExecutionEngineRegistry() {
    // Built-in Phase 11 registration.
    // Done here explicitly to avoid MSVC static library linker stripping.
    registerFactory("llama.cpp", "llama.cpp LLM backend", []() {
        return std::make_shared<LlamaCppExecutionEngine>();
    });
}

ExecutionEngineRegistry& ExecutionEngineRegistry::instance() {
    static ExecutionEngineRegistry s_instance;
    return s_instance;
}

void ExecutionEngineRegistry::registerFactory(const std::string& key,
                                              const std::string& description,
                                              EngineFactory factory) {
    // Intentionally not locked: call from one thread at startup.
    EngineRegistration reg;
    reg.key         = key;
    reg.description = description;
    reg.factory     = std::move(factory);
    factories_[key] = std::move(reg);
}

std::shared_ptr<IExecutionEngine> ExecutionEngineRegistry::create(
    const std::string& key) const
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = factories_.find(key);
    if (it == factories_.end()) return nullptr;
    return it->second.factory();
}

bool ExecutionEngineRegistry::isRegistered(const std::string& key) const {
    std::lock_guard<std::mutex> lk(mutex_);
    return factories_.count(key) > 0;
}

std::vector<std::string> ExecutionEngineRegistry::registeredKeys() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<std::string> keys;
    keys.reserve(factories_.size());
    for (auto& kv : factories_) keys.push_back(kv.first);
    return keys;
}

const EngineRegistration* ExecutionEngineRegistry::registration(
    const std::string& key) const
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = factories_.find(key);
    return (it == factories_.end()) ? nullptr : &it->second;
}

void ExecutionEngineRegistry::clear() {
    std::lock_guard<std::mutex> lk(mutex_);
    factories_.clear();
}

} // namespace agr
