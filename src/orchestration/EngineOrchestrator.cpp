// src/orchestration/EngineOrchestrator.cpp
// Phase 21: Engine Orchestrator

#include "orchestration/EngineOrchestrator.h"
#include "orchestration/HydroXLEngineAdapter.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace agr {
namespace orchestration {

EngineOrchestrator& EngineOrchestrator::instance() {
    static EngineOrchestrator inst;
    return inst;
}

EngineOrchestrator::EngineOrchestrator() {
    // Register default built-in engine factories
    factories_["hydroxl_cpu"] = []() {
        return std::make_shared<HydroXLCpuEngineAdapter>();
    };
    factories_["hydroxl_vulkan"] = []() {
        return std::make_shared<HydroXLVulkanEngineAdapter>();
    };
}

void EngineOrchestrator::register_backend(BackendCapabilities caps) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& b : backends_) {
        if (b.backend_id == caps.backend_id) {
            b = caps;
            return;
        }
    }
    backends_.push_back(std::move(caps));
}

void EngineOrchestrator::register_engine_factory(const std::string& backend_id, EngineFactory factory) {
    std::lock_guard<std::mutex> lk(mu_);
    factories_[backend_id] = std::move(factory);
}

void EngineOrchestrator::reset() {
    std::lock_guard<std::mutex> lk(mu_);
    backends_.clear();
    perf_history_.clear();
    rss_history_.clear();
    factories_.clear();
    factories_["hydroxl_cpu"] = []() {
        return std::make_shared<HydroXLCpuEngineAdapter>();
    };
    factories_["hydroxl_vulkan"] = []() {
        return std::make_shared<HydroXLVulkanEngineAdapter>();
    };
}

std::vector<BackendCapabilities> EngineOrchestrator::backends() const {
    std::lock_guard<std::mutex> lk(mu_);
    return backends_;
}

std::map<std::string, float> EngineOrchestrator::perf_history() const {
    std::lock_guard<std::mutex> lk(mu_);
    return perf_history_;
}

EngineOrchestrator::SelectionResult EngineOrchestrator::select_backend(const SelectionRequest& req) const {
    std::lock_guard<std::mutex> lk(mu_);

    SelectionResult res;
    res.is_fallback = false;
    res.used_benchmark_history = false;

    const auto& hw = HardwareDetector::instance();

    // 1. Forced backend override
    if (!req.forced_backend.empty()) {
        const BackendCapabilities* forced_cap = nullptr;
        for (const auto& b : backends_) {
            if (b.backend_id == req.forced_backend) {
                forced_cap = &b;
                break;
            }
        }

        if (forced_cap) {
            bool avail = forced_cap->is_available(hw);
            bool quant_ok = forced_cap->supports_quant(req.quant_type);
            bool arch_ok = forced_cap->supports_arch(req.model_arch);

            if (avail && quant_ok && arch_ok) {
                res.selected_backend = req.forced_backend;
                res.reason = "Forced backend override matched and verified";
                return res;
            } else {
                std::string rej = req.forced_backend + ": forced backend incompatible (";
                if (!avail) rej += "hardware unavailable; ";
                if (!quant_ok) rej += "unsupported quant " + std::to_string(req.quant_type) + "; ";
                if (!arch_ok) rej += "unsupported arch " + req.model_arch + "; ";
                rej += ")";
                res.rejected_backends.push_back(rej);

                if (req.allow_cpu_fallback) {
                    res.selected_backend = "hydroxl_cpu";
                    res.is_fallback = true;
                    res.reason = "Forced backend '" + req.forced_backend + "' unavailable; falling back to hydroxl_cpu";
                    return res;
                } else {
                    res.selected_backend = "";
                    res.reason = "Forced backend '" + req.forced_backend + "' is unavailable and fallback disallowed";
                    return res;
                }
            }
        } else {
            res.rejected_backends.push_back(req.forced_backend + ": backend not registered");
            if (req.allow_cpu_fallback) {
                res.selected_backend = "hydroxl_cpu";
                res.is_fallback = true;
                res.reason = "Forced backend '" + req.forced_backend + "' not found; falling back to hydroxl_cpu";
                return res;
            } else {
                res.selected_backend = "";
                res.reason = "Forced backend '" + req.forced_backend + "' is not registered";
                return res;
            }
        }
    }

    // 2. Filter and score candidate backends
    struct Candidate {
        const BackendCapabilities* caps;
        float score;
        bool used_history;
    };
    std::vector<Candidate> candidates;

    size_t memory_needed = static_cast<size_t>(req.model_size_bytes * 1.2);

    for (const auto& b : backends_) {
        // Check availability
        if (!b.is_available(hw)) {
            res.rejected_backends.push_back(b.backend_id + ": hardware requirements not satisfied");
            continue;
        }

        // Check quantization support
        if (!b.supports_quant(req.quant_type)) {
            res.rejected_backends.push_back(b.backend_id + ": quant type " + std::to_string(req.quant_type) + " not supported");
            continue;
        }

        // Check architecture support
        if (!b.supports_arch(req.model_arch)) {
            res.rejected_backends.push_back(b.backend_id + ": architecture '" + req.model_arch + "' not supported");
            continue;
        }

        // Check RAM / VRAM requirement
        if (memory_needed > 0) {
            if (b.requires_vulkan || b.requires_cuda) {
                const auto* gpu = hw.best_gpu();
                if (gpu && gpu->vram_bytes > 0 && gpu->vram_bytes < memory_needed) {
                    res.rejected_backends.push_back(b.backend_id + ": insufficient VRAM (have " +
                                                    std::to_string(gpu->vram_bytes / (1024 * 1024)) + " MB, need " +
                                                    std::to_string(memory_needed / (1024 * 1024)) + " MB)");
                    continue;
                }
            } else {
                if (hw.available_system_ram() < memory_needed) {
                    res.rejected_backends.push_back(b.backend_id + ": insufficient system RAM (have " +
                                                    std::to_string(hw.available_system_ram() / (1024 * 1024)) + " MB, need " +
                                                    std::to_string(memory_needed / (1024 * 1024)) + " MB)");
                    continue;
                }
            }
        }

        // Score computation: score = relative_throughput * ram_headroom_factor / init_cost_factor
        float throughput = b.relative_throughput;
        bool used_hist = false;
        auto hist_it = perf_history_.find(b.backend_id);
        if (hist_it != perf_history_.end() && hist_it->second > 0.0f) {
            // Boost throughput using observed benchmark history
            throughput = hist_it->second;
            used_hist = true;
        }

        float init_cost = (std::max)(50.0f, b.init_cost_ms);
        float init_cost_factor = 1.0f + (init_cost / 1000.0f);
        if (req.prefer_latency) {
            init_cost_factor *= 2.0f; // penalize high cold-start cost
        }

        float ram_headroom_factor = 1.0f;
        if (memory_needed > 0 && hw.available_system_ram() > 0) {
            float ratio = static_cast<float>(hw.available_system_ram()) / static_cast<float>(memory_needed);
            ram_headroom_factor = (std::min)(2.0f, (std::max)(1.0f, ratio / 2.0f));
        }

        float score = (throughput * ram_headroom_factor) / init_cost_factor + (b.priority * 0.001f);

        candidates.push_back({&b, score, used_hist});
    }

    if (!candidates.empty()) {
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.score > b.score;
        });

        res.selected_backend = candidates.front().caps->backend_id;
        res.used_benchmark_history = candidates.front().used_history;
        std::ostringstream oss;
        oss << "Auto-selected backend '" << res.selected_backend << "' (score="
            << std::fixed << std::setprecision(2) << candidates.front().score << ")";
        res.reason = oss.str();
        return res;
    }

    // 3. Fallback to CPU if allowed
    if (req.allow_cpu_fallback) {
        for (const auto& b : backends_) {
            if (b.backend_id == "hydroxl_cpu" || (!b.requires_vulkan && !b.requires_cuda)) {
                res.selected_backend = b.backend_id;
                res.is_fallback = true;
                res.reason = "Fallback to " + b.backend_id + " (no candidate passed all primary constraints)";
                return res;
            }
        }
    }

    res.selected_backend = "";
    res.reason = "No candidate backend satisfied model and hardware constraints";
    return res;
}

std::shared_ptr<IInferenceEngine> EngineOrchestrator::create_engine(
    const SelectionResult& sel,
    const SelectionRequest& req,
    std::string& error_out
) {
    std::lock_guard<std::mutex> lk(mu_);

    if (sel.selected_backend.empty()) {
        error_out = "Cannot create engine: no backend was selected (" + sel.reason + ")";
        return nullptr;
    }

    auto create_named = [&](const std::string& name) -> std::shared_ptr<IInferenceEngine> {
        auto it = factories_.find(name);
        if (it != factories_.end() && it->second) {
            return it->second();
        }
        return nullptr;
    };

    auto engine = create_named(sel.selected_backend);
    if (!engine) {
        error_out = "Factory for backend '" + sel.selected_backend + "' is not registered";
        if (req.allow_cpu_fallback && sel.selected_backend != "hydroxl_cpu") {
            auto fallback = create_named("hydroxl_cpu");
            if (fallback && fallback->initialize(req.model_path, error_out)) {
                return fallback;
            }
        }
        return nullptr;
    }

    if (engine->initialize(req.model_path, error_out)) {
        return engine;
    }

    // Primary initialization failed: try CPU fallback if permitted
    if (req.allow_cpu_fallback && sel.selected_backend != "hydroxl_cpu") {
        std::string fallback_err;
        auto fallback = create_named("hydroxl_cpu");
        if (fallback) {
            if (fallback->initialize(req.model_path, fallback_err)) {
                error_out.clear();
                return fallback;
            } else {
                error_out = "Primary backend '" + sel.selected_backend + "' failed (" + error_out +
                            "); CPU fallback also failed (" + fallback_err + ")";
                return nullptr;
            }
        }
    }

    return nullptr;
}

void EngineOrchestrator::record_performance(const std::string& backend_id,
                                            float tokens_per_second,
                                            size_t peak_rss_bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    float current = perf_history_[backend_id];
    perf_history_[backend_id] = (std::max)(current, tokens_per_second);
    rss_history_[backend_id] = (std::max)(rss_history_[backend_id], peak_rss_bytes);
}

void EngineOrchestrator::print_diagnostics(std::ostream& out) const {
    std::lock_guard<std::mutex> lk(mu_);

    out << "========================================================\n";
    out << " Engine Orchestrator Diagnostics\n";
    out << "========================================================\n";
    out << " Registered Backends (" << backends_.size() << "):\n";
    for (const auto& b : backends_) {
        out << "   * ID: " << b.backend_id << "\n";
        out << "     Requires Vulkan: " << (b.requires_vulkan ? "YES" : "NO")
            << ", CUDA: " << (b.requires_cuda ? "YES" : "NO") << "\n";
        out << "     Relative Throughput: " << b.relative_throughput
            << ", Init Cost: " << b.init_cost_ms << " ms\n";
        out << "     Priority: " << b.priority << "\n";
        if (!b.supported_quant_types.empty()) {
            out << "     Quant Types: ";
            for (int q : b.supported_quant_types) out << q << " ";
            out << "\n";
        }
        if (!b.supported_arch.empty()) {
            out << "     Architectures: ";
            for (const auto& a : b.supported_arch) out << a << " ";
            out << "\n";
        }
    }

    out << "\n Observed Performance History (" << perf_history_.size() << " entries):\n";
    for (const auto& [id, tok_s] : perf_history_) {
        out << "   * " << id << " -> " << tok_s << " tok/s";
        auto it = rss_history_.find(id);
        if (it != rss_history_.end() && it->second > 0) {
            out << " (Peak RSS: " << (it->second / (1024 * 1024)) << " MB)";
        }
        out << "\n";
    }
    out << "========================================================\n";
}

} // namespace orchestration
} // namespace agr
