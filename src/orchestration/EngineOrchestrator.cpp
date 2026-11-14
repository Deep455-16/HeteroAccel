// src/orchestration/EngineOrchestrator.cpp
// Phase 21: Engine Orchestrator

#include "orchestration/EngineOrchestrator.h"
#include "orchestration/HydroXLEngineAdapter.h"
#include "orchestration/LlamaCppEngineAdapter.h"
#include "orchestration/MemoryEstimator.h"
#include "colibri/ColibriEngine.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace agr {
namespace orchestration {

namespace {
// Registers the built-in engine factories. The "cuda" factory exists only in
// binaries compiled with HETEROACCEL_ENABLE_CUDA (llama.cpp GGML_CUDA).
void register_builtin_factories(std::map<std::string, EngineOrchestrator::EngineFactory>& f) {
    f["hydroxl_cpu"] = []() {
        return std::make_shared<HydroXLCpuEngineAdapter>();
    };
    f["hydroxl_vulkan"] = []() {
        return std::make_shared<HydroXLVulkanEngineAdapter>();
    };
    f["llamacpp_cpu"] = []() -> std::shared_ptr<IInferenceEngine> {
        return std::make_shared<LlamaCppEngineAdapter>("llamacpp_cpu", "cpu", 0);
    };
    f["llamacpp_vulkan"] = []() -> std::shared_ptr<IInferenceEngine> {
        return std::make_shared<LlamaCppEngineAdapter>("llamacpp_vulkan", "vulkan", 0);
    };
    f["colibri_cpu"] = []() -> std::shared_ptr<IInferenceEngine> {
        return std::make_shared<ColibriEngineAdapter>();
    };
#if defined(HETEROACCEL_CUDA_ENABLED) && HETEROACCEL_CUDA_ENABLED
    f["cuda"] = []() -> std::shared_ptr<IInferenceEngine> {
        int dev = 0;
        if (const auto* g = HardwareDetector::instance().best_cuda_gpu()) {
            if (g->cuda_device_index >= 0) dev = g->cuda_device_index;
        }
        return std::make_shared<LlamaCppEngineAdapter>("cuda", "cuda", dev);
    };
    f["llamacpp_cuda"] = f["cuda"];
#endif
}
} // namespace

EngineOrchestrator& EngineOrchestrator::instance() {
    static EngineOrchestrator inst;
    return inst;
}

EngineOrchestrator::EngineOrchestrator() {
    // Register default built-in engine factories
    register_builtin_factories(factories_);
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
    register_builtin_factories(factories_);
    last_report_ = CreationReport{};
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
            std::string nreg = req.forced_backend + ": backend not registered";
            if (req.forced_backend == "cuda") {
                nreg += HardwareDetector::cuda_inference_compiled()
                            ? " (CUDA inference is compiled in but no usable CUDA device was detected: " + hw.cuda_unavailable_reason() + ")"
                            : " (binary built without HETEROACCEL_ENABLE_CUDA)";
            }
            res.rejected_backends.push_back(nreg);
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

    size_t memory_needed = 0;
    if (req.n_layers > 0 && req.n_embd > 0 && req.vocab_size > 0) {
        auto est = MemoryEstimator::estimate(
            req.n_layers, req.n_embd, req.n_heads, req.n_kv_heads,
            req.n_ff, req.vocab_size, static_cast<uint32_t>(req.n_ctx)
        );
        memory_needed = est.total_estimated;
        res.estimated_memory_bytes = est.total_estimated;
    } else {
        memory_needed = static_cast<size_t>(req.model_size_bytes * 1.2);
        res.estimated_memory_bytes = memory_needed;
    }

    size_t avail_ram = MemoryEstimator::available_ram_bytes();
    if (avail_ram == 0) {
        avail_ram = hw.available_system_ram();
    }

    bool high_memory_pressure = (avail_ram > 0 && memory_needed > static_cast<size_t>(avail_ram * 0.90));
    if (high_memory_pressure) {
        std::cerr << "[HeteroAccel WARNING] Model estimated memory ("
                  << (memory_needed / (1024 * 1024)) << " MB) exceeds 90% of available RAM ("
                  << (avail_ram / (1024 * 1024)) << " MB); prioritizing CPU backend.\n";
    }

    for (const auto& b : backends_) {
        // If memory pressure is critical, skip high-risk GPU backends in favor of CPU
        if (high_memory_pressure && (b.requires_vulkan || b.requires_cuda)) {
            res.rejected_backends.push_back(b.backend_id + ": memory footprint exceeds 90% available RAM; CPU prioritized");
            continue;
        }

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
                const auto* gpu = b.requires_cuda ? hw.best_cuda_gpu() : nullptr;
                if (!gpu) gpu = hw.best_gpu();
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

    CreationReport report;
    report.requested_backend = req.forced_backend.empty() ? sel.selected_backend : req.forced_backend;
    report.selected_backend  = sel.selected_backend;
    report.fell_back         = sel.is_fallback;
    if (sel.is_fallback) {
        report.failure_reason = sel.reason;
        for (const auto& r : sel.rejected_backends) report.failure_reason += "; " + r;
    }

    if (sel.selected_backend.empty()) {
        error_out = "Cannot create engine: no backend was selected (" + sel.reason + ")";
        report.failure_reason = error_out;
        last_report_ = report;
        return nullptr;
    }

    auto create_named = [&](const std::string& name) -> std::shared_ptr<IInferenceEngine> {
        auto it = factories_.find(name);
        if (it != factories_.end() && it->second) {
            return it->second();
        }
        return nullptr;
    };

    auto try_cpu_fallback = [&](const std::string& primary_failure) -> std::shared_ptr<IInferenceEngine> {
        auto fallback = create_named("hydroxl_cpu");
        std::string fallback_err;
        if (fallback && fallback->initialize(req.model_path, fallback_err)) {
            error_out.clear();
            report.fell_back = true;
            report.failure_reason = primary_failure;
            report.actual_backend = fallback->backend_id();
            last_report_ = report;
            return fallback;
        }
        error_out = "Primary backend '" + sel.selected_backend + "' failed (" + primary_failure +
                    "); CPU fallback also failed (" + fallback_err + ")";
        report.failure_reason = error_out;
        last_report_ = report;
        return nullptr;
    };

    auto engine = create_named(sel.selected_backend);
    if (!engine) {
        std::string why = "Factory for backend '" + sel.selected_backend + "' is not registered";
        if (sel.selected_backend == "cuda") {
            why += " (this binary was built without HETEROACCEL_ENABLE_CUDA)";
        }
        error_out = why;
        if (req.allow_cpu_fallback && sel.selected_backend != "hydroxl_cpu") {
            return try_cpu_fallback(why);
        }
        report.failure_reason = why;
        last_report_ = report;
        return nullptr;
    }

    std::string init_err;
    if (engine->initialize(req.model_path, init_err)) {
        error_out.clear();
        report.actual_backend = engine->backend_id();
        last_report_ = report;
        return engine;
    }
    error_out = init_err;

    // Primary initialization failed: try CPU fallback if permitted
    if (req.allow_cpu_fallback && sel.selected_backend != "hydroxl_cpu") {
        return try_cpu_fallback(init_err);
    }

    report.failure_reason = init_err;
    last_report_ = report;
    return nullptr;
}

EngineOrchestrator::CreationReport EngineOrchestrator::last_creation_report() const {
    std::lock_guard<std::mutex> lk(mu_);
    return last_report_;
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
