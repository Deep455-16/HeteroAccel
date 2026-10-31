// src/orchestration/PerfProfile.cpp
// Phase 24: Persistent Performance Profiles for HeteroAccel

#define _CRT_SECURE_NO_WARNINGS
#include "orchestration/PerfProfile.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace agr {
namespace orchestration {

std::string PerfProfile::compute_model_hash(const std::string& path) {
    if (path.empty()) return "";
    try {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) return "";

        f.seekg(0, std::ios::end);
        uint64_t size = static_cast<uint64_t>(f.tellg());
        f.seekg(0, std::ios::beg);

        char buf[64] = {0};
        f.read(buf, sizeof(buf));
        std::streamsize bytes_read = f.gcount();

        // 64-bit FNV-1a hash of the first 64 bytes
        uint64_t hash = 14695981039346656037ULL;
        for (std::streamsize i = 0; i < bytes_read; ++i) {
            hash ^= static_cast<uint8_t>(buf[i]);
            hash *= 1099511628211ULL;
        }

        std::ostringstream ss;
        ss << std::hex << size << "_" << hash;
        return ss.str();
    } catch (...) {
        return "error";
    }
}

std::string PerfProfile::default_profile_path() {
#ifdef _WIN32
    const char* appdata = std::getenv("APPDATA");
    if (appdata && appdata[0] != '\0') {
        std::filesystem::path dir = std::filesystem::path(appdata) / "HeteroAccel";
        return (dir / "perf_profile.json").string();
    }
#else
    const char* home = std::getenv("HOME");
    if (home && home[0] != '\0') {
        std::filesystem::path dir = std::filesystem::path(home) / ".config" / "heteroaccel";
        return (dir / "perf_profile.json").string();
    }
#endif
    return "./heteroaccel_perf.json";
}

PerfProfile PerfProfile::load(const std::string& path) {
    PerfProfile prof;
    if (path.empty()) return prof;

    try {
        std::ifstream f(path);
        if (!f.is_open()) return prof;

        std::string line;
        while (std::getline(f, line)) {
            // Trim leading spaces
            size_t start = line.find_first_not_of(" \t\r\n");
            if (start == std::string::npos || line[start] == '#') continue;

            std::istringstream iss(line.substr(start));
            std::string token;
            PerfRecord rec;
            bool has_backend = false;
            bool has_toks = false;
            bool line_valid = true;

            while (iss >> token) {
                size_t eq = token.find('=');
                if (eq == std::string::npos) continue;
                std::string k = token.substr(0, eq);
                std::string v = token.substr(eq + 1);

                try {
                    if (k == "backend") {
                        rec.backend_id = v;
                        has_backend = true;
                    } else if (k == "model_hash") {
                        rec.model_hash = v;
                    } else if (k == "quant") {
                        rec.quant_type = static_cast<uint32_t>(std::stoul(v));
                    } else if (k == "layers") {
                        rec.n_layers = static_cast<uint32_t>(std::stoul(v));
                    } else if (k == "embd") {
                        rec.n_embd = static_cast<uint32_t>(std::stoul(v));
                    } else if (k == "toks_per_s") {
                        rec.tokens_per_second = std::stof(v);
                        has_toks = true;
                    } else if (k == "ts") {
                        rec.timestamp_unix = std::stoull(v);
                    } else if (k == "n") {
                        rec.sample_count = static_cast<uint32_t>(std::stoul(v));
                    }
                } catch (...) {
                    line_valid = false;
                    break;
                }
            }

            if (line_valid && has_backend && has_toks && !rec.backend_id.empty() && !rec.model_hash.empty()) {
                if (rec.sample_count == 0) rec.sample_count = 1;
                prof.records_.push_back(std::move(rec));
            }
        }
    } catch (...) {
        // Return empty on corrupt read
    }

    return prof;
}

bool PerfProfile::save(const std::string& path) const {
    if (path.empty()) return false;

    try {
        std::filesystem::path fs_path(path);
        if (fs_path.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(fs_path.parent_path(), ec);
        }

        std::ofstream f(path);
        if (!f.is_open()) return false;

        f << "# HeteroAccel performance profile v1\n";
        for (const auto& r : records_) {
            f << "backend=" << r.backend_id
              << " model_hash=" << (r.model_hash.empty() ? "none" : r.model_hash)
              << " quant=" << r.quant_type
              << " layers=" << r.n_layers
              << " embd=" << r.n_embd
              << " toks_per_s=" << std::fixed << std::setprecision(2) << r.tokens_per_second
              << " ts=" << r.timestamp_unix
              << " n=" << r.sample_count
              << "\n";
        }
        return true;
    } catch (...) {
        return false;
    }
}

void PerfProfile::record(const std::string& backend_id, const std::string& model_hash,
                         uint32_t quant_type, uint32_t n_layers, uint32_t n_embd,
                         float tokens_per_second) {
    if (backend_id.empty() || tokens_per_second <= 0.0f) return;

    uint64_t now_ts = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count()
    );

    for (auto& r : records_) {
        if (r.backend_id == backend_id && r.model_hash == model_hash &&
            r.quant_type == quant_type && r.n_layers == n_layers && r.n_embd == n_embd) {
            // Update running average
            float total = r.tokens_per_second * r.sample_count + tokens_per_second;
            r.sample_count += 1;
            r.tokens_per_second = total / r.sample_count;
            r.timestamp_unix = now_ts;
            return;
        }
    }

    PerfRecord nr;
    nr.backend_id = backend_id;
    nr.model_hash = model_hash;
    nr.quant_type = quant_type;
    nr.n_layers = n_layers;
    nr.n_embd = n_embd;
    nr.tokens_per_second = tokens_per_second;
    nr.timestamp_unix = now_ts;
    nr.sample_count = 1;
    records_.push_back(std::move(nr));
}

std::string PerfProfile::best_backend(const std::string& model_hash,
                                      uint32_t quant_type, uint32_t n_layers) const {
    std::string best_id;
    float max_speed = -1.0f;

    // First try exact match on model_hash
    if (!model_hash.empty() && model_hash != "none") {
        for (const auto& r : records_) {
            if (r.model_hash == model_hash && r.tokens_per_second > max_speed) {
                max_speed = r.tokens_per_second;
                best_id = r.backend_id;
            }
        }
        if (!best_id.empty()) return best_id;
    }

    // Secondary match: quant_type and layers
    for (const auto& r : records_) {
        if (r.quant_type == quant_type && (n_layers == 0 || r.n_layers == n_layers)) {
            if (r.tokens_per_second > max_speed) {
                max_speed = r.tokens_per_second;
                best_id = r.backend_id;
            }
        }
    }

    return best_id;
}

void PerfProfile::reset() {
    records_.clear();
}

} // namespace orchestration
} // namespace agr
