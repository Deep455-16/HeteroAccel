// src/profiler/Profiler.cpp
// Phase 8: Profiler implementation
#include "profiler/Profiler.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>

namespace agr {

Profiler::Profiler(PerformanceHistory& history, const HardwareInfo& hw)
    : history_(history), hw_(hw) {}

std::string Profiler::hardwareId() const {
    // Generate a simple unique string for this hardware combo
    return hw_.cpu.model_name;
}

ProfileKey Profiler::makeKey(const ProfileEvent& event) const {
    ProfileKey key;
    key.hardware_id = hardwareId();
    key.backend = event.backend;
    key.workload_type = event.workload_type;
    key.model_name = event.model_name;
    return key;
}

void Profiler::recordEvent(const ProfileEvent& event) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    events_.push_back(event);
    if (events_.size() > max_events_) {
        // Keep bounded
        events_.erase(events_.begin(), events_.begin() + (events_.size() - max_events_));
    }

    // Update history for online learning
    history_.recordEvent(makeKey(event), event);
}

std::vector<ProfileEvent> Profiler::getEvents() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return events_;
}

// Minimal JSON-like parser for persistence (to avoid heavy dependencies)
bool Profiler::saveProfile(const std::string& path) {
    auto stats = history_.getAllStats();
    
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
    }

    std::ofstream out(path);
    if (!out) return false;

    out << "{\n  \"version\": 1,\n  \"profiles\": [\n";
    bool first = true;
    for (const auto& kv : stats) {
        if (!first) out << ",\n";
        first = false;
        
        out << "    {\n";
        out << "      \"hardware_id\": \"" << kv.first.hardware_id << "\",\n";
        out << "      \"backend\": " << static_cast<int>(kv.first.backend) << ",\n";
        out << "      \"workload_type\": \"" << kv.first.workload_type << "\",\n";
        out << "      \"model_name\": \"" << kv.first.model_name << "\",\n";
        
        out << "      \"sample_count\": " << kv.second.sample_count << ",\n";
        out << "      \"ema_duration_ms\": " << kv.second.ema_duration_ms << ",\n";
        out << "      \"variance\": " << kv.second.variance << ",\n";
        out << "      \"min_duration_ms\": " << kv.second.min_duration_ms << ",\n";
        out << "      \"max_duration_ms\": " << kv.second.max_duration_ms << ",\n";
        out << "      \"failure_count\": " << kv.second.failure_count << ",\n";
        out << "      \"regression_detected\": " << (kv.second.regression_detected ? "true" : "false") << "\n";
        out << "    }";
    }
    out << "\n  ]\n}\n";
    return true;
}

// Simple key-value string parser for loading
bool Profiler::loadProfile(const std::string& path) {
    std::ifstream in(path);
    if (!in) return false;

    std::string line;
    std::unordered_map<ProfileKey, HistoricalStats> loaded;
    
    ProfileKey currentKey;
    HistoricalStats currentStats;
    bool inObj = false;

    // Extremely naive JSON parser for our specific format
    while (std::getline(in, line)) {
        if (line.find("{") != std::string::npos && line.find("version") == std::string::npos) {
            inObj = true;
            currentKey = ProfileKey{};
            currentStats = HistoricalStats{};
        } else if (line.find("}") != std::string::npos && inObj) {
            inObj = false;
            loaded[currentKey] = currentStats;
        } else if (inObj) {
            auto extractStr = [](const std::string& l) {
                auto start = l.find_first_of('"');
                if (start == std::string::npos) return std::string();
                start = l.find_first_of('"', start + 1);
                if (start == std::string::npos) return std::string();
                start = l.find_first_of('"', start + 1);
                if (start == std::string::npos) return std::string();
                auto end = l.find_first_of('"', start + 1);
                return l.substr(start + 1, end - start - 1);
            };
            auto extractNum = [](const std::string& l) {
                auto colon = l.find(':');
                if (colon == std::string::npos) return 0.0;
                return std::stod(l.substr(colon + 1));
            };

            if (line.find("\"hardware_id\"") != std::string::npos) currentKey.hardware_id = extractStr(line);
            else if (line.find("\"backend\"") != std::string::npos) currentKey.backend = static_cast<ComputeBackend>(extractNum(line));
            else if (line.find("\"workload_type\"") != std::string::npos) currentKey.workload_type = extractStr(line);
            else if (line.find("\"model_name\"") != std::string::npos) currentKey.model_name = extractStr(line);
            
            else if (line.find("\"sample_count\"") != std::string::npos) currentStats.sample_count = static_cast<uint64_t>(extractNum(line));
            else if (line.find("\"ema_duration_ms\"") != std::string::npos) currentStats.ema_duration_ms = extractNum(line);
            else if (line.find("\"variance\"") != std::string::npos) currentStats.variance = extractNum(line);
            else if (line.find("\"min_duration_ms\"") != std::string::npos) currentStats.min_duration_ms = extractNum(line);
            else if (line.find("\"max_duration_ms\"") != std::string::npos) currentStats.max_duration_ms = extractNum(line);
            else if (line.find("\"failure_count\"") != std::string::npos) currentStats.failure_count = static_cast<uint64_t>(extractNum(line));
            else if (line.find("\"regression_detected\"") != std::string::npos) currentStats.regression_detected = line.find("true") != std::string::npos;
        }
    }
    
    history_.setAllStats(loaded);
    return true;
}

void Profiler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    events_.clear();
    history_.setAllStats({});
}

} // namespace agr
