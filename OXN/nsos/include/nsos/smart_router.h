#ifndef SMART_ROUTER_H
#define SMART_ROUTER_H

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include "nsos_context.h"
#include "tensor.h"

namespace nsos {

struct InferenceMetrics {
    float avg_latency_ms;
    float success_rate;
    float cost_estimate; // Arbitrary units
};

enum class RouteMode {
    FAST_LOCAL,     // TurboQuant / Local GPU Small
    ACCURATE_GPU,   // Full Float32 Local GPU
    ULTRAPLAN,      // Parallel MCTS Rollouts (Slow but deep)
    EXTERNAL_API    // Cloud fallback (Gemini/Claude)
};

class SmartRouter {
public:
    SmartRouter() = default;

    // Select the best routing mode based on prompt complexity and current system load
    RouteMode select_route(const std::string& prompt, const Context& ctx) {
        // Heuristic: If prompt is long (> 1000 tokens-ish), use compressed or parallel
        if (prompt.length() > 4000) {
           return RouteMode::ULTRAPLAN;
        }

        // If system is under heavy load (measured by GPU queue/recent latency)
        if (get_avg_latency(RouteMode::ACCURATE_GPU) > 500.0f) {
           return RouteMode::FAST_LOCAL; // Downscale to TurboQuant
        }

        return RouteMode::ACCURATE_GPU;
    }

    void record_perf(RouteMode mode, float latency_ms, bool success) {
        auto& m = metrics_[mode];
        float alpha = 0.2f;
        m.avg_latency_ms = (1 - alpha) * m.avg_latency_ms + alpha * latency_ms;
        // Simple success rate update
        m.success_rate = (1 - alpha) * m.success_rate + alpha * (success ? 1.0f : 0.0f);
    }

    float get_avg_latency(RouteMode mode) const {
        if (metrics_.count(mode)) return metrics_.at(mode).avg_latency_ms;
        return 0.0f;
    }

private:
    std::map<RouteMode, InferenceMetrics> metrics_;
};

} // namespace nsos

#endif
