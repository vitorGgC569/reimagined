#ifndef CONTEXT_PREDICTOR_H
#define CONTEXT_PREDICTOR_H

#include <vector>
#include <map>
#include <string>

namespace uhk {
namespace ghost {

// Markov Chain based Context Predictor
class ContextPredictor {
public:
    ContextPredictor() {}

    void observe(const std::string& current_state, const std::string& next_state) {
        transitions_[current_state][next_state]++;
    }

    std::string predict(const std::string& current_state) {
        if (transitions_.find(current_state) == transitions_.end()) {
            return "";
        }

        const auto& next_states = transitions_[current_state];
        std::string likely_next = "";
        int max_count = -1;

        for (const auto& pair : next_states) {
            if (pair.second > max_count) {
                max_count = pair.second;
                likely_next = pair.first;
            }
        }
        return likely_next;
    }

private:
    // current_state -> { next_state -> count }
    std::map<std::string, std::map<std::string, int>> transitions_;
};

} // namespace ghost
} // namespace uhk

#endif // CONTEXT_PREDICTOR_H
