#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <sstream>

namespace pantheon {
namespace cognition {

    class ChainOfThought {
    public:
        // Chunk-Wise Training (CWT) Loss
        // rationales: vector of token IDs or embeddings representing reasoning steps
        // chunk_size: size of chunks to process independently
        // This simulates "Skip-Thinking" training where gradients are computed per chunk
        // to prevent smoothing over long chains.
        static float compute_chunk_wise_loss(const std::vector<float>& student_trace,
                                           const std::vector<float>& teacher_trace,
                                           int chunk_size) {

            if (student_trace.size() != teacher_trace.size()) {
                // In real CoT, lengths differ. Here we assume aligned/padded traces for loss calculation
                // or we compute loss on the common prefix.
                // For rigorous math test, we assume aligned.
                return -1.0f;
            }

            size_t n = student_trace.size();
            float total_loss = 0.0f;
            int chunks = 0;

            for (size_t i = 0; i < n; i += chunk_size) {
                float chunk_loss = 0.0f;
                size_t end = std::min(i + chunk_size, n);
                size_t count = 0;

                for (size_t j = i; j < end; ++j) {
                    // Simple MSE on embeddings/logits for the chunk
                    float diff = student_trace[j] - teacher_trace[j];
                    chunk_loss += diff * diff;
                    count++;
                }

                // Weight the chunk (e.g., later chunks might be more important, or uniform)
                // CWT emphasizes local coherence.
                if (count > 0) {
                    total_loss += chunk_loss / count;
                    chunks++;
                }
            }

            return chunks > 0 ? total_loss / chunks : 0.0f;
        }

        // MiCoTA: Mid-CoT Teacher Assistant - Granularity Adjustment
        // Filters a chain of thought based on student competence.
        // reasoning_steps: A list of "steps" (each step is a value or ID).
        // competence: 0.0 (Novice, needs all steps) to 1.0 (Expert, skips steps).
        static std::vector<int> adjust_granularity(const std::vector<int>& reasoning_steps, float competence) {
            std::vector<int> adjusted;

            // Algorithm: Keep steps based on importance/stride.
            // Simple heuristic: Expert skips every Nth step (skip-thinking).
            // Or: Expert only sees "milestone" steps.
            // Implementation: Stride = 1 + floor(competence * max_stride)

            int stride = 1 + static_cast<int>(competence * 3); // Max skip 3

            for (size_t i = 0; i < reasoning_steps.size(); ++i) {
                // Always keep the last step (the conclusion)
                if (i == reasoning_steps.size() - 1) {
                    adjusted.push_back(reasoning_steps[i]);
                    continue;
                }

                // Keep step if index % stride == 0
                if (i % stride == 0) {
                    adjusted.push_back(reasoning_steps[i]);
                }
            }
            return adjusted;
        }
    };

}
}
