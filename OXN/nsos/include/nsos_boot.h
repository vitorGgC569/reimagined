#ifndef NSOS_BOOT_H
#define NSOS_BOOT_H

#include <vector>
#include <string>
#include <map>
#include <chrono>

// V8.0: Bare Metal - Neural Operating System Scheduler
// Replaces traditional OS Preemptive Scheduling with Semantic Priority Queuing.

namespace nsos {

    enum ProcessPriority {
        BACKGROUND = 0,
        USER_INTERACTIVE = 1,
        CRITICAL_KNOWLEDGE = 2,
        SURVIVAL = 3
    };

    struct NeuralProcess {
        int pid;
        std::string intent; // e.g. "Generate Image", "Solve Math"
        ProcessPriority semantic_importance;
        float cpu_quota;
    };

    class NeuralScheduler {
        std::vector<NeuralProcess> run_queue;
        int next_pid = 1;

    public:
        int spawn(const std::string& intent) {
            NeuralProcess p;
            p.pid = next_pid++;
            p.intent = intent;
            
            // Heuristic Semantic Classification
            if (intent.find("critical") != std::string::npos || intent.find("save") != std::string::npos) {
                p.semantic_importance = SURVIVAL;
            } else if (intent.find("user") != std::string::npos) {
                p.semantic_importance = USER_INTERACTIVE;
            } else {
                p.semantic_importance = BACKGROUND;
            }
            
            run_queue.push_back(p);
            return p.pid;
        }

        // Context Switch Logic
        // Unlike Round Robin (Linux), we switch based on 'Thought Completion'
        int pick_next_task() {
            int best_idx = -1;
            int max_prio = -1;
            
            for(size_t i=0; i<run_queue.size(); ++i) {
                if (run_queue[i].semantic_importance > max_prio) {
                    max_prio = run_queue[i].semantic_importance;
                    best_idx = i;
                }
            }
            
            if (best_idx != -1) {
                return run_queue[best_idx].pid;
            }
            return 0; // Idle
        }
        
        void boot_kernel() {
            // Check Hardware Access
            // In real OS: probing PCI bus for GPU/NPU
            // Here: Just mock initialization
            spawn("kernel_init: load_drivers");
            spawn("system_2: mcts_daemon");
        }
    };
}
#endif
