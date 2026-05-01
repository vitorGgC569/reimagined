#include <iostream>
#include <cassert>
#include <chrono>
#include <thread>
#include <filesystem>
#include "nsos/coordinator.h"
#include "nsos/mailbox.h"

using namespace nsos::swarm;

int main() {
    std::cout << "--- Starting NSOS Swarm V2 (Task Claiming) Test ---\n";

    Coordinator& coordinator = Coordinator::get_instance();

    // 1. Register agents with specific "personalities" and models
    AgentIdentity grep_agent = {"agent_1", "SearchWorker", "cyan", "jamba-mini-q4", 0};
    AgentIdentity fix_agent  = {"agent_2", "RepairWorker", "magenta", "jamba-mini-q4", 0};

    coordinator.register_worker(grep_agent);
    coordinator.register_worker(fix_agent);

    std::cout << "[Test] Workers registered and standby (Task Claiming mode).\n";

    // 2. Submit tasks using real semantic routing types
    coordinator.submit_task("Analyze holographic memory leakage in cluster_0",
                            "critical_research");
    coordinator.submit_task("Prune stale MCTS nodes in branch_88",
                            "maintenance");
    coordinator.submit_task("Retrieve context for inference on query_42",
                            "retrieval");
    
    std::cout << "[Test] 3 tasks submitted (critical_research, maintenance, retrieval).\n";

    // 3. Poll for responses from the Coordinator's mailbox
    // We expect 2 ACKs and 2 Results
    int completed = 0;
    int acks = 0;
    int retries = 20; // 4 seconds total
    
    while (retries-- > 0 && completed < 2) {
        TaskNotification msg;
        if (coordinator.poll_responses(msg)) {
            if (msg.id.find("ack_") != std::string::npos) {
                acks++;
                std::cout << "[Test] Received ACK: " << msg.content << "\n";
            } else if (msg.id.find("resp_") != std::string::npos) {
                completed++;
                std::cout << "[Test] Received RESULT: " << msg.content << "\n";
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    // 4. Verify Persistence
    std::cout << "[Test] Verifying persistent mailbox storage...\n";
    if (std::filesystem::exists("mailboxes/agent_1/messages.log")) {
        std::cout << "[Test] agent_1 persistence OK.\n";
    } else {
        std::cerr << "[Test] FAILED: Persistence log not found!\n";
        return 1;
    }

    if (acks >= 2 && completed >= 2) {
        std::cout << "[Test] Swarm V2 Verification SUCCESS.\n";
    } else {
        std::cerr << "[Test] FAILED: Some tasks were not processed. ACKs: " << acks << ", Results: " << completed << "\n";
        return 1;
    }

    coordinator.shutdown();
    std::cout << "--- Swarm V2 Test Completed Successfully ---\n";
    return 0;
}
