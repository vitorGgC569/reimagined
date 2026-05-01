#pragma once

#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <chrono>
#include <fstream>
#include <filesystem>

namespace nsos {
namespace swarm {

struct TaskNotification {
    std::string id;
    std::string sender_id;
    std::string receiver_id; // "coordinator" or agent_name
    std::string content;
    std::string type; // "task", "status", "idle", "stop"
    std::string timestamp;
    std::string color;
    bool read = false;
};

class Mailbox {
public:
    static Mailbox& get_instance() {
        static Mailbox instance;
        return instance;
    }

    // Deliver a message to an agent's mailbox
    void deliver(const TaskNotification& notification) {
        std::lock_guard<std::mutex> lock(mutex_);
        queues_[notification.receiver_id].push(notification);
        persist_to_disk(notification);
        cv_.notify_all();
    }

    // New: Recover from disk on startup
    void load_agent(const std::string& agent_id) {
        std::string path = "mailboxes/" + agent_id + "/messages.log";
        try {
            if (std::filesystem::exists(path)) {
                // Here we would parse the fixed-format log to repopulate the queue.
            }
            std::filesystem::create_directories("mailboxes/" + agent_id);
        } catch (const std::filesystem::filesystem_error&) {
            // Mailbox persistence is best-effort; queue delivery still works in-memory.
        }
    }

    // Poll for messages (blocking with timeout)
    bool poll(const std::string& agent_id, TaskNotification& out_msg, int timeout_ms = 500) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
            [this, &agent_id] { return !queues_[agent_id].empty(); })) {
            
            out_msg = queues_[agent_id].front();
            queues_[agent_id].pop();
            return true;
        }
        return false;
    }

    // Helper to format as Teammate XML (Claude Code Style)
    static std::string wrap_xml(const std::string& from, const std::string& content, const std::string& color = "") {
        std::string xml = "<teammate-message teammate_id=\"" + from + "\"";
        if (!color.empty()) xml += " color=\"" + color + "\"";
        xml += ">\n" + content + "\n</teammate-message>";
        return xml;
    }

private:
    void persist_to_disk(const TaskNotification& n) {
        std::string dir = "mailboxes/" + n.receiver_id;
        try {
            std::filesystem::create_directories(dir);
            std::ofstream log(dir + "/messages.log", std::ios::app);
            if (log.is_open()) {
                log << "[" << n.timestamp << "] " << n.sender_id << " -> " << n.receiver_id 
                    << " [" << n.type << "] (" << n.id << ")\n"
                    << n.content << "\n---\n";
            }
        } catch (const std::filesystem::filesystem_error&) {
            // Persistence failures should not crash the worker swarm.
        }
    }
    Mailbox() = default;
    ~Mailbox() = default;
    Mailbox(const Mailbox&) = delete;
    Mailbox& operator=(const Mailbox&) = delete;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::unordered_map<std::string, std::queue<TaskNotification>> queues_;
};

} // namespace swarm
} // namespace nsos
