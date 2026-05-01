#pragma once

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <map>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include "mailbox.h"

namespace nsos {
namespace swarm {

struct AgentIdentity {
    std::string id;
    std::string name;
    std::string color;
    std::string model;
    int gpu_id = -1; // Specific GPU assignment
};

// Priority-based task comparison
struct TaskCompare {
    bool operator()(const TaskNotification& a, const TaskNotification& b) {
        // Lower ID or specific priority flags could go here
        return a.id > b.id; 
    }
};

class Coordinator; // Forward declaration

class Worker {
public:
    Worker(AgentIdentity identity, Coordinator* parent) 
        : identity_(std::move(identity)), parent_(parent), running_(false) {}
    
    ~Worker() {
        stop();
        if (thread_.joinable()) thread_.join();
    }

    void start() {
        bool expected = false;
        if (running_.compare_exchange_strong(expected, true)) {
            thread_ = std::thread(&Worker::run_loop, this);
        }
    }

    void stop() {
        running_ = false;
    }

    const AgentIdentity& identity() const { return identity_; }

private:
    void run_loop(); // Implementation in .cpp to avoid circularity
    void execute_reasoning(const TaskNotification& task);

    AgentIdentity identity_;
    Coordinator* parent_;
    std::atomic<bool> running_;
    std::thread thread_;
};

class Coordinator {
public:
    static Coordinator& get_instance() {
        static Coordinator instance;
        return instance;
    }

    void register_worker(const AgentIdentity& identity) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        auto worker = std::make_unique<Worker>(identity, this);
        Mailbox::get_instance().load_agent(identity.id);
        worker->start();
        workers_[identity.id] = std::move(worker);
    }

    // Task Claiming: Workers call this to get work
    bool claim_task(TaskNotification& out_task, const std::string& claimer_id) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        task_cv_.wait(lock, [this] { return !global_task_queue_.empty() || !running_; });
        
        if (!running_ || global_task_queue_.empty()) return false;
        
        out_task = global_task_queue_.top();
        global_task_queue_.pop();
        
        // Record receipt in the claimer's mailbox for persistence
        out_task.receiver_id = claimer_id;
        Mailbox::get_instance().deliver(out_task); 
        
        return true;
    }

    void submit_task(const std::string& content, const std::string& type = "task") {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        TaskNotification task;
        task.id = "task_" + std::to_string(task_id_counter_++);
        task.sender_id = "coordinator";
        task.receiver_id = "global_queue"; // Broadcast to all workers
        task.content = content;
        task.type = type;
        task.timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        
        global_task_queue_.push(task);
        task_cv_.notify_one();
    }

    bool poll_responses(TaskNotification& out_msg) {
        return Mailbox::get_instance().poll("coordinator", out_msg);
    }

    void shutdown() {
        running_ = false;
        task_cv_.notify_all();
        workers_.clear();
    }

private:
    Coordinator() : task_id_counter_(0), running_(true) {}
    ~Coordinator() { shutdown(); }

    std::map<std::string, std::unique_ptr<Worker>> workers_;
    std::priority_queue<TaskNotification, std::vector<TaskNotification>, TaskCompare> global_task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable task_cv_;
    std::atomic<int> task_id_counter_;
    std::atomic<bool> running_;
};

} // namespace swarm
} // namespace nsos
