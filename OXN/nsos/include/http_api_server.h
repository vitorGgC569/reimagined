#pragma once

#include "nsos_sdk.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#else
using SOCKET = int;
#endif

namespace nsos {

struct HttpApiServerConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string auth_token;
    int worker_threads = 0;
    size_t max_header_bytes = 64 * 1024;
    size_t max_body_bytes = 2 * 1024 * 1024;
    size_t max_queue_depth = 128;
    bool allow_unauthenticated_health = true;
    bool allow_unauthenticated_local = false;
    bool enable_admin_endpoints = false;
    int inference_replicas = 0;
    size_t rate_limit_requests_per_minute = 240;
    bool trust_proxy_headers = false;
    bool require_tls_proxy_header = false;
    int socket_timeout_ms = 10000;
    int max_json_depth = 32;
    int max_generate_tokens = 256;
    int max_context_tokens_per_request = 8192;
    size_t max_prompts_per_batch = 16;
    int max_train_steps = 64;
    int max_train_epochs = 4;
    int max_train_batch_size = 32;
    int max_train_seq_len = 512;
    int max_train_corpus_steps = 256;
    size_t max_train_text_bytes = 64 * 1024;
    std::string pack_output_root = "artifacts/model_packs";
    bool allow_pack_absolute_paths = false;
};

class HttpApiServer {
public:
    HttpApiServer(InferenceEngine& engine, HttpApiServerConfig config = {});
    ~HttpApiServer();

    bool start();
    void serve_forever();
    void stop();

    int port() const { return bound_port_; }
    const HttpApiServerConfig& config() const { return config_; }

private:
    InferenceEngine& engine_;
    HttpApiServerConfig config_;
    int bound_port_ = 0;
    SOCKET server_socket_;
    bool sockets_ready_ = false;

    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> workers_started_{false};
    std::atomic<uint64_t> request_counter_{0};
    std::atomic<uint64_t> total_requests_{0};
    std::atomic<uint64_t> auth_failures_{0};
    std::atomic<uint64_t> parse_failures_{0};
    std::atomic<uint64_t> rejected_requests_{0};
    std::atomic<uint64_t> rate_limited_requests_{0};
    std::atomic<uint64_t> internal_errors_{0};
    std::atomic<uint64_t> active_requests_{0};
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<SOCKET> client_queue_;
    std::vector<std::thread> workers_;
    std::chrono::steady_clock::time_point started_at_{};
    mutable std::shared_mutex model_state_mutex_;
    std::mutex replica_mutex_;
    std::condition_variable replica_cv_;
    std::vector<std::unique_ptr<InferenceEngine>> inference_replicas_;
    std::vector<size_t> idle_replicas_;
    GenerationMetrics latest_generation_metrics_{};
    mutable std::mutex latest_metrics_mutex_;
    std::mutex rate_limit_mutex_;
    std::unordered_map<std::string, std::deque<std::chrono::steady_clock::time_point>>
        recent_requests_by_client_;

    bool initialize_sockets();
    void cleanup_sockets();
    void close_socket(SOCKET socket);
    void start_workers();
    void initialize_inference_replicas();
    void sync_inference_replicas_locked();
    size_t desired_inference_replica_count() const;
    InferenceEngine* acquire_inference_replica(size_t& replica_index);
    void release_inference_replica(size_t replica_index);
    GenerationMetrics latest_generation_metrics_snapshot() const;
    void update_latest_generation_metrics(const GenerationMetrics& metrics);
    void worker_loop();
    void handle_client(SOCKET client_socket);
};

} // namespace nsos
