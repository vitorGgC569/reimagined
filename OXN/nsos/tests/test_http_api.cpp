#include "http_api_server.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

using namespace nsos;

namespace {

struct ServerThreadGuard {
    HttpApiServer* server = nullptr;
    std::thread* thread = nullptr;

    ~ServerThreadGuard() {
        if (server) {
            server->stop();
        }
        if (thread && thread->joinable()) {
            thread->join();
        }
    }
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void set_ipv4_address(sockaddr_in& address, const std::string& host) {
#ifdef _WIN32
    const int status = InetPtonA(AF_INET, host.c_str(), &address.sin_addr);
#else
    const int status = inet_pton(AF_INET, host.c_str(), &address.sin_addr);
#endif
    require(status == 1, "invalid IPv4 address: " + host);
}

void send_all_or_throw(SOCKET socket_fd, const std::string& bytes) {
    size_t sent_total = 0;
    while (sent_total < bytes.size()) {
        const int sent = send(socket_fd, bytes.data() + sent_total,
                              static_cast<int>(bytes.size() - sent_total), 0);
        require(sent > 0, "send failed");
        sent_total += static_cast<size_t>(sent);
    }
}

std::string send_http_request(const std::string& host, int port, const std::string& method,
                              const std::string& path, const std::string& body = {},
                              const std::vector<std::string>& extra_headers = {}) {
#ifdef _WIN32
    WSADATA wsa_data{};
    require(WSAStartup(MAKEWORD(2, 2), &wsa_data) == 0, "WSAStartup failed");
#endif

    SOCKET socket_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    require(socket_fd >= 0, "socket creation failed");

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(port));
    set_ipv4_address(address, host);
    require(connect(socket_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0,
            "connect failed");

    std::string request = method + " " + path + " HTTP/1.1\r\nHost: " + host +
                          "\r\nConnection: close\r\n";
    for (const auto& header : extra_headers) {
        request += header + "\r\n";
    }
    if (!body.empty()) {
        request += "Content-Type: application/json\r\n";
        request += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    }
    request += "\r\n";
    request += body;

    send_all_or_throw(socket_fd, request);

    std::string response;
    char buffer[4096];
    while (true) {
        const int received = recv(socket_fd, buffer, sizeof(buffer), 0);
        if (received <= 0) {
            break;
        }
        response.append(buffer, buffer + received);
    }

#ifdef _WIN32
    closesocket(socket_fd);
    WSACleanup();
#else
    close(socket_fd);
#endif

    return response;
}

std::string send_chunked_http_request(const std::string& host, int port, const std::string& method,
                                      const std::string& path,
                                      const std::vector<std::string>& chunks,
                                      const std::vector<std::string>& extra_headers = {}) {
#ifdef _WIN32
    WSADATA wsa_data{};
    require(WSAStartup(MAKEWORD(2, 2), &wsa_data) == 0, "WSAStartup failed");
#endif

    SOCKET socket_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    require(socket_fd >= 0, "socket creation failed");

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(port));
    set_ipv4_address(address, host);
    require(connect(socket_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0,
            "connect failed");

    std::string request = method + " " + path + " HTTP/1.1\r\nHost: " + host +
                          "\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n"
                          "Content-Type: application/json\r\n";
    for (const auto& header : extra_headers) {
        request += header + "\r\n";
    }
    request += "\r\n";
    for (const auto& chunk : chunks) {
        std::ostringstream chunk_size;
        chunk_size << std::hex << chunk.size();
        request += chunk_size.str();
        request += "\r\n";
        request += chunk;
        request += "\r\n";
    }
    request += "0\r\n\r\n";

    send_all_or_throw(socket_fd, request);

    std::string response;
    char buffer[4096];
    while (true) {
        const int received = recv(socket_fd, buffer, sizeof(buffer), 0);
        if (received <= 0) {
            break;
        }
        response.append(buffer, buffer + received);
    }

#ifdef _WIN32
    closesocket(socket_fd);
    WSACleanup();
#else
    close(socket_fd);
#endif

    return response;
}

} // namespace

int main() {
    try {
        ModelConfig config;
        config.num_layers = 1;
        config.d_model = 32;
        config.vocab_size = 256;
        config.n_heads = 4;
        config.n_kv_heads = 2;
        config.max_context_tokens = 64;

        InferenceEngine engine;
        require(engine.load_model("", config), "load_model failed");

        HttpApiServerConfig server_config;
        server_config.host = "127.0.0.1";
        server_config.port = 0;
        server_config.auth_token = "test-secret";
        server_config.worker_threads = 4;
        server_config.max_queue_depth = 32;
        server_config.inference_replicas = 2;
        server_config.enable_admin_endpoints = true;
        server_config.max_generate_tokens = 8;
        server_config.max_prompts_per_batch = 4;
        server_config.max_train_steps = 8;

        HttpApiServer server(engine, server_config);
        engine.request_training_cancellation();
        require(server.start(), "server.start failed");
        require(!engine.training_cancellation_requested(),
                "server.start retained a cancellation request from an old "
                "lifecycle");

        std::thread server_thread([&]() { server.serve_forever(); });
        ServerThreadGuard server_thread_guard{&server, &server_thread};
        std::this_thread::sleep_for(std::chrono::milliseconds(150));

        const int port = server.port();
        require(port > 0, "invalid bound port");
        const std::vector<std::string> auth_headers = {
            "Authorization: Bearer test-secret",
        };

        const std::string health = send_http_request("127.0.0.1", port, "GET", "/health");
        require(health.find("200 OK") != std::string::npos, "health endpoint failed");
        require(health.find("\"status\":\"alive\"") != std::string::npos, "health body mismatch");
        const std::string ready = send_http_request("127.0.0.1", port, "GET", "/ready");
        require(ready.find("200 OK") != std::string::npos &&
                    ready.find("\"status\":\"ready\"") != std::string::npos,
                "ready endpoint did not report the initialized model");

        const std::string unauthorized =
            send_http_request("127.0.0.1", port, "GET", "/metrics");
        require(unauthorized.find("401 Unauthorized") != std::string::npos,
                "metrics should require auth");

        const std::string invalid_json = send_http_request(
            "127.0.0.1", port, "POST", "/generate", "{\"prompt\":", auth_headers);
        require(invalid_json.find("400 Bad Request") != std::string::npos,
                "invalid JSON should return 400");

        const std::string train = send_http_request(
            "127.0.0.1", port, "POST", "/train-text",
            "{\"text\":\"0123012301230123\",\"steps\":8}", auth_headers);
        require(train.find("200 OK") != std::string::npos, "train-text endpoint failed");
        require(train.find("\"loss\":") != std::string::npos, "train-text missing loss");

        const std::string generate = send_http_request(
            "127.0.0.1", port, "POST", "/generate",
            "{\"prompt\":\"0123\",\"max_tokens\":4,\"temperature\":0.2}", auth_headers);
        require(generate.find("200 OK") != std::string::npos, "generate endpoint failed");
        require(generate.find("\"text\":") != std::string::npos,
                "generate response missing text");

        const std::string generate_with_server_default = send_http_request(
            "127.0.0.1", port, "POST", "/generate",
            "{\"prompt\":\"0123\",\"temperature\":0.0}", auth_headers);
        require(generate_with_server_default.find("200 OK") != std::string::npos,
                "generate without max_tokens must use the server-side cap");

        const std::string unicode_generate = send_http_request(
            "127.0.0.1", port, "POST", "/generate",
            "{\"prompt\":\"ol\\u00E1 \\uD83D\\uDE80\",\"max_tokens\":1}", auth_headers);
        require(unicode_generate.find("200 OK") != std::string::npos,
                "unicode JSON request failed: " + unicode_generate);

        const std::string too_many_tokens = send_http_request(
            "127.0.0.1", port, "POST", "/generate",
            "{\"prompt\":\"0123\",\"max_tokens\":999}", auth_headers);
        require(too_many_tokens.find("400 Bad Request") != std::string::npos,
                "generate should reject max_tokens above configured limit");

        const std::string chunked_generate = send_chunked_http_request(
            "127.0.0.1", port, "POST", "/generate",
            {"{\"prompt\":\"01\",", "\"max_tokens\":2,\"temperature\":0.2}"},
            auth_headers);
        require(chunked_generate.find("200 OK") != std::string::npos,
                "chunked generate endpoint failed: " + chunked_generate);

        const std::string info =
            send_http_request("127.0.0.1", port, "GET", "/info", {}, auth_headers);
        require(info.find("200 OK") != std::string::npos, "info endpoint failed");
        require(info.find("\"inference_replicas\":2") != std::string::npos,
                "info endpoint missing replica count");

        const std::string batch = send_http_request(
            "127.0.0.1", port, "POST", "/generate_batch",
            "{\"prompts\":[\"01\",\"1234\"],\"max_tokens\":2}", auth_headers);
        require(batch.find("200 OK") != std::string::npos, "generate_batch endpoint failed");
        require(batch.find("\"outputs\":") != std::string::npos,
                "generate_batch missing outputs");

        const std::string stream = send_http_request(
            "127.0.0.1", port, "POST", "/generate_stream",
            "{\"prompt\":\"01\",\"max_tokens\":2}", auth_headers);
        require(stream.find("200 OK") != std::string::npos, "generate_stream endpoint failed");
        require(stream.find("event: done") != std::string::npos,
                "generate_stream missing done event");
        const size_t stream_headers_end = stream.find("\r\n\r\n");
        require(stream_headers_end != std::string::npos,
                "generate_stream response missing header terminator");
        require(stream.substr(0, stream_headers_end).find("Content-Length:") ==
                    std::string::npos,
                "generate_stream must not declare a fixed content length");
        const size_t done_pos = stream.find("event: done");
        const size_t last_chunk_pos = stream.rfind("event: chunk");
        require(last_chunk_pos == std::string::npos || last_chunk_pos < done_pos,
                "generate_stream emitted a chunk after the done event");

        const std::string invalid_content_length = send_http_request(
            "127.0.0.1", port, "POST", "/generate", {},
            {"Authorization: Bearer test-secret", "Content-Length: 2x"});
        require(invalid_content_length.find("400 Bad Request") != std::string::npos,
                "invalid content-length suffix should be rejected");
        const std::string signed_content_length = send_http_request(
            "127.0.0.1", port, "POST", "/generate", {},
            {"Authorization: Bearer test-secret", "Content-Length: +0"});
        require(signed_content_length.find("400 Bad Request") != std::string::npos,
                "signed content-length should be rejected");
        const std::string invalid_header_name = send_http_request(
            "127.0.0.1", port, "GET", "/info", {},
            {"Authorization: Bearer test-secret", "Bad@Header: value"});
        require(invalid_header_name.find("400 Bad Request") != std::string::npos,
                "non-token HTTP header name should be rejected");

        const auto parameters_before_rejected_train = engine.model->parameters();
        require(!parameters_before_rejected_train.empty(), "model has no parameters");
        const float weight_before_rejected_train =
            parameters_before_rejected_train.front()->data.data()[0];
        const int step_before_rejected_train = engine.trainer->global_step_count;
        const std::string rejected_train_batch = send_http_request(
            "127.0.0.1", port, "POST", "/train-batch",
            "{\"texts\":[\"01230123\",\"A\"],\"epochs\":1}", auth_headers);
        require(rejected_train_batch.find("400 Bad Request") != std::string::npos,
                "train-batch should reject samples shorter than two tokens");
        require(engine.model->parameters().front()->data.data()[0] ==
                    weight_before_rejected_train &&
                    engine.trainer->global_step_count == step_before_rejected_train,
                "rejected administrative training mutated live model state");

        std::atomic<int> concurrent_ok{0};
        std::vector<std::thread> clients;
        for (int i = 0; i < 8; ++i) {
            clients.emplace_back([&, i]() {
                const std::string response = send_http_request(
                    "127.0.0.1", port, "POST", "/generate",
                    "{\"prompt\":\"0123\",\"max_tokens\":2,\"temperature\":0.2}", auth_headers);
                if (response.find("200 OK") != std::string::npos &&
                    response.find("\"request_id\":") != std::string::npos) {
                    concurrent_ok.fetch_add(1);
                }
            });
        }
        for (auto& client : clients) {
            client.join();
        }
        require(concurrent_ok.load() == 8, "concurrent generate requests failed");

        const std::string metrics =
            send_http_request("127.0.0.1", port, "GET", "/metrics", {}, auth_headers);
        require(metrics.find("200 OK") != std::string::npos, "metrics endpoint failed");
        require(metrics.find("\"requests_total\":") != std::string::npos,
                "metrics missing requests_total");
        require(metrics.find("\"auth_failures\":") != std::string::npos,
                "metrics missing auth_failures");
        require(metrics.find("\"rate_limited_requests\":") != std::string::npos,
                "metrics missing rate_limited_requests");

        server.stop();

        ModelConfig strict_config;
        strict_config.num_layers = 1;
        strict_config.d_model = 32;
        strict_config.vocab_size = 16;
        strict_config.n_heads = 4;
        strict_config.n_kv_heads = 2;
        strict_config.max_context_tokens = 32;

        InferenceEngine strict_engine;
        require(strict_engine.load_model("", strict_config), "strict load_model failed");

        HttpApiServerConfig strict_server_config;
        strict_server_config.host = "127.0.0.1";
        strict_server_config.port = 0;
        strict_server_config.auth_token = "strict-secret";
        strict_server_config.worker_threads = 2;
        strict_server_config.max_queue_depth = 8;
        strict_server_config.inference_replicas = 1;
        strict_server_config.rate_limit_requests_per_minute = 3;
        strict_server_config.enable_admin_endpoints = true;

        HttpApiServer strict_server(strict_engine, strict_server_config);
        require(strict_server.start(), "strict server.start failed");

        std::thread strict_server_thread([&]() { strict_server.serve_forever(); });
        ServerThreadGuard strict_thread_guard{&strict_server, &strict_server_thread};
        std::this_thread::sleep_for(std::chrono::milliseconds(150));

        const int strict_port = strict_server.port();
        require(strict_port > 0, "invalid strict bound port");
        const std::vector<std::string> strict_auth_headers = {
            "Authorization: Bearer strict-secret",
        };

        const std::string strict_train_corpus = send_http_request(
            "127.0.0.1", strict_port, "POST", "/train-corpus",
            "{\"corpus\":\"A\",\"epochs\":1,\"batch_size\":1,\"seq_len\":4}",
            strict_auth_headers);
        require(strict_train_corpus.find("500 Internal Server Error") != std::string::npos,
                "train-corpus internal model mismatch should return HTTP 500");
        require(strict_train_corpus.find("Tokenizer/model vocabulary mismatch") ==
                    std::string::npos,
                "train-corpus must not leak internal model details");

        const std::string info_one =
            send_http_request("127.0.0.1", strict_port, "GET", "/info", {}, strict_auth_headers);
        const std::string info_two =
            send_http_request("127.0.0.1", strict_port, "GET", "/info", {}, strict_auth_headers);
        const std::string info_three =
            send_http_request("127.0.0.1", strict_port, "GET", "/info", {}, strict_auth_headers);
        require(info_one.find("200 OK") != std::string::npos, "strict info request 1 failed");
        require(info_two.find("200 OK") != std::string::npos, "strict info request 2 failed");
        require(info_three.find("429 Too Many Requests") != std::string::npos,
                "strict info request 3 should be rate limited");

        strict_server.stop();

        InferenceEngine unauth_engine;
        require(unauth_engine.load_model("", config), "unauth load_model failed");
        HttpApiServerConfig unauth_config;
        unauth_config.host = "127.0.0.1";
        unauth_config.port = 0;
        HttpApiServer unauth_server(unauth_engine, unauth_config);
        require(!unauth_server.start(), "server without token should fail unless explicitly allowed");

        InferenceEngine untrusted_proxy_engine;
        require(untrusted_proxy_engine.load_model("", config), "proxy load_model failed");
        HttpApiServerConfig untrusted_proxy_config;
        untrusted_proxy_config.host = "127.0.0.1";
        untrusted_proxy_config.port = 0;
        untrusted_proxy_config.auth_token = "proxy-secret";
        untrusted_proxy_config.trust_proxy_headers = true;
        HttpApiServer untrusted_proxy_server(untrusted_proxy_engine, untrusted_proxy_config);
        require(!untrusted_proxy_server.start(),
                "proxy-header trust without an exact peer allowlist should fail");

        InferenceEngine locked_engine;
        require(locked_engine.load_model("", config), "locked load_model failed");
        HttpApiServerConfig locked_config;
        locked_config.host = "127.0.0.1";
        locked_config.port = 0;
        locked_config.auth_token = "locked-secret";
        locked_config.worker_threads = 1;
        HttpApiServer locked_server(locked_engine, locked_config);
        require(locked_server.start(), "locked server.start failed");
        std::thread locked_thread([&]() { locked_server.serve_forever(); });
        ServerThreadGuard locked_guard{&locked_server, &locked_thread};
        const int locked_port = locked_server.port();
        const std::vector<std::string> locked_auth_headers = {
            "Authorization: Bearer locked-secret",
        };
        const std::string disabled_train = send_http_request(
            "127.0.0.1", locked_port, "POST", "/train-text",
            "{\"text\":\"01230123\",\"steps\":1}", locked_auth_headers);
        require(disabled_train.find("403 Forbidden") != std::string::npos,
                "train-text should be disabled without admin flag");
        locked_server.stop();

        HttpApiServerConfig tls_config = locked_config;
        tls_config.auth_token = "container-health-secret";
        tls_config.allow_unauthenticated_health = false;
        tls_config.trust_proxy_headers = true;
        tls_config.require_tls_proxy_header = true;
        tls_config.trusted_proxy_ips = {"127.0.0.2"};
        HttpApiServer tls_server(locked_engine, tls_config);
        require(tls_server.start(), "TLS proxy server did not start");
        std::thread tls_thread([&]() { tls_server.serve_forever(); });
        ServerThreadGuard tls_guard{&tls_server, &tls_thread};
        const std::vector<std::string> tls_auth = {
            "Authorization: Bearer container-health-secret"};
        const auto local_ready = send_http_request("127.0.0.1", tls_server.port(),
            "GET", "/ready", {}, tls_auth);
        require(local_ready.find("200 OK") != std::string::npos,
                "authenticated loopback readiness must not require a proxy header");
        const auto no_auth_ready = send_http_request("127.0.0.1", tls_server.port(),
            "GET", "/ready", {}, {});
        require(no_auth_ready.find("401 Unauthorized") != std::string::npos,
                "loopback readiness must still require authentication");
        const auto insecure_info = send_http_request("127.0.0.1", tls_server.port(),
            "GET", "/info", {}, tls_auth);
        require(insecure_info.find("426 Upgrade Required") != std::string::npos &&
                    insecure_info.find("https_required") != std::string::npos,
                "health exception must not bypass TLS enforcement on other routes");
        tls_server.stop();

        std::cout << "HTTP API hardening test passed!" << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "HTTP API hardening test failed: " << ex.what() << std::endl;
        return 1;
    }
}
