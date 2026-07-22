#include "http_api_server.h"
#include "nsos/determinism.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <cstdlib>
#include <omp.h>

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

struct CircuitExample {
    std::string prompt;
    char expected;
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string json_escape(const std::string& text) {
    std::string escaped;
    for (char c : text) {
        switch (c) {
        case '\\': escaped += "\\\\"; break;
        case '"': escaped += "\\\""; break;
        case '\n': escaped += "\\n"; break;
        case '\r': escaped += "\\r"; break;
        case '\t': escaped += "\\t"; break;
        default: escaped.push_back(c); break;
        }
    }
    return escaped;
}

std::string make_training_line(const std::string& gate, int a, int b, int out) {
    return gate + std::to_string(a) + std::to_string(b) + "=" + std::to_string(out) + ";";
}

std::vector<std::string> make_truth_table_lines() {
    return {
        make_training_line("AND", 0, 0, 0),
        make_training_line("AND", 0, 1, 0),
        make_training_line("AND", 1, 0, 0),
        make_training_line("AND", 1, 1, 1),
        make_training_line("OR", 0, 0, 0),
        make_training_line("OR", 0, 1, 1),
        make_training_line("OR", 1, 0, 1),
        make_training_line("OR", 1, 1, 1),
        make_training_line("XOR", 0, 0, 0),
        make_training_line("XOR", 0, 1, 1),
        make_training_line("XOR", 1, 0, 1),
        make_training_line("XOR", 1, 1, 0),
        make_training_line("NAND", 0, 0, 1),
        make_training_line("NAND", 0, 1, 1),
        make_training_line("NAND", 1, 0, 1),
        make_training_line("NAND", 1, 1, 0),
    };
}

std::string make_training_corpus() {
    const auto lines = make_truth_table_lines();
    std::string corpus;
    for (int repeat = 0; repeat < 256; ++repeat) {
        for (const auto& line : lines) {
            corpus += line;
        }
    }
    return corpus;
}

std::vector<CircuitExample> make_eval_examples() {
    const auto lines = make_truth_table_lines();
    std::vector<CircuitExample> examples;
    std::string prefix;
    for (const auto& line : lines) {
        const auto eq = line.find('=');
        const auto out = line.find(';');
        if (eq == std::string::npos || out == std::string::npos || eq + 1 >= out) {
            continue;
        }

        const std::string lhs = line.substr(0, eq + 1);
        const char expected = line[eq + 1];
        examples.push_back({prefix + lhs, expected});
        prefix += line;
    }
    return examples;
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
    address.sin_addr.s_addr = inet_addr(host.c_str());
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

    require(send(socket_fd, request.data(), static_cast<int>(request.size()), 0) > 0,
            "send failed");

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

std::string response_body(const std::string& response) {
    const auto pos = response.find("\r\n\r\n");
    return pos == std::string::npos ? response : response.substr(pos + 4);
}

std::string extract_string_field(const std::string& body, const std::string& field) {
    const std::string needle = "\"" + field + "\":\"";
    const auto start = body.find(needle);
    if (start == std::string::npos) {
        return "";
    }
    const auto value_start = start + needle.size();
    const auto value_end = body.find('"', value_start);
    if (value_end == std::string::npos) {
        return "";
    }
    return body.substr(value_start, value_end - value_start);
}

std::vector<std::string> extract_outputs(const std::string& body) {
    std::vector<std::string> outputs;
    const std::string needle = "\"outputs\":[";
    const auto start = body.find(needle);
    if (start == std::string::npos) {
        return outputs;
    }
    auto pos = start + needle.size();
    while (pos < body.size() && body[pos] != ']') {
        if (body[pos] == '"') {
            const auto end = body.find('"', pos + 1);
            if (end == std::string::npos) {
                break;
            }
            outputs.push_back(body.substr(pos + 1, end - pos - 1));
            pos = end + 1;
        } else {
            ++pos;
        }
    }
    return outputs;
}

int evaluate_examples(const std::string& host, int port, const std::vector<std::string>& headers,
                      const std::vector<CircuitExample>& examples) {
    int hits = 0;
    for (const auto& example : examples) {
        const std::string payload =
            "{\"prompt\":\"" + json_escape(example.prompt) +
            "\",\"max_tokens\":1,\"temperature\":0.05,\"top_k\":1}";
        const std::string response =
            send_http_request(host, port, "POST", "/generate", payload, headers);
        require(response.find("200 OK") != std::string::npos, "generate request failed");
        const std::string text = extract_string_field(response_body(response), "text");
        std::cout << "[CircuitAPI] prompt='" << example.prompt << "' got='"
                  << (text.empty() ? "<empty>" : text) << "' expected='" << example.expected
                  << "'" << std::endl;
        if (!text.empty() && text.front() == example.expected) {
            ++hits;
        }
    }
    return hits;
}

} // namespace

int main() {
    std::thread server_thread;
    std::thread reloaded_thread;
    HttpApiServer* active_server = nullptr;
    HttpApiServer* active_reloaded_server = nullptr;
    try {
        // Determinism: a fixed global seed makes the tiny circuit model's init
        // reproducible (the tensor_rng() fix makes a global seed honor-able),
        // and a single OpenMP thread removes float reduction-order variance.
        // Together they make this train+accuracy gate deterministic instead of
        // flaky.  (DeterminismManager is internally mutex-guarded; safe here.)
        omp_set_num_threads(1);
        uint64_t circuit_seed = 7ull;  // a seed that converges deterministically
        if (const char* seed_env = std::getenv("NSOS_TEST_SEED")) {
            circuit_seed = std::strtoull(seed_env, nullptr, 10);
        }
        nsos::determinism::DeterminismManager::instance().set_global_seed(circuit_seed);
        const std::string host = "127.0.0.1";
        const std::string auth_token = "circuit-secret";
        const std::vector<std::string> auth_headers = {
            "Authorization: Bearer " + auth_token,
        };
        const auto training_corpus = make_training_corpus();
        const auto eval_examples = make_eval_examples();

        ModelConfig config;
        config.num_layers = 1;
        config.d_model = 64;
        config.vocab_size = 128;
        config.n_heads = 8;
        config.n_kv_heads = 4;
        config.max_context_tokens = 256;

        InferenceEngine engine;
        require(engine.load_model("", config), "initial load_model failed");

        HttpApiServerConfig server_config;
        server_config.host = host;
        server_config.port = 0;
        server_config.auth_token = auth_token;
        server_config.worker_threads = 4;
        server_config.enable_admin_endpoints = true;
        server_config.allow_pack_absolute_paths = true;
        server_config.max_train_epochs = 100;
        server_config.max_train_corpus_steps = 240;
        server_config.max_train_batch_size = 4;
        server_config.max_train_seq_len = 64;

        HttpApiServer server(engine, server_config);
        active_server = &server;
        require(server.start(), "server.start failed");
        server_thread = std::thread([&]() { server.serve_forever(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        std::cout << "[CircuitAPI] server started on " << server.port() << std::endl;

        const int port = server.port();
        require(port > 0, "invalid server port");

        const std::string corpus_json =
            "{\"corpus\":\"" + json_escape(training_corpus) +
            "\",\"epochs\":80,\"batch_size\":4,\"seq_len\":64,\"max_steps\":240,"
            "\"learning_rate\":0.003,\"weight_decay\":0.0,\"max_grad_norm\":2.0,"
            "\"warmup_steps\":10,\"min_learning_rate_scale\":0.2}";

        const std::string train_response =
            send_http_request(host, port, "POST", "/train-corpus", corpus_json, auth_headers);
        std::cout << "[CircuitAPI] train-corpus finished" << std::endl;
        require(train_response.find("200 OK") != std::string::npos, "train-corpus failed");
        require(train_response.find("\"token_count\":") != std::string::npos,
                "train-corpus missing token_count");

        const int prepack_hits = evaluate_examples(host, port, auth_headers, eval_examples);
        std::cout << "[CircuitAPI] prepack hits=" << prepack_hits << std::endl;
        require(prepack_hits >= 10,
                "circuit model accuracy before pack/reload is too low: " +
                    std::to_string(prepack_hits));

        const auto pack_dir = std::filesystem::temp_directory_path() / "nsos_circuit_api_pack";
        std::filesystem::remove_all(pack_dir);
        const std::string pack_payload =
            "{\"directory\":\"" + json_escape(pack_dir.string()) + "\"}";
        const std::string pack_response =
            send_http_request(host, port, "POST", "/pack", pack_payload, auth_headers);
        std::cout << "[CircuitAPI] pack saved" << std::endl;
        require(pack_response.find("200 OK") != std::string::npos, "pack endpoint failed");
        require(std::filesystem::exists(pack_dir / "manifest.nsos"), "manifest not written");

        server.stop();
        server_thread.join();
        std::cout << "[CircuitAPI] first server stopped" << std::endl;

        InferenceEngine reloaded;
        require(reloaded.load_model(pack_dir.string(), ModelConfig{}), "reload from pack failed");

        HttpApiServer reloaded_server(reloaded, server_config);
        active_reloaded_server = &reloaded_server;
        require(reloaded_server.start(), "reloaded_server.start failed");
        reloaded_thread = std::thread([&]() { reloaded_server.serve_forever(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        std::cout << "[CircuitAPI] reloaded server started on " << reloaded_server.port() << std::endl;

        const int reloaded_port = reloaded_server.port();
        require(reloaded_port > 0, "invalid reloaded server port");

        std::string batch_payload = "{\"prompts\":[";
        for (size_t i = 0; i < eval_examples.size(); ++i) {
            if (i > 0) {
                batch_payload += ",";
            }
            batch_payload += "\"" + json_escape(eval_examples[i].prompt) + "\"";
        }
        batch_payload += "],\"max_tokens\":1,\"temperature\":0.05,\"top_k\":1}";

        const std::string batch_response =
            send_http_request(host, reloaded_port, "POST", "/generate_batch", batch_payload,
                              auth_headers);
        std::cout << "[CircuitAPI] generate_batch after reload finished" << std::endl;
        require(batch_response.find("200 OK") != std::string::npos,
                "generate_batch after reload failed");
        const auto outputs = extract_outputs(response_body(batch_response));
        require(outputs.size() == eval_examples.size(), "unexpected number of batch outputs");

        int non_empty_batch_outputs = 0;
        for (const auto& output : outputs) {
            if (!output.empty()) {
                ++non_empty_batch_outputs;
            }
        }
        require(non_empty_batch_outputs >= static_cast<int>(eval_examples.size() / 2),
                "generate_batch returned too many empty outputs after reload");

        const int reload_hits =
            evaluate_examples(host, reloaded_port, auth_headers, eval_examples);
        require(reload_hits >= 10,
                "circuit model accuracy after pack/reload is too low: " +
                    std::to_string(reload_hits));
        std::cout << "[CircuitAPI] reload hits=" << reload_hits << std::endl;

        reloaded_server.stop();
        reloaded_thread.join();
        std::filesystem::remove_all(pack_dir);

        std::cout << "Circuit API pipeline test passed! prepack_hits=" << prepack_hits
                  << " reload_hits=" << reload_hits << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        if (active_reloaded_server) {
            active_reloaded_server->stop();
        }
        if (reloaded_thread.joinable()) {
            reloaded_thread.join();
        }
        if (active_server) {
            active_server->stop();
        }
        if (server_thread.joinable()) {
            server_thread.join();
        }
        std::cerr << "Circuit API pipeline test failed: " << ex.what() << std::endl;
        return 1;
    }
}
