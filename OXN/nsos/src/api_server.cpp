#include "../include/http_api_server.h"

#include <exception>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace nsos;

namespace {

std::string value_after_flag(int argc, char* argv[], const std::string& flag,
                             const std::string& fallback = "") {
    for (int i = 1; i + 1 < argc; ++i) {
        if (flag == argv[i]) {
            return argv[i + 1];
        }
    }
    return fallback;
}

bool has_flag(int argc, char* argv[], const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (flag == argv[i]) {
            return true;
        }
    }
    return false;
}

int int_after_flag(int argc, char* argv[], const std::string& flag, int fallback) {
    const std::string value = value_after_flag(argc, argv, flag, "");
    if (value.empty()) {
        return fallback;
    }
    try {
        return std::stoi(value);
    } catch (...) {
        return fallback;
    }
}

std::string env_or_empty(const char* key) {
    const char* value = std::getenv(key);
    return value ? std::string(value) : std::string();
}

bool is_loopback_host(const std::string& host) {
    return host.empty() || host == "localhost" || host == "127.0.0.1" || host == "::1";
}

void validate_server_config(const HttpApiServerConfig& config) {
    const bool loopback = is_loopback_host(config.host);
    if (config.auth_token.empty() && !(loopback && config.allow_unauthenticated_local)) {
        throw std::runtime_error(
            "HTTP API requires --auth-token/NSOS_API_TOKEN by default. "
            "Use --allow-unauthenticated-local only for loopback-only local development.");
    }
    if (!loopback && config.auth_token.empty()) {
        throw std::runtime_error("refusing to bind a non-loopback HTTP API without authentication");
    }
    if (!loopback && config.allow_unauthenticated_local) {
        throw std::runtime_error("--allow-unauthenticated-local is only valid with loopback hosts");
    }
    if (config.require_tls_proxy_header && !config.trust_proxy_headers) {
        throw std::runtime_error("--require-tls-proxy-header requires --trust-proxy-headers");
    }
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        ModelConfig config;
        config.num_layers = int_after_flag(argc, argv, "--layers", config.num_layers);
        config.d_model = int_after_flag(argc, argv, "--d-model", config.d_model);
        config.vocab_size = int_after_flag(argc, argv, "--vocab", config.vocab_size);
        config.max_context_tokens =
            int_after_flag(argc, argv, "--max-context", config.max_context_tokens);
        config.default_batch_size =
            int_after_flag(argc, argv, "--batch-size", config.default_batch_size);
        config.use_cuda = has_flag(argc, argv, "--cuda");

        const std::string model_path = value_after_flag(argc, argv, "--model", "");
        HttpApiServerConfig server_config;
        server_config.host = value_after_flag(argc, argv, "--host", "127.0.0.1");
        server_config.port = int_after_flag(argc, argv, "--port", 8080);
        server_config.auth_token =
            value_after_flag(argc, argv, "--auth-token", env_or_empty("NSOS_API_TOKEN"));
        server_config.worker_threads =
            int_after_flag(argc, argv, "--workers", server_config.worker_threads);
        server_config.inference_replicas =
            int_after_flag(argc, argv, "--inference-replicas", server_config.inference_replicas);
        server_config.max_body_bytes =
            static_cast<size_t>(int_after_flag(argc, argv, "--max-body-bytes",
                                               static_cast<int>(server_config.max_body_bytes)));
        server_config.max_header_bytes =
            static_cast<size_t>(int_after_flag(argc, argv, "--max-header-bytes",
                                               static_cast<int>(server_config.max_header_bytes)));
        server_config.max_queue_depth =
            static_cast<size_t>(int_after_flag(argc, argv, "--max-queue-depth",
                                               static_cast<int>(server_config.max_queue_depth)));
        server_config.rate_limit_requests_per_minute =
            static_cast<size_t>(int_after_flag(argc, argv, "--rate-limit-rpm",
                                               static_cast<int>(server_config.rate_limit_requests_per_minute)));
        server_config.socket_timeout_ms =
            int_after_flag(argc, argv, "--socket-timeout-ms", server_config.socket_timeout_ms);
        server_config.max_generate_tokens =
            int_after_flag(argc, argv, "--max-generate-tokens", server_config.max_generate_tokens);
        server_config.max_context_tokens_per_request =
            int_after_flag(argc, argv, "--max-request-context",
                           server_config.max_context_tokens_per_request);
        server_config.max_prompts_per_batch =
            static_cast<size_t>(int_after_flag(argc, argv, "--max-batch-prompts",
                                               static_cast<int>(server_config.max_prompts_per_batch)));
        server_config.max_train_steps =
            int_after_flag(argc, argv, "--max-train-steps", server_config.max_train_steps);
        server_config.max_train_epochs =
            int_after_flag(argc, argv, "--max-train-epochs", server_config.max_train_epochs);
        server_config.max_train_batch_size =
            int_after_flag(argc, argv, "--max-train-batch-size",
                           server_config.max_train_batch_size);
        server_config.max_train_seq_len =
            int_after_flag(argc, argv, "--max-train-seq-len", server_config.max_train_seq_len);
        server_config.max_train_corpus_steps =
            int_after_flag(argc, argv, "--max-train-corpus-steps",
                           server_config.max_train_corpus_steps);
        server_config.pack_output_root =
            value_after_flag(argc, argv, "--pack-root", server_config.pack_output_root);
        if (has_flag(argc, argv, "--health-auth")) {
            server_config.allow_unauthenticated_health = false;
        }
        if (has_flag(argc, argv, "--allow-unauthenticated-local")) {
            server_config.allow_unauthenticated_local = true;
        }
        if (has_flag(argc, argv, "--enable-admin-endpoints")) {
            server_config.enable_admin_endpoints = true;
        }
        if (has_flag(argc, argv, "--allow-pack-absolute-paths")) {
            server_config.allow_pack_absolute_paths = true;
        }
        if (has_flag(argc, argv, "--allow-cors")) {
            // Inject CORS headers so the Oxta browser UI (file://, localhost:3000, etc.)
            // can call /generate without Same-Origin Policy blocking the request.
            // Only use on local dev or trusted private network — never on a public server
            // without a proper CORS allow-list in front.
            server_config.allow_cors = true;
        }
        if (has_flag(argc, argv, "--trust-proxy-headers")) {
            server_config.trust_proxy_headers = true;
        }
        if (has_flag(argc, argv, "--require-tls-proxy-header")) {
            server_config.require_tls_proxy_header = true;
        }
        validate_server_config(server_config);

        InferenceEngine engine;
        if (!engine.load_model(model_path, config)) {
            std::cerr << "Failed to load model or pack from '" << model_path << "'" << std::endl;
            return 1;
        }

        HttpApiServer server(engine, server_config);
        if (!server.start()) {
            std::cerr << "Failed to start HTTP API server" << std::endl;
            return 1;
        }

        std::cout << "NSOS HTTP API listening on http://" << server_config.host << ":" << server.port()
                  << std::endl;
        server.serve_forever();
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "NSOS HTTP API failed: " << ex.what() << std::endl;
        return 1;
    }
}
