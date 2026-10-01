#include "../include/http_api_server.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace nsos {

namespace {

constexpr const char* kApiVersion = "nsos-http-v2";

#ifdef _WIN32
constexpr SOCKET kInvalidSocket = INVALID_SOCKET;
#else
constexpr SOCKET kInvalidSocket = -1;
#endif
constexpr size_t kInvalidReplicaIndex = static_cast<size_t>(-1);

void append_utf8_codepoint(std::string& out, uint32_t codepoint);
unsigned int parse_unicode_hex_quad(const std::string& input, size_t& pos);

// Thrown for malformed CLIENT input (bad JSON, out-of-range fields, unsafe
// paths).  handle_client maps it to HTTP 400 with the message; every OTHER
// exception is an internal fault -> HTTP 500 with a generic body (the real
// cause is logged server-side, never leaked to the client).
struct BadRequest : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct StreamAborted : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool bool_value = false;
    double number_value = 0.0;
    std::string string_value;
    std::vector<JsonValue> array_value;
    std::map<std::string, JsonValue> object_value;

    bool is_bool() const { return type == Type::Bool; }
    bool is_number() const { return type == Type::Number; }
    bool is_string() const { return type == Type::String; }
    bool is_array() const { return type == Type::Array; }
    bool is_object() const { return type == Type::Object; }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& input, int max_depth)
        : input_(input), max_depth_((std::max)(max_depth, 1)) {}

    JsonValue parse() {
        skip_ws();
        JsonValue value = parse_value(0);
        skip_ws();
        if (!eof()) {
            error("unexpected trailing content");
        }
        return value;
    }

private:
    const std::string& input_;
    int max_depth_ = 32;
    size_t pos_ = 0;

    bool eof() const { return pos_ >= input_.size(); }
    char peek() const { return eof() ? '\0' : input_[pos_]; }

    char consume() {
        if (eof()) {
            error("unexpected end of input");
        }
        return input_[pos_++];
    }

    void skip_ws() {
        while (!eof() && std::isspace(static_cast<unsigned char>(input_[pos_]))) {
            ++pos_;
        }
    }

    [[noreturn]] void error(const std::string& message) const {
        throw BadRequest("JSON parse error at position " + std::to_string(pos_) + ": " +
                         message);
    }

    void expect(char expected) {
        const char got = consume();
        if (got != expected) {
            error(std::string("expected '") + expected + "', got '" + got + "'");
        }
    }

    void expect_literal(const char* literal) {
        for (const char* ptr = literal; *ptr != '\0'; ++ptr) {
            if (consume() != *ptr) {
                error(std::string("expected literal ") + literal);
            }
        }
    }

    JsonValue parse_value(int depth) {
        if (depth > max_depth_) {
            error("JSON nesting exceeded configured limit");
        }
        skip_ws();
        if (eof()) {
            error("missing value");
        }

        switch (peek()) {
        case '{':
            return parse_object(depth + 1);
        case '[':
            return parse_array(depth + 1);
        case '"':
            return parse_string();
        case 't':
            return parse_true();
        case 'f':
            return parse_false();
        case 'n':
            return parse_null();
        default:
            if (peek() == '-' || std::isdigit(static_cast<unsigned char>(peek()))) {
                return parse_number();
            }
            error("invalid value");
        }
    }

    JsonValue parse_object(int depth) {
        expect('{');
        JsonValue value;
        value.type = JsonValue::Type::Object;
        skip_ws();
        if (peek() == '}') {
            consume();
            return value;
        }

        while (true) {
            skip_ws();
            JsonValue key = parse_string();
            skip_ws();
            expect(':');
            skip_ws();
            JsonValue member = parse_value(depth);
            const auto [unused, inserted] =
                value.object_value.emplace(key.string_value, std::move(member));
            if (!inserted) {
                error("duplicate object key");
            }
            skip_ws();
            if (peek() == '}') {
                consume();
                break;
            }
            expect(',');
        }
        return value;
    }

    JsonValue parse_array(int depth) {
        expect('[');
        JsonValue value;
        value.type = JsonValue::Type::Array;
        skip_ws();
        if (peek() == ']') {
            consume();
            return value;
        }

        while (true) {
            value.array_value.push_back(parse_value(depth));
            skip_ws();
            if (peek() == ']') {
                consume();
                break;
            }
            expect(',');
        }
        return value;
    }

    JsonValue parse_string() {
        expect('"');
        JsonValue value;
        value.type = JsonValue::Type::String;

        while (true) {
            if (eof()) {
                error("unterminated string");
            }
            const char ch = consume();
            if (ch == '"') {
                break;
            }
            if (ch != '\\') {
                if (static_cast<unsigned char>(ch) < 0x20u) {
                    error("unescaped control character in string");
                }
                value.string_value.push_back(ch);
                continue;
            }

            if (eof()) {
                error("unterminated escape");
            }
            const char escaped = consume();
            switch (escaped) {
            case '"':
            case '\\':
            case '/':
                value.string_value.push_back(escaped);
                break;
            case 'b':
                value.string_value.push_back('\b');
                break;
            case 'f':
                value.string_value.push_back('\f');
                break;
            case 'n':
                value.string_value.push_back('\n');
                break;
            case 'r':
                value.string_value.push_back('\r');
                break;
            case 't':
                value.string_value.push_back('\t');
                break;
            case 'u': {
                try {
                    uint32_t code = parse_unicode_hex_quad(input_, pos_);
                    if (code >= 0xD800 && code <= 0xDBFF) {
                        if (pos_ + 6 > input_.size() || input_[pos_] != '\\' || input_[pos_ + 1] != 'u') {
                            error("invalid unicode surrogate pair");
                        }
                        pos_ += 2;
                        const uint32_t low = parse_unicode_hex_quad(input_, pos_);
                        if (low < 0xDC00 || low > 0xDFFF) {
                            error("invalid unicode surrogate pair");
                        }
                        code = 0x10000 + (((code - 0xD800) << 10) | (low - 0xDC00));
                    } else if (code >= 0xDC00 && code <= 0xDFFF) {
                        error("unexpected low surrogate");
                    }
                    append_utf8_codepoint(value.string_value, code);
                } catch (const std::exception& ex) {
                    error(ex.what());
                }
                break;
            }
            default:
                error("unsupported escape sequence");
            }
        }

        return value;
    }

    JsonValue parse_number() {
        const size_t begin = pos_;
        if (peek() == '-') consume();
        if (!std::isdigit(static_cast<unsigned char>(peek()))) error("invalid number");
        if (peek() == '0') consume();
        else while (std::isdigit(static_cast<unsigned char>(peek()))) consume();
        if (peek() == '.') {
            consume();
            if (!std::isdigit(static_cast<unsigned char>(peek()))) error("invalid fractional number");
            while (std::isdigit(static_cast<unsigned char>(peek()))) consume();
        }
        if (peek() == 'e' || peek() == 'E') {
            consume();
            if (peek() == '+' || peek() == '-') consume();
            if (!std::isdigit(static_cast<unsigned char>(peek()))) error("invalid exponent");
            while (std::isdigit(static_cast<unsigned char>(peek()))) consume();
        }

        JsonValue value;
        value.type = JsonValue::Type::Number;
        try {
            value.number_value = std::stod(input_.substr(begin, pos_ - begin));
        } catch (const std::exception&) {
            error("number is outside the supported range");
        }
        if (!std::isfinite(value.number_value)) {
            error("number must be finite");
        }
        return value;
    }

    JsonValue parse_true() {
        expect_literal("true");
        JsonValue value;
        value.type = JsonValue::Type::Bool;
        value.bool_value = true;
        return value;
    }

    JsonValue parse_false() {
        expect_literal("false");
        JsonValue value;
        value.type = JsonValue::Type::Bool;
        value.bool_value = false;
        return value;
    }

    JsonValue parse_null() {
        expect_literal("null");
        return JsonValue{};
    }
};

struct HttpRequest {
    std::string method;
    std::string raw_target;
    std::string path;
    std::string version;
    std::map<std::string, std::string> headers;
    std::string body;
};

struct HttpResponse {
    int status_code = 200;
    std::string status_text = "OK";
    std::string content_type = "application/json";
    std::map<std::string, std::string> headers;
    std::string body;
};

struct RequestReadResult {
    bool success = false;
    HttpRequest request;
    HttpResponse error;
};

std::string trim_copy(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

bool is_http_token_char(unsigned char c) {
    if (std::isalnum(c)) return true;
    switch (c) {
    case '!': case '#': case '$': case '%': case '&': case '\'': case '*':
    case '+': case '-': case '.': case '^': case '_': case '`': case '|': case '~':
        return true;
    default:
        return false;
    }
}

bool is_valid_header_value(const std::string& value) {
    return std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return c == '\t' || (c >= 0x20u && c != 0x7fu);
    });
}

void append_utf8_codepoint(std::string& out, uint32_t codepoint) {
    if (codepoint > 0x10FFFF || (codepoint >= 0xD800 && codepoint <= 0xDFFF)) {
        throw std::runtime_error("invalid unicode codepoint");
    }
    if (codepoint <= 0x7F) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
}

unsigned int parse_unicode_hex_quad(const std::string& input, size_t& pos) {
    if (pos + 4 > input.size()) {
        throw std::runtime_error("invalid unicode escape");
    }
    unsigned int code = 0;
    for (int i = 0; i < 4; ++i) {
        code <<= 4;
        const char hex = input[pos++];
        if (hex >= '0' && hex <= '9') code |= static_cast<unsigned int>(hex - '0');
        else if (hex >= 'a' && hex <= 'f') code |= static_cast<unsigned int>(10 + hex - 'a');
        else if (hex >= 'A' && hex <= 'F') code |= static_cast<unsigned int>(10 + hex - 'A');
        else throw std::runtime_error("invalid unicode escape");
    }
    return code;
}

std::string to_lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string json_escape(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size() + 8);
    auto append_control_or_ascii = [&](unsigned char c) {
        switch (c) {
        case '\\':
            escaped += "\\\\";
            break;
        case '"':
            escaped += "\\\"";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            if (c < 0x20) {
                std::ostringstream out;
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
                escaped += out.str();
            } else {
                escaped.push_back(static_cast<char>(c));
            }
            break;
        }
    };

    auto append_replacement = [&]() {
        escaped += "\\uFFFD";
    };

    for (size_t i = 0; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < 0x80) {
            append_control_or_ascii(c);
            ++i;
            continue;
        }

        size_t needed = 0;
        uint32_t codepoint = 0;
        if ((c & 0xE0u) == 0xC0u) {
            needed = 2;
            codepoint = c & 0x1Fu;
        } else if ((c & 0xF0u) == 0xE0u) {
            needed = 3;
            codepoint = c & 0x0Fu;
        } else if ((c & 0xF8u) == 0xF0u) {
            needed = 4;
            codepoint = c & 0x07u;
        } else {
            append_replacement();
            ++i;
            continue;
        }

        if (i + needed > text.size()) {
            append_replacement();
            break;
        }

        bool valid = true;
        for (size_t j = 1; j < needed; ++j) {
            const unsigned char cont = static_cast<unsigned char>(text[i + j]);
            if ((cont & 0xC0u) != 0x80u) {
                valid = false;
                break;
            }
            codepoint = (codepoint << 6) | (cont & 0x3Fu);
        }

        const bool overlong =
            (needed == 2 && codepoint < 0x80u) ||
            (needed == 3 && codepoint < 0x800u) ||
            (needed == 4 && codepoint < 0x10000u);
        if (!valid || overlong || codepoint > 0x10FFFFu ||
            (codepoint >= 0xD800u && codepoint <= 0xDFFFu)) {
            append_replacement();
            ++i;
            continue;
        }

        escaped.append(text, i, needed);
        i += needed;
    }
    return escaped;
}

std::string json_bool(bool value) { return value ? "true" : "false"; }

} // namespace

namespace {

std::string json_error_body(uint64_t request_id, const std::string& code,
                            const std::string& message) {
    std::ostringstream body;
    body << "{"
         << "\"ok\":false,"
         << "\"request_id\":" << request_id << ','
         << "\"error\":{"
         << "\"code\":\"" << json_escape(code) << "\","
         << "\"message\":\"" << json_escape(message) << "\""
         << "}"
         << "}";
    return body.str();
}

std::string join_json_fields(const std::vector<std::string>& fields) {
    std::ostringstream json;
    json << "{";
    for (size_t i = 0; i < fields.size(); ++i) {
        if (i > 0) {
            json << ',';
        }
        json << fields[i];
    }
    json << "}";
    return json.str();
}

std::string generation_metrics_json(const GenerationMetrics& metrics) {
    return join_json_fields({
        "\"prompt_tokens_total\":" + std::to_string(metrics.prompt_tokens_total),
        "\"prompt_tokens_used\":" + std::to_string(metrics.prompt_tokens_used),
        "\"generated_tokens\":" + std::to_string(metrics.generated_tokens),
        "\"batch_size\":" + std::to_string(metrics.batch_size),
        "\"elapsed_ms\":" + std::to_string(metrics.elapsed_ms),
        "\"prefill_ms\":" + std::to_string(metrics.prefill_ms),
        "\"decode_ms\":" + std::to_string(metrics.decode_ms),
        "\"sampler_ms\":" + std::to_string(metrics.sampler_ms),
        "\"prompt_tokens_per_sec\":" + std::to_string(metrics.prompt_tokens_per_sec),
        "\"decode_tokens_per_sec\":" + std::to_string(metrics.decode_tokens_per_sec),
        "\"total_tokens_per_sec\":" + std::to_string(metrics.total_tokens_per_sec),
        "\"used_streaming\":" + json_bool(metrics.used_streaming),
        "\"loaded_from_pack\":" + json_bool(metrics.loaded_from_pack),
        "\"mamba_fast_path_hits\":" + std::to_string(metrics.mamba_fast_path_hits),
        "\"mamba_fast_path_fallbacks\":" + std::to_string(metrics.mamba_fast_path_fallbacks),
        "\"mamba_stream_priming_gpu_calls\":" +
            std::to_string(metrics.mamba_stream_priming_gpu_calls),
        "\"mamba_stream_priming_host_fallbacks\":" +
            std::to_string(metrics.mamba_stream_priming_host_fallbacks),
        "\"mamba_last_fallback_reason\":\"" + json_escape(metrics.mamba_last_fallback_reason) + "\"",
    });
}

std::optional<const JsonValue*> object_lookup(const JsonValue& object, const std::string& key) {
    if (!object.is_object()) {
        return std::nullopt;
    }
    const auto it = object.object_value.find(key);
    if (it == object.object_value.end()) {
        return std::nullopt;
    }
    return &it->second;
}

std::optional<std::string> json_string(const JsonValue& object, const std::string& key) {
    const auto value = object_lookup(object, key);
    if (!value || !(*value)->is_string()) {
        return std::nullopt;
    }
    return (*value)->string_value;
}

std::optional<int> json_int(const JsonValue& object, const std::string& key) {
    const auto value = object_lookup(object, key);
    if (!value || !(*value)->is_number()) {
        return std::nullopt;
    }
    const double number = (*value)->number_value;
    if (!std::isfinite(number) || std::trunc(number) != number ||
        number < static_cast<double>((std::numeric_limits<int>::min)()) ||
        number > static_cast<double>((std::numeric_limits<int>::max)())) {
        throw BadRequest("field '" + key + "' must be an integer in range");
    }
    return static_cast<int>(number);
}

std::optional<float> json_float(const JsonValue& object, const std::string& key) {
    const auto value = object_lookup(object, key);
    if (!value || !(*value)->is_number()) {
        return std::nullopt;
    }
    const double number = (*value)->number_value;
    if (!std::isfinite(number) ||
        number < -static_cast<double>((std::numeric_limits<float>::max)()) ||
        number > static_cast<double>((std::numeric_limits<float>::max)())) {
        throw BadRequest("field '" + key + "' must be a finite float in range");
    }
    return static_cast<float>(number);
}

std::vector<std::string> json_string_array(const JsonValue& object, const std::string& key) {
    const auto value = object_lookup(object, key);
    if (!value || !(*value)->is_array()) {
        return {};
    }
    std::vector<std::string> out;
    for (const auto& item : (*value)->array_value) {
        if (!item.is_string()) {
            throw BadRequest("field '" + key + "' must contain only strings");
        }
        out.push_back(item.string_value);
    }
    return out;
}

std::string serialize_http_response(const HttpResponse& response,
                                    bool omit_content_length,
                                    bool allow_cors) {
    std::map<std::string, std::string> headers = response.headers;
    headers.try_emplace("X-Content-Type-Options", "nosniff");
    headers.try_emplace("X-Frame-Options", "DENY");
    headers.try_emplace("Referrer-Policy", "no-referrer");
    if (allow_cors) {
        // Permissive CORS — only use in local dev or trusted LAN.
        headers["Access-Control-Allow-Origin"]  = "*";
        headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS";
        headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization";
        headers["Access-Control-Max-Age"]        = "86400";
        // Override same-origin policies when CORS is active.
        headers["Cross-Origin-Resource-Policy"]  = "cross-origin";
        headers["Cross-Origin-Opener-Policy"]    = "unsafe-none";
    } else {
        headers.try_emplace("Cross-Origin-Resource-Policy", "same-origin");
        headers.try_emplace("Cross-Origin-Opener-Policy", "same-origin");
    }
    std::ostringstream raw;
    raw << "HTTP/1.1 " << response.status_code << ' ' << response.status_text << "\r\n";
    raw << "Content-Type: " << response.content_type << "\r\n";
    // Streaming responses (SSE) must NOT declare a fixed Content-Length: the
    // body is open-ended and terminated by the connection close.  Emitting
    // "Content-Length: 0" before the events made the response an invalid SSE
    // stream (a conforming client stops reading at 0 bytes).
    if (!omit_content_length) {
        raw << "Content-Length: " << response.body.size() << "\r\n";
    }
    raw << "Connection: close\r\n";
    for (const auto& [key, value] : headers) {
        raw << key << ": " << value << "\r\n";
    }
    raw << "\r\n";
    raw << response.body;
    return raw.str();
}

bool send_all(SOCKET socket, const std::string& bytes) {
    size_t sent_total = 0;
    while (sent_total < bytes.size()) {
        const size_t remaining = bytes.size() - sent_total;
        const int chunk = static_cast<int>((std::min)(
            remaining, static_cast<size_t>((std::numeric_limits<int>::max)())));
        const int sent = send(socket, bytes.data() + sent_total, chunk, 0);
        if (sent <= 0) {
            return false;
        }
        sent_total += static_cast<size_t>(sent);
    }
    return true;
}

struct JoinThreadGuard {
    std::thread* thread = nullptr;

    ~JoinThreadGuard() {
        if (thread && thread->joinable()) {
            thread->join();
        }
    }
};

bool recv_append(SOCKET socket, std::array<char, 4096>& buffer, std::string& raw) {
    const int received = recv(socket, buffer.data(), static_cast<int>(buffer.size()), 0);
    if (received <= 0) {
        return false;
    }
    raw.append(buffer.data(), static_cast<size_t>(received));
    return true;
}

bool ensure_available(SOCKET socket,
                      std::array<char, 4096>& buffer,
                      std::string& raw,
                      size_t required_size,
                      size_t max_body_bytes) {
    while (raw.size() < required_size) {
        if (!recv_append(socket, buffer, raw)) {
            return false;
        }
        if (raw.size() > max_body_bytes + 8192) {
            return false;
        }
    }
    return true;
}

bool decode_chunked_body(SOCKET socket,
                         std::array<char, 4096>& buffer,
                         std::string& pending,
                         size_t max_body_bytes,
                         std::string& decoded_body,
                         std::string& error_message) {
    size_t cursor = 0;
    decoded_body.clear();

    while (true) {
        size_t line_end = pending.find("\r\n", cursor);
        while (line_end == std::string::npos) {
            // Bound the buffered bytes while scanning for the chunk-size line's
            // terminator — otherwise a client that never sends "\r\n" grows
            // `pending` without limit (pre-handler memory exhaustion).
            if (pending.size() > max_body_bytes) {
                error_message = "chunked request chunk header exceeded limit";
                return false;
            }
            if (!recv_append(socket, buffer, pending)) {
                error_message = "chunked request ended before chunk header";
                return false;
            }
            line_end = pending.find("\r\n", cursor);
        }

        std::string size_text = trim_copy(pending.substr(cursor, line_end - cursor));
        const auto ext_pos = size_text.find(';');
        if (ext_pos != std::string::npos) {
            size_text.resize(ext_pos);
            size_text = trim_copy(size_text);
        }
        if (size_text.empty()) {
            error_message = "chunked request contained empty chunk size";
            return false;
        }
        if (!std::all_of(size_text.begin(), size_text.end(), [](unsigned char c) {
                return std::isxdigit(c) != 0;
            })) {
            error_message = "chunked request contained invalid chunk size";
            return false;
        }

        size_t chunk_size = 0;
        try {
            size_t parsed_chars = 0;
            const unsigned long long parsed =
                std::stoull(size_text, &parsed_chars, 16);
            if (parsed_chars != size_text.size() ||
                parsed > static_cast<unsigned long long>(
                             (std::numeric_limits<size_t>::max)())) {
                throw std::out_of_range("invalid chunk size");
            }
            chunk_size = static_cast<size_t>(parsed);
        } catch (const std::exception&) {
            error_message = "chunked request contained invalid chunk size";
            return false;
        }

        cursor = line_end + 2;
        if (chunk_size >
                max_body_bytes - (std::min)(decoded_body.size(), max_body_bytes) ||
            chunk_size > (std::numeric_limits<size_t>::max)() - cursor - 2) {
            error_message = "request body exceeded configured limit";
            return false;
        }
        const size_t required_size = cursor + chunk_size + 2;
        if (!ensure_available(socket, buffer, pending, required_size, max_body_bytes)) {
            error_message = "chunked request ended before chunk payload";
            return false;
        }
        if (chunk_size > 0) {
            if (decoded_body.size() + chunk_size > max_body_bytes) {
                error_message = "request body exceeded configured limit";
                return false;
            }
            decoded_body.append(pending.data() + static_cast<std::ptrdiff_t>(cursor), chunk_size);
        }
        cursor += chunk_size;
        if (pending.compare(cursor, 2, "\r\n") != 0) {
            error_message = "chunked request missing chunk terminator";
            return false;
        }
        cursor += 2;

        if (chunk_size == 0) {
            if (cursor == pending.size()) {
                return true;
            }
            while (true) {
                size_t trailer_end = pending.find("\r\n", cursor);
                while (trailer_end == std::string::npos) {
                    // Same bound for trailer lines: cap buffered bytes so an
                    // unterminated / endless trailer section cannot grow
                    // `pending` without limit before the request is handled.
                    if (pending.size() > max_body_bytes) {
                        error_message = "chunked request trailers exceeded limit";
                        return false;
                    }
                    if (!recv_append(socket, buffer, pending)) {
                        error_message = "chunked request ended before trailer terminator";
                        return false;
                    }
                    trailer_end = pending.find("\r\n", cursor);
                }
                if (trailer_end == cursor) {
                    return true;
                }
                cursor = trailer_end + 2;
            }
        }

        if (cursor > 8192) {
            pending.erase(0, cursor);
            cursor = 0;
        }
    }
}

HttpResponse make_json_response(int status_code, const std::string& status_text,
                                const std::string& body) {
    HttpResponse response;
    response.status_code = status_code;
    response.status_text = status_text;
    response.body = body;
    response.headers["Cache-Control"] = "no-store";
    return response;
}

HttpResponse make_error_response(int status_code, const std::string& status_text,
                                 uint64_t request_id, const std::string& code,
                                 const std::string& message) {
    HttpResponse response = make_json_response(status_code, status_text,
                                               json_error_body(request_id, code, message));
    response.headers["X-Request-ID"] = std::to_string(request_id);
    return response;
}

std::string extract_path(std::string target) {
    const auto query_sep = target.find('?');
    if (query_sep != std::string::npos) {
        target.resize(query_sep);
    }
    return target;
}

std::string socket_peer_ip(SOCKET socket) {
    sockaddr_storage address{};
    #ifdef _WIN32
    int length = static_cast<int>(sizeof(address));
    #else
    socklen_t length = sizeof(address);
    #endif
    if (getpeername(socket, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
        return "unknown";
    }

    char buffer[INET6_ADDRSTRLEN] = {};
    if (address.ss_family == AF_INET) {
        const auto* addr4 = reinterpret_cast<const sockaddr_in*>(&address);
        if (inet_ntop(AF_INET, &addr4->sin_addr, buffer, sizeof(buffer)) != nullptr) {
            return buffer;
        }
    } else if (address.ss_family == AF_INET6) {
        const auto* addr6 = reinterpret_cast<const sockaddr_in6*>(&address);
        if (inet_ntop(AF_INET6, &addr6->sin6_addr, buffer, sizeof(buffer)) != nullptr) {
            return buffer;
        }
    }

    return "unknown";
}

bool is_trusted_proxy_peer(SOCKET socket, const HttpApiServerConfig& config) {
    if (!config.trust_proxy_headers) return false;
    const std::string peer = socket_peer_ip(socket);
    return std::find(config.trusted_proxy_ips.begin(), config.trusted_proxy_ips.end(), peer) !=
           config.trusted_proxy_ips.end();
}

std::string request_client_identity(const HttpRequest& request,
                                    SOCKET client_socket,
                                    const HttpApiServerConfig& config) {
    if (is_trusted_proxy_peer(client_socket, config)) {
        const auto forwarded = request.headers.find("x-forwarded-for");
        if (forwarded != request.headers.end()) {
            std::string identity = forwarded->second;
            const auto comma = identity.find(',');
            if (comma != std::string::npos) {
                identity.resize(comma);
            }
            identity = trim_copy(identity);
            if (!identity.empty()) {
                return identity;
            }
        }
    }
    return socket_peer_ip(client_socket);
}

bool request_declares_https(const HttpRequest& request,
                            SOCKET client_socket,
                            const HttpApiServerConfig& config) {
    if (!config.require_tls_proxy_header) {
        return true;
    }
    // A local container readiness probe has no TLS proxy hop. Only these two
    // GET endpoints may bypass transport checks; normal auth still applies.
    if (request.method == "GET" &&
        (request.path == "/health" || request.path == "/ready")) {
        const std::string peer = socket_peer_ip(client_socket);
        if (peer == "127.0.0.1" || peer == "::1" || peer == "::ffff:127.0.0.1") {
            return true;
        }
    }
    if (!is_trusted_proxy_peer(client_socket, config)) {
        return false;
    }
    const auto forwarded = request.headers.find("x-forwarded-proto");
    if (forwarded == request.headers.end()) {
        return false;
    }
    return to_lower_copy(trim_copy(forwarded->second)) == "https";
}

bool is_loopback_host(const std::string& host) {
    return host.empty() || host == "localhost" || host == "localhost." ||
           host == "127.0.0.1" || host == "::1" || host == "0:0:0:0:0:0:0:1";
}

bool server_auth_configuration_is_valid(const HttpApiServerConfig& config) {
    const bool loopback = is_loopback_host(config.host);
    if (!loopback && config.auth_token.empty()) {
        return false;
    }
    if (!loopback && config.allow_unauthenticated_local) {
        return false;
    }
    if (config.auth_token.empty() && !(loopback && config.allow_unauthenticated_local)) {
        return false;
    }
    if (config.require_tls_proxy_header && !config.trust_proxy_headers) {
        return false;
    }
    if (config.trust_proxy_headers && config.trusted_proxy_ips.empty()) {
        return false;
    }
    for (const auto& peer : config.trusted_proxy_ips) {
        if (peer.empty() || peer.size() > INET6_ADDRSTRLEN ||
            peer.find_first_of(" \t\r\n,") != std::string::npos) {
            return false;
        }
    }
    if (!loopback && (!config.require_tls_proxy_header || !config.trust_proxy_headers ||
                      config.auth_token.size() < 16)) {
        return false;
    }
    if (config.port < 0 || config.port > 65535 || config.worker_threads < 0 ||
        config.worker_threads > 256 || config.inference_replicas < 0 ||
        config.inference_replicas > 64 || config.max_header_bytes == 0 ||
        config.max_header_bytes > 1024 * 1024 || config.max_body_bytes == 0 ||
        config.max_body_bytes > 64 * 1024 * 1024 || config.max_queue_depth == 0 ||
        config.max_queue_depth > 65536 || config.socket_timeout_ms <= 0 ||
        config.rate_limit_requests_per_minute > 1000000 ||
        config.max_rate_limit_clients == 0 || config.max_rate_limit_clients > 1000000 ||
        config.max_stream_pending_bytes == 0 ||
        config.max_stream_pending_bytes > 64 * 1024 * 1024 ||
        config.max_json_depth <= 0 || config.max_json_depth > 128 ||
        config.max_generate_tokens < 0 || config.max_generate_tokens > (1 << 20) ||
        config.max_context_tokens_per_request <= 0 ||
        config.max_context_tokens_per_request > (1 << 24) ||
        config.max_prompts_per_batch == 0 || config.max_prompts_per_batch > 1024 ||
        config.max_train_steps <= 0 || config.max_train_steps > 1000000 ||
        config.max_train_epochs <= 0 || config.max_train_epochs > 10000 ||
        config.max_train_batch_size <= 0 || config.max_train_batch_size > 65536 ||
        config.max_train_seq_len < 2 || config.max_train_seq_len > (1 << 20) ||
        config.max_train_corpus_steps <= 0 || config.max_train_corpus_steps > 10000000 ||
        config.max_train_text_bytes == 0 ||
        config.max_train_text_bytes > config.max_body_bytes ||
        (config.enable_admin_endpoints && config.auth_token.empty()) ||
        config.auth_token.find_first_of("\r\n\0", 0, 3) != std::string::npos) {
        return false;
    }
    return true;
}

bool constant_time_equals(const std::string& lhs, const std::string& rhs) {
    size_t diff = lhs.size() ^ rhs.size();
    const size_t extent = (std::max)(lhs.size(), rhs.size());
    for (size_t i = 0; i < extent; ++i) {
        const unsigned char left =
            i < lhs.size() ? static_cast<unsigned char>(lhs[i]) : 0;
        const unsigned char right =
            i < rhs.size() ? static_cast<unsigned char>(rhs[i]) : 0;
        diff |= static_cast<size_t>(left ^ right);
    }
    return diff == 0;
}

bool expects_json_body(const HttpRequest& request) {
    return request.method == "POST" || request.method == "PUT" || request.method == "PATCH";
}

int bounded_int(const JsonValue& payload,
                const std::string& key,
                int fallback,
                int minimum,
                int maximum) {
    int value = json_int(payload, key).value_or(fallback);
    if (value < minimum || value > maximum) {
        throw BadRequest("field '" + key + "' must be between " +
                                 std::to_string(minimum) + " and " +
                                 std::to_string(maximum));
    }
    return value;
}

void validate_generation_options(GenerationOptions& options,
                                 const HttpApiServerConfig& config,
                                 int vocab_size) {
    if (options.max_tokens < 0 || options.max_tokens > config.max_generate_tokens) {
        throw BadRequest("field 'max_tokens' must be between 0 and " +
                                 std::to_string(config.max_generate_tokens));
    }
    if (options.max_context_tokens <= 0) {
        options.max_context_tokens = config.max_context_tokens_per_request;
    }
    if (options.max_context_tokens > config.max_context_tokens_per_request) {
        throw BadRequest("field 'max_context_tokens' exceeds configured limit " +
                                 std::to_string(config.max_context_tokens_per_request));
    }
    if (!std::isfinite(options.temperature) || options.temperature < 0.0f ||
        options.temperature > 5.0f) {
        throw BadRequest("field 'temperature' must be finite and between 0 and 5");
    }
    if (!std::isfinite(options.top_p) || options.top_p <= 0.0f || options.top_p > 1.0f) {
        throw BadRequest("field 'top_p' must be finite and between 0 and 1");
    }
    if (options.top_k < 0 || options.top_k > vocab_size) {
        throw BadRequest("field 'top_k' must be between 0 and the model vocabulary size");
    }
    if (options.eos_token_id < 0 || options.eos_token_id >= vocab_size) {
        throw BadRequest("field 'eos_token_id' is outside the model vocabulary");
    }
}

HttpResponse admin_disabled_response(uint64_t request_id) {
    return make_error_response(403, "Forbidden", request_id, "admin_endpoint_disabled",
                               "this endpoint is disabled; start the server with "
                               "--enable-admin-endpoints to use it");
}

std::filesystem::path resolve_pack_output_directory(const HttpApiServerConfig& config,
                                                    const std::string& requested) {
    namespace fs = std::filesystem;
    fs::path requested_path(requested);
    if (requested_path.empty()) {
        throw BadRequest("field 'directory' must not be empty");
    }
    if (requested_path.is_absolute()) {
        if (!config.allow_pack_absolute_paths) {
            throw BadRequest("absolute pack directories are disabled for the HTTP API");
        }
        return requested_path.lexically_normal();
    }
    for (const auto& part : requested_path) {
        if (part == "..") {
            throw BadRequest("pack directory must not contain '..'");
        }
    }
    fs::path root(config.pack_output_root.empty() ? "artifacts/model_packs"
                                                  : config.pack_output_root);
    std::error_code ec;
    const fs::path canonical_root = fs::weakly_canonical(root, ec);
    if (ec) throw BadRequest("pack output root could not be canonicalized");
    const fs::path target = fs::weakly_canonical(root / requested_path, ec);
    if (ec) throw BadRequest("pack output directory could not be canonicalized");
    const fs::path relative = target.lexically_relative(canonical_root);
    if (relative.is_absolute()) {
        throw BadRequest("pack output directory resolves outside configured root");
    }
    for (const auto& part : relative) {
        if (part == "..") {
            throw BadRequest("pack output directory resolves outside configured root");
        }
    }
    return target;
}

void configure_socket_timeouts(SOCKET socket, int timeout_ms) {
    if (timeout_ms <= 0) {
        return;
    }
#ifdef _WIN32
    DWORD timeout = static_cast<DWORD>(timeout_ms);
    setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout),
               sizeof(timeout));
    setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&timeout),
               sizeof(timeout));
#else
    timeval timeout{};
    timeout.tv_sec = timeout_ms / 1000;
    timeout.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
#endif
}

} // namespace

namespace {

RequestReadResult read_http_request(SOCKET socket, const HttpApiServerConfig& config,
                                    uint64_t request_id) {
    RequestReadResult result;
    std::string raw;
    std::array<char, 4096> buffer{};
    size_t header_end = std::string::npos;

    while ((header_end = raw.find("\r\n\r\n")) == std::string::npos) {
        const int received = recv(socket, buffer.data(), static_cast<int>(buffer.size()), 0);
        if (received <= 0) {
            result.error = make_error_response(400, "Bad Request", request_id, "invalid_request",
                                               "failed to read request");
            return result;
        }
        raw.append(buffer.data(), static_cast<size_t>(received));
        if (raw.size() > config.max_header_bytes) {
            result.error = make_error_response(431, "Request Header Fields Too Large", request_id,
                                               "headers_too_large",
                                               "request headers exceeded configured limit");
            return result;
        }
    }

    std::istringstream stream(raw.substr(0, header_end));
    std::string request_line;
    if (!std::getline(stream, request_line)) {
        result.error = make_error_response(400, "Bad Request", request_id, "invalid_request_line",
                                           "missing request line");
        return result;
    }
    if (!request_line.empty() && request_line.back() == '\r') {
        request_line.pop_back();
    }

    std::istringstream request_line_stream(request_line);
    request_line_stream >> result.request.method >> result.request.raw_target >> result.request.version;
    std::string unexpected_request_line_field;
    request_line_stream >> unexpected_request_line_field;
    if (result.request.method.empty() || result.request.raw_target.empty() ||
        result.request.version.empty() || !unexpected_request_line_field.empty() ||
        (result.request.version != "HTTP/1.1" && result.request.version != "HTTP/1.0") ||
        result.request.raw_target.empty() || result.request.raw_target.front() != '/') {
        result.error = make_error_response(400, "Bad Request", request_id, "invalid_request_line",
                                           "malformed request line");
        return result;
    }

    std::transform(result.request.method.begin(), result.request.method.end(),
                   result.request.method.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    result.request.path = extract_path(result.request.raw_target);

    size_t header_count = 0;
    std::string header_line;
    while (std::getline(stream, header_line)) {
        if (!header_line.empty() && header_line.back() == '\r') {
            header_line.pop_back();
        }
        if (header_line.empty()) {
            continue;
        }
        if (++header_count > 64) {
            result.error = make_error_response(431, "Request Header Fields Too Large", request_id,
                                               "too_many_headers",
                                               "request included too many headers");
            return result;
        }
        const auto sep = header_line.find(':');
        if (sep == std::string::npos) {
            result.error = make_error_response(400, "Bad Request", request_id, "invalid_header",
                                               "malformed header line");
            return result;
        }
        const std::string key = to_lower_copy(trim_copy(header_line.substr(0, sep)));
        const std::string value = trim_copy(header_line.substr(sep + 1));
        if (key.empty() ||
            !std::all_of(key.begin(), key.end(), [](unsigned char c) {
                return is_http_token_char(c);
            }) ||
            !is_valid_header_value(value)) {
            result.error = make_error_response(400, "Bad Request", request_id, "invalid_header",
                                               "header name is invalid");
            return result;
        }
        const auto [unused, inserted] = result.request.headers.emplace(key, value);
        if (!inserted) {
            result.error = make_error_response(400, "Bad Request", request_id,
                                               "duplicate_header",
                                               "duplicate request headers are not accepted");
            return result;
        }
    }

    bool transfer_chunked = false;
    const auto transfer = result.request.headers.find("transfer-encoding");
    if (transfer != result.request.headers.end()) {
        const std::string encoding = to_lower_copy(trim_copy(transfer->second));
        if (encoding != "chunked") {
            result.error = make_error_response(400, "Bad Request", request_id,
                                               "unsupported_transfer_encoding",
                                               "only transfer-encoding: chunked is supported");
            return result;
        }
        if (result.request.headers.count("content-length") != 0) {
            result.error = make_error_response(400, "Bad Request", request_id,
                                               "ambiguous_body_length",
                                               "content-length and transfer-encoding cannot coexist");
            return result;
        }
        transfer_chunked = true;
    }

    size_t content_length = 0;
    if (!transfer_chunked) {
        const auto content_length_it = result.request.headers.find("content-length");
        if (content_length_it != result.request.headers.end()) {
            try {
                if (content_length_it->second.empty() ||
                    !std::all_of(content_length_it->second.begin(),
                                 content_length_it->second.end(),
                                 [](unsigned char c) { return std::isdigit(c) != 0; })) {
                    throw std::invalid_argument("invalid content length");
                }
                size_t parsed_chars = 0;
                const unsigned long long parsed =
                    std::stoull(content_length_it->second, &parsed_chars, 10);
                if (parsed_chars != content_length_it->second.size() ||
                    parsed > static_cast<unsigned long long>(
                                 (std::numeric_limits<size_t>::max)())) {
                    throw std::out_of_range("invalid content length");
                }
                content_length = static_cast<size_t>(parsed);
            } catch (const std::exception&) {
                result.error = make_error_response(400, "Bad Request", request_id,
                                                   "invalid_content_length",
                                                   "content-length header is invalid");
                return result;
            }
        } else if (expects_json_body(result.request)) {
            result.error = make_error_response(411, "Length Required", request_id,
                                               "length_required",
                                               "request body requires content-length or chunked encoding");
            return result;
        }
    }

    if (!transfer_chunked && content_length > config.max_body_bytes) {
        result.error = make_error_response(413, "Payload Too Large", request_id, "body_too_large",
                                           "request body exceeded configured limit");
        return result;
    }

    result.request.body = raw.substr(header_end + 4);
    if (transfer_chunked) {
        std::string decoded_body;
        std::string chunk_error;
        if (!decode_chunked_body(socket, buffer, result.request.body, config.max_body_bytes,
                                 decoded_body, chunk_error)) {
            result.error = make_error_response(400, "Bad Request", request_id,
                                               "invalid_chunked_body", chunk_error);
            return result;
        }
        result.request.body = std::move(decoded_body);
    } else {
        while (result.request.body.size() < content_length) {
            const int received = recv(socket, buffer.data(), static_cast<int>(buffer.size()), 0);
            if (received <= 0) {
                result.error = make_error_response(400, "Bad Request", request_id, "incomplete_body",
                                                   "request body ended early");
                return result;
            }
            result.request.body.append(buffer.data(), static_cast<size_t>(received));
            if (result.request.body.size() > config.max_body_bytes) {
                result.error = make_error_response(413, "Payload Too Large", request_id,
                                                   "body_too_large",
                                                   "request body exceeded configured limit");
                return result;
            }
        }
        if (result.request.body.size() > content_length) {
            result.request.body.resize(content_length);
        }
    }

    result.success = true;
    return result;
}

} // namespace

namespace {

bool request_requires_auth(const HttpApiServerConfig& config, const HttpRequest& request) {
    if (config.allow_unauthenticated_health &&
        (request.path == "/health" || request.path == "/ready")) {
        return false;
    }
    if (config.auth_token.empty()) {
        return !(config.allow_unauthenticated_local && is_loopback_host(config.host));
    }
    return true;
}

bool is_authorized(const HttpApiServerConfig& config, const HttpRequest& request) {
    if (!request_requires_auth(config, request)) {
        return true;
    }

    const auto auth_it = request.headers.find("authorization");
    if (auth_it != request.headers.end()) {
        const std::string prefix = "Bearer ";
        if (auth_it->second.rfind(prefix, 0) == 0) {
            return constant_time_equals(auth_it->second.substr(prefix.size()), config.auth_token);
        }
    }

    const auto api_key_it = request.headers.find("x-api-key");
    if (api_key_it != request.headers.end()) {
        return constant_time_equals(api_key_it->second, config.auth_token);
    }

    return false;
}

JsonValue parse_json_body_or_throw(const HttpRequest& request, const HttpApiServerConfig& config) {
    const auto content_type = request.headers.find("content-type");
    if (content_type != request.headers.end()) {
        const std::string lowered = to_lower_copy(content_type->second);
        if (lowered.find("application/json") == std::string::npos) {
            throw BadRequest("content-type must be application/json");
        }
    }
    if (request.body.empty()) {
        throw BadRequest("missing JSON body");
    }
    return JsonParser(request.body, config.max_json_depth).parse();
}

std::string server_metrics_json(const HttpApiServer& server, uint64_t uptime_ms,
                                uint64_t total_requests, uint64_t auth_failures,
                                uint64_t parse_failures, uint64_t rejected_requests,
                                uint64_t rate_limited_requests,
                                uint64_t internal_errors, uint64_t active_requests,
                                size_t queue_depth, const GenerationMetrics& generation_metrics) {
    return join_json_fields({
        "\"api\":\"" + std::string(kApiVersion) + "\"",
        "\"uptime_ms\":" + std::to_string(uptime_ms),
        "\"requests_total\":" + std::to_string(total_requests),
        "\"auth_failures\":" + std::to_string(auth_failures),
        "\"parse_failures\":" + std::to_string(parse_failures),
        "\"rejected_requests\":" + std::to_string(rejected_requests),
        "\"rate_limited_requests\":" + std::to_string(rate_limited_requests),
        "\"internal_errors\":" + std::to_string(internal_errors),
        "\"active_requests\":" + std::to_string(active_requests),
        "\"queue_depth\":" + std::to_string(queue_depth),
        "\"server_port\":" + std::to_string(server.port()),
        "\"configured_inference_replicas\":" +
            std::to_string((std::max)(server.config().inference_replicas, 0)),
        "\"generation\":" + generation_metrics_json(generation_metrics),
    });
}

} // namespace

HttpApiServer::HttpApiServer(InferenceEngine& engine, HttpApiServerConfig config)
    : engine_(engine), config_(std::move(config)), server_socket_(kInvalidSocket) {}

HttpApiServer::~HttpApiServer() {
    stop();
    cleanup_sockets();
}

bool HttpApiServer::initialize_sockets() {
    if (sockets_ready_) {
        return true;
    }
#ifdef _WIN32
    WSADATA wsa_data{};
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
        return false;
    }
#endif
    sockets_ready_ = true;
    return true;
}

void HttpApiServer::cleanup_sockets() {
    if (!sockets_ready_) {
        return;
    }
#ifdef _WIN32
    WSACleanup();
#endif
    sockets_ready_ = false;
}

void HttpApiServer::close_socket(SOCKET socket) {
    if (socket == kInvalidSocket) {
        return;
    }
#ifdef _WIN32
    closesocket(socket);
#else
    close(socket);
#endif
}

size_t HttpApiServer::desired_inference_replica_count() const {
    if (config_.inference_replicas > 0) {
        return static_cast<size_t>(config_.inference_replicas);
    }
    // Replicating a complete model on one accelerator multiplies VRAM without
    // creating independent GPU hardware. Operators can still opt in explicitly
    // for multi-GPU/process-specific deployments.
    if (engine_.config.use_cuda) {
        return 1;
    }
    const unsigned int hardware = std::thread::hardware_concurrency();
    const int desired_workers =
        config_.worker_threads > 0 ? config_.worker_threads
                                   : static_cast<int>(hardware > 0 ? hardware : 4);
    return static_cast<size_t>((std::max)(1, (std::min)(desired_workers, 4)));
}

void HttpApiServer::sync_inference_replicas_locked() {
    std::vector<std::unique_ptr<InferenceEngine>> next_replicas;
    const size_t replica_count = desired_inference_replica_count();
    next_replicas.reserve(replica_count);
    for (size_t index = 0; index < replica_count; ++index) {
        next_replicas.push_back(engine_.clone_for_inference());
    }

    {
        std::lock_guard<std::mutex> lock(replica_mutex_);
        inference_replicas_ = std::move(next_replicas);
        idle_replicas_.clear();
        idle_replicas_.reserve(inference_replicas_.size());
        for (size_t index = 0; index < inference_replicas_.size(); ++index) {
            idle_replicas_.push_back(index);
        }
    }
    update_latest_generation_metrics(engine_.last_generation_metrics());
    replica_cv_.notify_all();
}

void HttpApiServer::initialize_inference_replicas() {
    std::unique_lock<std::shared_mutex> state_lock(model_state_mutex_);
    sync_inference_replicas_locked();
}

InferenceEngine* HttpApiServer::acquire_inference_replica(size_t& replica_index) {
    std::unique_lock<std::mutex> lock(replica_mutex_);
    replica_cv_.wait(lock, [&]() { return stop_requested_.load() || !idle_replicas_.empty(); });
    if (stop_requested_.load() && idle_replicas_.empty()) {
        replica_index = kInvalidReplicaIndex;
        return nullptr;
    }
    replica_index = idle_replicas_.back();
    idle_replicas_.pop_back();
    return inference_replicas_[replica_index].get();
}

void HttpApiServer::release_inference_replica(size_t replica_index) {
    if (replica_index == kInvalidReplicaIndex) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(replica_mutex_);
        idle_replicas_.push_back(replica_index);
    }
    replica_cv_.notify_one();
}

GenerationMetrics HttpApiServer::latest_generation_metrics_snapshot() const {
    std::lock_guard<std::mutex> lock(latest_metrics_mutex_);
    return latest_generation_metrics_;
}

void HttpApiServer::update_latest_generation_metrics(const GenerationMetrics& metrics) {
    std::lock_guard<std::mutex> lock(latest_metrics_mutex_);
    latest_generation_metrics_ = metrics;
}

void HttpApiServer::start_workers() {
    if (workers_started_.exchange(true)) {
        return;
    }

    const unsigned int hardware = std::thread::hardware_concurrency();
    const int desired = config_.worker_threads > 0
                            ? config_.worker_threads
                            : static_cast<int>(hardware > 0 ? hardware : 4);
    const int worker_count = (std::max)(desired, 1);
    workers_.reserve(static_cast<size_t>(worker_count));
    for (int i = 0; i < worker_count; ++i) {
        workers_.emplace_back([this]() { worker_loop(); });
    }
}

bool HttpApiServer::start() {
    if (server_socket_ != kInvalidSocket) {
        return true;
    }
    if (!server_auth_configuration_is_valid(config_)) {
        return false;
    }
    if (!initialize_sockets()) {
        return false;
    }

    server_socket_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (server_socket_ == kInvalidSocket) {
        return false;
    }

    int reuse = 1;
    setsockopt(server_socket_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&reuse),
               sizeof(reuse));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(config_.port));
    const std::string bind_host =
        config_.host == "localhost" || config_.host == "localhost." ||
                config_.host == "::1" || config_.host == "0:0:0:0:0:0:0:1"
            ? "127.0.0.1"
            : config_.host;
    if (bind_host == "0.0.0.0") {
        address.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (bind_host.empty()) {
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    } else if (inet_pton(AF_INET, bind_host.c_str(), &address.sin_addr) != 1) {
        close_socket(server_socket_);
        server_socket_ = kInvalidSocket;
        return false;
    }

    if (bind(server_socket_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        close_socket(server_socket_);
        server_socket_ = kInvalidSocket;
        return false;
    }

    if (listen(server_socket_, static_cast<int>((std::max)(config_.max_queue_depth, size_t{16}))) !=
        0) {
        close_socket(server_socket_);
        server_socket_ = kInvalidSocket;
        return false;
    }

    sockaddr_in bound{};
    socklen_t bound_len = sizeof(bound);
    if (getsockname(server_socket_, reinterpret_cast<sockaddr*>(&bound), &bound_len) == 0) {
        bound_port_ = ntohs(bound.sin_port);
    } else {
        bound_port_ = config_.port;
    }

    started_at_ = std::chrono::steady_clock::now();
    stop_requested_.store(false);
    // stop() sets this flag to interrupt an in-flight administrative job.
    // A successful restart begins a new server lifecycle and must not inherit
    // that old cooperative-cancellation request.
    engine_.clear_training_cancellation();
    try {
        initialize_inference_replicas();
    } catch (...) {
        close_socket(server_socket_);
        server_socket_ = kInvalidSocket;
        throw;
    }
    start_workers();
    return true;
}

void HttpApiServer::serve_forever() {
    const auto build_http_response = [&](const HttpResponse& response,
                                         bool omit_content_length = false) {
        return serialize_http_response(response, omit_content_length, config_.allow_cors);
    };
    if (!start()) {
        throw std::runtime_error("failed to start HTTP API server");
    }

    while (!stop_requested_.load()) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        SOCKET client_socket =
            accept(server_socket_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_socket == kInvalidSocket) {
            if (!stop_requested_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            continue;
        }
        configure_socket_timeouts(client_socket, config_.socket_timeout_ms);

        bool queued = false;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (client_queue_.size() < config_.max_queue_depth) {
                client_queue_.push_back(client_socket);
                queued = true;
            }
        }

        if (!queued) {
            rejected_requests_.fetch_add(1);
            const uint64_t request_id = ++request_counter_;
            const HttpResponse response = make_error_response(
                503, "Service Unavailable", request_id, "queue_full",
                "server request queue is full; retry later");
            send_all(client_socket, build_http_response(response));
            close_socket(client_socket);
            continue;
        }

        queue_cv_.notify_one();
    }
}

void HttpApiServer::stop() {
    const bool was_stopping = stop_requested_.exchange(true);
    engine_.request_training_cancellation();
    {
        std::lock_guard<std::mutex> lock(
            active_training_mutex_);
        for (InferenceEngine* training_engine :
             active_training_engines_) {
            if (training_engine) {
                training_engine->
                    request_training_cancellation();
            }
        }
    }
    if (!was_stopping && server_socket_ != kInvalidSocket) {
        close_socket(server_socket_);
        server_socket_ = kInvalidSocket;
    }

    queue_cv_.notify_all();
    replica_cv_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
    workers_started_.store(false);
    {
        std::lock_guard<std::mutex> lock(replica_mutex_);
        idle_replicas_.clear();
        inference_replicas_.clear();
    }
    replica_cv_.notify_all();

    std::deque<SOCKET> pending;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        pending.swap(client_queue_);
    }
    for (SOCKET socket : pending) {
        close_socket(socket);
    }
}

void HttpApiServer::worker_loop() {
    while (true) {
        SOCKET client_socket = kInvalidSocket;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [&]() { return stop_requested_.load() || !client_queue_.empty(); });
            if (stop_requested_.load() && client_queue_.empty()) {
                return;
            }
            client_socket = client_queue_.front();
            client_queue_.pop_front();
        }

        active_requests_.fetch_add(1);
        handle_client(client_socket);
        close_socket(client_socket);
        active_requests_.fetch_sub(1);
    }
}

void HttpApiServer::handle_client(SOCKET client_socket) {
    const auto build_http_response = [&](const HttpResponse& response,
                                         bool omit_content_length = false) {
        return serialize_http_response(response, omit_content_length, config_.allow_cors);
    };
    const uint64_t request_id = ++request_counter_;
    const RequestReadResult read_result = read_http_request(client_socket, config_, request_id);
    if (!read_result.success) {
        parse_failures_.fetch_add(1);
        send_all(client_socket, build_http_response(read_result.error));
        return;
    }

    total_requests_.fetch_add(1);
    const HttpRequest& request = read_result.request;

    // Handle CORS preflight: browsers send OPTIONS before cross-origin POST.
    // Respond 204 with CORS headers (build_http_response will inject them
    // when config_.allow_cors is true, so a 204 body-less response suffices).
    if (request.method == "OPTIONS") {
        HttpResponse response;
        response.status_code = 204;
        response.status_text = "No Content";
        response.body = "";
        response.headers["X-Request-ID"] = std::to_string(request_id);
        send_all(client_socket, build_http_response(response));
        return;
    }

    if (request.method != "GET" && request.method != "POST") {
        HttpResponse response = make_error_response(405, "Method Not Allowed", request_id,
                                                    "method_not_allowed",
                                                    "only GET, POST, and OPTIONS are supported");
        response.headers["Allow"] = "GET, POST, OPTIONS";
        send_all(client_socket, build_http_response(response));
        return;
    }

    if (!request_declares_https(request, client_socket, config_)) {
        rejected_requests_.fetch_add(1);
        HttpResponse response = make_error_response(
            426, "Upgrade Required", request_id, "https_required",
            "request must arrive through an HTTPS-terminating proxy and send x-forwarded-proto=https");
        response.headers["Upgrade"] = "TLS/1.3, HTTPS/1.1";
        send_all(client_socket, build_http_response(response));
        return;
    }

    const bool exempt_from_rate_limit =
        request.path == "/health" || request.path == "/ready";
    if (!exempt_from_rate_limit && config_.rate_limit_requests_per_minute > 0) {
        const std::string client_identity = request_client_identity(request, client_socket, config_);
        const auto now = std::chrono::steady_clock::now();
        const auto window = std::chrono::minutes(1);
        bool allowed = false;
        size_t remaining = 0;
        {
            std::lock_guard<std::mutex> lock(rate_limit_mutex_);
            // Bound the rate-limit map: periodically (and whenever it grows
            // large) sweep out every client whose window has fully expired.
            // Without this, an endless stream of distinct identities (e.g. a
            // spoofed X-Forwarded-For per request) leaks one map entry each,
            // forever — a memory-exhaustion DoS.  Runs under rate_limit_mutex_.
            if (((++rate_limit_sweep_counter_) & 0x3FFu) == 0 ||
                recent_requests_by_client_.size() >= config_.max_rate_limit_clients) {
                for (auto it = recent_requests_by_client_.begin();
                     it != recent_requests_by_client_.end();) {
                    auto& dq = it->second;
                    while (!dq.empty() && now - dq.front() > window) {
                        dq.pop_front();
                    }
                    if (dq.empty()) {
                        it = recent_requests_by_client_.erase(it);
                    } else {
                        ++it;
                    }
                }
            }

            auto client_it = recent_requests_by_client_.find(client_identity);
            if (client_it == recent_requests_by_client_.end()) {
                // Never use operator[] before enforcing the cardinality cap:
                // an attacker controlling X-Forwarded-For could otherwise
                // grow this map even when every existing identity is active.
                if (recent_requests_by_client_.size() < config_.max_rate_limit_clients) {
                    client_it = recent_requests_by_client_
                                    .emplace(client_identity,
                                             std::deque<std::chrono::steady_clock::time_point>{})
                                    .first;
                }
            }
            if (client_it != recent_requests_by_client_.end()) {
                auto& timestamps = client_it->second;
                while (!timestamps.empty() && now - timestamps.front() > window) {
                    timestamps.pop_front();
                }
                if (timestamps.size() < config_.rate_limit_requests_per_minute) {
                    timestamps.push_back(now);
                    allowed = true;
                    remaining = config_.rate_limit_requests_per_minute - timestamps.size();
                }
            }
        }
        if (!allowed) {
            rate_limited_requests_.fetch_add(1);
            HttpResponse response = make_error_response(
                429, "Too Many Requests", request_id, "rate_limited",
                "client exceeded the configured request budget; retry later");
            response.headers["Retry-After"] = "60";
            response.headers["X-RateLimit-Limit"] =
                std::to_string(config_.rate_limit_requests_per_minute);
            response.headers["X-RateLimit-Remaining"] = "0";
            send_all(client_socket, build_http_response(response));
            return;
        }
        // Expose coarse budget to callers even when accepted.
        (void)remaining;
    }

    if (!is_authorized(config_, request)) {
        auth_failures_.fetch_add(1);
        HttpResponse response = make_error_response(401, "Unauthorized", request_id,
                                                    "unauthorized",
                                                    "missing or invalid authentication token");
        response.headers["WWW-Authenticate"] = "Bearer realm=\"nsos\"";
        send_all(client_socket, build_http_response(response));
        return;
    }

    try {
        auto with_read_engine = [&](const auto& fn) {
            std::shared_lock<std::shared_mutex> state_lock(model_state_mutex_);
            fn(engine_);
        };
        auto with_transactional_training_engine = [&](const auto& fn) {
            std::unique_lock<std::shared_mutex> state_lock(model_state_mutex_);
            // Administrative training is opt-in and prioritizes correctness:
            // mutate a deep model+optimizer clone and publish it only after the
            // full operation succeeds with finite outputs. Any exception leaves
            // the serving model and its replicas byte-for-byte untouched.
            std::unique_ptr<InferenceEngine> staged = engine_.clone_for_training();
            {
                std::lock_guard<std::mutex> lock(
                    active_training_mutex_);
                if (stop_requested_.load()) {
                    throw AbortException();
                }
                active_training_engines_.insert(staged.get());
            }
            try {
                fn(*staged);
                std::vector<std::unique_ptr<InferenceEngine>>
                    next_replicas;
                const size_t replica_count =
                    desired_inference_replica_count();
                next_replicas.reserve(replica_count);
                for (size_t index = 0;
                     index < replica_count; ++index) {
                    if (stop_requested_.load()) {
                        throw AbortException();
                    }
                    next_replicas.push_back(
                        staged->clone_for_inference());
                }
                std::vector<size_t> next_idle_replicas(
                    replica_count);
                std::iota(
                    next_idle_replicas.begin(),
                    next_idle_replicas.end(), size_t{0});
                GenerationMetrics next_metrics =
                    staged->last_generation_metrics();
                static_assert(
                    std::is_nothrow_move_assignable_v<
                        InferenceEngine>,
                    "Transactional HTTP publication requires noexcept "
                    "InferenceEngine move assignment");
                static_assert(
                    std::is_nothrow_move_assignable_v<
                        GenerationMetrics>,
                    "Transactional HTTP publication requires noexcept "
                    "GenerationMetrics move assignment");
                {
                    std::lock_guard<std::mutex> lock(
                        active_training_mutex_);
                    if (stop_requested_.load()) {
                        throw AbortException();
                    }
                    engine_ = std::move(*staged);
                    {
                        std::lock_guard<std::mutex> replica_lock(
                            replica_mutex_);
                        inference_replicas_ =
                            std::move(next_replicas);
                        idle_replicas_ =
                            std::move(next_idle_replicas);
                    }
                    {
                        std::lock_guard<std::mutex> metrics_lock(
                            latest_metrics_mutex_);
                        latest_generation_metrics_ =
                            std::move(next_metrics);
                    }
                    active_training_engines_.erase(
                        staged.get());
                }
                replica_cv_.notify_all();
            } catch (...) {
                std::lock_guard<std::mutex> lock(
                    active_training_mutex_);
                active_training_engines_.erase(staged.get());
                throw;
            }
        };
        auto with_inference_replica = [&](const auto& fn) {
            std::shared_lock<std::shared_mutex> state_lock(model_state_mutex_);
            size_t replica_index = kInvalidReplicaIndex;
            InferenceEngine* replica = acquire_inference_replica(replica_index);
            if (!replica) {
                throw std::runtime_error("inference replica pool is unavailable");
            }
            try {
                fn(*replica);
            } catch (...) {
                release_inference_replica(replica_index);
                throw;
            }
            release_inference_replica(replica_index);
        };

        if (request.method == "GET" &&
            (request.path == "/health" || request.path == "/ready")) {
            const auto now = std::chrono::steady_clock::now();
            const uint64_t uptime_ms = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(now - started_at_).count());
            const size_t queue_depth = [&]() {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                return client_queue_.size();
            }();

            bool ready = true;
            if (request.path == "/ready") {
                std::shared_lock<std::shared_mutex> state_lock(model_state_mutex_);
                std::lock_guard<std::mutex> replica_lock(replica_mutex_);
                ready = workers_started_.load() && !stop_requested_.load() &&
                        engine_.model != nullptr && !inference_replicas_.empty() &&
                        inference_replicas_.size() == desired_inference_replica_count();
            }
            const std::string status = request.path == "/health"
                                           ? "alive"
                                           : (ready ? "ready" : "not_ready");
            const std::string body = join_json_fields({
                "\"ok\":" + json_bool(ready),
                "\"request_id\":" + std::to_string(request_id),
                "\"status\":\"" + status + "\"",
                "\"api\":\"" + std::string(kApiVersion) + "\"",
                "\"uptime_ms\":" + std::to_string(uptime_ms),
                "\"active_requests\":" + std::to_string(active_requests_.load()),
                "\"queue_depth\":" + std::to_string(queue_depth),
            });
            HttpResponse response = make_json_response(
                ready ? 200 : 503, ready ? "OK" : "Service Unavailable", body);
            response.headers["X-Request-ID"] = std::to_string(request_id);
            send_all(client_socket, build_http_response(response));
            return;
        }

        if (request.method == "GET" && (request.path == "/" || request.path == "/info")) {
            GenerationMetrics metrics = latest_generation_metrics_snapshot();
            size_t memory_bytes = 0;
            with_read_engine([&](InferenceEngine& engine) { memory_bytes = engine.get_memory_usage(); });

            const std::string body = join_json_fields({
                "\"ok\":true",
                "\"request_id\":" + std::to_string(request_id),
                "\"api\":\"" + std::string(kApiVersion) + "\"",
                "\"config\":" +
                    join_json_fields({
                        "\"num_layers\":" + std::to_string(engine_.config.num_layers),
                        "\"d_model\":" + std::to_string(engine_.config.d_model),
                        "\"vocab_size\":" + std::to_string(engine_.config.vocab_size),
                        "\"max_context_tokens\":" +
                            std::to_string(engine_.config.max_context_tokens),
                        "\"default_batch_size\":" +
                            std::to_string(engine_.config.default_batch_size),
                        "\"use_cuda\":" + json_bool(engine_.config.use_cuda),
                        "\"inference_replicas\":" +
                            std::to_string(static_cast<int>(desired_inference_replica_count())),
                    }),
                "\"memory_bytes\":" + std::to_string(memory_bytes),
                "\"generation\":" + generation_metrics_json(metrics),
            });

            HttpResponse response = make_json_response(200, "OK", body);
            response.headers["X-Request-ID"] = std::to_string(request_id);
            send_all(client_socket, build_http_response(response));
            return;
        }

        if (request.method == "GET" && request.path == "/metrics") {
            const auto now = std::chrono::steady_clock::now();
            const uint64_t uptime_ms = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(now - started_at_).count());
            const size_t queue_depth = [&]() {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                return client_queue_.size();
            }();

            GenerationMetrics generation_metrics = latest_generation_metrics_snapshot();

            const std::string body = join_json_fields({
                "\"ok\":true",
                "\"request_id\":" + std::to_string(request_id),
                "\"metrics\":" + server_metrics_json(
                                     *this, uptime_ms, total_requests_.load(),
                                     auth_failures_.load(), parse_failures_.load(),
                                     rejected_requests_.load(), rate_limited_requests_.load(),
                                     internal_errors_.load(),
                                     active_requests_.load(), queue_depth, generation_metrics),
            });
            HttpResponse response = make_json_response(200, "OK", body);
            response.headers["X-Request-ID"] = std::to_string(request_id);
            send_all(client_socket, build_http_response(response));
            return;
        }

        if (request.method == "POST" && request.path == "/generate") {
            const JsonValue payload = parse_json_body_or_throw(request, config_);
            const std::string prompt = json_string(payload, "prompt").value_or("");
            GenerationOptions options;
            options.max_tokens =
                (std::min)(options.max_tokens, config_.max_generate_tokens);
            if (const auto value = json_int(payload, "max_tokens")) options.max_tokens = *value;
            if (const auto value = json_float(payload, "temperature")) options.temperature = *value;
            if (const auto value = json_float(payload, "top_p")) options.top_p = *value;
            if (const auto value = json_int(payload, "top_k")) options.top_k = *value;
            if (const auto value = json_int(payload, "eos_token_id")) options.eos_token_id = *value;
            if (const auto value = json_int(payload, "max_context_tokens")) options.max_context_tokens = *value;
            validate_generation_options(options, config_, engine_.config.vocab_size);

            std::string text;
            GenerationMetrics metrics;
            with_inference_replica([&](InferenceEngine& engine) {
                text = engine.generate(prompt, options);
                metrics = engine.last_generation_metrics();
            });
            update_latest_generation_metrics(metrics);

            const std::string body = join_json_fields({
                "\"ok\":true",
                "\"request_id\":" + std::to_string(request_id),
                "\"text\":\"" + json_escape(text) + "\"",
                "\"metrics\":" + generation_metrics_json(metrics),
            });
            HttpResponse response = make_json_response(200, "OK", body);
            response.headers["X-Request-ID"] = std::to_string(request_id);
            send_all(client_socket, build_http_response(response));
            return;
        }

        if (request.method == "POST" && request.path == "/generate_batch") {
            const JsonValue payload = parse_json_body_or_throw(request, config_);
            GenerationOptions options;
            options.max_tokens =
                (std::min)(options.max_tokens, config_.max_generate_tokens);
            if (const auto value = json_int(payload, "max_tokens")) options.max_tokens = *value;
            if (const auto value = json_float(payload, "temperature")) options.temperature = *value;
            if (const auto value = json_float(payload, "top_p")) options.top_p = *value;
            if (const auto value = json_int(payload, "top_k")) options.top_k = *value;
            if (const auto value = json_int(payload, "eos_token_id")) options.eos_token_id = *value;
            if (const auto value = json_int(payload, "max_context_tokens")) options.max_context_tokens = *value;
            validate_generation_options(options, config_, engine_.config.vocab_size);
            const auto prompts = json_string_array(payload, "prompts");
            if (prompts.empty()) {
                throw BadRequest("field 'prompts' must contain at least one prompt");
            }
            if (prompts.size() > config_.max_prompts_per_batch) {
                throw BadRequest("field 'prompts' exceeds configured batch limit " +
                                         std::to_string(config_.max_prompts_per_batch));
            }

            std::vector<std::string> outputs;
            GenerationMetrics metrics;
            with_inference_replica([&](InferenceEngine& engine) {
                outputs = engine.generate_batch(prompts, options);
                metrics = engine.last_generation_metrics();
            });
            update_latest_generation_metrics(metrics);

            std::ostringstream output_json;
            output_json << '[';
            for (size_t i = 0; i < outputs.size(); ++i) {
                if (i > 0) output_json << ',';
                output_json << '"' << json_escape(outputs[i]) << '"';
            }
            output_json << ']';

            const std::string body = join_json_fields({
                "\"ok\":true",
                "\"request_id\":" + std::to_string(request_id),
                "\"outputs\":" + output_json.str(),
                "\"metrics\":" + generation_metrics_json(metrics),
            });
            HttpResponse response = make_json_response(200, "OK", body);
            response.headers["X-Request-ID"] = std::to_string(request_id);
            send_all(client_socket, build_http_response(response));
            return;
        }

        if (request.method == "POST" && request.path == "/generate_stream") {
            const JsonValue payload = parse_json_body_or_throw(request, config_);
            const std::string prompt = json_string(payload, "prompt").value_or("");
            GenerationOptions options;
            options.max_tokens =
                (std::min)(options.max_tokens, config_.max_generate_tokens);
            options.stream = true;
            if (const auto value = json_int(payload, "max_tokens")) options.max_tokens = *value;
            if (const auto value = json_float(payload, "temperature")) options.temperature = *value;
            if (const auto value = json_float(payload, "top_p")) options.top_p = *value;
            if (const auto value = json_int(payload, "top_k")) options.top_k = *value;
            if (const auto value = json_int(payload, "eos_token_id")) options.eos_token_id = *value;
            if (const auto value = json_int(payload, "max_context_tokens")) options.max_context_tokens = *value;
            validate_generation_options(options, config_, engine_.config.vocab_size);

            HttpResponse head;
            head.status_code = 200;
            head.status_text = "OK";
            head.content_type = "text/event-stream";
            head.body = "";
            head.headers["Cache-Control"] = "no-cache";
            head.headers["X-Request-ID"] = std::to_string(request_id);
            if (!send_all(client_socket,
                          build_http_response(head, /*omit_content_length=*/true))) {
                return;
            }

            std::string final_text;
            GenerationMetrics metrics;
            std::mutex stream_mutex;
            std::condition_variable stream_cv;
            std::deque<std::string> pending_chunks;
            size_t pending_bytes = 0;
            bool stream_finished = false;
            bool stream_failed = false;
            std::thread sender([&] {
                while (true) {
                    std::string chunk;
                    {
                        std::unique_lock<std::mutex> lock(stream_mutex);
                        stream_cv.wait(lock, [&] {
                            return stream_finished || stream_failed ||
                                   !pending_chunks.empty();
                        });
                        if (stream_failed) return;
                        if (pending_chunks.empty()) {
                            if (stream_finished) return;
                            continue;
                        }
                        chunk = std::move(pending_chunks.front());
                        pending_chunks.pop_front();
                    }

                    const bool sent = send_all(client_socket,
                                               "event: chunk\n"
                                               "data: {\"request_id\":" +
                                                   std::to_string(request_id) +
                                                   ",\"chunk\":\"" +
                                                   json_escape(chunk) + "\"}\n\n");
                    bool finished = false;
                    {
                        std::lock_guard<std::mutex> lock(stream_mutex);
                        pending_bytes -= (std::min)(pending_bytes, chunk.size());
                        if (!sent) {
                            stream_failed = true;
                            pending_chunks.clear();
                            pending_bytes = 0;
                        }
                        finished = stream_finished && pending_chunks.empty();
                    }
                    stream_cv.notify_all();
                    if (!sent || finished) return;
                }
            });
            JoinThreadGuard sender_guard{&sender};
            // Guarantee the sender is told to finish before sender_guard joins
            // it — even if generate_stream throws.  Without this, an exception
            // leaves stream_finished == false, the sender blocks forever on
            // stream_cv, and JoinThreadGuard::~JoinThreadGuard deadlocks the
            // worker.  Declared AFTER sender_guard so its destructor runs FIRST
            // (reverse order): signal + notify, THEN the join wakes and returns.
            struct FinishSignal {
                std::mutex& m;
                std::condition_variable& cv;
                bool& done;
                ~FinishSignal() {
                    {
                        std::lock_guard<std::mutex> lock(m);
                        done = true;
                    }
                    cv.notify_all();
                }
            } finish_signal{stream_mutex, stream_cv, stream_finished};
            bool aborted = false;
            try {
                with_inference_replica([&](InferenceEngine& engine) {
                    final_text = engine.generate_stream(
                        prompt, options, [&](const std::string& chunk) {
                            if (chunk.empty()) return;
                            if (chunk.size() > config_.max_stream_pending_bytes) {
                                throw StreamAborted("stream chunk exceeded configured buffer");
                            }
                            std::unique_lock<std::mutex> lock(stream_mutex);
                            stream_cv.wait(lock, [&] {
                                return stream_failed || stop_requested_.load() ||
                                       pending_bytes <=
                                           config_.max_stream_pending_bytes - chunk.size();
                            });
                            if (stream_failed || stop_requested_.load()) {
                                throw StreamAborted("stream client disconnected");
                            }
                            pending_bytes += chunk.size();
                            pending_chunks.push_back(chunk);
                            lock.unlock();
                            stream_cv.notify_one();
                        });
                    metrics = engine.last_generation_metrics();
                });
            } catch (const StreamAborted&) {
                aborted = true;
            }
            {
                std::lock_guard<std::mutex> lock(stream_mutex);
                stream_finished = true;
            }
            stream_cv.notify_all();
            // Drain and join the sender before emitting the terminal event.
            // Otherwise the main thread can write "done" while the sender is
            // still writing the final chunk, interleaving bytes on the socket
            // or placing "done" before a queued chunk.
            if (sender.joinable()) {
                sender.join();
            }
            if (aborted || stream_failed) {
                return;
            }
            update_latest_generation_metrics(metrics);

            send_all(client_socket,
                     "event: done\n"
                     "data: {\"request_id\":" +
                         std::to_string(request_id) + ",\"text\":\"" +
                         json_escape(final_text) + "\",\"metrics\":" +
                         generation_metrics_json(metrics) + "}\n\n");
            return;
        }

        if (request.method == "POST" && request.path == "/train-text") {
            if (!config_.enable_admin_endpoints) {
                send_all(client_socket, build_http_response(admin_disabled_response(request_id)));
                return;
            }
            const JsonValue payload = parse_json_body_or_throw(request, config_);
            const std::string text = json_string(payload, "text").value_or("");
            const int steps = bounded_int(payload, "steps", 1, 1, config_.max_train_steps);
            if (text.empty()) {
                throw BadRequest("field 'text' must not be empty");
            }
            if (text.size() > config_.max_train_text_bytes) {
                throw BadRequest("field 'text' exceeds configured byte limit");
            }

            float loss = 0.0f;
            with_transactional_training_engine([&](InferenceEngine& engine) {
                const std::vector<int> tokens =
                    engine.sanitize_token_ids(engine.tokenizer.encode(text));
                if (tokens.size() < 2) {
                    throw BadRequest("field 'text' must encode to at least two tokens");
                }
                for (int i = 0; i < steps; ++i) {
                    loss = engine.train_step(tokens, {});
                }
                if (!std::isfinite(loss)) {
                    throw std::runtime_error("training produced a non-finite loss");
                }
            });

            const std::string body = join_json_fields({
                "\"ok\":true",
                "\"request_id\":" + std::to_string(request_id),
                "\"loss\":" + std::to_string(loss),
                "\"steps\":" + std::to_string(steps),
                "\"text_bytes\":" + std::to_string(text.size()),
            });
            HttpResponse response = make_json_response(200, "OK", body);
            response.headers["X-Request-ID"] = std::to_string(request_id);
            send_all(client_socket, build_http_response(response));
            return;
        }

        if (request.method == "POST" && request.path == "/train-batch") {
            if (!config_.enable_admin_endpoints) {
                send_all(client_socket, build_http_response(admin_disabled_response(request_id)));
                return;
            }
            const JsonValue payload = parse_json_body_or_throw(request, config_);
            const auto texts = json_string_array(payload, "texts");
            const int epochs = bounded_int(payload, "epochs", 1, 1, config_.max_train_epochs);
            if (texts.empty()) {
                throw BadRequest("field 'texts' must contain at least one text sample");
            }
            if (texts.size() > config_.max_prompts_per_batch) {
                throw BadRequest("field 'texts' exceeds configured batch limit " +
                                         std::to_string(config_.max_prompts_per_batch));
            }
            for (const auto& text : texts) {
                if (text.empty()) {
                    throw BadRequest("field 'texts' must not contain empty samples");
                }
                if (text.size() > config_.max_train_text_bytes) {
                    throw BadRequest("field 'texts' contains an item exceeding configured byte limit");
                }
            }

            float last_loss = 0.0f;
            size_t updates = 0;
            with_transactional_training_engine([&](InferenceEngine& engine) {
                std::vector<std::vector<int>> token_batches;
                token_batches.reserve(texts.size());
                for (const auto& text : texts) {
                    std::vector<int> tokens =
                        engine.sanitize_token_ids(engine.tokenizer.encode(text));
                    if (tokens.size() < 2) {
                        throw BadRequest(
                            "each item in field 'texts' must encode to at least two tokens");
                    }
                    token_batches.push_back(std::move(tokens));
                }
                for (int epoch = 0; epoch < epochs; ++epoch) {
                    for (const auto& tokens : token_batches) {
                        last_loss = engine.train_step(tokens, {});
                        ++updates;
                    }
                }
                if (!std::isfinite(last_loss)) {
                    throw std::runtime_error("training produced a non-finite loss");
                }
            });

            const std::string body = join_json_fields({
                "\"ok\":true",
                "\"request_id\":" + std::to_string(request_id),
                "\"loss\":" + std::to_string(last_loss),
                "\"epochs\":" + std::to_string(epochs),
                "\"samples\":" + std::to_string(texts.size()),
                "\"updates\":" + std::to_string(updates),
            });
            HttpResponse response = make_json_response(200, "OK", body);
            response.headers["X-Request-ID"] = std::to_string(request_id);
            send_all(client_socket, build_http_response(response));
            return;
        }

        if (request.method == "POST" && request.path == "/train-corpus") {
            if (!config_.enable_admin_endpoints) {
                send_all(client_socket, build_http_response(admin_disabled_response(request_id)));
                return;
            }
            const JsonValue payload = parse_json_body_or_throw(request, config_);
            const std::string corpus = json_string(payload, "corpus").value_or("");
            const int epochs = bounded_int(payload, "epochs", 1, 1, config_.max_train_epochs);
            const int batch_size =
                bounded_int(payload, "batch_size", 4, 1, config_.max_train_batch_size);
            const int seq_len = bounded_int(payload, "seq_len", 32, 2, config_.max_train_seq_len);
            const int max_steps =
                bounded_int(payload, "max_steps", 120, 1, config_.max_train_corpus_steps);
            if (corpus.empty()) {
                throw BadRequest("field 'corpus' must not be empty");
            }
            if (corpus.size() > config_.max_body_bytes) {
                throw BadRequest("field 'corpus' exceeds configured byte limit");
            }
            const auto learning_rate = json_float(payload, "learning_rate");
            const auto weight_decay = json_float(payload, "weight_decay");
            const auto max_grad_norm = json_float(payload, "max_grad_norm");
            const auto min_learning_rate_scale =
                json_float(payload, "min_learning_rate_scale");
            const auto warmup_steps = json_int(payload, "warmup_steps");
            if (learning_rate && (*learning_rate <= 0.0f || *learning_rate > 1.0f)) {
                throw BadRequest("field 'learning_rate' must be in (0, 1]");
            }
            if (weight_decay && (*weight_decay < 0.0f || *weight_decay > 10.0f)) {
                throw BadRequest("field 'weight_decay' must be between 0 and 10");
            }
            if (max_grad_norm && (*max_grad_norm <= 0.0f || *max_grad_norm > 1000000.0f)) {
                throw BadRequest("field 'max_grad_norm' must be in (0, 1000000]");
            }
            if (min_learning_rate_scale &&
                (*min_learning_rate_scale <= 0.0f || *min_learning_rate_scale > 1.0f)) {
                throw BadRequest("field 'min_learning_rate_scale' must be in (0, 1]");
            }
            if (warmup_steps && (*warmup_steps < 0 || *warmup_steps > 100000000)) {
                throw BadRequest("field 'warmup_steps' must be between 0 and 100000000");
            }

            float last_loss = 0.0f;
            size_t token_count = 0;
            with_transactional_training_engine([&](InferenceEngine& engine) {
                if (!engine.trainer) {
                    throw std::runtime_error("trainer is not initialized");
                }

                std::vector<int> tokens =
                    engine.sanitize_token_ids(engine.tokenizer.encode(corpus));
                token_count = tokens.size();
                if (token_count < 2) {
                    throw BadRequest("field 'corpus' must encode to at least two tokens");
                }

                // Apply overrides only after all input-derived validation has
                // passed; the whole staged engine is discarded on later error.
                if (learning_rate) engine.trainer->learning_rate = *learning_rate;
                if (weight_decay) engine.trainer->weight_decay = *weight_decay;
                if (max_grad_norm) engine.trainer->max_grad_norm = *max_grad_norm;
                if (min_learning_rate_scale) {
                    engine.trainer->min_learning_rate_scale = *min_learning_rate_scale;
                }
                if (warmup_steps) engine.trainer->warmup_steps = *warmup_steps;

                engine.trainer->train_loop(
                    tokens, epochs, batch_size, seq_len,
                    [&](int, float loss) { last_loss = loss; }, max_steps);
                if (!std::isfinite(last_loss)) {
                    throw std::runtime_error("training produced a non-finite loss");
                }
            });

            const std::string body = join_json_fields({
                "\"ok\":true",
                "\"request_id\":" + std::to_string(request_id),
                "\"loss\":" + std::to_string(last_loss),
                "\"epochs\":" + std::to_string(epochs),
                "\"batch_size\":" + std::to_string(batch_size),
                "\"seq_len\":" + std::to_string(seq_len),
                "\"max_steps\":" + std::to_string(max_steps),
                "\"token_count\":" + std::to_string(token_count),
            });
            HttpResponse response = make_json_response(200, "OK", body);
            response.headers["X-Request-ID"] = std::to_string(request_id);
            send_all(client_socket, build_http_response(response));
            return;
        }

        if (request.method == "POST" && request.path == "/pack") {
            if (!config_.enable_admin_endpoints) {
                send_all(client_socket, build_http_response(admin_disabled_response(request_id)));
                return;
            }
            const JsonValue payload = parse_json_body_or_throw(request, config_);
            const std::string directory = json_string(payload, "directory").value_or("");
            if (directory.empty()) {
                throw BadRequest("field 'directory' must not be empty");
            }
            const std::filesystem::path output_directory =
                resolve_pack_output_directory(config_, directory);

            bool saved = false;
            with_read_engine([&](InferenceEngine& engine) {
                saved = engine.save_model_pack(output_directory.string());
            });
            if (!saved) {
                throw std::runtime_error("model pack persistence failed");
            }

            const std::string body = join_json_fields({
                "\"ok\":" + json_bool(saved),
                "\"request_id\":" + std::to_string(request_id),
                "\"saved\":" + json_bool(saved),
                "\"directory\":\"" + json_escape(output_directory.string()) + "\"",
            });
            HttpResponse response = make_json_response(200, "OK", body);
            response.headers["X-Request-ID"] = std::to_string(request_id);
            send_all(client_socket, build_http_response(response));
            return;
        }

        const HttpResponse response = make_error_response(
            404, "Not Found", request_id, "unknown_endpoint", "requested endpoint was not found");
        send_all(client_socket, build_http_response(response));
    } catch (const AbortException& ex) {
        rejected_requests_.fetch_add(1);
        const HttpResponse response = make_error_response(
            503, "Service Unavailable", request_id,
            "training_cancelled", ex.what());
        send_all(client_socket, build_http_response(response));
    } catch (const OptimizerStatePoisonedException&) {
        internal_errors_.fetch_add(1);
        const HttpResponse response = make_error_response(
            503, "Service Unavailable", request_id,
            "optimizer_state_requires_recovery",
            "training is unavailable until a validated checkpoint recovery");
        send_all(client_socket, build_http_response(response));
    } catch (const BadRequest& ex) {
        // Explicit client-input error: surface the message with 400 (the
        // project deliberately returns actionable validation messages, and the
        // HTTP gate asserts this).  Distinct telemetry from internal faults.
        parse_failures_.fetch_add(1);
        const HttpResponse response =
            make_error_response(400, "Bad Request", request_id, "bad_request", ex.what());
        send_all(client_socket, build_http_response(response));
    } catch (const std::exception& ex) {
        internal_errors_.fetch_add(1);
        std::cerr << "HTTP request " << request_id
                  << " failed internally: " << ex.what() << std::endl;
        const HttpResponse response =
            make_error_response(500, "Internal Server Error", request_id,
                                "internal_error",
                                "the request could not be completed");
        send_all(client_socket, build_http_response(response));
    }
}

} // namespace
