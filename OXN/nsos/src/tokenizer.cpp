#include "../include/tokenizer.h"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <unordered_set>

namespace nsos {

namespace {

constexpr int kMaxTokenizerVocabSize = 2'000'000;
constexpr size_t kMaxTokenizerTokenBytes = 16 * 1024;
constexpr uintmax_t kMaxTokenizerFileBytes = 256ull * 1024ull * 1024ull;
constexpr size_t kMaxTokenizerEncodeBytes = 64ull * 1024ull * 1024ull;
constexpr size_t kMaxTokenizerDecodeBytes = 256ull * 1024ull * 1024ull;

void ensure_tokenizer_file_is_bounded(const std::string& path) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec) || ec) {
    throw std::runtime_error("Tokenizer path is not a regular file: " + path);
  }
  const uintmax_t bytes = std::filesystem::file_size(path, ec);
  if (ec || bytes > kMaxTokenizerFileBytes) {
    throw std::runtime_error("Tokenizer file exceeds configured size limit");
  }
}

bool read_u32_le(std::istream& input, uint32_t& value) {
  unsigned char bytes[4]{};
  input.read(reinterpret_cast<char*>(bytes), 4);
  if (!input) return false;
  value = static_cast<uint32_t>(bytes[0]) |
          (static_cast<uint32_t>(bytes[1]) << 8) |
          (static_cast<uint32_t>(bytes[2]) << 16) |
          (static_cast<uint32_t>(bytes[3]) << 24);
  return true;
}

bool has_suffix(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string hex_encode(const std::string& input) {
  static constexpr char kHex[] = "0123456789ABCDEF";
  std::string out;
  out.reserve(input.size() * 2);
  for (unsigned char ch : input) {
    out.push_back(kHex[(ch >> 4) & 0xF]);
    out.push_back(kHex[ch & 0xF]);
  }
  return out;
}

std::string hex_decode(const std::string& input) {
  if ((input.size() % 2) != 0) {
    throw std::runtime_error("Invalid tokenizer hex payload");
  }
  if (input.size() / 2 > kMaxTokenizerTokenBytes) {
    throw std::runtime_error("Tokenizer token payload exceeds configured limit");
  }

  auto from_hex = [](char ch) -> unsigned char {
    if (ch >= '0' && ch <= '9') return static_cast<unsigned char>(ch - '0');
    if (ch >= 'A' && ch <= 'F') return static_cast<unsigned char>(10 + ch - 'A');
    if (ch >= 'a' && ch <= 'f') return static_cast<unsigned char>(10 + ch - 'a');
    throw std::runtime_error("Invalid tokenizer hex character");
  };

  std::string out;
  out.reserve(input.size() / 2);
  for (size_t i = 0; i < input.size(); i += 2) {
    const unsigned char hi = from_hex(input[i]);
    const unsigned char lo = from_hex(input[i + 1]);
    out.push_back(static_cast<char>((hi << 4) | lo));
  }
  return out;
}

void append_utf8_replacement(std::string& out) {
  out.push_back(static_cast<char>(0xEF));
  out.push_back(static_cast<char>(0xBF));
  out.push_back(static_cast<char>(0xBD));
}

std::string sanitize_utf8(const std::string& input) {
  std::string out;
  out.reserve(input.size());
  for (size_t i = 0; i < input.size();) {
    const unsigned char c = static_cast<unsigned char>(input[i]);
    if (c < 0x80) {
      out.push_back(static_cast<char>(c));
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
      append_utf8_replacement(out);
      ++i;
      continue;
    }

    if (i + needed > input.size()) {
      append_utf8_replacement(out);
      break;
    }

    bool valid = true;
    for (size_t j = 1; j < needed; ++j) {
      const unsigned char cont = static_cast<unsigned char>(input[i + j]);
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
      append_utf8_replacement(out);
      ++i;
      continue;
    }

    out.append(input, i, needed);
    i += needed;
  }
  return out;
}

void reset_tokenizer_state(Tokenizer& tokenizer) {
  tokenizer.bpe_ranks.clear();
  tokenizer.token_to_id.clear();
  tokenizer.id_to_token.clear();
  tokenizer.special_tokens.clear();
  tokenizer.sorted_specials.clear();
  tokenizer.vocab_size = 0;
}

void rebuild_special_cache(Tokenizer& tokenizer) {
  tokenizer.sorted_specials.clear();
  for (const auto& token : tokenizer.special_tokens) {
    // An empty special token matches every position without advancing the
    // cursor.  Keep the cache defensive even if a caller mutates the public
    // set directly.
    if (token.empty()) {
      continue;
    }
    tokenizer.sorted_specials.push_back(token);
  }
  std::sort(tokenizer.sorted_specials.begin(), tokenizer.sorted_specials.end(),
            [](const std::string& a, const std::string& b) {
              return a.length() > b.length();
            });
}

bool is_whitespace_byte(unsigned char ch) {
  return std::isspace(ch) != 0;
}

bool is_word_byte(unsigned char ch) {
  return (ch >= '0' && ch <= '9') ||
         (ch >= 'A' && ch <= 'Z') ||
         (ch >= 'a' && ch <= 'z') ||
         ch == '_' || ch == '-' || ch == '/' ||
         ch >= 0x80;
}

void replace_all_inplace(std::string& text, const std::string& needle, const std::string& repl) {
  if (needle.empty()) {
    return;
  }
  size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    text.replace(pos, needle.size(), repl);
    pos += repl.size();
  }
}

std::string normalize_for_tokenization(std::string text) {
  replace_all_inplace(text, "\r\n", "\n");
  replace_all_inplace(text, "\r", "\n");
  replace_all_inplace(text, "\xC2\xA0", " ");
  replace_all_inplace(text, "\xE2\x80\x89", " ");
  replace_all_inplace(text, "\xE2\x80\x8A", " ");
  replace_all_inplace(text, "\xE2\x80\xAF", " ");
  replace_all_inplace(text, "\xE2\x80\x98", "'");
  replace_all_inplace(text, "\xE2\x80\x99", "'");
  replace_all_inplace(text, "\xE2\x80\x9C", "\"");
  replace_all_inplace(text, "\xE2\x80\x9D", "\"");
  replace_all_inplace(text, "\xE2\x80\x93", "-");
  replace_all_inplace(text, "\xE2\x80\x94", "-");
  replace_all_inplace(text, "\xE2\x88\x92", "-");

  std::string normalized;
  normalized.reserve(text.size());
  bool previous_space = false;
  for (unsigned char ch : text) {
    if (is_whitespace_byte(ch)) {
      if (!previous_space) {
        normalized.push_back(ch == '\n' ? '\n' : ' ');
      }
      previous_space = true;
      continue;
    }
    previous_space = false;
    normalized.push_back(static_cast<char>(ch));
  }
  return normalized;
}

std::vector<std::string> pretokenize_segment(const std::string& segment) {
  std::vector<std::string> pieces;
  size_t cursor = 0;
  while (cursor < segment.size()) {
    const unsigned char current = static_cast<unsigned char>(segment[cursor]);
    const bool whitespace = is_whitespace_byte(current);
    const bool word = is_word_byte(current);
    size_t end = cursor + 1;
    if (whitespace) {
      while (end < segment.size() &&
             is_whitespace_byte(static_cast<unsigned char>(segment[end]))) {
        ++end;
      }
    } else if (word) {
      while (end < segment.size() &&
             is_word_byte(static_cast<unsigned char>(segment[end]))) {
        ++end;
      }
    }
    pieces.push_back(segment.substr(cursor, end - cursor));
    cursor = end;
  }
  return pieces;
}

} // namespace

// =============================================================================
// BPE Tokenizer (Full Inference Implementation with OX3 Support)
// =============================================================================

Tokenizer::Tokenizer() {
  // 1. Initialize Byte-to-Token Map (Base Vocabulary)
  for (int i = 0; i < 256; ++i) {
    std::string s(1, static_cast<char>(i));
    token_to_id[s] = i;
    id_to_token[i] = s;
  }
  vocab_size = 256;
}

void Tokenizer::add_special_tokens(const std::vector<std::string> &tokens) {
  for (const auto &t : tokens) {
    if (t.empty()) {
      throw std::invalid_argument("Tokenizer special tokens must not be empty");
    }
    if (t.size() > kMaxTokenizerTokenBytes) {
      throw std::invalid_argument("Tokenizer special token exceeds configured limit");
    }
    if (token_to_id.find(t) == token_to_id.end()) {
      if (vocab_size >= kMaxTokenizerVocabSize) {
        throw std::invalid_argument("Tokenizer vocabulary exceeds configured limit");
      }
      int id = vocab_size++;
      token_to_id[t] = id;
      id_to_token[id] = t;
    }
    special_tokens.insert(t);
  }
  rebuild_special_cache(*this);
}

void Tokenizer::load(const std::string &path) {
  ensure_tokenizer_file_is_bounded(path);
  Tokenizer staged;

  std::cout << "[Tokenizer] Loading BPE model from " << path << "..."
            << std::endl;

  if (has_suffix(path, ".nsos") || has_suffix(path, ".tok") ||
      has_suffix(path, "tokenizer.nsos")) {
    staged.load_pack(path);
  } else if (path.length() >= 4 && path.substr(path.length() - 4) == ".ox3") {
    staged.load_ox3(path);
  } else {
    staged.load_text(path);
  }
  *this = std::move(staged);
  std::cout << "[Tokenizer] Loaded vocab size: " << vocab_size << std::endl;
}

void Tokenizer::save_pack(const std::string& path) const {
  if (vocab_size < 256 || vocab_size > kMaxTokenizerVocabSize ||
      id_to_token.size() != static_cast<size_t>(vocab_size) ||
      token_to_id.size() != static_cast<size_t>(vocab_size)) {
    throw std::runtime_error("Tokenizer state has an invalid vocabulary size");
  }
  for (int id = 0; id < vocab_size; ++id) {
    const auto by_id = id_to_token.find(id);
    if (by_id == id_to_token.end() || by_id->second.empty() ||
        by_id->second.size() > kMaxTokenizerTokenBytes) {
      throw std::runtime_error("Tokenizer state has an incomplete or invalid id space");
    }
    const auto by_token = token_to_id.find(by_id->second);
    if (by_token == token_to_id.end() || by_token->second != id) {
      throw std::runtime_error("Tokenizer forward and reverse vocabularies disagree");
    }
  }
  for (int byte = 0; byte < 256; ++byte) {
    const std::string expected(1, static_cast<char>(byte));
    const auto it = id_to_token.find(byte);
    if (it == id_to_token.end() || it->second != expected ||
        special_tokens.count(expected) != 0) {
      throw std::runtime_error("Tokenizer state has an invalid base byte vocabulary");
    }
  }
  std::vector<uint8_t> seen_ranks(bpe_ranks.size(), 0);
  for (const auto& [pair, rank] : bpe_ranks) {
    if (rank < 0 || static_cast<size_t>(rank) >= bpe_ranks.size() ||
        seen_ranks[static_cast<size_t>(rank)] || pair.first.empty() ||
        pair.second.empty() ||
        pair.first.size() > kMaxTokenizerTokenBytes -
                                (std::min)(pair.second.size(), kMaxTokenizerTokenBytes)) {
      throw std::runtime_error("Tokenizer state has invalid or non-contiguous BPE ranks");
    }
    seen_ranks[static_cast<size_t>(rank)] = 1;
  }
  for (const auto& token : special_tokens) {
    if (token_to_id.count(token) == 0) {
      throw std::runtime_error("Tokenizer special-token set references an unknown token");
    }
  }

  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("Could not open tokenizer pack for writing");
  }

  out << "NSOS_TOKENIZER_V2\t" << vocab_size << "\t" << bpe_ranks.size() << "\n";

  std::vector<std::pair<int, std::string>> ordered;
  ordered.reserve(id_to_token.size());
  for (const auto& [id, token] : id_to_token) {
    ordered.emplace_back(id, token);
  }
  std::sort(ordered.begin(), ordered.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

  for (const auto& [id, token] : ordered) {
    out << "T\t" << id << "\t" << hex_encode(token) << "\t"
        << (special_tokens.count(token) ? 1 : 0) << "\n";
  }
  struct OrderedMerge {
    int rank = 0;
    std::string left;
    std::string right;
  };
  std::vector<OrderedMerge> merges;
  merges.reserve(bpe_ranks.size());
  for (const auto& [pair, rank] : bpe_ranks) {
    merges.push_back({rank, pair.first, pair.second});
  }
  std::sort(merges.begin(), merges.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.rank != rhs.rank) return lhs.rank < rhs.rank;
    if (lhs.left != rhs.left) return lhs.left < rhs.left;
    return lhs.right < rhs.right;
  });
  for (const auto& merge : merges) {
    out << "M\t" << merge.rank << "\t" << hex_encode(merge.left) << "\t"
        << hex_encode(merge.right) << "\n";
  }
  out.close();
  if (!out) {
    throw std::runtime_error("Could not flush tokenizer pack");
  }
}

void Tokenizer::load_pack(const std::string& path) {
  ensure_tokenizer_file_is_bounded(path);

  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Could not open tokenizer pack");
  }

  std::string header;
  if (!std::getline(in, header)) {
    throw std::runtime_error("Invalid tokenizer pack header");
  }
  int declared_vocab = 0;
  int declared_merges = 0;
  bool version_two = false;
  {
    std::istringstream header_stream(header);
    std::string magic;
    const bool base_fields_ok = static_cast<bool>(header_stream >> magic >> declared_vocab);
    version_two = magic == "NSOS_TOKENIZER_V2";
    bool merge_field_ok = true;
    if (version_two) {
      merge_field_ok = static_cast<bool>(header_stream >> declared_merges);
    }
    std::string trailing;
    const bool has_trailing = static_cast<bool>(header_stream >> trailing);
    if ((magic != "NSOS_TOKENIZER_V1" && !version_two) ||
        !base_fields_ok || !merge_field_ok || has_trailing || declared_vocab < 256 ||
        declared_vocab > kMaxTokenizerVocabSize || declared_merges < 0 ||
        declared_merges > kMaxTokenizerVocabSize) {
      throw std::runtime_error("Tokenizer pack declared vocab exceeds configured limit");
    }
  }

  std::unordered_map<std::string, int> next_token_to_id;
  std::unordered_map<int, std::string> next_id_to_token;
  std::unordered_map<std::pair<std::string, std::string>, int, PairHash> next_bpe_ranks;
  std::set<std::string> next_special_tokens;
  std::vector<char> seen_ids(static_cast<size_t>(declared_vocab), 0);
  std::vector<char> seen_ranks(static_cast<size_t>(declared_merges), 0);
  int max_id = -1;
  size_t line_count = 0;
  size_t token_line_count = 0;
  size_t merge_line_count = 0;
  auto parse_bounded_int = [](const std::string& text, const char* label) {
    try {
      size_t consumed = 0;
      const int value = std::stoi(text, &consumed);
      if (consumed != text.size()) throw std::runtime_error("trailing characters");
      return value;
    } catch (const std::exception&) {
      throw std::runtime_error(std::string("Tokenizer pack contains an invalid ") + label);
    }
  };
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (++line_count > static_cast<size_t>(declared_vocab) +
                           static_cast<size_t>(declared_merges)) {
      throw std::runtime_error("Tokenizer pack contains too many entries");
    }

    std::istringstream ss(line);
    std::string kind;
    std::string id_str;
    std::string token_hex;
    std::string special_flag;
    if (version_two) {
      if (!std::getline(ss, kind, '\t')) {
        throw std::runtime_error("Malformed tokenizer pack line");
      }
      if (kind == "M") {
        std::string rank_text;
        std::string left_hex;
        std::string right_hex;
        if (!std::getline(ss, rank_text, '\t') ||
            !std::getline(ss, left_hex, '\t') ||
            !std::getline(ss, right_hex) || right_hex.find('\t') != std::string::npos) {
          throw std::runtime_error("Malformed tokenizer merge line");
        }
        const int rank = parse_bounded_int(rank_text, "merge rank");
        if (rank < 0 || rank >= declared_merges ||
            seen_ranks[static_cast<size_t>(rank)]) {
          throw std::runtime_error("Tokenizer pack merge rank is duplicated or out of range");
        }
        std::string left = hex_decode(left_hex);
        std::string right = hex_decode(right_hex);
        if (left.empty() || right.empty() ||
            left.size() > kMaxTokenizerTokenBytes -
                              (std::min)(right.size(), kMaxTokenizerTokenBytes)) {
          throw std::runtime_error("Tokenizer pack contains an invalid merge token");
        }
        const auto [unused, inserted] =
            next_bpe_ranks.emplace(std::make_pair(std::move(left), std::move(right)), rank);
        if (!inserted) {
          throw std::runtime_error("Tokenizer pack contains a duplicate merge pair");
        }
        seen_ranks[static_cast<size_t>(rank)] = 1;
        ++merge_line_count;
        continue;
      }
      if (kind != "T") {
        throw std::runtime_error("Tokenizer pack contains an unknown record type");
      }
    }
    if (!std::getline(ss, id_str, '\t') ||
        !std::getline(ss, token_hex, '\t') ||
        !std::getline(ss, special_flag) || special_flag.find('\t') != std::string::npos) {
      throw std::runtime_error("Malformed tokenizer pack line");
    }

    const int id = parse_bounded_int(id_str, "token id");
    if (id < 0 || id >= declared_vocab) {
      throw std::runtime_error("Tokenizer pack token id is outside configured limit");
    }
    std::string token = hex_decode(token_hex);
    if (token.empty()) {
      throw std::runtime_error("Tokenizer pack contains an empty token");
    }
    if (special_flag != "0" && special_flag != "1") {
      throw std::runtime_error("Tokenizer pack contains an invalid special flag");
    }
    if (seen_ids[static_cast<size_t>(id)] || next_token_to_id.count(token) != 0) {
      throw std::runtime_error("Tokenizer pack contains duplicate ids or tokens");
    }
    seen_ids[static_cast<size_t>(id)] = 1;
    next_token_to_id.emplace(token, id);
    next_id_to_token.emplace(id, token);
    if (special_flag == "1") {
      next_special_tokens.insert(token);
    }
    max_id = std::max(max_id, id);
    ++token_line_count;
  }

  if (token_line_count != static_cast<size_t>(declared_vocab) ||
      merge_line_count != static_cast<size_t>(declared_merges) ||
      max_id + 1 != declared_vocab ||
      std::find(seen_ids.begin(), seen_ids.end(), 0) != seen_ids.end() ||
      std::find(seen_ranks.begin(), seen_ranks.end(), 0) != seen_ranks.end()) {
    throw std::runtime_error("Tokenizer pack id space is incomplete");
  }
  for (int byte = 0; byte < 256; ++byte) {
    const std::string expected(1, static_cast<char>(byte));
    const auto it = next_id_to_token.find(byte);
    if (it == next_id_to_token.end() || it->second != expected ||
        next_special_tokens.count(expected) != 0) {
      throw std::runtime_error("Tokenizer pack base byte vocabulary is invalid");
    }
  }
  token_to_id = std::move(next_token_to_id);
  id_to_token = std::move(next_id_to_token);
  bpe_ranks = std::move(next_bpe_ranks);
  special_tokens = std::move(next_special_tokens);
  vocab_size = declared_vocab;
  rebuild_special_cache(*this);
}

void Tokenizer::load_text(const std::string &vocab_path) {
  ensure_tokenizer_file_is_bounded(vocab_path);
  std::ifstream f(vocab_path);
  if (!f.is_open()) {
    throw std::runtime_error("Tokenizer text vocab could not be opened: " +
                             vocab_path);
  }

  Tokenizer base;
  auto next_bpe_ranks = std::move(base.bpe_ranks);
  auto next_token_to_id = std::move(base.token_to_id);
  auto next_id_to_token = std::move(base.id_to_token);
  int next_vocab_size = base.vocab_size;
  std::string line;
  int rank = 0;
  size_t line_count = 0;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    if (++line_count > static_cast<size_t>(kMaxTokenizerVocabSize) ||
        line.size() > 2 * kMaxTokenizerTokenBytes + 32) {
      throw std::runtime_error("Tokenizer text vocab exceeds configured limits");
    }
    std::stringstream ss(line);
    std::string p1, p2;
    ss >> p1 >> p2;

    if (!p1.empty() && !p2.empty()) {
      if (p1.size() > kMaxTokenizerTokenBytes ||
          p2.size() > kMaxTokenizerTokenBytes ||
          p1.size() > kMaxTokenizerTokenBytes -
                          (std::min)(p2.size(), kMaxTokenizerTokenBytes)) {
        throw std::runtime_error("Tokenizer text merge token exceeds configured limit");
      }
      const auto [unused, inserted] = next_bpe_ranks.emplace(std::make_pair(p1, p2), rank++);
      if (!inserted) {
        throw std::runtime_error("Tokenizer text vocab contains a duplicate merge");
      }
      std::string merged = p1 + p2;
      if (next_token_to_id.find(merged) == next_token_to_id.end()) {
        if (next_vocab_size >= kMaxTokenizerVocabSize) {
          throw std::runtime_error(
              "Tokenizer vocab exceeds configured limit");
        }
        int id = next_vocab_size++;
        next_token_to_id[merged] = id;
        next_id_to_token[id] = merged;
      }
    } else {
      throw std::runtime_error("Tokenizer text vocab contains a malformed merge");
    }
  }
  bpe_ranks = std::move(next_bpe_ranks);
  token_to_id = std::move(next_token_to_id);
  id_to_token = std::move(next_id_to_token);
  special_tokens.clear();
  sorted_specials.clear();
  vocab_size = next_vocab_size;
}

void Tokenizer::load_ox3(const std::string &path) {
  ensure_tokenizer_file_is_bounded(path);
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("Tokenizer OX3 file could not be opened: " + path);
  }

  // Header: OX3\0 (4 bytes) + Version (4 bytes)
  char magic[4];
  if (!f.read(magic, 4)) {
    throw std::runtime_error("Tokenizer OX3 header could not be read");
  }
  if (std::memcmp(magic, "OX3\0", 4) != 0) {
    throw std::runtime_error("Tokenizer OX3 has invalid magic bytes");
  }

  uint32_t version = 0;
  if (!read_u32_le(f, version)) {
    throw std::runtime_error("Tokenizer OX3 version could not be read");
  }
  if (version != 1) {
    throw std::runtime_error("Tokenizer OX3 version is unsupported");
  }

  Tokenizer base;
  auto next_bpe_ranks = std::move(base.bpe_ranks);
  auto next_token_to_id = std::move(base.token_to_id);
  auto next_id_to_token = std::move(base.id_to_token);
  int next_vocab_size = base.vocab_size;
  size_t entry_count = 0;

  while (f.peek() != EOF) {
    if (++entry_count > static_cast<size_t>(kMaxTokenizerVocabSize)) {
      throw std::runtime_error("Tokenizer OX3 contains too many merges");
    }
    uint32_t rank = 0;
    uint32_t s1_len = 0;
    uint32_t s2_len = 0;

    if (!read_u32_le(f, rank) || !read_u32_le(f, s1_len)) {
      throw std::runtime_error("Tokenizer OX3 is truncated before a merge");
    }

    if (s1_len > kMaxTokenizerTokenBytes || s1_len == 0) {
      throw std::runtime_error("Tokenizer OX3 contains invalid first token length");
    }

    std::vector<char> b1(s1_len);
    if (!f.read(b1.data(), s1_len)) {
      throw std::runtime_error("Tokenizer OX3 truncated while reading first token");
    }

    if (!read_u32_le(f, s2_len))
      throw std::runtime_error("Tokenizer OX3 truncated before second token length");

    if (s2_len > kMaxTokenizerTokenBytes || s2_len == 0 ||
        s1_len > kMaxTokenizerTokenBytes - s2_len) {
      throw std::runtime_error("Tokenizer OX3 contains invalid second token length");
    }

    std::vector<char> b2(s2_len);
    if (!f.read(b2.data(), s2_len)) {
      throw std::runtime_error("Tokenizer OX3 truncated while reading second token");
    }

    std::string s1(b1.begin(), b1.end());
    std::string s2(b2.begin(), b2.end());

    if (rank > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("Tokenizer OX3 merge rank exceeds configured limit");
    }
    if (rank != static_cast<uint32_t>(entry_count - 1)) {
      throw std::runtime_error("Tokenizer OX3 merge ranks must be contiguous and ordered");
    }
    const auto [unused, inserted] =
        next_bpe_ranks.emplace(std::make_pair(s1, s2), static_cast<int>(rank));
    if (!inserted) {
      throw std::runtime_error("Tokenizer OX3 contains a duplicate merge pair");
    }
    std::string merged = s1 + s2;
    if (next_token_to_id.find(merged) == next_token_to_id.end()) {
      if (next_vocab_size >= kMaxTokenizerVocabSize) {
        // Same cap as load_text/load_pack — an adversarial .ox3 must not grow
        // the vocab (and the id space) without bound.
        throw std::runtime_error("Tokenizer vocab exceeds configured limit");
      }
      int id = next_vocab_size++;
      next_token_to_id[merged] = id;
      next_id_to_token[id] = merged;
    }
  }
  bpe_ranks = std::move(next_bpe_ranks);
  token_to_id = std::move(next_token_to_id);
  id_to_token = std::move(next_id_to_token);
  special_tokens.clear();
  sorted_specials.clear();
  vocab_size = next_vocab_size;
}

std::vector<int> Tokenizer::encode(const std::string &text) {
  if (text.empty())
    return {};
  if (text.size() > kMaxTokenizerEncodeBytes) {
    throw std::invalid_argument("Tokenizer input exceeds configured byte limit");
  }
  // Public state exists for backward compatibility; rebuild the derived cache
  // so direct special_tokens mutations cannot leave matching stale.
  rebuild_special_cache(*this);

  const std::string normalized = normalize_for_tokenization(text);
  std::vector<int> result;
  size_t pos = 0;

  while (pos < normalized.length()) {
    // Try to match special token first (longest match)
    bool matched = false;
    for (const auto &st : sorted_specials) {
      if (normalized.compare(pos, st.length(), st) == 0) {
        const auto token = token_to_id.find(st);
        if (token == token_to_id.end()) {
          throw std::runtime_error("Tokenizer special-token cache is inconsistent");
        }
        result.push_back(token->second);
        pos += st.length();
        matched = true;
        break;
      }
    }
    if (matched)
      continue;

    // Find next segment of non-special text
    size_t segment_start = pos;
    while (pos < normalized.length()) {
      bool is_special_start = false;
      for (const auto &st : sorted_specials) {
        if (normalized.compare(pos, st.length(), st) == 0) {
          is_special_start = true;
          break;
        }
      }
      if (is_special_start)
        break;
      ++pos;
    }

    // Encode the non-special segment using BPE
    if (pos > segment_start) {
      std::string segment = normalized.substr(segment_start, pos - segment_start);
      for (const auto& piece : pretokenize_segment(segment)) {
        if (piece.empty()) {
          continue;
        }
        if (is_whitespace_byte(static_cast<unsigned char>(piece[0]))) {
          for (unsigned char value : piece) {
            result.push_back(static_cast<int>(value));
          }
        } else {
          std::vector<int> segment_ids = bpe_encode_word(piece);
          result.insert(result.end(), segment_ids.begin(), segment_ids.end());
        }
      }
    }
  }

  return result;
}

std::vector<int> Tokenizer::bpe_encode_word(const std::string &word) {
  if (word.empty())
    return {};

  // Efficient BPE merge: a doubly-linked list of symbols + a min-heap of merge
  // candidates keyed by (rank, left position).  Byte-identical to the previous
  // O(n^2) "scan-all-pairs, merge the global-min-rank pair, ties -> leftmost"
  // algorithm: initial positions preserve left-to-right order, so the heap's
  // min-(rank, pos) is exactly the lowest-current-index min-rank pair; stale
  // entries are skipped by re-checking the pair's rank on pop.  This removes the
  // quadratic blow-up on a long no-space / CJK run (one giant pretokenized word).
  const int n = static_cast<int>(word.length());
  std::vector<std::string> node(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    node[static_cast<size_t>(i)] = std::string(1, word[static_cast<size_t>(i)]);
  }
  std::vector<int> prev_idx(static_cast<size_t>(n));
  std::vector<int> next_idx(static_cast<size_t>(n));
  std::vector<char> alive(static_cast<size_t>(n), 1);
  for (int i = 0; i < n; ++i) {
    prev_idx[static_cast<size_t>(i)] = i - 1;
    next_idx[static_cast<size_t>(i)] = (i + 1 < n) ? i + 1 : -1;
  }

  struct MergeCand {
    int rank;
    int pos;
  };
  const auto cand_cmp = [](const MergeCand &a, const MergeCand &b) {
    return a.rank != b.rank ? a.rank > b.rank : a.pos > b.pos;
  };
  std::priority_queue<MergeCand, std::vector<MergeCand>, decltype(cand_cmp)> heap(
      cand_cmp);
  const auto push_pair = [&](int i) {
    if (i < 0 || next_idx[static_cast<size_t>(i)] < 0) return;
    auto it = bpe_ranks.find(
        {node[static_cast<size_t>(i)],
         node[static_cast<size_t>(next_idx[static_cast<size_t>(i)])]});
    if (it != bpe_ranks.end()) heap.push({it->second, i});
  };
  for (int i = 0; i < n; ++i) push_pair(i);

  while (!heap.empty()) {
    const MergeCand cand = heap.top();
    heap.pop();
    const int i = cand.pos;
    if (!alive[static_cast<size_t>(i)]) continue;
    const int j = next_idx[static_cast<size_t>(i)];
    if (j < 0 || !alive[static_cast<size_t>(j)]) continue;
    auto it = bpe_ranks.find(
        {node[static_cast<size_t>(i)], node[static_cast<size_t>(j)]});
    if (it == bpe_ranks.end() || it->second != cand.rank) continue;  // stale
    node[static_cast<size_t>(i)] += node[static_cast<size_t>(j)];
    alive[static_cast<size_t>(j)] = 0;
    next_idx[static_cast<size_t>(i)] = next_idx[static_cast<size_t>(j)];
    if (next_idx[static_cast<size_t>(j)] >= 0) {
      prev_idx[static_cast<size_t>(next_idx[static_cast<size_t>(j)])] = i;
    }
    push_pair(i);
    push_pair(prev_idx[static_cast<size_t>(i)]);
  }

  // Surviving symbols in left-to-right order (position 0 is never merged away,
  // so it heads the chain); the id conversion below consumes this unchanged.
  std::vector<std::string> symbols;
  symbols.reserve(static_cast<size_t>(n));
  for (int i = (n > 0 ? 0 : -1); i >= 0; i = next_idx[static_cast<size_t>(i)]) {
    symbols.push_back(node[static_cast<size_t>(i)]);
  }

  // Convert symbols to IDs
  std::vector<int> ids;
  ids.reserve(symbols.size());
  for (const auto &sym : symbols) {
    auto it = token_to_id.find(sym);
    if (it != token_to_id.end()) {
      ids.push_back(it->second);
    } else {
      // Fallback to bytes (shouldn't happen with proper base vocab)
      for (unsigned char c : sym) {
        std::string byte_str(1, static_cast<char>(c));
        auto byte_it = token_to_id.find(byte_str);
        if (byte_it != token_to_id.end()) {
          ids.push_back(byte_it->second);
        }
      }
    }
  }

  return ids;
}

std::string Tokenizer::decode(const std::vector<int> &ids) const {
  std::string text;
  // Reserve space to avoid reallocations
  size_t total_len = 0;
  for (int id : ids) {
    auto it = id_to_token.find(id);
    if (it != id_to_token.end()) {
      if (it->second.length() > kMaxTokenizerDecodeBytes -
                                    (std::min)(total_len, kMaxTokenizerDecodeBytes)) {
        throw std::invalid_argument("Tokenizer decoded output exceeds configured byte limit");
      }
      total_len += it->second.length();
    }
  }
  text.reserve(total_len);

  for (int id : ids) {
    auto it = id_to_token.find(id);
    if (it != id_to_token.end()) {
      text += it->second;
    }
  }
  return sanitize_utf8(text);
}

} // namespace nsos
