#include "../include/tokenizer.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

namespace nsos {

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
    if (token_to_id.find(t) == token_to_id.end()) {
      int id = vocab_size++;
      token_to_id[t] = id;
      id_to_token[id] = t;
      special_tokens.insert(t);
    }
  }
  // Rebuild sorted specials cache (longest first for greedy matching)
  sorted_specials.clear();
  for (const auto &t : special_tokens) {
    sorted_specials.push_back(t);
  }
  std::sort(sorted_specials.begin(), sorted_specials.end(),
            [](const std::string &a, const std::string &b) {
              return a.length() > b.length();
            });
}

void Tokenizer::load(const std::string &path) {
  // Clear previous state (strictly resetting metadata)
  bpe_ranks.clear();
  token_to_id.clear();
  id_to_token.clear();
  special_tokens.clear();
  sorted_specials.clear();

  // Reinitialize base vocabulary
  for (int i = 0; i < 256; ++i) {
    std::string s(1, static_cast<char>(i));
    token_to_id[s] = i;
    id_to_token[i] = s;
  }
  vocab_size = 256;

  std::cout << "[Tokenizer] Loading BPE model from " << path << "..."
            << std::endl;

  // Check if it's .ox3 or text
  if (path.length() >= 4 && path.substr(path.length() - 4) == ".ox3") {
    load_ox3(path);
  } else {
    load_text(path);
  }
  std::cout << "[Tokenizer] Loaded vocab size: " << vocab_size << std::endl;
}

void Tokenizer::load_text(const std::string &vocab_path) {
  std::ifstream f(vocab_path);
  if (!f.is_open()) {
    std::cerr << "[Tokenizer] Error: Could not open " << vocab_path
              << std::endl;
    return;
  }

  std::string line;
  int rank = 0;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::stringstream ss(line);
    std::string p1, p2;
    ss >> p1 >> p2;

    if (!p1.empty() && !p2.empty()) {
      bpe_ranks[{p1, p2}] = rank++;
      std::string merged = p1 + p2;
      if (token_to_id.find(merged) == token_to_id.end()) {
        int id = vocab_size++;
        token_to_id[merged] = id;
        id_to_token[id] = merged;
      }
    }
  }
}

void Tokenizer::load_ox3(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    std::cerr << "[Tokenizer] Error: Could not open " << path << std::endl;
    return;
  }

  // Header: OX3\0 (4 bytes) + Version (4 bytes)
  char magic[4];
  if (!f.read(magic, 4)) {
    std::cerr << "[Tokenizer] Error: Could not read header" << std::endl;
    return;
  }
  if (std::strncmp(magic, "OX3", 3) != 0) {
    std::cerr << "[Tokenizer] Error: Invalid magic bytes" << std::endl;
    return;
  }

  uint32_t version;
  if (!f.read(reinterpret_cast<char *>(&version), 4)) {
    std::cerr << "[Tokenizer] Error: Could not read version" << std::endl;
    return;
  }

  constexpr uint32_t MAX_STRING_LEN = 1024 * 1024; // 1MB limit for safety

  while (f.peek() != EOF) {
    uint32_t rank;
    uint32_t s1_len, s2_len;

    if (!f.read(reinterpret_cast<char *>(&rank), 4))
      break;
    if (!f.read(reinterpret_cast<char *>(&s1_len), 4))
      break;

    if (s1_len > MAX_STRING_LEN || s1_len == 0) {
      if (s1_len != 0)
        std::cerr << "[Tokenizer] Error: Invalid s1 length" << std::endl;
      break;
    }

    std::vector<char> b1(s1_len);
    if (!f.read(b1.data(), s1_len))
      break;

    if (!f.read(reinterpret_cast<char *>(&s2_len), 4))
      break;

    if (s2_len > MAX_STRING_LEN || s2_len == 0) {
      if (s2_len != 0)
        std::cerr << "[Tokenizer] Error: Invalid s2 length" << std::endl;
      break;
    }

    std::vector<char> b2(s2_len);
    if (!f.read(b2.data(), s2_len))
      break;

    std::string s1(b1.begin(), b1.end());
    std::string s2(b2.begin(), b2.end());

    bpe_ranks[{s1, s2}] = static_cast<int>(rank);
    std::string merged = s1 + s2;
    if (token_to_id.find(merged) == token_to_id.end()) {
      int id = vocab_size++;
      token_to_id[merged] = id;
      id_to_token[id] = merged;
    }
  }
}

std::vector<int> Tokenizer::encode(const std::string &text) {
  if (text.empty())
    return {};

  std::vector<int> result;
  size_t pos = 0;

  while (pos < text.length()) {
    // Try to match special token first (longest match)
    bool matched = false;
    for (const auto &st : sorted_specials) {
      if (text.compare(pos, st.length(), st) == 0) {
        result.push_back(token_to_id[st]);
        pos += st.length();
        matched = true;
        break;
      }
    }
    if (matched)
      continue;

    // Find next segment of non-special text
    size_t segment_start = pos;
    while (pos < text.length()) {
      bool is_special_start = false;
      for (const auto &st : sorted_specials) {
        if (text.compare(pos, st.length(), st) == 0) {
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
      std::string segment = text.substr(segment_start, pos - segment_start);
      std::vector<int> segment_ids = bpe_encode_word(segment);
      result.insert(result.end(), segment_ids.begin(), segment_ids.end());
    }
  }

  return result;
}

std::vector<int> Tokenizer::bpe_encode_word(const std::string &word) {
  if (word.empty())
    return {};

  // Initialize symbols as individual characters
  std::vector<std::string> symbols;
  symbols.reserve(word.length());
  for (unsigned char c : word) {
    symbols.emplace_back(1, static_cast<char>(c));
  }

  // BPE merge loop
  while (symbols.size() >= 2) {
    // Find the pair with minimum rank (highest priority)
    int best_rank = std::numeric_limits<int>::max();
    size_t best_idx = static_cast<size_t>(-1);

    for (size_t i = 0; i < symbols.size() - 1; ++i) {
      auto it = bpe_ranks.find({symbols[i], symbols[i + 1]});
      if (it != bpe_ranks.end() && it->second < best_rank) {
        best_rank = it->second;
        best_idx = i;
      }
    }

    if (best_idx == static_cast<size_t>(-1))
      break; // No more merges possible

    // Perform merge
    symbols[best_idx] += symbols[best_idx + 1];
    symbols.erase(symbols.begin() + best_idx + 1);
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

std::string Tokenizer::decode(const std::vector<int> &ids) {
  std::string text;
  // Reserve space to avoid reallocations
  size_t total_len = 0;
  for (int id : ids) {
    auto it = id_to_token.find(id);
    if (it != id_to_token.end()) {
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
  return text;
}

} // namespace nsos
