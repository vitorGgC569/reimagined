#include "../include/tokenizer.h"
#include <cassert>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace nsos;

void test_basic_encoding() {
  std::cout << "Testing basic encoding..." << std::endl;
  Tokenizer tok;

  // Add some BPE ranks manually for testing
  // vocab: a, b, c, ab, abc
  tok.bpe_ranks[{"a", "b"}] = 0;
  tok.bpe_ranks[{"ab", "c"}] = 1;

  // We need to ensure that ab and abc are in token_to_id
  tok.token_to_id["ab"] = 256;
  tok.id_to_token[256] = "ab";
  tok.token_to_id["abc"] = 257;
  tok.id_to_token[257] = "abc";
  tok.vocab_size = 258;

  std::vector<int> ids = tok.encode("abc");
  // Should merge a+b -> ab, then ab+c -> abc
  assert(ids.size() == 1);
  assert(ids[0] == 257);

  std::string decoded = tok.decode(ids);
  assert(decoded == "abc");
  std::cout << "OK" << std::endl;
}

void test_special_tokens() {
  std::cout << "Testing special tokens..." << std::endl;
  Tokenizer tok;
  tok.add_special_tokens({"<|endoftext|>", "<|system|>"});

  tok.bpe_ranks[{"h", "e"}] = 0;
  tok.token_to_id["he"] = 258;
  tok.id_to_token[258] = "he";

  std::string text = "<|system|>hello<|endoftext|>";
  std::vector<int> ids = tok.encode(text);

  // <|system|> (257), h+e->he (258), l (108), l (108), o (111), <|endoftext|>
  // (256) IDs: [257, 258, 108, 108, 111, 256]
  assert(ids.size() == 6);
  assert(ids[0] == 257);
  assert(ids[1] == 258);
  assert(ids[ids.size() - 1] == 256);

  assert(tok.decode(ids) == text);
  std::cout << "OK" << std::endl;
}

void test_decode_sanitizes_invalid_utf8() {
  std::cout << "Testing invalid UTF-8 decode sanitization..." << std::endl;
  Tokenizer tok;
  std::string decoded = tok.decode({255});
  assert(decoded == "\xEF\xBF\xBD");
  std::cout << "OK" << std::endl;
}

void test_utf8_word_merges() {
  std::cout << "Testing UTF-8 word merges..." << std::endl;
  Tokenizer tok;
  const std::string o_acute = "\xC3\xB3";
  tok.bpe_ranks[{std::string("\xC3", 1), std::string("\xB3", 1)}] = 0;
  tok.token_to_id[o_acute] = 256;
  tok.id_to_token[256] = o_acute;
  tok.vocab_size = 257;

  std::vector<int> ids = tok.encode(o_acute);
  assert(ids.size() == 1);
  assert(ids[0] == 256);
  assert(tok.decode(ids) == o_acute);
  std::cout << "OK" << std::endl;
}

// Reference implementation: the ORIGINAL O(n^2) BPE merge, kept here to prove
// the production efficient bpe_encode_word (heap + linked list) is byte-identical.
// Uses the tokenizer's public maps so it sees the exact same merge table.
static std::vector<int> naive_bpe_reference(const Tokenizer &tok,
                                            const std::string &word) {
  if (word.empty()) return {};
  std::vector<std::string> symbols;
  for (unsigned char c : word) symbols.emplace_back(1, static_cast<char>(c));
  while (symbols.size() >= 2) {
    int best_rank = std::numeric_limits<int>::max();
    size_t best_idx = static_cast<size_t>(-1);
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      auto it = tok.bpe_ranks.find({symbols[i], symbols[i + 1]});
      if (it != tok.bpe_ranks.end() && it->second < best_rank) {
        best_rank = it->second;
        best_idx = i;
      }
    }
    if (best_idx == static_cast<size_t>(-1)) break;
    symbols[best_idx] += symbols[best_idx + 1];
    symbols.erase(symbols.begin() + best_idx + 1);
  }
  std::vector<int> ids;
  for (const auto &s : symbols) {
    auto it = tok.token_to_id.find(s);
    if (it != tok.token_to_id.end()) {
      ids.push_back(it->second);
    } else {
      for (unsigned char c : s) {
        auto b = tok.token_to_id.find(std::string(1, static_cast<char>(c)));
        if (b != tok.token_to_id.end()) ids.push_back(b->second);
      }
    }
  }
  return ids;
}

void test_bpe_efficient_matches_naive() {
  std::cout << "Testing efficient BPE byte-identity vs naive reference..."
            << std::endl;
  Tokenizer tok;
  std::mt19937 rng(12345u);
  std::uniform_int_distribution<int> letter(0, 25);
  int next_id = 256;
  int rank = 0;
  std::vector<std::string> pieces;
  for (char c = 'a'; c <= 'z'; ++c) pieces.emplace_back(1, c);
  auto add_merge = [&](const std::string &x, const std::string &y) {
    if (tok.bpe_ranks.count({x, y})) return;
    tok.bpe_ranks[{x, y}] = rank++;
    const std::string merged = x + y;
    if (!tok.token_to_id.count(merged)) {
      tok.token_to_id[merged] = next_id;
      tok.id_to_token[next_id] = merged;
      pieces.push_back(merged);
      ++next_id;
    }
  };
  // Level 1: letter bigrams.  Level 2: an existing piece + a letter (creates
  // overlapping, multi-level merges that exercise tie-breaking + staleness).
  for (int k = 0; k < 150; ++k)
    add_merge(std::string(1, char('a' + letter(rng))),
              std::string(1, char('a' + letter(rng))));
  for (int k = 0; k < 150; ++k) {
    const std::string &x =
        pieces[std::uniform_int_distribution<size_t>(0, pieces.size() - 1)(rng)];
    add_merge(x, std::string(1, char('a' + letter(rng))));
  }
  tok.vocab_size = next_id;

  std::uniform_int_distribution<int> len_dist(1, 300);
  for (int trial = 0; trial < 500; ++trial) {
    const int len = len_dist(rng);
    std::string word;
    word.reserve(static_cast<size_t>(len));
    for (int i = 0; i < len; ++i) word.push_back(char('a' + letter(rng)));
    const std::vector<int> got = tok.encode(word);
    const std::vector<int> ref = naive_bpe_reference(tok, word);
    if (got != ref) {
      std::cerr << "BPE mismatch (len " << len << "): efficient " << got.size()
                << " ids vs naive " << ref.size() << std::endl;
      assert(false && "efficient BPE diverged from naive reference");
    }
  }
  std::cout << "OK (500 random words up to len 300, byte-identical)" << std::endl;
}

int main() {
  try {
    test_basic_encoding();
    test_special_tokens();
    test_decode_sanitizes_invalid_utf8();
    test_utf8_word_merges();
    test_bpe_efficient_matches_naive();
    std::cout << "\nAll Tokenizer tests passed!" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Test failed: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
