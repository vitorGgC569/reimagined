#include "../include/tokenizer.h"
#include <cassert>
#include <iostream>
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

int main() {
  try {
    test_basic_encoding();
    test_special_tokens();
    test_decode_sanitizes_invalid_utf8();
    std::cout << "\nAll Tokenizer tests passed!" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Test failed: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
