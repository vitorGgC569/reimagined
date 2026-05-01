#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <cstddef>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nsos {

struct PairHash {
  std::size_t operator()(const std::pair<std::string, std::string> &p) const {
    return std::hash<std::string>{}(p.first) ^
           std::hash<std::string>{}(p.second);
  }
};

class Tokenizer {
public:
  std::unordered_map<std::string, int> token_to_id;
  std::unordered_map<int, std::string> id_to_token;
  std::unordered_map<std::pair<std::string, std::string>, int, PairHash>
      bpe_ranks;
  std::set<std::string> special_tokens;
  std::vector<std::string> sorted_specials;
  int vocab_size;

  Tokenizer();
  void load(const std::string &path);
  void load_text(const std::string &path);
  void load_ox3(const std::string &path);
  void load_pack(const std::string &path);
  void save_pack(const std::string &path) const;

  // Special Tokens
  void add_special_tokens(const std::vector<std::string> &tokens);

  std::vector<int> encode(const std::string &text);
  std::string decode(const std::vector<int> &ids) const;

private:
  std::vector<int> bpe_encode_word(const std::string &word);
};

} // namespace nsos

#endif
