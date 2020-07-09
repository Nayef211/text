#pragma once

#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace torchtext {

class Dictionary {
protected:
  // static const int32_t MAX_VOCAB_SIZE = 30000000;
  static const uint32_t MAX_VOCAB_SIZE = 100000;

  uint32_t find(const std::string &) const;
  uint32_t find(const std::string &, uint32_t h) const;

  int32_t word2int_[MAX_VOCAB_SIZE];
  std::vector<std::string> words_;

  int32_t size_;

public:
  explicit Dictionary();
  int32_t size() const;
  int32_t getId(const std::string &) const;
  int32_t getId(const std::string &, uint32_t) const;
  std::string getWord(uint32_t) const;
  std::vector<std::string> getWords() const;
  uint32_t hash(const std::string &) const;
  void add(const std::string &);
  void add(const std::string &w, uint32_t);
  void insert(const std::string &, uint32_t);
  void dump(std::ostream &) const;
};
} // namespace torchtext