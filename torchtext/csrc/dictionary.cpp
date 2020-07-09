#include "dictionary.h"

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <fstream>
#include <iostream>

namespace torchtext {

Dictionary::Dictionary() : size_(0) {
  // set all indices to -1
  for (uint32_t i = 0; i < MAX_VOCAB_SIZE; i++) {
    word2int_[i] = -1;
  }
}

uint32_t Dictionary::find(const std::string &w) const {
  return find(w, hash(w));
}

uint32_t Dictionary::find(const std::string &w, uint32_t h) const {
  uint32_t id = h % MAX_VOCAB_SIZE;
  while (word2int_[id] != -1 && words_[word2int_[id]] != w) {
    id = (id + 1) % MAX_VOCAB_SIZE;
  }
  return id;
}

void Dictionary::add(const std::string &w) {
  uint32_t h = find(w);
  if (word2int_[h] == -1) {
    words_.push_back(w);
    word2int_[h] = size_++;
  }
}

void Dictionary::add(const std::string &w, uint32_t h) {
  if (word2int_[h] == -1) {
    words_.push_back(w);
    word2int_[h] = size_++;
  }
}

void Dictionary::insert(const std::string &w, uint32_t id) {
  assert(id < size_);

  // ensure word not in dictionary
  uint32_t h = find(w);
  assert(word2int_[h] == -1);

  // need to offset all words greater than index by 1
  for (uint32_t i = id; i < static_cast<uint32_t>(size_); i++) {
    h = find(words_[i]);
    word2int_[h]++;
  }

  h = find(w);
  word2int_[h] = id;
  auto it = words_.begin() + id;
  words_.insert(it, std::move(w));
  size_++;
}

int32_t Dictionary::size() const { return size_; }

int32_t Dictionary::getId(const std::string &w) const {
  uint32_t h = find(w);
  return word2int_[h];
}

int32_t Dictionary::getId(const std::string &w, uint32_t h) const {
  return word2int_[h];
}

std::string Dictionary::getWord(uint32_t id) const {
  assert(id < size_);
  return words_[id];
}

std::vector<std::string> Dictionary::getWords() const { return words_; }

uint32_t Dictionary::hash(const std::string &str) const {
  uint32_t h = 2166136261;
  for (size_t i = 0; i < str.size(); i++) {
    h = h ^ uint32_t(uint8_t(str[i]));
    h = h * 16777619;
  }
  return h;
}

void Dictionary::dump(std::ostream &out) const {
  out << words_.size() << std::endl;
  for (auto word : words_) {
    uint32_t id = getId(word);
    out << word << ": " << id << std::endl;
  }
}

} // namespace torchtext