#include "dictionary.h"
#include <stdexcept>
#include <string>
#include <torch/script.h>

namespace torchtext {
namespace {

using c10::Dict;

struct Vocab : torch::CustomClassHolder {
private:
  int64_t unk_index_;
  // Dict<std::string, int64_t> stoi_;

public:
  // stoi_, and unordered_map holds the serialized params passed in
  // during initialization. We need this because we need to be able to serialize
  // the model so that we can save the scripted object. Pickle will get the
  // serialized model from these members, thus they needs to be public.
  // std::vector<std::string> itos_;
  Dictionary stoi_;
  std::string unk_token_;

  explicit Vocab(const std::vector<std::string> &tokens,
                 const std::string &unk_token)
      : unk_token_(std::move(unk_token)) {
    for (std::size_t i = 0; i < tokens.size(); i++) {
      // tokens should not have any duplicates
      if (stoi_.getId(tokens[i]) != -1) {
        throw std::runtime_error("Duplicate token found in tokens list: " +
                                 tokens[i]);
      }
      stoi_.add(std::move(tokens[i]));
    }
    unk_index_ = stoi_.getId(unk_token);
  }

  int64_t __len__() const { return stoi_.size(); }

  int64_t __getitem__(const std::string &token) const {
    const int32_t index = stoi_.getId(token);
    if (index != -1) {
      return index;
    }
    return unk_index_;
  }

  void append_token(const std::string &token) { stoi_.add(std::move(token)); }

  void insert_token(const std::string &token, const int64_t &index) {
    if (index < 0 || index > stoi_.size()) {
      throw std::runtime_error(
          "Specified index " + std::to_string(index) +
          " is out of bounds of the size of stoi dictionary: " +
          std::to_string(stoi_.size()) + ".");
    }

    const int32_t id = stoi_.getId(token);
    // if item already in stoi we throw an error
    if (id != -1) {
      throw std::runtime_error("Token " + token +
                               " already exists in the Vocab with index: " +
                               std::to_string(id) + ".");
    }
    stoi_.insert(token, index);
    unk_index_ = stoi_.getId(unk_token_);
  }

  std::string lookup_token(const int64_t &index) {
    if (index < 0 || index > stoi_.size()) {
      throw std::runtime_error(
          "Specified index " + std::to_string(index) +
          " is out of bounds of the size of itos dictionary: " +
          std::to_string(stoi_.size()) + ".");
    }

    return stoi_.getWord(index);
  }

  std::vector<std::string> lookup_tokens(const std::vector<int64_t> &indices) {
    std::vector<std::string> tokens(indices.size());
    for (size_t i = 0; i < indices.size(); i++) {
      tokens[i] = lookup_token(indices[i]);
    }
    return tokens;
  }

  std::vector<int64_t> lookup_indices(const std::vector<std::string> &tokens) {
    std::vector<int64_t> indices(tokens.size());
    for (size_t i = 0; i < tokens.size(); i++) {
      indices[i] = __getitem__(tokens[i]);
    }
    return indices;
  }

  Dict<std::string, int64_t> get_stoi() const {
    Dict<std::string, int64_t> stoi;
    stoi.reserve(stoi_.size());
    for (const std::string word : stoi_.getWords()) {
      stoi.insert(word, stoi_.getId(word));
    }
    return stoi;
  }
  std::vector<std::string> get_itos() const { return stoi_.getWords(); }
};

// Registers our custom class with torch.
static auto vocab =
    torch::class_<Vocab>("torchtext", "Vocab")
        .def(torch::init<std::vector<std::string>, std::string>())
        .def("__getitem__", &Vocab::__getitem__)
        .def("__len__", &Vocab::__len__)
        .def("insert_token", &Vocab::insert_token)
        .def("append_token", &Vocab::append_token)
        .def("lookup_token", &Vocab::lookup_token)
        .def("lookup_tokens", &Vocab::lookup_tokens)
        .def("lookup_indices", &Vocab::lookup_indices)
        .def("get_stoi", &Vocab::get_stoi)
        .def("get_itos", &Vocab::get_itos)
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<Vocab> &self)
                -> std::tuple<std::vector<std::string>, std::string> {
              std::tuple<std::vector<std::string>, std::string> states(
                  self->stoi_.getWords(), self->unk_token_);
              return states;
            },
            // __setstate__
            [](std::tuple<std::vector<std::string>, std::string> states)
                -> c10::intrusive_ptr<Vocab> {
              return c10::make_intrusive<Vocab>(std::move(std::get<0>(states)),
                                                std::move(std::get<1>(states)));
            });
} // namespace
} // namespace torchtext
