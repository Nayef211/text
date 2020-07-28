from collections import (Counter, OrderedDict)
import time

import torch
from torchtext.experimental.datasets import AG_NEWS, DBpedia, IMDB, YahooAnswers
from torchtext.experimental.vocab import Vocab as VocabExperimental
from torchtext.vocab import Vocab


def benchmark_experimental_vocab():
    def _run_benchmark_lookup_list_tokens(tokens_lists, vocab):
        t0 = time.monotonic()
        for _ in range(1):
            for cur_tokens in tokens_lists:
                vocab.lookup_indices(cur_tokens)
        print("Lookup time:", time.monotonic() - t0)

    def _run_benchmark_lookup(tokens, vocab):
        t0 = time.monotonic()
        for _ in range(1):
            for token in tokens:
                vocab[token]
        print("Lookup time:", time.monotonic() - t0)

    train, = AG_NEWS(data_select='train')

    # all tokens
    vocab = train.get_vocab()
    tokens = []
    tokens_lists = []

    for (label, text) in train:
        cur_tokens = []
        for id in text.tolist():
            cur_tokens.append(vocab.itos[id])
        tokens_lists.append(cur_tokens)
        tokens += cur_tokens

    print("num tokens", len(tokens))
    print("num tokens list", len(tokens_lists))
    # print(tokens_lists[:3])

    counter = Counter(tokens)
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    # # existing Vocab construction
    # t0 = time.monotonic()
    # v_existing = Vocab(counter)
    # print("Construction time:", time.monotonic() - t0)

    # experimental Vocab construction
    print("Vocab Experimental")
    t0 = time.monotonic()
    v_experimental = VocabExperimental(ordered_dict)

    print("Construction time:", time.monotonic() - t0)

    # # existing Vocab not jit lookup
    # print("Vocab - Not Jit Mode")
    # _run_benchmark_lookup(tokens, v_existing)

    # # experimental Vocab not jit lookup
    # print("Vocab Experimental - Not Jit Mode")
    # _run_benchmark_lookup_list_tokens(tokens_lists, v_experimental)
    # # _run_benchmark_lookup(tokens, v_experimental)

    # experimental Vocab jit lookup
    print("Vocab Experimental - Jit Mode")
    jit_v_experimental = torch.jit.script(v_experimental)
    _run_benchmark_lookup(tokens, jit_v_experimental)
    _run_benchmark_lookup_list_tokens(tokens_lists, jit_v_experimental)


if __name__ == "__main__":
    benchmark_experimental_vocab()
