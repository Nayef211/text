from collections import (Counter, OrderedDict)
import time
from typing import List

import torch
from torchtext.experimental.datasets import AG_NEWS
from torchtext.experimental.vocab import Vocab as VocabExperimental


def _run_benchmark_lookup(tokens, vocab: VocabExperimental, num_iters=1):
    t0 = time.monotonic()
    for _ in range(num_iters):
        # list lookup
        if isinstance(tokens, list) and isinstance(tokens[0], list):
            for tokens_list in tokens:
                vocab.lookup_indices(tokens_list)
        # single token lookup
        elif isinstance(tokens, list):
            for token in tokens:
                vocab[token]
        else:
            raise RuntimeError("Received tokens of incorrect type {}.".format(type(toks)))
    print("Lookup time:", time.monotonic() - t0)


def _run_benchmark_lookup_jit_for_loop(tokens, vocab, num_iters=1):
    @torch.jit.script
    def _run_benchmark_single_token(toks: List[str], v: VocabExperimental):
        for token in toks:
            v[token]

    @torch.jit.script
    def _run_benchmark_lists(tok_lists: List[List[str]], v: VocabExperimental):
        for tokens_list in tok_lists:
            v.lookup_indices(tokens_list)

    t0 = time.monotonic()
    # list lookup
    if isinstance(tokens, list) and isinstance(tokens[0], list):
        for _ in range(num_iters):
            _run_benchmark_lists(tokens, vocab)
    # single token lookup
    elif isinstance(tokens, list):
        for _ in range(num_iters):
            _run_benchmark_single_token(tokens, vocab)
    else:
        raise RuntimeError("Received tokens of incorrect type {}.".format(type(tokens)))
    print("Lookup time:", time.monotonic() - t0)


def benchmark_experimental_vocab():
    train, = AG_NEWS(data_select='train')
    vocab = train.get_vocab()
    tokens: List[str] = []
    tokens_lists: List[List[str]] = []

    for (_, text) in train:
        cur_tokens = []
        for id in text.tolist():
            cur_tokens.append(vocab.itos[id])
        tokens_lists.append(cur_tokens)
        tokens += cur_tokens

    print("Tokens size:", len(tokens))
    print("Tokens list size:", len(tokens_lists))

    counter = Counter(tokens)
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    # experimental vocab construction
    print("Experimental Vocabulary")
    t0 = time.monotonic()
    v_experimental = VocabExperimental(ordered_dict)
    print("Construction time:", time.monotonic() - t0)

    jit_v_experimental = torch.jit.script(v_experimental)

    # experimental Vocab eager lookup
    print("Experimental Vocabulary - Eager Mode")
    _run_benchmark_lookup(tokens, v_experimental)
    _run_benchmark_lookup([tokens], v_experimental)
    _run_benchmark_lookup(tokens_lists, v_experimental)

    # experimental Vocab jit lookup
    print("Experimental Vocabulary - Jit Mode")
    _run_benchmark_lookup(tokens, jit_v_experimental)
    _run_benchmark_lookup([tokens], jit_v_experimental)
    _run_benchmark_lookup(tokens_lists, jit_v_experimental)

    # experimental Vocab JITed for loop
    print("Experimental Vocabulary - Jit For Loop")
    _run_benchmark_lookup_jit_for_loop(tokens, jit_v_experimental)
    _run_benchmark_lookup_jit_for_loop([tokens], jit_v_experimental)
    _run_benchmark_lookup_jit_for_loop(tokens_lists, jit_v_experimental)


if __name__ == "__main__":
    benchmark_experimental_vocab()
