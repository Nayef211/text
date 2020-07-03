import logging
from typing import Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Vocab(nn.Module):
    r"""Creates a vocab object which maps tokens to indices.

    Arguments:
        ordered_dict (collections.OrderedDict): object holding the frequencies of each token found in the data.
        min_freq: The minimum frequency needed to include a token in the vocabulary.
            Values less than 1 will be set to 1. Default: 1.
        specials: The tuple of special tokens (e.g., padding or eos) that will be prepended/postpended to the vocabulary.
            based on the `specials_first` flag. The ordering of the tuple will be preserved. Default: [<pad>']
        specials_first: Whether to add special tokens into the vocabulary at first. If it is False,
            they are added into the vocabulary at last. Default: True.

    Raises:
        ValueError: if a default `unk_token` isn't provided.
    """

    def __init__(self, ordered_dict, min_freq=1, unk_token='<unk>', specials=('<pad>',), specials_first=True):
        super(Vocab, self).__init__()

        if not unk_token:
            raise ValueError("A default unk token wasn't provided.")

        tokens = []
        for token, freq in ordered_dict.items():
            if freq >= min_freq:
                tokens.append(token)

        # assume unk_token and special tokens dont appear in ordered_dict
        if specials_first:
            tokens = [unk_token] + list(specials) + tokens
        else:
            tokens += [unk_token] + list(specials)

        self.vocab = torch.classes.torchtext.Vocab(tokens, unk_token)

    @torch.jit.export
    def __len__(self) -> int:
        r"""Returns:
            length (int): the length of the vocab
        """
        return len(self.vocab)

    @torch.jit.export
    def __getitem__(self, token: str) -> int:
        r"""
        Args:
            token (str): the token used to lookup the corresponding index.

        Returns:
            index (int): the index corresponding to the associated token.
        """
        return self.vocab[token]

    @torch.jit.export
    def __setitem__(self, token: str, index: int) -> None:
        r"""
        Args:
            token (str): the token used to lookup the corresponding index.
            index (int): the index corresponding to the associated token.

        Raises:
            RuntimeError: if `index` not between [0, Vocab.size()] or if token already exists in the vocab.
        """
        self.vocab[token] = index

    @torch.jit.export
    def add_token(self, token: str) -> None:
        r"""
        Args:
            token (str): the token used to lookup the corresponding index.
            index (int): the index corresponding to the associated token.
        """
        self.vocab.add_token(token)

    @torch.jit.export
    def lookup_token(self, index: int) -> str:
        r"""
        Args:
            index (int): the index corresponding to the associated token.

        Returns:
            token (str): the token used to lookup the corresponding index.

        Raises:
            RuntimeError: if `index` not between [0, itos.size()].
        """
        return self.vocab.lookup_token(index)

    @torch.jit.export
    def lookup_tokens(self, indices: List[int]) -> List[str]:
        r"""
        Args:
            indices (List[int]): the `indices` used to lookup their corresponding`tokens`.

        Returns:
            tokens (List[str]): the `tokens` associated with `indices`.

        Raises:
            RuntimeError: if an index within `indices` is not between [0, itos.size()].
        """
        return self.vocab.lookup_tokens(indices)

    @torch.jit.export
    def lookup_indices(self, tokens: List[str]) -> List[int]:
        r"""
        Args:
            tokens (List[str]): the tokens used to lookup their corresponding `indices`.

        Returns:
            indices (List[int]): the 'indices` associated with `tokens`.
        """
        return self.vocab.lookup_indices(tokens)

    @torch.jit.export
    def get_stoi(self) -> Dict[str, int]:
        r"""
        Returns:
            stoi (dict): dictionary mapping tokens to indices.
        """
        return self.vocab.get_stoi()

    @torch.jit.export
    def get_itos(self) -> List[str]:
        r"""
        Returns:
            stoi (dict): dictionary mapping indices to tokens.
        """
        return self.vocab.get_itos()
