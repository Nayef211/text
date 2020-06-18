import csv
import io
import os

import torch
from torch import Tensor
import torch.nn as nn

from torchtext.utils import (
    download_from_url,
    extract_archive
)


def fast_text(language="en", unk_tensor=None, root='.data'):
    r"""Create a fast text Vectors object.

    Args:
        language (str): the language to use for FastText.
        unk_tensor (Tensor): a 1d tensor representing the vector associated with an unknown token
        root: folder used to store downloaded files in (.data)

    Returns:
        Vectors: a Vectors object.

    """
    url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{}.vec'.format(language)
    vectors_file_path = os.path.join(root, os.path.basename(url) + '.pt')
    if os.path.isfile(vectors_file_path):
        return(torch.load(vectors_file_path))

    downloaded_file_path = download_from_url(url, root=root)

    tokens = []
    vectors = []
    with open(downloaded_file_path, 'r') as f1:
        for line in f1:
            tokens.append(line.split(' ', 1)[0])
            vectors.append(torch.tensor([float(c) for c in line.split(' ', 1)[1].split()], dtype=torch.float))
    
    vectors_obj = Vectors(tokens, vectors, unk_tensor=unk_tensor)
    torch.save(vectors_obj, vectors_file_path)
    return vectors_obj


def glo_ve(name="840B", dim=300, unk_tensor=None, root='.data'):
    r"""Create a GloVe Vectors object.

    Args:
        name (str): the language to use for FastText.
        unk_tensor (Tensor): a 1d tensor representing the vector associated with an unknown token.
        root: folder used to store downloaded files in (.data)

    Returns:
        Vectors: a Vectors object.

    """
    urls = {
        '42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        '840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        'twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        '6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
    }

    url = urls[name]
    vectors_file_path = os.path.join(root, 'glove.{}.{}d.pt'.format(name, str(dim)))
    if os.path.isfile(vectors_file_path):
        # print('loading from cache')
        return(torch.load(vectors_file_path))

    downloaded_file_path = download_from_url(url, root=root)
    # print('downloaded_file_path', downloaded_file_path)
    extracted_file_path = extract_archive(downloaded_file_path)[0]
    # print('extracted_file_path', extracted_file_path)

    stovec = {}
    tokens = []
    vectors = []
    with io.open(extracted_file_path, encoding="utf8") as f1:
    # with open(extracted_file_path, 'r') as f1:
        for line in f1:
            token = line.split(' ', 1)[0]
            vector = torch.tensor([float(c) for c in line.split(' ', 1)[1].split()], dtype=torch.float)
            
            # try:
            #     if isinstance(token, bytes):
            #         token = token.decode('utf-8')
            # except UnicodeDecodeError:
            #     logger.info("Skipping non-UTF8 token {}".format(repr(word)))
            #     print("Current line:", len(vectors))
            #     continue

            if token in stovec:
                print("Existing vector:", stovec[token][1][:20])
                print("New vector:", line.split(' ', 1)[1][:20])
                print("Found dupe for token:", token)
                print("Past line:", stovec[token][0])
                print("Current line:", len(vectors))
                continue

            stovec[token] = (len(vectors), line.split(' ', 1)[1])
            tokens.append(token)
            vectors.append(vector)

    vectors_obj = Vectors(tokens, vectors, unk_tensor=unk_tensor)
    torch.save(vectors_obj, vectors_file_path)
    return vectors_obj
    

def vectors_from_file_object(file_like_object, unk_tensor=None):
    r"""Create a Vectors object from a csv file like object.

    Note that the tensor corresponding to each vector is of type `torch.float`.

    Format for csv file:
        token1,num1 num2 num3
        token2,num4 num5 num6
        ...
        token_n,num_m num_j num_k

    Args:
        file_like_object (FileObject): a file like object to read data from.
        unk_tensor (Tensor): a 1d tensor representing the vector associated with an unknown token.

    Returns:
        Vectors: a Vectors object.

    """
    readCSV = csv.reader(file_like_object, delimiter=',')

    tokens = []
    vectors = []
    for row in readCSV:
        tokens.append(row[0])
        vectors.append(torch.tensor([float(c) for c in row[1].split()], dtype=torch.float))

    return Vectors(tokens, vectors, unk_tensor=unk_tensor)


class Vectors(nn.Module):
    r"""Creates a vectors object which maps tokens to vectors.

    Arguments:
        tokens (List[str]): a list of tokens.
        vectors (List[torch.Tensor]): a list of 1d tensors representing the vector associated with each token.
        unk_tensor (torch.Tensor): a 1d tensors representing the vector associated with an unknown token.

    Raises:
        ValueError: if `vectors` is empty and a default `unk_tensor` isn't provided.
        RuntimeError: if `tokens` and `vectors` have different sizes or `tokens` has duplicates.
        TypeError: if all tensors within`vectors` are not of data type `torch.float`.
    """

    def __init__(self, tokens, vectors, unk_tensor=None):
        super(Vectors, self).__init__()

        if unk_tensor is None and not vectors:
            raise ValueError("The vectors list is empty and a default unk_tensor wasn't provided.")

        if not all(vector.dtype == torch.float for vector in vectors):
            raise TypeError("All tensors within `vectors` should be of data type `torch.float`.")

        unk_tensor = unk_tensor if unk_tensor is not None else torch.zeros(vectors[0].size(), dtype=torch.float)

        self.vectors = torch.classes.torchtext.Vectors(tokens, vectors, unk_tensor)

    @torch.jit.export
    def __getitem__(self, token: str) -> Tensor:
        r"""
        Args:
            token (str): the token used to lookup the corresponding vector.
        Returns:
            vector (Tensor): a tensor (the vector) corresponding to the associated token.
        """
        return self.vectors.GetItem(token)

    @torch.jit.export
    def __setitem__(self, token: str, vector: Tensor):
        r"""
        Args:
            token (str): the token used to lookup the corresponding vector.
            vector (Tensor): a 1d tensor representing a vector associated with the token.

        Raises:
            TypeError: if `vector` is not of data type `torch.float`.
        """
        if vector.dtype != torch.float:
            raise TypeError("`vector` should be of data type `torch.float` but it's of type " + vector.dtype)

        self.vectors.AddItem(token, vector.float())
