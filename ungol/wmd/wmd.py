#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from ungol.models import embcodr

import attr
import numpy as np
from tqdm import tqdm as _tqdm
from tabulate import tabulate

import pickle
import argparse
import functools

from typing import Dict
from typing import List
from typing import Tuple


# ---

tqdm = functools.partial(_tqdm, ncols=80)

# ---


@attr.s
class Doc:

    tokens: List[str] = attr.ib()  # (n, )
    codes: np.ndarray = attr.ib()  # (n, bytes)
    dists: np.ndarray = attr.ib()  # (n, knn)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, key):
        return self.codes[key], self.dists[key]

    def __attrs_post_init__(self):
        assert len(self.tokens) == self.codes.shape[0]
        assert len(self.tokens) == self.dists.shape[0]


# ---


def _assert_hamming_input(code1, code2):
    assert code1.shape == code2.shape
    assert code1.dtype == np.uint8
    assert len(code1.shape) == 1


def hamming_bincount(code1: '(bits, )', code2: '(bits, )') -> int:
    _assert_hamming_input(code1, code2)

    dist = 0
    for char in code1 ^ code2:
        dist += bin(char).count('1')

    return dist


def hamming_bitmask(code1: '(bits, )', code2: '(bits, )') -> int:
    _assert_hamming_input(code1, code2)

    mask = 1
    dist = 0

    for char in code1 ^ code2:
        while char:
            dist += char & mask
            char >>= 1

    return dist


# ---


def __print_distance_matrix(T, doc1, doc2):

    # doc1: rows, doc2: columns
    tab_data = [('', ) + tuple(doc2.tokens)]
    for idx in range(len(doc1)):
        word = (doc1.tokens[idx], )
        dists = tuple(T[idx])
        tab_data.append(word + dists)

    print(tabulate(tab_data))


def distance_matrix_loop(doc1: Doc, doc2: Doc) -> '(n1, n2)':

    def _norm_dist(hamming_dist, c_bits: int, maxdist: int = None):
        # clip too large distance
        dist = min(hamming_dist, maxdist)

        # normalize
        normed = dist / min(c_bits, maxdist)

        assert 0 <= normed and normed <= 1
        return normed

    # ---

    n1, n2 = doc1.codes.shape[0], doc2.codes.shape[0]
    T = np.zeros((n1, n2))

    c_bits = doc1.codes.shape[1] * 8

    # target matrix is of shape (n1, n2) -> build
    # a meshgrid for all possible index combinations
    # grid: '(n1, n2)' = np.meshgrid(np.arange(n1), np.arange(n2))

    # compute distance for every possible combination of words
    for i in range(n1):
        c1, d1 = doc1[i]

        for j in range(n2):
            c2, d2 = doc2[j]

            hamming_dist = hamming_bincount(c1, c2)
            normed = _norm_dist(hamming_dist, c_bits, 100)
            T[i][j] = normed

    return T


# ---


def __print_hamming_nn(doc1, doc2, min_idx, min_dist):
    assert len(doc1.tokens) == len(min_idx)
    assert len(doc1.tokens) == len(min_dist)

    nn = [doc2.tokens[idx] for idx in min_idx]

    tab_data = sorted(zip(doc1.tokens, nn, min_dist), key=lambda t: t[2])
    print('\n', tabulate(tab_data, headers=['token', 'neighbour', 'distance']))


def dist(doc1: Doc, doc2: Doc) -> float:
    T = distance_matrix_loop(doc1, doc2)

    doc1_idx = np.argmin(T, axis=1)
    doc2_idx = np.argmin(T, axis=0)

    assert doc1.codes.shape[0] == doc1_idx.shape[0]
    assert doc2.codes.shape[0] == doc2_idx.shape[0]

    # note: this returns the _first_ occurence if there
    # are multiple codes with the same distance
    doc1_dists = T[np.arange(T.shape[0]), doc1_idx]
    doc2_dists = T.T[np.arange(T.shape[1]), doc2_idx]

    print('\nhamming distances:'.upper())
    __print_hamming_nn(doc1, doc2, doc1_idx, doc1_dists)
    __print_hamming_nn(doc2, doc1, doc2_idx, doc2_dists)

    score = max(doc1_dists.mean(), doc2_dists.mean())
    return score


def _load_vocabulary(fname: str) -> Dict[str, int]:
    with open(fname, 'rb') as fd:
        vocab = pickle.load(fd)

    assert len(vocab)
    print('read vocabulary')
    return vocab


def _map_to_codes(
        tokens: List[str],
        vocab: Dict[str, int],
        codes: np.array,
        dists: np.array) -> Tuple['codes', 'dists']:

    selection = [vocab[token] for token in tokens]

    return Doc(
        tokens=tokens,
        codes=codes[selection],
        dists=dists[selection], )


def __print_doc(tokens, vocab, meta, doc):
    print('\ndocument of length: {}'.format(len(tokens)))

    tab_data = []

    header_knn = tuple('{}-nn'.format(k) for k in meta['knn'])
    header = ('word', 'index', 'code sum', ) + header_knn

    assert doc.codes.shape[0] == len(tokens)
    assert doc.codes.shape[0] == doc.dists.shape[0]

    for token, code, dist in zip(tokens, doc.codes, doc.dists):
        assert code.shape[0] == doc.codes.shape[1]

        idx = vocab[token]
        tab_data.append((token, idx, code.sum()) + tuple(dist))

    print('\n', tabulate(tab_data, headers=header), '\n')


def main(args):
    print('\n', 'welcome to vngol wmd'.upper(), '\n')
    print('please note: binary data loaded is not checked for malicious')
    print('content - never load anything you did not produce!\n')

    tokens1 = 'hallo', 'welt', 'wie', 'gehen'
    tokens2 = 'moin', 'ich', 'laufe', 'gern', 'kühlschrank'

    vocab = _load_vocabulary(args.vocabulary)
    meta, dists, codes = embcodr.load_codes_bin(args.codes)

    assert 'knn' in meta
    assert len(vocab) == dists.shape[0]
    assert len(vocab) == codes.shape[0]

    doc1 = _map_to_codes(tokens1, vocab, codes, dists)
    doc2 = _map_to_codes(tokens2, vocab, codes, dists)

    print('\ndocuments:'.upper())
    __print_doc(tokens1, vocab, meta, doc1)
    __print_doc(tokens2, vocab, meta, doc2)

    score = dist(doc1, doc2)
    print('\nscore: {}'.format(score))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'codes', type=str,
        help='binary code file produced by ungol-models/ungol.embcodr', )

    parser.add_argument(
        'vocabulary', type=str,
        help='pickle produced by ungol-models/ungol.analyze.vocabulary', )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
