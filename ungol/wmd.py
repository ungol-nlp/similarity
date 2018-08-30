#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import h5py
import numpy as np
from tqdm import tqdm as _tqdm
from tabulate import tabulate

import pickle
import argparse
import functools

from typing import Dict
from typing import List


# ---

tqdm = functools.partial(_tqdm, ncols=80)

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


def transport_matrix_loop(
        doc1: '(n1, bits)',
        doc2: '(n2, bits)') -> '(n1, n2)':

    n1, n2 = doc1.shape[0], doc2.shape[0]
    T = np.zeros((n1, n2))

    # target matrix is of shape (n1, n2) -> build
    # a meshgrid for all possible index combinations
    # grid: '(n1, n2)' = np.meshgrid(np.arange(n1), np.arange(n2))

    # compute distance for every possible combination of words
    for i in range(n1):
        c1 = doc1[i]
        for j in range(n2):
            c2 = doc2[j]
            T[i][j] = hamming_bincount(c1, c2)

    return T


def dist(doc1: np.ndarray, doc2: np.ndarray) -> float:
    T = transport_matrix_loop(doc1, doc2)

    doc1_min = np.argmin(T, axis=1)
    doc2_min = np.argmin(T, axis=0)

    assert doc1.shape[0] == doc1_min.shape[0]
    assert doc2.shape[0] == doc2_min.shape[0]

    # note: this returns the _first_ occurence if there
    # are multiple codes with the same distance

    print(T)
    print('\ndoc1', doc1_min)
    print('dists', T[np.arange(T.shape[0]), doc1_min])

    print('doc1', doc2_min)
    print('dists', T.T[np.arange(T.shape[1]), doc2_min])


def _load_vocabulary(fname: str) -> Dict[str, int]:
    with open(fname, 'rb') as fd:
        vocab = pickle.load(fd)

    assert len(vocab)
    print('read vocabulary')
    return vocab


def _load_codes(fname: str, bits: int, words: int) -> np.ndarray:
    with open(fname, 'rb') as fd:
        buf = fd.read()

    raw = np.frombuffer(buf, dtype=np.uint8)
    codes = raw.reshape(-1, bits // 8)

    print('read {} codes'.format(len(codes)))
    assert words == codes.shape[0], 'words=({}) =/= codes({})'.format(
        words, codes.shape[0])

    return codes


def _map_to_codes(
        words: List[str],
        vocab: Dict[str, int],
        codes: np.array) -> np.ndarray:

    mapping = [codes[vocab[word]] for word in words]
    return np.vstack(mapping)


def __print_doc(tokens, vocab, codes, __ranges):
    print('document of length: {}'.format(len(tokens)))

    tab_data = []
    header = ('word', 'index', 'code sum', 'knn start', 'knn end')

    for token, code in zip(tokens, codes):
        assert code.shape[0] == codes.shape[1]

        idx = vocab[token]
        tab_data.append((token, idx, code.sum(),
                         __ranges[0][idx], __ranges[1][idx]))

    print('\n', tabulate(tab_data, headers=header), '\n')


def main(args):
    print('\n', 'welcome to vngol wmd'.upper(), '\n')

    tokens1 = 'hallo', 'welt', 'wie', 'gehen'
    tokens2 = 'moin', 'ich', 'laufe', 'gern', 'kühlschrank'

    vocab = _load_vocabulary(args.vocabulary)
    codes = _load_codes(args.codes, args.bits, len(vocab))

    assert len(vocab) == codes.shape[0]

    # --- tmp

    print('__loading ranges')
    __KNN = 50
    __fd = h5py.File(args.knn, 'r')
    __ranges = __fd['dists'][:, 1], __fd['dists'][:, __KNN]

    # ---

    doc1 = _map_to_codes(tokens1, vocab, codes)
    doc2 = _map_to_codes(tokens2, vocab, codes)

    __print_doc(tokens1, vocab, doc1, __ranges)
    __print_doc(tokens2, vocab, doc2, __ranges)

    assert doc1.shape[0] == len(doc1)
    assert doc2.shape[0] == len(doc2)

    dist(doc1, doc2)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'codes', type=str,
        help='binary code file produced by ungol-models/ungol.embcodr', )

    parser.add_argument(
        'bits', type=int,
        help='how many bits the codes have', )

    parser.add_argument(
        'vocabulary', type=str,
        help='pickle produced by ungol-models/ungol.analyze.vocabulary', )

    # This file is included temporarily to play around with
    # thresholding/weighting by distance. It may be included in the
    # binary code file at some later point.
    parser.add_argument(
        'knn', type=str,
        help='knn file TEMPORARY!')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
