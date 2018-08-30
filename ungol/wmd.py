#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from tqdm import tqdm as _tqdm

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


def _transport_matrix(doc1: '(n1, bits)', doc2: '(n2, bits)') -> '(n1, n2)':
    print('doc1', doc1.shape)
    print('doc2', doc2.shape)

    n1, n2 = doc1.shape[0], doc2.shape[0]

    # target matrix is of shape (n1, n2) -> build a meshgrid
    # for all possible index combinations
    grid: '(n1, n2)' = np.meshgrid(np.arange(n1), np.arange(n2))

    print('>', grid[0])


def dist(doc1: np.ndarray, doc2: np.ndarray) -> float:
    T1 = _transport_matrix(doc1, doc2)

    print(T1)

    # 2. compute transport matrices

    # 3. compute wmd
    # print('-' * 40)
    # print(doc1)
    # print('-' * 40)
    # print(doc2)
    # print('-' * 40)


def _load_vocabulary(fname: str) -> Dict[str, int]:
    with open(fname, 'rb') as fd:
        vocab = pickle.load(fd)

    assert len(vocab)
    print('read vocabulary')
    return vocab


def _load_codes(fname: str, bits: int, words: int) -> np.ndarray:
    with open(fname, 'rb') as fd:
        buf = fd.read()

    # dt = np.dtype(np.uint8)
    # dt = dt.newbyteorder('>')

    raw = np.frombuffer(buf, dtype=np.uint8)
    codes = raw.reshape(-1, bits // 8)

    print('read {} codes'.format(len(codes)))
    assert words == codes.shape[0], 'words=({}) =/= codes({})'.format(
        words, codes.shape[0])

    return np.array(codes)


def _map_to_codes(
        words: List[str],
        vocab: Dict[str, int],
        codes: np.array) -> np.ndarray:

    mapping = [codes[vocab[word]] for word in words]
    return np.vstack(mapping)


def main(args):
    print('\n', 'welcome to vngol wmd'.upper(), '\n')

    tokens1 = 'hallo', 'welt', 'wie', 'gehts'
    tokens2 = 'moin', 'ich', 'laufe', 'gern', 'kühlschrank'

    vocab = _load_vocabulary(args.vocabulary)
    codes = _load_codes(args.codes, args.bits, len(vocab))

    assert len(vocab) == codes.shape[0]

    doc1 = _map_to_codes(tokens1, vocab, codes)
    doc2 = _map_to_codes(tokens2, vocab, codes)

    assert doc1.shape[0] == len(doc1)
    assert doc2.shape[0] == len(doc2)

    dist(doc1, doc2)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'codes', type=str, nargs=1,
        help='binary code file produced by ungol-models/ungol.embcodr', )

    parser.add_argument(
        'bits', type=int, nargs=1,
        help='how many bits the codes have', )

    parser.add_argument(
        'vocabulary', type=str,
        help='pickle produced by ungol-models/ungol.analyze.vocabulary', )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
