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


# ---

tqdm = functools.partial(_tqdm, ncols=80)

# ---


def bincount(x: int):
    return bin(x).count('1')

# ---


@attr.s
class Doc:

    name:         str = attr.ib()
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

    @staticmethod
    def create(
            name: str,
            content: str,
            vocab: Dict[str, int],
            codemap: np.ndarray,
            distmap: np.ndarray, ):

        words = content.split(' ')
        filtered = [word for word in words if word in vocab]

        tokens = [token.lower().strip() for token in filtered]
        selection = [vocab[token] for token in tokens]

        return Doc(
            name=name,
            tokens=tokens,
            codes=codemap[selection],
            dists=distmap[selection], )


# ---


def _assert_hamming_input(code1, code2):
    assert code1.shape == code2.shape
    assert code1.dtype == np.uint8
    assert len(code1.shape) == 1


def hamming_bincount(code1: '(bits, )', code2: '(bits, )') -> int:
    _assert_hamming_input(code1, code2)

    dist = 0
    for char in code1 ^ code2:
        dist += bincount(char)

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


_hamming_lookup = np.array([bincount(x) for x in range(0x100)])


def hamming_lookup(code1: '(bits, )', code2: '(bits, )') -> int:
    _assert_hamming_input(code1, code2)
    return _hamming_lookup[code1 ^ code2].sum()

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

    # compute distance for every possible combination of words
    for i in range(n1):
        c1, d1 = doc1[i]

        for j in range(n2):
            c2, d2 = doc2[j]

            hamming_dist = hamming_bincount(c1, c2)
            normed = _norm_dist(hamming_dist, c_bits, 100)
            T[i][j] = normed

    return T


# dmv: distance matrix vectorized


_DMV_MESHGRID_SIZE = int(1e4)
_dmv_meshgrid = np.meshgrid(
    np.arange(_DMV_MESHGRID_SIZE),
    np.arange(_DMV_MESHGRID_SIZE), )


_dmv_vectorized_bincount = np.vectorize(lambda x: bin(x).count('1'))


def distance_matrix_vectorized(doc1: Doc, doc2: Doc) -> '(n1, n2)':

    idx_y = _dmv_meshgrid[0][:len(doc1), :len(doc2)]
    idx_x = _dmv_meshgrid[1][:len(doc1), :len(doc2)]

    # FIXME: why is there a second array returned?
    C1 = doc1[idx_x][0]
    C2 = doc2[idx_y][0]

    T = _dmv_vectorized_bincount(C1 ^ C2).sum(axis=-1)
    return T / (doc1.codes.shape[1] * 8)


def distance_matrix_lookup(doc1: Doc, doc2: Doc) -> '(n1, n2)':

    idx_y = _dmv_meshgrid[0][:len(doc1), :len(doc2)]
    idx_x = _dmv_meshgrid[1][:len(doc1), :len(doc2)]

    # FIXME: why is there a second array returned?
    C1 = doc1[idx_x][0]
    C2 = doc2[idx_y][0]

    T = _hamming_lookup[C1 ^ C2].sum(axis=-1)
    return T / (doc1.codes.shape[1] * 8)


# ---


def __print_hamming_nn(doc1, doc2, min_idx, min_dist):
    assert len(doc1.tokens) == len(min_idx)
    assert len(doc1.tokens) == len(min_dist)

    nn = [doc2.tokens[idx] for idx in min_idx]

    tab_data = sorted(zip(doc1.tokens, nn, min_dist), key=lambda t: t[2])
    print('\n', tabulate(tab_data, headers=['token', 'neighbour', 'distance']))


def dist(doc1: Doc, doc2: Doc) -> float:
    T = distance_matrix_lookup(doc1, doc2)

    doc1_idx = np.argmin(T, axis=1)
    doc2_idx = np.argmin(T, axis=0)

    assert doc1.codes.shape[0] == doc1_idx.shape[0]
    assert doc2.codes.shape[0] == doc2_idx.shape[0]

    # note: this returns the _first_ occurence if there
    # are multiple codes with the same distance
    doc1_dists = T[np.arange(T.shape[0]), doc1_idx]
    doc2_dists = T.T[np.arange(T.shape[1]), doc2_idx]

    print('\nhamming distances:'.upper())
    if VERBOSE:
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


def __print_doc(doc, vocab, meta):
    print('\ndocument of length: {}'.format(len(doc)))

    tab_data = []

    header_knn = tuple('{}-nn'.format(k) for k in meta['knn'])
    header = ('word', 'index', 'code sum', ) + header_knn

    assert doc.codes.shape[0] == doc.dists.shape[0]

    for token, code, dist in zip(doc.tokens, doc.codes, doc.dists):
        assert code.shape[0] == doc.codes.shape[1]

        idx = vocab[token]
        tab_data.append((token, idx, code.sum()) + tuple(dist))

    print('\n', tabulate(tab_data, headers=header), '\n')


def _gen_combinations(pool: List[Doc]):
    for i in range(len(pool)):
        for doc in pool[i:]:
            yield pool[i], doc


def calculate_distances(vocab, codemap, distmap, meta):

    docs = []
    for fname in args.docs:
        with open(fname, 'r') as fd:
            doc = Doc.create(fd.name, fd.read(), vocab, codemap, distmap)
            docs.append(doc)

        if VERBOSE:
            print('\nloaded document:'.upper(), doc)
            for doc in docs:
                __print_doc(doc, vocab, meta)

    for doc1, doc2 in _gen_combinations(docs):
        score = dist(doc1, doc2)
        print('\nscore: {}'.format(score))
        break


def main(args):
    global VERBOSE
    VERBOSE = args.verbose

    print('\n', 'welcome to vngol wmd'.upper(), '\n')
    print('please note: binary data loaded is not checked for malicious')
    print('content - never load anything you did not produce!\n')

    vocab = _load_vocabulary(args.vocabulary)
    meta, distmap, codemap = embcodr.load_codes_bin(args.codemap)

    assert 'knn' in meta
    assert len(vocab) == distmap.shape[0]
    assert len(vocab) == codemap.shape[0]

    assert args.docs, 'no documents provided'

    calculate_distances(vocab, codemap, distmap, meta)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'codemap', type=str,
        help='binary code file produced by ungol-models/ungol.embcodr', )

    parser.add_argument(
        'vocabulary', type=str,
        help='pickle produced by ungol-models/ungol.analyze.vocabulary', )

    parser.add_argument(
        '--docs', type=str, nargs='+',
        help='documents to compare'
    )

    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='verbose output with tables and such'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
