# =*= coding: utf-8 -*-

from ungol.wmd import wmd

import numpy as np
from tabulate import tabulate

import enum


def bincount(x: int):
    return bin(x).count('1')


class Strategy(enum.Enum):

    # selecting max(score(d1, d2), score(d2, d1))
    MAX = enum.auto()

    # selecting min(score(d1, d2), score(d2, d1))
    MIN = enum.auto()

    # only use score(ds, dl), where ds = argmin(|d1|, |d2|)
    # and dl = argmax(|d1|, |d2|)
    ADAPTIVE_SMALL = enum.auto()

    # only use score(dl, ds), where ds = argmin(|d1|, |d2|)
    # and dl = argmax(|d1|, |d2|)
    ADAPTIVE_BIG = enum.auto()


#
#  -------------------- HAMMING CALCULATIONS
#


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


# FIXME: this impairs startup time, just save a big array with the values here
_hamming_lookup = np.array([bincount(x) for x in range(0x100)])


def hamming_lookup(code1: '(bits, )', code2: '(bits, )') -> int:
    _assert_hamming_input(code1, code2)
    return _hamming_lookup[code1 ^ code2].sum()


#
#  -------------------- DISTANCE CALCULATIONS
#


def __print_distance_matrix(T, doc1, doc2):

    # doc1: rows, doc2: columns
    tab_data = [('', ) + tuple(doc2.tokens)]
    for idx in range(len(doc1)):
        word = (doc1.tokens[idx], )
        dists = tuple(T[idx])
        tab_data.append(word + dists)

    print(tabulate(tab_data))


def distance_matrix_loop(doc1: wmd.Doc, doc2: wmd.Doc) -> '(n1, n2)':

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
        c1 = doc1[i]

        for j in range(n2):
            c2 = doc2[j]

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


def distance_matrix_vectorized(doc1: wmd.Doc, doc2: wmd.Doc) -> '(n1, n2)':

    idx_y = _dmv_meshgrid[0][:len(doc1), :len(doc2)]
    idx_x = _dmv_meshgrid[1][:len(doc1), :len(doc2)]

    C1 = doc1[idx_x]
    C2 = doc2[idx_y]

    T = _dmv_vectorized_bincount(C1 ^ C2).sum(axis=-1)
    return T / (doc1.codes.shape[1] * 8)


def distance_matrix_lookup(doc1: wmd.Doc, doc2: wmd.Doc) -> '(n1, n2)':

    idx_y = _dmv_meshgrid[0][:len(doc1), :len(doc2)]
    idx_x = _dmv_meshgrid[1][:len(doc1), :len(doc2)]

    C1 = doc1[idx_x]
    C2 = doc2[idx_y]

    T = _hamming_lookup[C1 ^ C2].sum(axis=-1)
    return T / (doc1.codes.shape[1] * 8)
