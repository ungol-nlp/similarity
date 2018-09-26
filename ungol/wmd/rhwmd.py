# =*= coding: utf-8 -*-

from ungol.wmd import wmd

import numpy as np
from tabulate import tabulate

import enum
from typing import Tuple


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

    # score(d1, d2) + score(d2, d1)
    SUM = enum.auto()


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

    # FIXME: can be faster not overwriting char (?)

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

            hamming_dist = hamming_lookup(c1, c2)
            # hamming_dist = hamming_bincount(c1, c2)
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


# ---


def retrieve_nn(doc1: wmd.Doc, doc2: wmd.Doc):
    # Compute the distance matrix.
    T = distance_matrix_lookup(doc1, doc2)

    doc1_idxs = np.argmin(T, axis=1)
    doc2_idxs = np.argmin(T.T, axis=1)

    # Select the nearest neighbours per file Note: this returns the
    # _first_ occurence if there are multiple codes with the same
    # distance (not important for further computation...)  This value
    # is inverted to further work with 'similarity' instead of
    # distance (lead to confusion formerly as to where distance ended
    # and similarity began)
    a_sims1 = 1 - T[np.arange(T.shape[0]), doc1_idxs]
    a_sims2 = 1 - T.T[np.arange(T.shape[1]), doc2_idxs]

    a_sims = a_sims1, a_sims2

    # transform
    # G = 1
    # a_sims = tuple((np.e ** (a * G)) / np.e ** G for a in (a_sims1, a_sims2))

    # G = 1.8
    # a_sims = tuple((G * a - G) for a in (a_sims1, a_sims2))
    # for a in a_sims:
    #     a[a < 0] = 0

    # from tabulate import tabulate
    # print(tabulate(zip(a_sims1, a_sims[0])))

    return a_sims, (doc1_idxs, doc2_idxs)


#
#  MOCKING SECTION - used by tests/benchmarks
#


def mock_codes(bytecount: int) -> Tuple['code1', 'code2']:
    """
    To test wmd.hamming_*
    """
    code1 = (np.random.randn(bytecount) * 255).astype(dtype=np.uint8)
    code2 = (np.random.randn(bytecount) * 255).astype(dtype=np.uint8)
    return code1, code2


def mock_doc(n: int, bytecount: int = 32) -> wmd.Doc:
    """
    To test wmd.distance_matrix_*
    """
    codemap = (np.random.randn(n, bytecount) * 255).astype(dtype=np.uint8)
    distmap = (np.random.randn(n, 1) * 255).astype(dtype=np.int)

    idx = np.arange(n).astype(np.uint)
    np.random.shuffle(idx)

    vocab = {str(i): i for i in idx}

    ref = wmd.DocReferences(
        meta={'knn': [1]},
        vocabulary=vocab,
        codemap=codemap,
        distmap=distmap, )

    doc = wmd.Doc(idx=idx[:n], cnt=np.ones(n).astype(dtype=np.uint), ref=ref)
    return doc
