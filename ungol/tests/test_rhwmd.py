# -*- coding: utf-8

from ungol.similarity import rhwmd
from ungol.index import index as uii

import numpy as np

from typing import Callable


DistanceMatrix = Callable[[uii.Doc, uii.Doc], np.array]


def _distance_matrix_test(fn: DistanceMatrix):

    vocab = {'non': 0, 'all': 1, 'som': 2, 'non2': 3}

    codemap = np.array([
        [0x00, 0x00],
        [0xff, 0xff],
        [0x00, 0xff],
        [0x00, 0x00],
    ]).astype(np.uint8)

    # # whatever the normalization
    # # step does: providing the whole
    # # domain should be robust enough
    # distmap = np.array([
    #     [0x0, 2 * 0xff],
    #     [0x0, 2 * 0xff],
    #     [0x0, 2 * 0xff],
    #     [0x0, 2 * 0xff],
    # ]).astype(np.uint8)

    ref = uii.References(
        meta={'knn': ['1', '∞']},
        vocabulary=vocab,
        stopwords=set(),
        codemap=codemap,
    )

    # using different sizes to make sure
    # no dimensions are switched by accident
    doc1 = uii.Doc(idx=np.array([0, 1, 2]).astype(np.uint),
                   cnt=np.array([1, 1, 1]).astype(np.uint),
                   ref=ref, )

    doc2 = uii.Doc(idx=np.array([0, 1, 2, 3]).astype(np.uint),
                   cnt=np.array([1, 1, 1, 1]).astype(np.uint),
                   ref=ref, )

    T = fn(doc1, doc2)

    assert T.shape[0] == len(doc1)
    assert T.shape[1] == len(doc2)

    # now the tedious job of checking all combinations:

    #          0   1   2   3
    #        non all som non
    # 0  non   0   1  .5   0
    # 1  all   1   0  .5   1
    # 2  som  .5  .5   0  .5

    # word with itself has a distance of 0
    for i, j in (0, 0), (0, 3), (1, 1), (2, 2):
        assert T[i][j] == 0, 'expected 0 at {}, {}'.format(i, j)

    # non (all zeroes) and all (all ones) have the maximum distance
    for i, j in (0, 1), (1, 0), (1, 3):
        assert T[i][j] == 1, 'expected 1 at {}, {}'.format(i, j)

    # the rest are (non, som) and (all, som) combinations
    for i, j in (0, 2), (1, 2), (2, 0), (2, 1), (2, 3):
        assert T[i][j] == 0.5, 'expected 0.5 at {}, {}'.format(i, j)


def test_distance_matrix_loop():
    fn = rhwmd.distance_matrix_loop
    _distance_matrix_test(fn)


def test_distance_matrix_vectorized():
    fn = rhwmd.distance_matrix_vectorized
    _distance_matrix_test(fn)


def test_distance_matrix_lookup():
    fn = rhwmd.distance_matrix_lookup
    _distance_matrix_test(fn)


if __name__ == '__main__':
    # for temporary stuff
    pass
