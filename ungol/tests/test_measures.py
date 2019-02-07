from ungol.index import index as uii
from ungol.similarity import measures as usm

import np

from typing import Callable


Hamming = Callable[[np.array, np.array], int]


def _hamming_test_min(fn: Hamming):
    c1 = c2 = np.zeros(5, dtype=np.uint8)
    dist = fn(c1, c2)
    assert dist == 0, dist

    c1 = c2 = np.ones(5, dtype=np.uint8)
    dist = fn(c1, c2)
    assert dist == 0, dist


def _hamming_test_max(fn: Hamming):
    codelen = 3
    c1 = np.zeros(codelen, dtype=np.uint8)
    c2 = np.ones(codelen, dtype=np.uint8) * 255
    dist = fn(c1, c2)
    assert dist == codelen * 8, dist


def _hamming_test_combinations(fn: Hamming):
    c1 = np.array([5], dtype=np.uint8)  # 0101
    c2 = np.array([9], dtype=np.uint8)  # 1001

    dist = fn(c1, c2)
    assert dist == 2, dist


def test_hamming_vectorized():
    fn = usm.hamming_vectorized
    _hamming_test_min(fn)
    _hamming_test_max(fn)
    _hamming_test_combinations(fn)


def test_hamming_vtov():
    fn = usm.hamming_vtv
    _hamming_test_min(fn)
    _hamming_test_max(fn)
    _hamming_test_combinations(fn)
