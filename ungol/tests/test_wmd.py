# -*- coding: utf-8

from ungol.wmd import wmd

import numpy as np

from typing import Callable
Hamming = Callable[[np.array, np.array], int]


def _hamming_test_min(h: Hamming):
    c1 = c2 = np.zeros(5, dtype=np.uint8)
    dist = h(c1, c2)
    assert dist == 0, dist

    c1 = c2 = np.ones(5, dtype=np.uint8)
    dist = h(c1, c2)
    assert dist == 0, dist


def _hamming_test_max(h: Hamming):
    codelen = 3
    c1 = np.zeros(codelen, dtype=np.uint8)
    c2 = np.ones(codelen, dtype=np.uint8) * 255
    dist = h(c1, c2)
    assert dist == codelen * 8, dist


def _hamming_test_combinations(h: Hamming):
    c1 = np.array([5], dtype=np.uint8)  # 0101
    c2 = np.array([9], dtype=np.uint8)  # 1001

    dist = h(c1, c2)
    assert dist == 2, dist


def test_hamming_bincount():
    h = wmd.hamming_bincount
    _hamming_test_min(h)
    _hamming_test_max(h)
    _hamming_test_combinations(h)


def test_hamming_bitmask():
    h = wmd.hamming_bitmask
    _hamming_test_min(h)
    _hamming_test_max(h)
    _hamming_test_combinations(h)
