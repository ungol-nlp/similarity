from ungol.similarity import measures as usm

import numpy as np

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

    fn = usm.hamming

    _hamming_test_min(fn)
    _hamming_test_max(fn)
    _hamming_test_combinations(fn)


def test_hamming_vtov():
    fn = usm.hamming_vtov

    _hamming_test_min(fn)
    _hamming_test_max(fn)
    _hamming_test_combinations(fn)


def test_hamming_vtov_canonical():
    fn = usm.hamming

    _hamming_test_min(fn)
    _hamming_test_max(fn)
    _hamming_test_combinations(fn)


# |--------------------  V TO A


def _create_vtoa_sample():
    x = np.array(
        [0x0, 0x11, 0xff],
        dtype=np.uint8)

    Y = np.array([
        [0x0,  0x0, 0x0],    # 0 + 2 + 8 -> 10
        [0x0, 0x10, 0xff],   # 0 + 1 + 0 ->  1
        [0x0, 0x11, 0xff],   # 0 + 0 + 0 ->  0
        [0xff, 0xff, 0xff],  # 8 + 6 + 0 -> 14
    ], dtype=np.uint8)

    return x, Y


def _assert_hamming_vtoa(Y, dist, idx):
    n = Y.shape[0]

    assert len(dist) == n, f'{len(dist)} != {n}'
    assert dist[0] == 0
    assert dist[1] == 1
    assert dist[2] == 10
    assert dist[3] == 14

    assert len(idx) == n, f'{len(idx)} != {n}'
    assert idx[0] == 2
    assert idx[1] == 1
    assert idx[2] == 0
    assert idx[3] == 3


def _assert_hamming_vtoa_k(k, dist, idx):
    assert len(dist) == k, f'{len(dist)} != {k}'
    assert dist[0] == 0
    assert dist[1] == 1

    assert len(idx) == k, f'{len(idx)} != {k}'
    assert idx[0] == 2
    assert idx[1] == 1


def _test_hamming_vtoa(fn):
    x, Y = _create_vtoa_sample()
    dist, idx = fn(x, Y)
    _assert_hamming_vtoa(Y, dist, idx)


def _test_hamming_vtoa_k(fn):
    k = 2
    x, Y = _create_vtoa_sample()
    dist, idx = fn(x, Y, k=k)
    _assert_hamming_vtoa_k(k, dist, idx)


# ---


def test_hamming_vtoa():
    _test_hamming_vtoa(usm.hamming_vtoa)


def test_hamming_vtoa_canonical():
    _test_hamming_vtoa(usm.hamming)


def test_hamming_vtoa_k():
    _test_hamming_vtoa_k(usm.hamming_vtoa)


def test_hamming_vtoa_k_canonical():
    _test_hamming_vtoa_k(usm.hamming)


# |--------------------  A TO A


def _create_atoa_sample():
    X = np.array([
        [0x0, 0x0, 0x0],
        [0xff, 0xff, 0xff],
    ], dtype=np.uint8)

    Y = np.array([
        [0x0, 0x0, 0x0],
        [0x0, 0x11, 0x0],
        [0xff, 0xff, 0xff],
    ], dtype=np.uint8)

    return X, Y


def _assert_hamming_atoa(X, Y, dist, idx):
    n, m = X.shape[0], Y.shape[0]

    assert dist.shape == (n, m), f'{dist.shape} != ({n}, {m})'

    assert dist[0][0] == 0   # 0, 0
    assert dist[0][1] == 2   # 0, 1
    assert dist[0][2] == 24  # 0, 2

    assert dist[1][0] == 0   # 1, 2
    assert dist[1][1] == 22  # 1, 1
    assert dist[1][2] == 24  # 1, 0

    assert idx.shape == (n, m), f'{idx.shape} != ({n}, {m})'

    assert idx[0][0] == 0
    assert idx[0][1] == 1
    assert idx[0][2] == 2

    assert idx[1][0] == 2
    assert idx[1][1] == 1
    assert idx[1][2] == 0


def _assert_hamming_atoa_k(X, Y, k, dist, idx):
    n = X.shape[0]

    assert dist.shape == (n, k), f'{dist.shape} != ({n}, {k})'

    assert dist[0][0] == 0   # 0, 0
    assert dist[0][1] == 2   # 0, 1

    assert dist[1][0] == 0   # 1, 2
    assert dist[1][1] == 22  # 1, 1

    assert idx.shape == (n, k), f'{idx.shape} != ({n}, {k})'

    assert idx[0][0] == 0
    assert idx[0][1] == 1

    assert idx[1][0] == 2
    assert idx[1][1] == 1


def _test_hamming_atoa(fn):
    X, Y = _create_atoa_sample()
    dist, idx = fn(X, Y)
    _assert_hamming_atoa(X, Y, dist, idx)


def _test_hamming_atoa_k(fn):
    k = 2
    X, Y = _create_atoa_sample()
    dist, idx = fn(X, Y, k=k)
    _assert_hamming_atoa_k(X, Y, k, dist, idx)


# ---


def test_hamming_atoa():
    _test_hamming_atoa(usm.hamming_atoa)


def test_hamming_atoa_canonical():
    _test_hamming_atoa(usm.hamming)


def test_hamming_atoa_k():
    _test_hamming_atoa_k(usm.hamming_atoa)


def test_hamming_atoa_k_canonical():
    _test_hamming_atoa_k(usm.hamming)
