#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

A collection of different similarity and distance measure implementations.
Sometimes batched or gpu accelerated variants exist.

"""

from ungol.common import logger

import numpy as np


log = logger.get('similarity.measures')


# def m_cosine(train_data, test_data, tqdm=lambda x: x, max_k=100):
#     dists, train, test = None, None, None

#     try:
#         train = torch.from_numpy(train_data).to(device=DEV)
#         test  = torch.from_numpy(test_data).to(device=DEV)

#         train /= train.norm(dim=1).unsqueeze(1)
#         test /= test.norm(dim=1).unsqueeze(1)

#         dists = torch.stack([
#             (1-train.matmul(t).squeeze())
#             for t in tqdm(test)])

#         topkek = dists.topk(k=max_k, largest=False, dim=1)
#         sortdists, sortindices = map(lambda t: t.cpu().numpy(), topkek)

#     finally:
#         del dists, train, test
#         torch.cuda.empty_cache()

#     return sortdists, sortindices


def topk(a, k):
    """
    Return the top k elements and indexes of vector a with length n.
    The resulting k elements are sorted ascending and the returned
    indexes correspond to the original array a.

    This runs in O(n log k).

    FIXME: For further speedup use the "bottleneck" implementation.

    """
    a_idx = np.argpartition(a, k)[:k]
    s_idx = np.argsort(a[a_idx])

    idx = a_idx[s_idx]
    return a[idx], idx


#  HAMMING SPACE
#  ----------------------------------------|
#
#  take care to use the correct input encoding per
#  function (bit-encoding, byte-encoding, one-hot)
#


_hamming_lookup = np.array([bin(x).count('1') for x in range(0xff)])


def _hamming_fn(x, Y):
    return _hamming_lookup[x ^ Y].sum(axis=-1)


def hamming_vtov(x, y):
    """
    see hamming()
    """
    return _hamming_fn(x, y)


def hamming_vtoa(x, Y, k: int = None):
    return topk(_hamming_fn(x, Y), k)


def hamming_atoa(X, Y, k: int = None):
    """
    see hamming()
    """
    n, m = X.shape[0], Y.shape[0]
    k = m if k is None else k

    top_d, top_i = np.zeros((n, k)), np.zeros((n, k))
    for i, x in enumerate(X):
        top_d[i], top_i[i] = hamming_vtoa(x, Y, k=k)

    return top_d, top_i


def hamming(X, Y, **kwargs):
    """

    Most comfortable, but uses heuristics to select the correct
    function - might marginally impair execution time.

    These function family always returns both distance values
    and position indexes of the original Y array if Y is a matrix.

    Possible combinations:
      atoa: given two matrices X: (n, b), Y: (m, b)
        for every x in X the distance to all y in Y
        result: (n, m) distances and indexes

      vtoa: given a vector and a matrix x: (b, ), Y: (m, b)
        compute distance of x to all y in Y
        result: (m, ) distances and indexes

      vtov: distance between x: (b, ) and y: (b, )
        result: number

    Accepted keyword arguments:
      k: int - only return k nearest neighbours (invalid for vtov)

    """
    dx, dy = len(X.shape), len(Y.shape)

    if dx == 2 and dy == 2:
        return hamming_atoa(X, Y, **kwargs)
    elif dx == 1 and dy == 2:
        return hamming_vtoa(X, Y, **kwargs)
    elif dx == 1 and dy == 1:
        return hamming_vtov(X, Y, **kwargs)

    assert False, 'unknown input size'


if __name__ == '__main__':
    pass