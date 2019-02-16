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


def topk(a, k: int = None):
    """
    Return the top k elements and indexes of vector a with length n.
    The resulting k elements are sorted ascending and the returned
    indexes correspond to the original array a.

    This runs in O(n log k).

    FIXME: For further speedup use the "bottleneck" implementation.

    """
    a_idx = np.arange(len(a)) if k is None else np.argpartition(a, k)[:k]
    s_idx = np.argsort(a[a_idx])

    idx = a_idx[s_idx]
    return a[idx], idx


#  HAMMING SPACE
#  |----------------------------------------
#
#  take care to use the correct input encoding per
#  function (bit-encoding, byte-encoding, one-hot)
#


# NUMPY VECTORIZED OPERATIONS


# for b bit the lowest possible distance is 0 and the highest
# possible distance is b - thus this vector is of size b + 1
_hamming_lookup = np.array([bin(x).count('1') for x in range(0x100)])


# |--------------------  see measures.hamming


def hamming_vectorized(X, Y):
    assert X.dtype == np.uint8, f'X ({X.dtype}) not uint8'
    assert Y.dtype == np.uint8, f'Y ({Y.dtype}) not uint8'

    return _hamming_lookup[X ^ Y].sum(axis=-1)


def hamming_vtov(x, y):
    return hamming_vectorized(x, y)


def hamming_vtoa(x, Y, k: int = None):
    return topk(hamming_vectorized(x, Y), k)


def hamming_atoa(X, Y, k: int = None):
    n = X.shape[0]
    m = Y.shape[0] if k is None else k

    # even if it might be possible to fully vectorize it
    # this approach keeps a somewhat sane memory profile
    top_d, top_i = np.zeros((n, m)), np.zeros((n, m), dtype=np.int)
    for i, x in enumerate(X):
        top_d[i], top_i[i] = hamming_vtoa(x, Y, k=k)

    return top_d, top_i


# |--------------------  canonical


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
