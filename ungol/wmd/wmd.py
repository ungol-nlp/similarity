# -*- coding: utf-8 -*-

from ungol.models import embcodr

import attr
import nltk
import numpy as np
from tqdm import tqdm as _tqdm
from tabulate import tabulate

import pickle
import pathlib
import functools
from collections import defaultdict

from typing import Set
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple


# ---

tqdm = functools.partial(_tqdm, ncols=80)

# global flag in hope of being as non-intrusive as possible
# regarding benchmarks...
STATS = False


# --- utility


def bincount(x: int):
    return bin(x).count('1')


def basename(fname: str):
    return pathlib.Path(fname).name


def load_stopwords(f_stopwords: List[str] = None) -> Set[str]:
    stopwords: Set[str] = set()

    if f_stopwords is None:
        return stopwords

    def clean_line(raw: str) -> str:
        return raw.strip()

    def filter_line(token: str) -> bool:
        cond = any((
            len(token) == 0,
            token.startswith(';'),
            token.startswith('#'), ))

        return not cond

    for fname in f_stopwords:
        with open(fname, 'r') as fd:
            raw = fd.readlines()

        stopwords |= set(filter(filter_line, map(clean_line, raw)))

    print('loaded {} stopwords'.format(len(stopwords)))
    return stopwords


# ---


@attr.s
class DocReferences:
    """
    It got pretty tedious to pass these arguments around
    - hence this collection type for use by wmd.Doc
    """

    # see ungol.models.embcodr.load_codes_bin for meta
    meta:       Dict[str, Any] = attr.ib()
    vocabulary: Dict[str, int] = attr.ib()
    lookup:     Dict[int, str] = attr.ib()
    stopwords:        Set[str] = attr.ib()

    codemap: np.ndarray = attr.ib()  # (Vocabulary, bytes); np.uint8
    distmap: np.ndarray = attr.ib()  # (Vocabulary, knn);   np.uint8

    @staticmethod
    def from_files(
            f_codemap: str,  # usually codemap.h5 produced by embcodr
            f_vocab: str,    # pickled dictionary mapping str -> int
            f_stopwords: List[str] = None):

        with open(f_vocab, 'rb') as fd:
            vocab = pickle.load(fd)

        meta, dists, codes = embcodr.load_codes_bin(f_codemap)
        stopwords = load_stopwords(f_stopwords)
        lookup = {v: k for k, v in vocab.items()}

        return DocReferences(
            meta=meta,
            vocabulary=vocab,
            lookup=lookup,
            stopwords=stopwords,
            codemap=codes,
            distmap=dists)


class DocumentEmptyException(Exception):
    pass


@attr.s
class Doc:

    # document specific attributes
    idx:        np.array = attr.ib()  # (n, ) code indexes
    cnt:        np.array = attr.ib()  # (n, ) word frequency (normalized)
    ref:   DocReferences = attr.ib()

    # optional (mainly for use in this module)
    name: str = attr.ib(default=None)

    # ---

    @property
    def tokens(self) -> Tuple[str]:
        return [self.ref.lookup[idx] for idx in self.idx]

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, key: int):
        return self.codes[key], self.cnt[key], self.dists[key]

    def __attrs_post_init__(self):

        # type checks
        assert self.idx.dtype == np.dtype('uint')
        assert self.cnt.dtype == np.dtype('float')
        assert round(sum(self.cnt), 5) == 1, round(sum(self.cnt), 5)

        # shape checks
        assert len(self.tokens) == self.codes.shape[0]
        assert len(self.tokens) == self.dists.shape[0]
        assert self.codes.shape[1] == self.ref.codemap.shape[1]
        assert self.dists.shape[1] == self.ref.distmap.shape[1]

    def __str__(self):
        str_buf = ['\ndocument of length: {}'.format(len(self))]
        tab_data = []

        header_knn = tuple('{}-nn'.format(k) for k in self.ref.meta['knn'])
        header = ('word', 'frequency (%)', 'code sum', ) + header_knn

        assert self.codes.shape[0] == self.dists.shape[0]

        row_data = self.tokens, self.cnt, self.codes, self.dists
        for token, freq, code, dist in zip(*row_data):
            assert code.shape[0] == self.codes.shape[1]

            freq = freq * 100
            code = code.sum()

            tab_data.append((token, freq, code) + tuple(dist))

        str_buf += [tabulate(tab_data, headers=header)]
        return '\n'.join(str_buf)

    @property
    def dists(self) -> '(words, retained k distances)':
        return self.ref.distmap[self.idx, ]

    @property
    def codes(self) -> '(words, bytes)':
        return self.ref.codemap[self.idx, ]

    @staticmethod
    def from_tokens(name: str, tokens: List[str], ref: DocReferences):
        sanitized = (token.strip().lower() for token in tokens)
        filtered = [token for token in sanitized if all([
            token in ref.vocabulary,
            token not in ref.stopwords])]

        countmap = defaultdict(lambda: 0)
        for token in filtered:
            countmap[ref.vocabulary[token]] += 1

        countlist = sorted(countmap.items(), key=lambda t: t[1], reverse=True)
        if not len(countlist):
            raise DocumentEmptyException

        idx, cnt = zip(*countlist)

        a_idx = np.array(idx).astype(np.uint)
        a_cnt_raw = np.array(cnt).astype(np.float)
        a_cnt = a_cnt_raw / a_cnt_raw.sum()

        return Doc(name=name, idx=a_idx, cnt=a_cnt, ref=ref)

    @staticmethod
    def from_text(name: str, text: str, ref: DocReferences):
        tokenize = nltk.word_tokenize
        tokens = tokenize(text)
        return Doc.from_tokens(name, tokens, ref)


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


def dist(doc1: Doc, doc2: Doc, verbose: bool = False) -> float:
    # FIXME assert correct dimensions in every step!
    # compute the distance matrix
    T = distance_matrix_lookup(doc1, doc2)

    # weight the distance matrix by term frequency per document
    # and select the minimum distances per document
    # doc1_idx = np.argmin(T * doc1.cnt)
    # doc2_idx = np.argmin(T.T * doc2.cnt)
    doc1_idx = np.argmin(T, axis=1)
    doc2_idx = np.argmin(T, axis=0)

    assert doc1.codes.shape[0] == doc1_idx.shape[0]
    assert doc2.codes.shape[0] == doc2_idx.shape[0]

    # note: this returns the _first_ occurence if there
    # are multiple codes with the same distance
    doc1_dists = T[np.arange(T.shape[0]), doc1_idx]
    doc2_dists = T.T[np.arange(T.shape[1]), doc2_idx]

    # for manual inspection:
    # print('\nhamming distances:'.upper())
    # __print_hamming_nn(doc1, doc2, doc1_idx, doc1_dists)
    # __print_hamming_nn(doc2, doc1, doc2_idx, doc2_dists)

    score = max(doc1_dists.mean(), doc2_dists.mean())

    if verbose:
        return score, T, doc1_dists, doc2_dists

    return score
