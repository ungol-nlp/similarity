# -*- coding: utf-8 -*-

from ungol.models import embcodr

import attr
import nltk
import numpy as np
from tqdm import tqdm as _tqdm
from tabulate import tabulate

import enum
import pickle
import pathlib
import functools
from collections import defaultdict

from typing import Set
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union


# ---

tqdm = functools.partial(_tqdm, ncols=80)

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
    To effectively separate shared memory from individual document
    information for Doc instances, this class wraps information shared
    by all documents.

    TODO: currently, meta['knn'] and distmap must always be
    provided. They are in fact completely optional -> so make them
    default to None and handle this case downstream.

    """

    # see ungol.models.embcodr.load_codes_bin for meta
    meta:       Dict[str, Any] = attr.ib()
    vocabulary: Dict[str, int] = attr.ib()

    codemap: np.ndarray = attr.ib()  # (Vocabulary, bytes); np.uint8
    distmap: np.ndarray = attr.ib()  # (Vocabulary, knn);   np.uint8

    stopwords: Set[str] = attr.ib(default=attr.Factory(set))

    # populated when filling the database
    termfreqs: Dict[int. int] = attr.ib(default=attr.Factory(dict))
    docfreqs:  Dict[int, int] = attr.ib(default=attr.Factory(dict))

    # ---

    def __attrs_post_init__(self):
        self.lookup = {v: k for k, v in self.vocabulary.items()}

    def __str__(self) -> str:
        sbuf = ['VNGOL meta information:']

        sbuf.append('  vocabulary: {} words'.format(
            len(self.vocabulary)))

        sbuf.append('  filtering: {} stopwords'.format(
            len(self.stopwords)))

        sbuf.append('  code size: {} bits'.format(
            self.codemap.shape[1] * 8))

        return '\n'.join(sbuf)

    @staticmethod
    def from_files(
            f_codemap: str,  # usually codemap.h5 produced by embcodr
            f_vocab: str,    # pickled dictionary mapping str -> int
            f_stopwords: List[str] = None):

        with open(f_vocab, 'rb') as fd:
            vocab = pickle.load(fd)

        meta, dists, codes = embcodr.load_codes_bin(f_codemap)
        stopwords = load_stopwords(f_stopwords)

        return DocReferences(
            meta=meta,
            vocabulary=vocab,
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

    # optionally filled
    unknown:  Dict[str, int] = attr.ib(default=attr.Factory(dict))
    unwanted:            int = attr.ib(default=0)

    # mainly for use in this module
    name: str = attr.ib(default=None)

    # ---

    @property
    def tokens(self) -> Tuple[str]:
        return [self.ref.lookup[idx] for idx in self.idx]

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, key: int) -> Tuple[int, int, int]:
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
        str_buf = ['Document: "{}"'.format(
            self.name if self.name else 'Unknown')]

        str_buf += ['tokens: {}, unknown: {}, unwanted: {}\n'.format(
            len(self), len(self.unknown), self.unwanted)]

        header_knn = tuple('{}-nn'.format(k) for k in self.ref.meta['knn'])
        header = ('word', 'frequency (%)', 'code sum', ) + header_knn

        assert self.codes.shape[0] == self.dists.shape[0]

        tab_data = []
        row_data = self.tokens, self.cnt, self.codes, self.dists
        for token, freq, code, dist in zip(*row_data):
            assert code.shape[0] == self.codes.shape[1]
            tab_data.append((token, freq * 100, code.sum()) + tuple(dist))

        str_buf += [tabulate(tab_data, headers=header)]

        if len(self.unknown):
            str_buf += ['\nUnknown Words:']
            for word, freq in self.unknown.items():
                str_buf += [f'{word}: {freq}']

        str_buf.append('')
        return '\n'.join(str_buf)

    @property
    def dists(self) -> '(words, retained k distances)':
        return self.ref.distmap[self.idx, ]

    @property
    def codes(self) -> '(words, bytes)':
        return self.ref.codemap[self.idx, ]

    @staticmethod
    def from_tokens(name: str, tokens: List[str], ref: DocReferences):

        # partition tokens
        tok_known, tok_unknown, tok_unwanted = [], {}, 0

        for token in tokens:
            if token in ref.stopwords:
                tok_unwanted += 1
            elif token in ref.vocabulary:
                tok_known.append(token)
            else:
                tok_unknown[token] = tok_unknown.get(token, 0) + 1

        # aggregate frequencies; builds mapping of idx -> freq,
        # sort by frequency and build fast lookup arrays
        # with indexes and normalized distribution values
        countmap = defaultdict(lambda: 0)
        for token in tok_known:
            countmap[ref.vocabulary[token]] += 1

        countlist = sorted(countmap.items(), key=lambda t: t[1], reverse=True)
        if not len(countlist):
            raise DocumentEmptyException

        idx, cnt = zip(*countlist)

        a_idx = np.array(idx).astype(np.uint)
        a_cnt_raw = np.array(cnt).astype(np.float)
        a_cnt = a_cnt_raw / a_cnt_raw.sum()

        return Doc(name=name, idx=a_idx, cnt=a_cnt, ref=ref,
                   unknown=tok_unknown, unwanted=tok_unwanted)

    @staticmethod
    def from_text(name: str, text: str, ref: DocReferences):
        tokenize = nltk.word_tokenize
        tokens = tokenize(text.lower())
        return Doc.from_tokens(name, tokens, ref)


@attr.s
class Database:

    docref:   DocReferences = attr.ib()
    mapping: Dict[str, Doc] = attr.ib()
    skipped:      List[str] = attr.ib(default=None)

    def __str__(self) -> str:
        sbuf = ['VNGOL database']

        sbuf.append('  containing: {} documents'.format(
            len(self.mapping)))

        sbuf.append('  vocabulary: {} words'.format(
            len(self.docref.vocabulary)))

        sbuf.append('  filtering: {} stopwords'.format(
            len(self.docref.stopwords)))

        sbuf.append('  code size: {} bits'.format(
            self.docref.codemap.shape[1] * 8))

        sbuf.append('  skipped: {} documents'.format(
            len(self.skipped or [])))

        return '\n' + '\n'.join(sbuf) + '\n'

    def to_file(self, fname: str):
        with open(fname, 'wb') as fd:
            pickle.dump(self, fd)

    @staticmethod
    def from_file(fname: str):
        with open(fname, 'rb') as fd:
            return pickle.load(fd)


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
        c1, _, d1 = doc1[i]

        for j in range(n2):
            c2, _, d2 = doc2[j]

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

    C1 = doc1[idx_x][0]
    C2 = doc2[idx_y][0]

    T = _dmv_vectorized_bincount(C1 ^ C2).sum(axis=-1)
    return T / (doc1.codes.shape[1] * 8)


def distance_matrix_lookup(doc1: Doc, doc2: Doc) -> '(n1, n2)':

    idx_y = _dmv_meshgrid[0][:len(doc1), :len(doc2)]
    idx_x = _dmv_meshgrid[1][:len(doc1), :len(doc2)]

    C1 = doc1[idx_x][0]
    C2 = doc2[idx_y][0]

    T = _hamming_lookup[C1 ^ C2].sum(axis=-1)
    return T / (doc1.codes.shape[1] * 8)


# ---


@attr.s
class Score:
    """
    This class explains the score if dist(verbose=True)
    """

    # given l1 = len(doc1), l2 = len(doc2)

    value:   float = attr.ib()  # based on the strategy
    T:  np.ndarray = attr.ib()  # (l1, l2)

    doc1: Doc = attr.ib()
    doc2: Doc = attr.ib()

    doc1_idx: np.array = attr.ib()  # (l1, )
    doc2_idx: np.array = attr.ib()  # (l2, )

    doc1_dists: np.array = attr.ib()  # (l1, )
    doc2_dists: np.array = attr.ib()  # (l2, )

    # raw mean without weighting
    doc1_raw_mean: float = attr.ib()
    doc2_raw_mean: float = attr.ib()

    # distances weighted by frequency
    doc1_mean: float = attr.ib()
    doc2_mean: float = attr.ib()

    # scores reduced by fraction of matching terms
    doc1_score: float = attr.ib()
    doc2_score: float = attr.ib()

    # ---

    def _str_hamming_nn(self, ref_doc, cmp_doc, min_idx, min_dist) -> str:
        assert len(ref_doc.tokens) == len(min_idx)
        assert len(ref_doc.tokens) == len(min_dist)
        nn = [cmp_doc.tokens[idx] for idx in min_idx]

        headers = ['token', 'neighbour', 'distance']
        tab_data = list(zip(ref_doc.tokens, nn, min_dist))
        tab_data.sort(key=lambda t: t[2])

        return tabulate(tab_data, headers=headers)

    def __str__(self) -> str:
        sbuf = ['wmd score - {:.3f}'.upper().format(self.value), '']

        # doc1 / doc2

        fmt = '\ncomparing "{}" to "{}" [mean raw {:.3f}, '
        fmt += 'weighted: {:.3f}, +unknown: {:3f}]'

        sbuf.append(fmt.format(
            self.doc1.name, self.doc2.name,
            self.doc1_raw_mean, self.doc2_mean, self.doc2_score))

        sbuf.append(self._str_hamming_nn(
            self.doc1, self.doc2, self.doc1_idx, self.doc1_dists))

        # doc2 / doc1

        sbuf.append(fmt.format(
            self.doc2.name, self.doc1.name,
            self.doc2_raw_mean, self.doc2_mean, self.doc2_score))

        sbuf.append(self._str_hamming_nn(
            self.doc2, self.doc1, self.doc2_idx, self.doc2_dists))

        return '\n'.join(sbuf)


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


def dist(
        doc1: Doc, doc2: Doc,
        strategy: Strategy = Strategy.MAX,
        # ------------------------------------ optional
        docref: DocReferences = None,
        idf: bool = False,
        verbose: bool = False, ) -> Union[float, Score]:
    """

    Calculate the RWMD score based on hamming distances for two
    documents. Higher is better.

    :param doc1: Doc - first document
    :param doc2: Doc - second document
    :param verbose: bool - if True: return a Score object
    :param strategy: Strategy - see wmd.Strategy, defaults to Strategy.MAX
                                as defined in the paper

    """

    # Compute the distance matrix.
    T = distance_matrix_lookup(doc1, doc2)

    assert T.shape[0] == len(doc1)
    assert T.shape[1] == len(doc2)

    l1, l2 = T.shape

    doc1_idx = np.argmin(T, axis=1)
    doc2_idx = np.argmin(T.T, axis=1)

    assert len(doc1_idx.shape) == 1
    assert len(doc2_idx.shape) == 1
    assert doc1_idx.shape[0] == l1
    assert doc2_idx.shape[0] == l2

    # Select the nearest neighbours per file Note: this returns the
    # _first_ occurence if there are multiple codes with the same
    # distance (not important for further computation...)
    doc1_dists = T[np.arange(T.shape[0]), doc1_idx]
    doc2_dists = T.T[np.arange(T.shape[1]), doc2_idx]

    assert len(doc1_dists.shape) == 1
    assert len(doc2_dists.shape) == 1
    assert doc1_dists.shape[0] == l1
    assert doc2.dists.shape[0] == l2

    # --------------------

    doc1_raw_mean = doc1_dists.mean()
    doc2_raw_mean = doc2_dists.mean()

    # Weight by term frequency.
    doc1_mean = (doc1_dists * doc1.cnt).sum()
    doc2_mean = (doc2_dists * doc2.cnt).sum()

    # --------------------

    # Include terms not found in the code database: an exact match
    # would produce a T[i, j] = 0 value which results in its frequency
    # weighting to become 0 and thus reducing the absolute score
    # value. This approach reduces the total score by the fraction
    # of matching unknown terms weighted by frequency.
    common_unknown = doc1.unknown.keys() & doc2.unknown.keys()

    doc1_un_weight = sum(doc1.unknown[tok] for tok in common_unknown)
    doc2_un_weight = sum(doc2.unknown[tok] for tok in common_unknown)

    doc1_unknown_ratio = doc1_un_weight / (doc1_un_weight + len(doc1))
    doc2_unknown_ratio = doc2_un_weight / (doc2_un_weight + len(doc2))

    doc1_score = (1 - doc1_unknown_ratio) * doc1_mean
    doc2_score = (1 - doc2_unknown_ratio) * doc2_mean

    assert 0 <= doc1_score and doc1_score <= doc1_mean
    assert 0 <= doc2_score and doc2_score <= doc2_mean
    if len(common_unknown):
        assert doc1_score < doc1_mean
        assert doc2_score < doc2_mean

    # --------------------

    # Apply the inverse document frequency if this information is
    # provided by the DocReferences.
    if idf:
        assert docref is not None

        # FIXME

    # --------------------

    # produce the score by applying a strategy
    if strategy is Strategy.MAX:
        score = max(doc1_score, doc2_score)

    if strategy is Strategy.MIN:
        score = min(doc1_score, doc2_score)

    if strategy is Strategy.ADAPTIVE_SMALL:
        score = doc1_score if len(doc1) < len(doc2) else doc2_score

    if strategy is Strategy.ADAPTIVE_BIG:
        score = doc2_score if len(doc1) < len(doc2) else doc1_score

    # --------------------

    score = 1 - score

    if not verbose:
        return score
    else:
        return Score(
            value=score, T=T,
            doc1=doc1, doc2=doc2,
            doc1_raw_mean=doc1_raw_mean, doc2_raw_mean=doc2_raw_mean,
            doc1_mean=doc1_mean, doc2_mean=doc2_mean,
            doc1_score=doc1_score, doc2_score=doc2_score,
            doc1_idx=doc1_idx, doc2_idx=doc2_idx,
            doc1_dists=doc1_dists, doc2_dists=doc2_dists, )


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


def mock_doc(n: int, bytecount: int = 32) -> Doc:
    """
    To test wmd.distance_matrix_*
    """
    codemap = (np.random.randn(n, bytecount) * 255).astype(dtype=np.uint8)
    distmap = (np.random.randn(n, 1) * 255).astype(dtype=np.int)

    idx = np.arange(n).astype(np.uint)
    np.random.shuffle(idx)

    vocab = {str(i): i for i in idx}

    ref = DocReferences(
        meta={'knn': [1]},
        vocabulary=vocab,
        codemap=codemap,
        distmap=distmap, )

    doc = Doc(idx=idx[:n], cnt=np.ones(n) / n, ref=ref)
    return doc
