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

    # --- populated when filling the database (db + Doc)

    termfreqs: Dict[int, int] = attr.ib(default=attr.Factory(dict))
    docfreqs:  Dict[int, int] = attr.ib(default=attr.Factory(dict))

    # documents not added to the database
    skipped:        List[str] = attr.ib(default=attr.Factory(list))

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
    idx:  np.array = attr.ib()  # (n, ) code indexes
    cnt:  np.array = attr.ib()  # (n, ) word frequency counts
#   freq: np,array = attr.ib()  # calculated by __attr_post_init__

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
        return self.codes[key]

    def __attrs_post_init__(self):
        self.freq = self.cnt.astype(np.float) / self.cnt.sum()

        # type checks
        assert self.idx.dtype == np.dtype('uint')
        assert self.cnt.dtype == np.dtype('uint')
        assert self.freq.dtype == np.dtype('float')
        assert round(sum(self.freq), 5) == 1, round(sum(self.freq), 5)

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
        header = ('word', 'term count', 'code sum', ) + header_knn

        assert self.codes.shape[0] == self.dists.shape[0]

        tab_data = []
        row_data = self.tokens, self.cnt, self.codes, self.dists
        for token, cnt, code, dist in zip(*row_data):
            assert code.shape[0] == self.codes.shape[1]
            tab_data.append((token, cnt, code.sum()) + tuple(dist))

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
        a_cnt = np.array(cnt).astype(np.uint)

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
    mapping: Dict[str, Doc] = attr.ib(default=attr.Factory(dict))

    def __add__(self, doc: Doc):
        assert len(doc.idx), 'the document has no content'
        assert doc.name, 'document needs the name attribute set'
        assert doc.name not in self.mapping, f'"{doc.name}" already indexed'

        for i, idx in enumerate(doc.idx):
            tf = self.docref.termfreqs
            tf[idx] = tf.get(idx, 0) + doc.cnt[i]

        for idx in set(doc.idx):
            df = self.docref.docfreqs
            df[idx] = df.get(idx, 0) + 1

        self.mapping[doc.name] = doc

        return self

    # ---

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

        sbuf.append('  tokens: {}'.format(
            len(self.docref.termfreqs)))

        sbuf.append('  skipped: {}'.format(
            len(self.docref.skipped)))

        return '\n' + '\n'.join(sbuf) + '\n'

    def to_file(self, fname: str):
        with open(fname, 'wb') as fd:
            pickle.dump(self, fd)

    def doc_info(self, doc_id: str, legend: bool = True) -> str:
        assert doc_id in self.mapping, f'"doc_id" not found in db'
        doc = self.mapping[doc_id]

        buf = [f'Information about "{doc_id}"', '']

        headers = 'word', 'idx', 'dtf (%)', 'dtf', 'gtf', 'df'
        tab_data = zip(
            doc.tokens,
            doc.idx,
            (f'{freq:.2f}' for freq in doc.freq),
            doc.cnt,
            (self.docref.termfreqs[idx] for idx in doc.idx),
            (self.docref.docfreqs[idx] for idx in doc.idx), )

        buf.append(tabulate(tab_data, headers=headers))

        if legend:
            buf.append('')
            buf.append('Legend\n' +
                       '  idx: embedding matrix index\n' +
                       '  dtf: document term frequency\n' +
                       '  gtf: global term frequency\n' +
                       '  df: document frequency')

            buf.append('')
        return '\n'.join(buf)

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


def distance_matrix_vectorized(doc1: Doc, doc2: Doc) -> '(n1, n2)':

    idx_y = _dmv_meshgrid[0][:len(doc1), :len(doc2)]
    idx_x = _dmv_meshgrid[1][:len(doc1), :len(doc2)]

    C1 = doc1[idx_x]
    C2 = doc2[idx_y]

    T = _dmv_vectorized_bincount(C1 ^ C2).sum(axis=-1)
    return T / (doc1.codes.shape[1] * 8)


def distance_matrix_lookup(doc1: Doc, doc2: Doc) -> '(n1, n2)':

    idx_y = _dmv_meshgrid[0][:len(doc1), :len(doc2)]
    idx_x = _dmv_meshgrid[1][:len(doc1), :len(doc2)]

    C1 = doc1[idx_x]
    C2 = doc2[idx_y]

    T = _hamming_lookup[C1 ^ C2].sum(axis=-1)
    return T / (doc1.codes.shape[1] * 8)


# ---


@attr.s
class ScoreData:
    """
    This class explains the score if dist(verbose=True)

    Note that no logic must be implemented here. To find bugs
    this class only accepts the real data produced by dist().
    """

    # given l1 = len(doc1), l2 = len(doc2)

    score:   float = attr.ib()  # based on the strategy

    # all values are length 2 tuples

    docs: Tuple[Doc] = attr.ib()
    strategy: str = attr.ib()
    common_unknown: Set[str] = attr.ib()  # tokens shared

    T:            np.ndarray = attr.ib()  # distance matrix
    n_sims:     Tuple[float] = attr.ib()  # raw similarity
    n_weighted: Tuple[float] = attr.ib()  # with weighting (idf)
    n_scores:   Tuple[float] = attr.ib()  # per-document score

    a_idxs: Tuple[np.array] = attr.ib()  # ((l1, ), (l2, ))
    a_sims: Tuple[np.array] = attr.ib()  # raw distance values

    a_tfs:      Tuple[np.array] = attr.ib()  # raw tf values
    a_idfs:     Tuple[np.array] = attr.ib()  # raw idf values
    a_weighted: Tuple[np.array] = attr.ib()  # distances weighted

    # ---

    def __attrs_post_init__(self):
        for i in (0, 1):
            assert len(self.docs[i].tokens) == len(self.a_idxs[i])
            assert len(self.docs[i].tokens) == len(self.a_tfs[i])
            assert len(self.docs[i].tokens) == len(self.a_idfs[i])
            assert len(self.docs[i].tokens) == len(self.a_weighted[i])

    def _draw_border(self, s: str) -> str:
        lines = s.splitlines()
        maxl = max(len(l) for l in lines)

        # clock-wise, starting north (west)
        borders = '-', '  |', '-', ' |  '
        edges = ' +--', '--+', '--+', ' +--'

        first = f'{edges[0]}' + f'{borders[0]}' * maxl + f'{edges[1]}'
        midfmt = f'{borders[3]}%-' + str(maxl) + f's{borders[1]}'
        last = f'{edges[3]}' + f'{borders[2]}' * maxl + f'{edges[2]}'

        mid = [midfmt % s for s in lines]
        return '\n'.join([first] + mid + [last])

    def _str_common_unknown(self, docbuf: List[str], a: int) -> None:
        docbuf.append('\n')
        docbuf.append('Common unknown words:\n')
        headers = ('token', 'count')
        tab_data = [(tok, self.docs[a].unknown[tok])
                    for tok in self.common_unknown]

        unknown_table = tabulate(tab_data, headers=headers)
        docbuf.append(unknown_table)

    def _str_sim_table(self, docbuf: List[str], a: int, b: int) -> None:
        headers = ('no', 'token', 'neighbour', 'sim', 'f(sim)',
                   'tf', 'idf', )

        doc1, doc2 = self.docs[a], self.docs[b]

        tab_data = list(zip(
            doc1.tokens,
            [doc2.tokens[idx] for idx in self.a_idxs[a]],
            self.a_sims[a],
            self.a_weighted[a],
            self.a_tfs[a],
            self.a_idfs[a], ))

        tab_data.sort(key=lambda t: t[2], reverse=True)

        sims_table = tabulate(
            tab_data, headers=headers,
            showindex='always', floatfmt=".4f")

        docbuf.append(sims_table)

    def _str_score_table(self, docbuf: List[str], a: int, b: int) -> None:
        name1, name2 = self.docs[a].name, self.docs[b].name
        docbuf.append(f'\ncomparing: "{name1}" to "{name2}"\n')

        headers, tab_data = zip(*(
            ('', 'score:'),
            ('similarity', self.n_sims[a]),
            ('weighted', self.n_weighted[a]),
            ('score', self.n_scores[a]), ))

        docbuf.append(tabulate([tab_data], headers=headers))
        docbuf += ['', '']

    def __str__(self) -> str:
        sbuf = [f'\nWMD SCORE : {self.score}']
        sbuf.append(f'selection strategy: {self.strategy}\n')
        sbuf.append(self.docstr(first=True) + '\n')
        sbuf.append(self.docstr(first=False) + '\n')
        return '\n'.join(sbuf)

    def docstr(self, first: bool = True) -> str:
        a, b = (0, 1) if first else (1, 0)

        docbuf = []

        self._str_score_table(docbuf, a, b)
        self._str_sim_table(docbuf, a, b)
        self._str_common_unknown(docbuf, a)

        docbuf += ['', '']

        return self._draw_border('\n'.join(docbuf))


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
#  FIXME: speech about what I learned about ripping apart the calculation.
#
#  Progress:
#
#    - [X] re-implement status quo without noise
#    - [X] re-implement verbose=True
#    - [ ] devise oov strategy
#    - [ ] adjust tests, add assertions
#
#  Notes:
#
#    - prefixes: s_* for str, a_* for np.array, n_ for scalars
#
def _similarity(db: Database, doc1: Doc, doc2: Doc, verbose: bool) -> float:

    # ----------------------------------------
    # this is the important part
    #

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

    # ---  IDF

    def idf(doc) -> np.array:
        a_df = np.array([db.docref.docfreqs[idx] for idx in doc.idx])
        a_idf = np.log(len(db.mapping) / a_df)
        a_idf = a_idf / a_idf.sum()
        return a_idf

    a_idf1 = idf(doc1)
    a_idf2 = idf(doc2)

    # ---  COMMON OOV

    # information currently unused.
    common_unknown = doc1.unknown.keys() & doc2.unknown.keys()

    # ---  WEIGHTING

    a_weighted_doc1 = a_sims1 * a_idf1
    a_weighted_doc2 = a_sims2 * a_idf2

    n_sim_weighted_doc1 = a_weighted_doc1.sum()
    n_sim_weighted_doc2 = a_weighted_doc2.sum()

    # ---  FINAL SCORE

    n_score1 = n_sim_weighted_doc1
    n_score2 = n_sim_weighted_doc2

    #
    # the important part ends here
    # ----------------------------------------

    if not verbose:
        return n_score1, n_score2, None

    # ---

    # create data object for the scorer to explain itself
    # look at this mess :(
    return n_score1, n_score2, ScoreData(
        score=None, strategy=None,  # set by similarity()
        docs=(doc1, doc2), T=T,
        n_sims=(a_sims1.mean(), a_sims2.mean()),
        a_idxs=(doc1_idxs, doc2_idxs),
        a_sims=(a_sims1, a_sims2),
        n_weighted=(n_sim_weighted_doc1, n_sim_weighted_doc2),
        a_tfs=(doc1.freq, doc2.freq),
        a_idfs=(a_idf1, a_idf2),
        a_weighted=(a_weighted_doc1, a_weighted_doc2),
        n_scores=(n_score1, n_score2),
        common_unknown=common_unknown)


def similarity(db: Database, s_doc1: str, s_doc2: str,
               strategy: Strategy = Strategy.ADAPTIVE_SMALL,
               verbose: bool = False) -> Union[float, ScoreData]:

    assert s_doc1 in db.mapping, f'"{s_doc1}" not in database'
    assert s_doc2 in db.mapping, f'"{s_doc2}" not in database'

    # calculate score

    doc1, doc2 = db.mapping[s_doc1], db.mapping[s_doc2]
    score1, score2, scoredata = _similarity(db, doc1, doc2, verbose)

    # select score based on a strategy

    if strategy is Strategy.MIN:
        score = min(score1, score2)

    if strategy is Strategy.MAX:
        score = max(score1, score2)

    elif strategy is Strategy.ADAPTIVE_SMALL:
        score = score1 if len(doc1) < len(doc2) else score2

    elif strategy is Strategy.ADAPTIVE_BIG:
        score = score2 if len(doc1) < len(doc2) else score1

    if scoredata is not None:
        scoredata.score = score
        scoredata.strategy = strategy.name

    return scoredata if verbose else score


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
