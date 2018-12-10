# -*- coding: utf-8 -*-

from ungol.models import embcodr

import attr
import nltk
import numpy as np
from tqdm import tqdm as _tqdm
from tabulate import tabulate

import pickle
import functools
from collections import defaultdict

from typing import Set
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple


# ---

tqdm = functools.partial(_tqdm, ncols=80)

# --- utility


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
class References:
    """
    To effectively separate shared memory from individual document
    information for Doc instances, this class wraps information shared
    by all documents.
    """

    # see ungol.models.embcodr.load_codes_bin for meta
    meta:       Dict[str, Any] = attr.ib()
    vocabulary: Dict[str, int] = attr.ib()  # (N, )
    codemap:        np.ndarray = attr.ib()  # (N, bytes); np.uint8

    stopwords: Set[str] = attr.ib(default=attr.Factory(set))

    # --- populated when filling the index (db + Doc)

    termfreqs: Dict[int, int] = attr.ib(default=attr.Factory(dict))
    docfreqs:  Dict[int, int] = attr.ib(default=attr.Factory(dict))
    unknown:   Dict[str, int] = attr.ib(default=attr.Factory(dict))  # FIXME

    # documents not added to the index
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
            f_stopwords: List[str] = None, ):

        with open(f_vocab, 'rb') as fd:
            vocab = pickle.load(fd)

        meta, codes = embcodr.load_codes_bin(f_codemap)
        stopwords = load_stopwords(f_stopwords)

        return References(
            meta=meta,
            vocabulary=vocab,
            codemap=codes,
            stopwords=stopwords, )


class DocumentEmptyException(Exception):
    pass


@attr.s
class Doc:

    # document specific attributes
    idx:  np.array = attr.ib()  # (n, ) code indexes
    cnt:  np.array = attr.ib()  # (n, ) word frequency counts
#   freq: np,array = attr.ib()  # calculated by __attr_post_init__

    ref:   References = attr.ib()

    # optionally filled
    unknown:  Dict[str, int] = attr.ib(default=attr.Factory(dict))
    unwanted:            int = attr.ib(default=0)

    # mainly for use in this module
    name: str = attr.ib(default=None)

    # ---

    @property
    def tokens(self) -> List[str]:
        return [self.ref.lookup[idx] for idx in self.idx]

    @property
    def docfreqs(self) -> List[int]:
        return [self.ref.docfreqs[idx] for idx in self.idx]

    @property
    def codes(self) -> np.ndarray:
        return self.ref.codemap[self.idx, ]

    # ---

    def __len__(self) -> int:
        return self.idx.shape[0]

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
        assert self.codes.shape[1] == self.ref.codemap.shape[1]

    def __str__(self):
        str_buf = ['Document: "{}"'.format(
            self.name if self.name else 'Unknown')]

        str_buf += ['tokens: {}, unknown: {}, unwanted: {}\n'.format(
            len(self), len(self.unknown), self.unwanted)]

        header = 'word', 'term count', 'code sum',

        tab_data = []
        row_data = self.tokens, self.cnt, self.codes
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

    @staticmethod
    def from_tokens(name: str, tokens: List[str], ref: References):

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
    def from_text(name: str, text: str, ref: References):
        tokenize = nltk.word_tokenize
        tokens = tokenize(text.lower())
        return Doc.from_tokens(name, tokens, ref)


@attr.s
class Index:

    ref: References = attr.ib()
    mapping: Dict[str, Doc] = attr.ib(default=attr.Factory(dict))

    @property
    def avg_doclen(self):
        if self.valid:
            return self._avg_doclen

        print('recalculating avg doclen')
        count = sum(len(d) for d in self.mapping.values())
        self._avg_doclen = count / len(self.mapping)

        self.valid = True
        return self.avg_doclen

    def __attrs_post_init__(self):
        self.valid = False

    def __add__(self, doc: Doc):
        assert len(doc.idx), 'the document has no content'
        assert doc.name, 'document needs the name attribute set'
        assert doc.name not in self.mapping, f'"{doc.name}" already indexed'

        self.valid = False

        for i, idx in enumerate(doc.idx):
            tf = self.ref.termfreqs
            tf[idx] = tf.get(idx, 0) + doc.cnt[i]

        for idx in set(doc.idx):
            df = self.ref.docfreqs
            df[idx] = df.get(idx, 0) + 1

        self.mapping[doc.name] = doc

        return self

    # ---

    def __str__(self) -> str:
        sbuf = ['VNGOL INDEX']

        sbuf.append('  containing: {} documents'.format(
            len(self.mapping)))

        sbuf.append('  vocabulary: {} words'.format(
            len(self.ref.vocabulary)))

        sbuf.append('  filtering: {} stopwords'.format(
            len(self.ref.stopwords)))

        sbuf.append('  code size: {} bits'.format(
            self.ref.codemap.shape[1] * 8))

        sbuf.append('  tokens: {}'.format(
            len(self.ref.termfreqs)))

        sbuf.append('  avg. doc length: {:.5f}'.format(
            self.avg_doclen))

        sbuf.append('  skipped: {}'.format(
            len(self.ref.skipped)))

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
            (self.ref.termfreqs[idx] for idx in doc.idx),
            (self.ref.docfreqs[idx] for idx in doc.idx), )

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
        if not len(self.common_unknown):
            docbuf.append('Common unknown words: None\n')
            return

        docbuf.append('Common unknown words:\n')
        headers = ('token', 'count')
        tab_data = [(tok, self.docs[a].unknown[tok])
                    for tok in self.common_unknown]

        unknown_table = tabulate(tab_data, headers=headers)
        docbuf.append(unknown_table)

    def _str_sim_table(self, docbuf: List[str], a: int, b: int) -> None:
        headers = ('no', 'token', 'neighbour', 'sim', 'tf', 'idf', 'f(sim)')

        doc1, doc2 = self.docs[a], self.docs[b]

        tab_data = list(zip(
            doc1.tokens,
            [doc2.tokens[idx] for idx in self.a_idxs[a]],
            self.a_sims[a],
            self.a_tfs[a],
            self.a_idfs[a],
            self.a_weighted[a],))

        tab_data.sort(key=lambda t: t[-1], reverse=True)

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
