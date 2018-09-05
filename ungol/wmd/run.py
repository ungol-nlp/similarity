#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Play around with wmd.py from the command line

"""

from ungol.wmd import wmd

import attr
from tabulate import tabulate

import pickle
import pathlib
import argparse
from datetime import datetime

from typing import List
from typing import Callable


# ---


@attr.s
class Stats:
    """
    Each Stats object holds information about
    a single binary distance computation

    """

    doc1: wmd.Doc = attr.ib()  # len -> n1
    doc2: wmd.Doc = attr.ib()  # len -> n2

    score: float = attr.ib()
    delta: float = attr.ib()

    score: wmd.Score = attr.ib()

    def __str__(self) -> str:
        sbuf = ['\n']
        a = sbuf.append
        hline = ['', '-' * 40, '']

        names = self.doc1.name, self.doc2.name
        a('comparing "{}" with "{}"'.format(*names))
        a('  - overall score: {}'.format(self.score.value))
        a('  - doc1 length: {}'.format(len(self.doc1.idx)))
        a('  - doc2 length: {}'.format(len(self.doc2.idx)))
        sbuf += hline

        # add document information

        a('doc1: {}'.format(self.doc1.name))
        a(str(self.doc1))
        sbuf += hline

        a('doc2: {}'.format(self.doc2.name))
        a(str(self.doc2))
        sbuf += hline

        # print (R)WMD calculation details

        a(str(self.score))

        return '\n'.join(sbuf) + '\n'

# ---


def timer() -> Callable[[], float]:
    stamp = datetime.now()

    def proxy():
        delta = datetime.now() - stamp
        return delta.total_seconds() * 1000

    return proxy


# currently unused.
def _gen_combinations(pool: List[wmd.Doc]):
    for i in range(len(pool) - 1):
        for doc in pool[i + 1:]:
            yield pool[i], doc


def _load_query(f_query, argv):

    with open(f_query, 'rb') as fd:
        topic = pickle.load(fd)
        f_tok = 'topic.description.tok'
        f_content = str(pathlib.Path(f_query).parent / f_tok)
        query = wmd.Doc.from_text(topic['title'], f_content, *argv)

    return query


def _load_docs(f_docs, argv, kwargv):

    doc_paths = [pathlib.Path(fname) for fname in f_docs]
    docs: List[wmd.Doc] = []

    for path in doc_paths:
        doc = wmd.Doc.from_text(path.name, str(path), *argv)
        docs.append(doc)

    return docs


def _calculate_distances(query: wmd.Doc, docs: List[wmd.Doc]) -> Stats:

    delta = timer()
    gen = ((query, doc) for doc in docs)

    stats = []
    for doc1, doc2 in gen:
        dist_delta = timer()

        # --- critical section

        score: wmd.Score = wmd.dist(doc1, doc2, verbose=True)

        # --- gather stats

        stat = Stats(
            doc1=doc1, doc2=doc2,
            score=score, delta=dist_delta(), )

        stats.append(stat)

    print('computation took {:.3f}ms\n'.format(delta()))
    stats.sort(key=lambda s: s.score, reverse=True)
    return stats


def main(args):
    global STATS
    STATS = True

    print('\n', 'welcome to vngol wmd'.upper(), '\n')
    print('please note: binary data loaded is not checked for malicious')
    print('content - never load anything you did not produce!\n')

    refs = wmd.DocReferences.from_files(
        args.codemap, args.vocabulary, args.stopwords)

    # not using the narrative because it usually enumerates
    # whats _not_ to be retrieved... (maybe this can be used)
    with open(args.query, 'r') as fd:
        res = fd.read().split('\n\n')
        title, desc, narrative = res
        query = wmd.Doc.from_text(title, desc, refs)

    docs = []
    for f_doc in args.docs:
        with open(f_doc, 'r') as fd:
            docs.append(wmd.Doc.from_text(f_doc, fd.read(), refs))

    stats = _calculate_distances(query, docs)

    print(tabulate(
        [(s.doc1.name, s.doc2.name, s.score.value) for s in stats],
        headers=('doc1', 'doc2', 'score'), ))

    for stat in stats:
        print(str(stat))

    # FIXME: write stats to file


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'codemap', type=str,
        help='binary code file produced by ungol-models/ungol.embcodr', )

    parser.add_argument(
        'vocabulary', type=str,
        help='pickle produced by ungol-models/ungol.analyze.vocabulary', )

    parser.add_argument(
        '-d', '--docs', type=str, nargs='+',
        help='documents to compare', )

    parser.add_argument(
        '-q', '--query', type=str,
        help='query for the database', )

    parser.add_argument(
        'out', type=str,
        help='file to write results and statistics to')

    # optional

    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='verbose output with tables and such', )

    parser.add_argument(
        '-s', '--stopwords', nargs='*',
        help='lists of words to ignore', )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
