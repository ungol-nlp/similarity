# -*- coding: utf-8 -*-

"""

This is an implementation of the Okapi BM25 relevance score using the
ungol.wmd infrastructure.

"""

from ungol.wmd import wmd

import numpy as np


def similarity(db: wmd.Database, s_doc1: str, s_doc2: str, k1=1.56, b=0.45):
    ref = db.docref

    assert s_doc1 in db.mapping, f'"{s_doc1}" not in database'
    assert s_doc2 in db.mapping, f'"{s_doc2}" not in database'

    #  this can be optimized performance wise,
    # but i'll leave it for now

    doc1, doc2 = db.mapping[s_doc1], db.mapping[s_doc2]

    # gather common tokens
    common = set(doc1.tokens) & set(doc2.tokens)

    # get code indexes
    a_common_idx = np.array([ref.vocabulary[t] for t in common])

    # find corresponding document frequencies
    a_df = np.array([ref.docfreqs[idx] for idx in a_common_idx])

    # calculate idf value
    a_idf = np.log(len(db.mapping) / a_df)

    # find corresponding token counts
    a_tf = doc2.cnt[np.nonzero(a_common_idx[:, None] == doc2.idx)[1]]
    assert len(a_tf) == len(common)

    # calculate numerator
    a_num = (k1 + 1) * a_tf

    # calculate denominator
    n_len = len(doc2) / db.avg_doclen
    a_den = k1 * ((1-b) + b * n_len) + a_tf

    # weight each idf value
    a_res = a_idf * a_num / a_den

    return a_res.sum()
