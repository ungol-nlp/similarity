# -*- coding: utf-8 -*-


from ungol.wmd import wmd
from ungol.wmd import stats
from ungol.wmd import rhwmd as _rhwmd

import numpy as np


Strategy = _rhwmd.Strategy


def _get_docs(db: wmd.Database, s_doc1: str, s_doc2: str):
    assert s_doc1 in db.mapping, f'"{s_doc1}" not in database'
    assert s_doc2 in db.mapping, f'"{s_doc2}" not in database'
    return db.mapping[s_doc1], db.mapping[s_doc2]


# --- DISTANCE SCHEMES

#
#
#  HR-WMD |----------------------------------------
#
#
#
#  TODO: speech about what I learned about ripping apart the calculation.
#  Notes:
#    - prefixes: s_* for str, a_* for np.array, n_ for scalars
#
def _rhwmd_similarity(
        db: wmd.Database,
        doc1: wmd.Doc, doc2: wmd.Doc,
        verbose: bool) -> float:

    # ----------------------------------------
    # this is the important part
    #

    a_sims, a_idxs = _rhwmd.retrieve_nn(doc1, doc2)

    # phony
    # a_sims1 = np.ones(doc1_idxs.shape[0])
    # a_sims2 = np.ones(doc2_idxs.shape[0])

    # ---  COMMON OOV

    common_unknown = doc1.unknown.keys() & doc2.unknown.keys()
    # U = len(common_unknown)
    U = 0
    a_unknown = np.ones(U)

    # ---  IDF

    def idf(doc) -> np.array:
        a_df = np.hstack((a_unknown, np.array(doc.docfreqs)))
        N = len(db.mapping)  # FIXME add unknown tokens
        a_idf = np.log(N / a_df)
        return a_idf

    a_idf1 = idf(doc1)
    a_idf2 = idf(doc2)

    # phony
    # a_idf1 = np.ones(len(doc1)) / len(doc1)
    # a_idf2 = np.ones(len(doc2)) / len(doc2)

    # ---  WEIGHTING

    boost = 1

    def weighted(a_sims, a_idf):
        a = np.hstack((np.ones(U), a_sims)) * a_idf
        s1, s2 = a[:U].sum(), a[U:].sum()
        return a, boost * s1 + s2

    a_idf1_norm = a_idf1 / a_idf1.sum()
    a_idf2_norm = a_idf2 / a_idf2.sum()

    a_weighted_doc1, n_score1 = weighted(a_sims[0], a_idf1_norm)
    a_weighted_doc2, n_score2 = weighted(a_sims[1], a_idf2_norm)

    # assert 0 <= n_sim_weighted_doc1 and n_sim_weighted_doc1 <= 1
    # assert 0 <= n_sim_weighted_doc2 and n_sim_weighted_doc2 <= 1

    #
    # the important part ends here
    # ----------------------------------------

    if not verbose:
        return n_score1, n_score2, None

    # ---

    # create data object for the scorer to explain itself.
    # the final score is set by the caller.
    scoredata = stats.ScoreData(
        name='rh-wmd', score=None, docs=(doc1, doc2),
        common_unknown=common_unknown)

    # ---

    scoredata.add_local_row('score', n_score1, n_score2)

    # ---

    scoredata.add_local_column(
        'token',
        np.array(doc1.tokens),
        np.array(doc2.tokens), )

    scoredata.add_local_column(
        'nn',
        np.array(doc2.tokens)[a_idxs[0]],
        np.array(doc1.tokens)[a_idxs[1]], )

    scoredata.add_local_column('sim', *a_sims)
    scoredata.add_local_column('tf', doc1.freq, doc2.freq)
    scoredata.add_local_column('idf', a_idf1, a_idf2)
    scoredata.add_local_column('weight', a_weighted_doc1, a_weighted_doc2)

    return n_score1, n_score2, scoredata


def rhwmd(db: wmd.Database, s_doc1: str, s_doc2: str,
          strategy: Strategy = Strategy.ADAPTIVE_SMALL,
          verbose: bool = False):

    doc1, doc2 = _get_docs(db, s_doc1, s_doc2)
    score1, score2, scoredata = _rhwmd_similarity(db, doc1, doc2, verbose)

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
        scoredata.add_global_row('strategy', strategy.name)

    return scoredata if verbose else score


#
#
#  OKAPI BM25 |----------------------------------------
#
#
def _bm25_normalization(a_tf, n_len: int, k1: float, b: float):
    # calculate numerator
    a_num = (k1 + 1) * a_tf
    # calculate denominator
    a_den = k1 * ((1-b) + b * n_len) + a_tf
    return a_num / a_den


def bm25(db: wmd.Database, s_doc1: str, s_doc2: str, k1=1.56, b=0.45):
    doc1, doc2 = _get_docs(db, s_doc1, s_doc2)
    ref = db.docref

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

    n_len = len(doc2) / db.avg_doclen
    a_norm = _bm25_normalization(a_tf, n_len, k1, b)

    # weight each idf value
    a_res = a_idf * a_norm

    return a_res.sum()


#
#
#  RH-WMD-25 |----------------------------------------
#
#
def rhwmd25(db: wmd.Database, s_doc1: str, s_doc2: str,
            k1=1.56, b=0.45,
            verbose: bool = False, ):

    doc1, doc2 = _get_docs(db, s_doc1, s_doc2)
    a_nn_sims, a_nn_idxs = _rhwmd.retrieve_nn(doc1, doc2)

    a_df = np.array([db.docref.docfreqs[idx] for idx in doc1.idx])
    a_idf = np.log(len(db.mapping) / a_df)
    a_tf = np.array([doc2.cnt[i] for i in a_nn_idxs[0]])

    n_len = len(doc2) / db.avg_doclen
    a_norm = _bm25_normalization(a_tf, n_len, k1, b)

    a_res = a_idf * a_norm * a_nn_sims[0]
    return a_res.sum()
