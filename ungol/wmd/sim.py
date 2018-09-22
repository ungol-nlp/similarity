# -*- coding: utf-8 -*-


from ungol.wmd import wmd
from ungol.wmd import stats
from ungol.wmd import rhwmd as _rhwmd

import numpy as np

from typing import Union


# --- DISTANCE SCHEMES

#
#
#  HR-WMD |----------------------------------------
#
#
#
#  FIXME: speech about what I learned about ripping apart the calculation.
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

    # Compute the distance matrix.
    T = _rhwmd.distance_matrix_lookup(doc1, doc2)

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
        a_known_df = np.array([db.docref.docfreqs[idx] for idx in doc.idx])
        a_df = np.hstack((a_unknown, a_known_df))

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

    a_weighted_doc1, n_score1 = weighted(a_sims1, a_idf1_norm)
    a_weighted_doc2, n_score2 = weighted(a_sims2, a_idf2_norm)

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
        name='hr-wmd', score=None, docs=(doc1, doc2),
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
        np.array(doc2.tokens)[doc1_idxs],
        np.array(doc1.tokens)[doc2_idxs], )

    scoredata.add_local_column('sim', a_sims1, a_sims2)
    scoredata.add_local_column('tf', doc1.freq, doc2.freq)
    scoredata.add_local_column('idf', a_idf1, a_idf2)
    scoredata.add_local_column('weight', a_weighted_doc1, a_weighted_doc2)

    return n_score1, n_score2, scoredata

    #     n_sims=(a_sims1.mean(), a_sims2.mean()),

    #     a_idxs=(doc1_idxs, doc2_idxs),
    #     n_weighted=(n_sim_weighted_doc1, n_sim_weighted_doc2),
    #     a_tfs=(doc1.freq, doc2.freq),
    #     a_idfs=(a_idf1[U:], a_idf2[U:]),
    #     a_weighted=(a_weighted_doc1[U:], a_weighted_doc2[U:]),
    #     n_scores=(n_score1, n_score2),
    #     common_unknown=common_unknown)


def rhwmd(db: wmd.Database, s_doc1: str, s_doc2: str,
          strategy: _rhwmd.Strategy = _rhwmd.Strategy.ADAPTIVE_SMALL,
          verbose: bool = False) -> Union[float, stats.ScoreData]:

    assert s_doc1 in db.mapping, f'"{s_doc1}" not in database'
    assert s_doc2 in db.mapping, f'"{s_doc2}" not in database'

    # calculate score

    doc1, doc2 = db.mapping[s_doc1], db.mapping[s_doc2]
    score1, score2, scoredata = _rhwmd_similarity(db, doc1, doc2, verbose)

    # select score based on a strategy

    if strategy is _rhwmd.Strategy.MIN:
        score = min(score1, score2)

    if strategy is _rhwmd.Strategy.MAX:
        score = max(score1, score2)

    elif strategy is _rhwmd.Strategy.ADAPTIVE_SMALL:
        score = score1 if len(doc1) < len(doc2) else score2

    elif strategy is _rhwmd.Strategy.ADAPTIVE_BIG:
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
def bm25(db: wmd.Database, s_doc1: str, s_doc2: str, k1=1.56, b=0.45):
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

