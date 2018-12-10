# -*- coding: utf-8 -*-


from ungol.index import index as uii
from ungol.similarity import stats
from ungol.similarity import rhwmd as _rhwmd

import numpy as np


Strategy = _rhwmd.Strategy


def _get_docs(db: uii.Index, s_doc1: str, s_doc2: str):
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
        index: uii.Index,
        doc1: uii.Doc, doc2: uii.Doc,
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
        N = len(index.mapping)  # FIXME add unknown tokens
        a_idf = np.log(N / a_df)
        return a_idf

    # --- idf combinations

    a_idf_doc1 = idf(doc1)
    a_idf_doc2 = idf(doc2)

    a_idf_nn1 = a_idf_doc2[a_idxs[0]]
    a_idf_nn2 = a_idf_doc1[a_idxs[1]]

    # a_idf1 = a_idf_doc1 + a_idf_nn1
    # a_idf2 = a_idf_doc2 + a_idf_nn2

    # --- query idf

    a_idf1 = a_idf_doc1
    a_idf2 = a_idf_doc2

    # --- phony

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
        name='rhwmd', score=None, docs=(doc1, doc2),
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
    scoredata.add_local_column('tf(token)', doc1.freq, doc2.freq)
    scoredata.add_local_column('idf(token)', a_idf_doc1, a_idf_doc2)
    scoredata.add_local_column('idf(nn)', a_idf_nn1, a_idf_nn2)
    scoredata.add_local_column('idf', a_idf1, a_idf2)
    scoredata.add_local_column('weight', a_weighted_doc1, a_weighted_doc2)

    return n_score1, n_score2, scoredata


def rhwmd(index: uii.Index, s_doc1: str, s_doc2: str,
          strategy: Strategy = Strategy.ADAPTIVE_SMALL,
          verbose: bool = False):

    doc1, doc2 = _get_docs(index, s_doc1, s_doc2)
    score1, score2, scoredata = _rhwmd_similarity(index, doc1, doc2, verbose)

    # select score based on a strategy

    if strategy is Strategy.MIN:
        score = min(score1, score2)

    elif strategy is Strategy.MAX:
        score = max(score1, score2)

    elif strategy is Strategy.ADAPTIVE_SMALL:
        score = score1 if len(doc1) < len(doc2) else score2

    elif strategy is Strategy.ADAPTIVE_BIG:
        score = score2 if len(doc1) < len(doc2) else score1

    elif strategy is Strategy.SUM:
        score = score1 + score2

    else:
        assert False, f'unknown strategy: "{strategy}"'

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


def bm25(index: uii.Index, s_doc1: str, s_doc2: str,
         k1=1.56, b=0.45, verbose: bool = False):

    doc1, doc2 = _get_docs(index, s_doc1, s_doc2)
    ref = index.ref

    # gather common tokens
    common = set(doc1.tokens) & set(doc2.tokens)
    if not len(common):
        return 0 if not verbose else stats.ScoreData(
            name='bm25', score=0, docs=(doc1, doc2))

    # get code indexes
    a_common_idx = np.array([ref.vocabulary[t] for t in common])

    # find corresponding document frequencies
    a_df = np.array([ref.docfreqs[idx] for idx in a_common_idx])

    # calculate idf value
    a_idf = np.log(len(index.mapping) / a_df)

    # find corresponding token counts
    # note: a[:, None] == np.array([a]).T
    a_tf = doc2.cnt[np.nonzero(a_common_idx[:, None] == doc2.idx)[1]]
    assert len(a_tf) == len(common)

    n_len = len(doc2) / index.avg_doclen
    a_norm = _bm25_normalization(a_tf, n_len, k1, b)

    # weight each idf value
    a_res = a_idf * a_norm
    score = a_res.sum()

    if not verbose:
        return score

    # ---

    scoredata = stats.ScoreData(name='bm25', score=score, docs=(doc1, doc2), )

    # to preserve order
    a_common_words = np.array([ref.lookup[idx] for idx in a_common_idx])

    col_data = (
        ('token', a_common_words),
        ('tf', a_tf),
        ('df', a_df),
        ('idf', a_idf),
        ('norm', a_norm),
        ('weight', a_res), )

    for t in col_data:
        scoredata.add_global_column(*t)

    return scoredata


#
#
#  RH-WMD-25 |----------------------------------------
#
#
def rhwmd25(index: uii.Index, s_doc1: str, s_doc2: str,
            k1=1.56, b=0.45,
            verbose: bool = False, ):

    doc1, doc2 = _get_docs(index, s_doc1, s_doc2)
    (a_nn_sim, _), (a_nn_idx, _) = _rhwmd.retrieve_nn(doc1, doc2)

    a_df = np.array([index.ref.docfreqs[idx] for idx in doc1.idx])
    a_idf = np.log(len(index.mapping) / a_df)
    a_tf = np.array([doc2.cnt[i] for i in a_nn_idx])

    n_len = len(doc2) / index.avg_doclen
    a_norm = _bm25_normalization(a_tf, n_len, k1, b)

    a_res = a_idf * a_norm * a_nn_sim
    score = a_res.sum()

    if not verbose:
        return score

    # ---

    scoredata = stats.ScoreData(
        name='rhwmd25', score=score, docs=(doc1, doc2),
        common_unknown=set(), )

    col_data = (
        ('token', np.array(doc1.tokens)),
        ('nn', np.array(doc2.tokens)[a_nn_idx]),
        ('sims', a_nn_sim),
        ('df(token)', a_df),
        ('idf(token)', a_idf),
        ('tf(nn)', a_tf),
        ('norm(nn)', a_norm),
        ('weight', a_res), )

    for t in col_data:
        scoredata.add_global_column(*t)

    return scoredata


#
#
#  TF-IDF |----------------------------------------
#
#
def tfidf(index: uii.Index, s_doc1: str, s_doc2: str, verbose: bool = False, ):

    doc1, doc2 = _get_docs(index, s_doc1, s_doc2)

    common = list(set(doc1.tokens) & set(doc2.tokens))
    if not len(common):
        return 0 if not verbose else stats.ScoreData(
            name='tfidf', score=0, docs=(doc1, doc2))

    # get code indexes
    a_common_idx = np.array([index.ref.vocabulary[t] for t in common])

    a_df = np.array([index.ref.docfreqs[idx] for idx in a_common_idx])
    a_idf = np.log(len(index.mapping) / a_df)

    mapping = np.hstack([np.where(doc2.idx == i) for i in a_common_idx])[0]
    a_tf = np.array([doc2.cnt[i] for i in mapping])

    a_res = a_idf * a_tf
    score = a_res.sum()

    if not verbose:
        return score

    # ---

    scoredata = stats.ScoreData(
        name='tfidf', score=score, docs=(doc1, doc2),
        common_unknown=set(), )

    col_data = (
        ('token', np.array(common)),
        ('idx', a_common_idx),
        ('tf(token)', a_tf),
        ('df(token)', a_df),
        ('idf(token)', a_idf),
        ('result', a_res, ), )

    for t in col_data:
        scoredata.add_global_column(*t)

    return scoredata
