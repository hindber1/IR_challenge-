"""
Fast hybrid experiment using scipy sparse BM25.
Scores all 100 queries against 20k docs in one matrix multiply per index.
"""
import sys, json, math, re
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

DATA_DIR   = Path("../data")
queries    = pd.read_parquet(DATA_DIR / "queries.parquet")
corpus     = pd.read_parquet(DATA_DIR / "corpus.parquet")
with open(DATA_DIR / "qrels.json", encoding="utf-8") as f:
    qrels = json.load(f)
query_ids  = queries["doc_id"].tolist()
corpus_ids = corpus["doc_id"].tolist()

# ── Text helpers ──────────────────────────────────────────────────────────────
def fmt_ta(row):
    t = str(row.get("title","") or "").strip()
    a = str(row.get("abstract","") or "").strip()
    return (t + " " + a).strip()

def fmt_full(row, cap=8000):
    ta = fmt_ta(row)
    ft = str(row.get("full_text","") or "").strip()
    return (ta + " " + ft[:cap]).strip()

def corpus_texts(cap=None):
    if cap is None:
        return [fmt_ta(r) for _, r in corpus.iterrows()]
    return [fmt_full(r, cap) for _, r in corpus.iterrows()]

def query_texts(cap=None):
    if cap is None:
        return [fmt_ta(r) for _, r in queries.iterrows()]
    return [fmt_full(r, cap) for _, r in queries.iterrows()]

# ── Fast scipy BM25 ───────────────────────────────────────────────────────────
def build_bm25(corpus_txt, k1=1.5, b=0.75, min_df=2, max_df=0.95):
    """Return (bm25_matrix, vectorizer).
    bm25_matrix shape: (n_docs, n_terms)  — precomputed BM25 weights.
    """
    cv = CountVectorizer(
        token_pattern=r"[a-z0-9]+",
        min_df=min_df, max_df=max_df,
        lowercase=True,
    )
    tf = cv.fit_transform(corpus_txt).astype(np.float32)   # (n_docs, n_terms)
    n_docs = tf.shape[0]

    # Document frequency per term
    df = np.diff(tf.tocsc().indptr).astype(np.float32)
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    # Document lengths
    dl     = np.asarray(tf.sum(axis=1)).ravel()
    avgdl  = dl.mean()

    # Apply BM25 TF normalization element-wise on the sparse matrix
    tf_csr   = tf.tocsr()
    rows, cols = tf_csr.nonzero()
    tf_vals  = np.asarray(tf_csr[rows, cols]).ravel()
    dl_norm  = k1 * (1.0 - b + b * dl[rows] / avgdl)
    bm25_val = idf[cols] * tf_vals * (k1 + 1.0) / (tf_vals + dl_norm)

    bm25 = sp.csr_matrix((bm25_val, (rows, cols)), shape=tf.shape, dtype=np.float32)
    return bm25, cv


def retrieve_fast(bm25_mat, cv, query_txt, top_n=100):
    """Score all queries at once; return {qid: [doc_id, ...]}."""
    q_tf  = cv.transform(query_txt).astype(np.float32)        # (n_q, n_terms)
    # Binary query: presence, not frequency (standard BM25 query side)
    q_bin = (q_tf > 0).astype(np.float32)
    # (n_q, n_terms) @ (n_terms, n_docs) = (n_q, n_docs)
    scores = (q_bin @ bm25_mat.T).toarray()                    # dense (100, 20000)
    result = {}
    for i, qid in enumerate(query_ids):
        top_idx = np.argsort(-scores[i])[:top_n]
        result[qid] = [corpus_ids[j] for j in top_idx]
    return result


# ── Metrics ───────────────────────────────────────────────────────────────────
def ev(sub):
    r10, n10, r100, ap_list = [], [], [], []
    for qid, ranked in sub.items():
        rel = set(qrels.get(qid, []))
        if not rel:
            continue
        r10.append(len(set(ranked[:10]) & rel) / len(rel))
        r100.append(len(set(ranked[:100]) & rel) / len(rel))
        dcg  = sum(1./math.log2(i+2) for i,d in enumerate(ranked[:10]) if d in rel)
        idcg = sum(1./math.log2(i+2) for i in range(min(len(rel),10)))
        n10.append(dcg/idcg if idcg else 0.)
        h, s = 0, 0.
        for i, d in enumerate(ranked):
            if d in rel:
                h += 1; s += h/(i+1)
        ap_list.append(s/len(rel))
    return np.mean(r10), np.mean(r100), np.mean(n10), np.mean(ap_list)

def show(label, sub):
    r10, r100, n10, ap = ev(sub)
    marker = " <-- BEST" if r10 >= 0.60 else ""
    print(f"  {label:<55}  r@10={r10:.4f}  r@100={r100:.4f}  n@10={n10:.4f}  map={ap:.4f}{marker}")


def rrf(*subs, k=60):
    result = {}
    for qid in query_ids:
        scores = {}
        for sub in subs:
            for rank, d in enumerate(sub.get(qid, [])):
                scores[d] = scores.get(d, 0.) + 1./(k + rank + 1)
        result[qid] = sorted(scores, key=scores.__getitem__, reverse=True)[:100]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# A. MiniLM dense (pre-computed)
# ─────────────────────────────────────────────────────────────────────────────
EMB = DATA_DIR / "embeddings" / "sentence-transformers_all-MiniLM-L6-v2"
q_e = np.load(EMB / "query_embeddings.npy").astype(np.float32)
c_e = np.load(EMB / "corpus_embeddings.npy").astype(np.float32)
with open(EMB / "query_ids.json")  as f: eq = json.load(f)
with open(EMB / "corpus_ids.json") as f: ec = json.load(f)
sim = q_e @ c_e.T
top = np.argsort(-sim, axis=1)[:, :100]
mLM = {qid: [ec[j] for j in top[i]] for i, qid in enumerate(eq)}
show("A. MiniLM-L6-v2 (dense baseline)", mLM)
print()

# ─────────────────────────────────────────────────────────────────────────────
# B. Build BM25 indices (corpus side)
# ─────────────────────────────────────────────────────────────────────────────
print("Building BM25 indices (corpus)...")

# Title-only index
bm_title_mat, cv_title = build_bm25(
    [str(r.get("title","") or "") for _, r in corpus.iterrows()],
    min_df=1, max_df=1.0,
)
print(f"  title index:  {bm_title_mat.shape}  nnz={bm_title_mat.nnz:,}")

# TA index
bm_ta_mat, cv_ta = build_bm25(corpus_texts(cap=None))
print(f"  TA index:     {bm_ta_mat.shape}  nnz={bm_ta_mat.nnz:,}")

# FT-8k index
bm_ft8_mat, cv_ft8 = build_bm25(corpus_texts(cap=8000))
print(f"  FT-8k index:  {bm_ft8_mat.shape}  nnz={bm_ft8_mat.nnz:,}")

# FT-full index
bm_ftfull_mat, cv_ftfull = build_bm25(corpus_texts(cap=None))
print(f"  FT-full index:{bm_ftfull_mat.shape}  nnz={bm_ftfull_mat.nnz:,}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# C. Query text variants
# ─────────────────────────────────────────────────────────────────────────────
q_ta    = query_texts(cap=None)
q_3k    = query_texts(cap=3000)
q_5k    = query_texts(cap=5000)
q_full  = query_texts(cap=None)   # full query text (same as q_ta since we cap the full_text portion)

# Actually build proper full-query text (TA + full_text capped)
q_full_3k  = [fmt_full(r, 3000) for _, r in queries.iterrows()]
q_full_5k  = [fmt_full(r, 5000) for _, r in queries.iterrows()]
q_full_10k = [fmt_full(r, 10000) for _, r in queries.iterrows()]

# ─────────────────────────────────────────────────────────────────────────────
# D. BM25 single-model results
# ─────────────────────────────────────────────────────────────────────────────
print("=== Single BM25 models ===")

# Title index, different query texts
b_tit_qta   = retrieve_fast(bm_title_mat, cv_title, q_ta)
b_tit_q3k   = retrieve_fast(bm_title_mat, cv_title, q_full_3k)
b_tit_q5k   = retrieve_fast(bm_title_mat, cv_title, q_full_5k)
b_tit_q10k  = retrieve_fast(bm_title_mat, cv_title, q_full_10k)
show("B1. BM25 title / TA-query",          b_tit_qta)
show("B2. BM25 title / FT-3k-query",       b_tit_q3k)
show("B3. BM25 title / FT-5k-query",       b_tit_q5k)
show("B4. BM25 title / FT-10k-query",      b_tit_q10k)

# TA index
b_ta_qta    = retrieve_fast(bm_ta_mat,    cv_ta,    q_ta)
b_ta_q5k    = retrieve_fast(bm_ta_mat,    cv_ta,    q_full_5k)
show("B5. BM25 TA    / TA-query",          b_ta_qta)
show("B6. BM25 TA    / FT-5k-query",       b_ta_q5k)

# FT-8k index
b_ft8_qta   = retrieve_fast(bm_ft8_mat,   cv_ft8,   q_ta)
b_ft8_q3k   = retrieve_fast(bm_ft8_mat,   cv_ft8,   q_full_3k)
b_ft8_q5k   = retrieve_fast(bm_ft8_mat,   cv_ft8,   q_full_5k)
b_ft8_q10k  = retrieve_fast(bm_ft8_mat,   cv_ft8,   q_full_10k)
show("B7. BM25 FT-8k / TA-query  [prev best BM25]", b_ft8_qta)
show("B8. BM25 FT-8k / FT-3k-query",               b_ft8_q3k)
show("B9. BM25 FT-8k / FT-5k-query",               b_ft8_q5k)
show("B10.BM25 FT-8k / FT-10k-query",              b_ft8_q10k)

# FT-full index
b_ftf_qta   = retrieve_fast(bm_ftfull_mat, cv_ftfull, q_ta)
b_ftf_q5k   = retrieve_fast(bm_ftfull_mat, cv_ftfull, q_full_5k)
b_ftf_q10k  = retrieve_fast(bm_ftfull_mat, cv_ftfull, q_full_10k)
show("B11.BM25 FT-full/ TA-query",                  b_ftf_qta)
show("B12.BM25 FT-full/ FT-5k-query",               b_ftf_q5k)
show("B13.BM25 FT-full/ FT-10k-query",              b_ftf_q10k)
print()

# ─────────────────────────────────────────────────────────────────────────────
# E. Hybrid RRF combinations
# ─────────────────────────────────────────────────────────────────────────────
print("=== Hybrid RRF combinations ===")

show("C1.  RRF(MiniLM + FT8/TA)              [prev best]", rrf(mLM, b_ft8_qta))
show("C2.  RRF(MiniLM + FT8/FT-5k)",                      rrf(mLM, b_ft8_q5k))
show("C3.  RRF(MiniLM + FT8/FT-10k)",                     rrf(mLM, b_ft8_q10k))
show("C4.  RRF(MiniLM + title/FT-5k)",                    rrf(mLM, b_tit_q5k))
show("C5.  RRF(MiniLM + title/FT-10k)",                   rrf(mLM, b_tit_q10k))
show("C6.  RRF(MiniLM + FT8/TA  + title/FT-5k)",         rrf(mLM, b_ft8_qta, b_tit_q5k))
show("C7.  RRF(MiniLM + FT8/FT-5k + title/FT-5k)",       rrf(mLM, b_ft8_q5k, b_tit_q5k))
show("C8.  RRF(MiniLM + FT8/FT-10k + title/FT-10k)",     rrf(mLM, b_ft8_q10k, b_tit_q10k))
show("C9.  RRF(MiniLM + FT8/TA + FT8/FT-5k + title/FT-5k)",
                                                           rrf(mLM, b_ft8_qta, b_ft8_q5k, b_tit_q5k))
show("C10. RRF(MiniLM + FTfull/TA + FTfull/FT-5k + title/FT-5k)",
                                                           rrf(mLM, b_ftf_qta, b_ftf_q5k, b_tit_q5k))
show("C11. RRF(MiniLM + FT8/TA + FT8/FT-3k + FT8/FT-5k + FT8/FT-10k + title/FT-10k)",
                                                           rrf(mLM, b_ft8_qta, b_ft8_q3k, b_ft8_q5k, b_ft8_q10k, b_tit_q10k))
show("C12. RRF(MiniLM + FTfull/FT-5k + title/FT-10k)",
                                                           rrf(mLM, b_ftf_q5k, b_tit_q10k))
print()
print("Done.")
