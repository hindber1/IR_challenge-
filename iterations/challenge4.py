#!/usr/bin/env python3
"""
iteration_next_candidate — 5-signal adaptive hybrid retrieval pipeline.

Retrieval signals:
  Dense  = (1 - eta) * MiniLM  +  eta * SPECTER2
           Corpus: loaded from precomputed .npy caches under data/embeddings/.
                   SPECTER2 is encoded and cached automatically on first run.
           Queries: always encoded at runtime with title+abstract+full_text
                    (cached in submissions/iteration_next_candidate/).

  Sparse = sparse_beta * TF-IDF(full_text ≤ 80k chars)
         + (1 - sparse_beta) * TF-IDF(title+abstract)
           Both channels: sublinear_tf, min_df=2, max_df=0.95,
           ngram_range=(1,2), stop_words='english'.

  Metadata boosts applied to raw dense and sparse scores before per-query
  min-max normalisation:
    domain_boost      if query.domain == corpus.domain
    venue_boost       if query.venue == corpus.venue (skip empty)
    year_boost        if query.year == corpus.year (exact match)
    year_close_boost  if |query.year - corpus.year| == 1 (adjacent)

  Adaptive alpha (per query):
    alpha_q  = clip(alpha - alpha_domain_boost * rare_flag, 0.20, 0.90)
    Adjusted by +/- alpha_gap_delta based on top-1/top-2 sparse score gap.
    Charlotte = alpha_q * norm(dense_aug) + (1 - alpha_q) * norm(sparse_aug)

  RRF second stage (lightweight, rank-only):
    rrf_score = 1/(RRF_K + rank_dense) + 1/(RRF_K + rank_sparse), k=60
    final     = (1 - gamma_rrf) * charlotte + gamma_rrf * norm(rrf)

  Optional hard_domain_first: same-domain docs ranked before others.
  Self-citation removal: query doc_id always stripped from its ranked list.

Two-stage hyperparameter tuning on public qrels (NDCG@10):
  Stage 1: joint grid over all params with gamma_rrf=0 fixed.
  Stage 2: tune eta_rrf and gamma_rrf with stage-1 best params fixed.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# iterations/ → challenge root → scripts/
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from submission_utils import (
    DEFAULT_TOP_K,
    challenge_dir,
    create_submission_zip,
    data_paths,
    iteration_submission_paths,
    load_queries_corpus,
    save_submission,
    validate_doc_ids_in_corpus,
    validate_submission,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ITERATION_NAME = "iteration_next_candidate"
TOP_K = DEFAULT_TOP_K

MINILM_EMB_SUBDIR = "sentence-transformers_all-MiniLM-L6-v2"
SPECTER_EMB_SUBDIR = "allenai_specter2_base"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPECTER_MODEL = "allenai/specter2_base"

MINILM_ENCODE_BATCH = 128
SPECTER_ENCODE_BATCH = 32
RRF_K = 60.0
SPARSE_MAX_CHARS = 80_000


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------

def _s(v) -> str:
    return str(v or "").strip()


def format_minilm_query(row: pd.Series) -> str:
    """Full-text concatenation for MiniLM query encoding."""
    parts = [_s(row.get(f)) for f in ("title", "abstract", "full_text")]
    return " ".join(p for p in parts if p)


def format_specter_query(row: pd.Series, sep: str) -> str:
    """SPECTER-style segments joined with the model's sep_token."""
    parts = [_s(row.get(f)) for f in ("title", "abstract", "full_text")]
    parts = [p for p in parts if p]
    return f" {sep} ".join(parts) if parts else ""


def format_sparse_full(row: pd.Series) -> str:
    """Full-text for sparse indexing, capped at SPARSE_MAX_CHARS characters."""
    parts = [_s(row.get(f)) for f in ("title", "abstract", "full_text")]
    text = " ".join(p for p in parts if p)
    return text[:SPARSE_MAX_CHARS]


def format_sparse_ta(row: pd.Series) -> str:
    """Title + abstract for the TA sparse channel."""
    parts = [_s(row.get("title")), _s(row.get("abstract"))]
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def years_as_int64(series: pd.Series) -> np.ndarray:
    """-1 for missing or non-parseable year values."""
    out = np.full(len(series), -1, dtype=np.int64)
    for i, v in enumerate(series.to_numpy()):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        try:
            out[i] = int(v)
        except (TypeError, ValueError):
            pass
    return out


def corpus_df_ordered(corpus: pd.DataFrame, corpus_ids: np.ndarray) -> pd.DataFrame:
    """Reindex corpus to match the order given by the embedding-cache corpus_ids."""
    ids = [str(x) for x in corpus_ids.tolist()]
    indexed = corpus.set_index(corpus["doc_id"].astype(str), drop=False)
    missing = set(ids) - set(indexed.index)
    if missing:
        raise ValueError(
            f"{len(missing)} corpus_ids from the embedding cache are missing in corpus.parquet"
        )
    return indexed.loc[ids]


# ---------------------------------------------------------------------------
# Embedding encode / load helpers
# ---------------------------------------------------------------------------

def _encode_minilm(texts: List[str], device=None) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MINILM_MODEL, **({"device": device} if device else {}))
    return model.encode(
        texts,
        batch_size=MINILM_ENCODE_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def _encode_specter(model, texts: List[str]) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=SPECTER_ENCODE_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def load_or_encode_minilm(
    path: Path, texts: List[str], expected_rows: int, device=None
) -> np.ndarray:
    """Load MiniLM embeddings from cache, re-encoding if missing or shape mismatch."""
    if path.exists():
        arr = np.load(path).astype(np.float32)
        if arr.shape[0] == expected_rows:
            print(f"  [cache] MiniLM {path.name} ({expected_rows} rows)")
            return arr
        print(f"  [stale] {path.name} has shape {arr.shape}; re-encoding …")
    print(f"  Encoding MiniLM ({expected_rows} texts) → {path.name} …")
    arr = _encode_minilm(texts, device)
    np.save(path, arr)
    return arr


def load_or_encode_specter(
    path: Path, model, texts: List[str], expected_rows: int
) -> np.ndarray:
    """Load SPECTER2 embeddings from cache, re-encoding if missing or shape mismatch."""
    if path.exists():
        arr = np.load(path).astype(np.float32)
        if arr.shape[0] == expected_rows:
            print(f"  [cache] SPECTER2 {path.name} ({expected_rows} rows)")
            return arr
        print(f"  [stale] {path.name} has shape {arr.shape}; re-encoding …")
    print(f"  Encoding SPECTER2 ({expected_rows} texts) → {path.name} …")
    arr = _encode_specter(model, texts)
    np.save(path, arr)
    return arr


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

def ndcg_at_k(ranked: List[str], relevant: Iterable[str], k: int = 10) -> float:
    rel = set(relevant)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:k], start=1)
        if doc_id in rel
    )
    ideal = min(len(rel), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_ndcg10(
    submission: Dict[str, List[str]], qrels: Dict[str, List[str]]
) -> float:
    return float(np.mean([ndcg_at_k(submission[qid], rels) for qid, rels in qrels.items()]))


def minmax_per_query(scores: np.ndarray) -> np.ndarray:
    """Per-query min-max normalisation to [0, 1]."""
    lo = scores.min(axis=1, keepdims=True)
    hi = scores.max(axis=1, keepdims=True)
    return (scores - lo) / (hi - lo + 1e-12)


def rrf_from_full_scores(
    dense: np.ndarray, sparse: np.ndarray, k: float = RRF_K
) -> np.ndarray:
    """
    Reciprocal Rank Fusion over the full corpus ranking.

    score(doc) = 1/(k + rank_dense) + 1/(k + rank_sparse)

    Returns min-max normalised scores (nq, nd).
    The inner loops over corpus size are intentional (spec requirement).
    """
    nq, nd = dense.shape
    out = np.zeros((nq, nd), dtype=np.float32)
    for i in range(nq):
        rd = np.argsort(-dense[i], kind="stable")
        rs = np.argsort(-sparse[i], kind="stable")
        s = np.zeros(nd, dtype=np.float32)
        for rank, j in enumerate(rd, start=1):
            s[j] += 1.0 / (k + rank)
        for rank, j in enumerate(rs, start=1):
            s[j] += 1.0 / (k + rank)
        out[i] = s
    return minmax_per_query(out)


# ---------------------------------------------------------------------------
# Core scoring and ranking
# ---------------------------------------------------------------------------

def build_submission(
    query_ids: List[str],
    corpus_ids: np.ndarray,
    minilm_scores: np.ndarray,    # (nq, nd) — L2-normalised dot products
    specter_scores: np.ndarray,   # (nq, nd)
    sparse_scores: np.ndarray,    # (nq, nd) — already beta-mixed FT + TA channels
    query_domains: np.ndarray,
    corpus_domains: np.ndarray,
    query_venues: np.ndarray,
    corpus_venues: np.ndarray,
    query_years: np.ndarray,      # int64, -1 = missing
    corpus_years: np.ndarray,
    eta: float,
    alpha: float,
    alpha_domain_boost: float,
    alpha_gap_delta: float,
    conf_quantile: float,
    domain_boost: float,
    venue_boost: float,
    year_boost: float,
    year_close_boost: float,
    gamma_rrf: float,
    rrf_norm: np.ndarray,         # (nq, nd) — already min-max normalised
    hard_domain_first: bool,
) -> Dict[str, List[str]]:
    nq = len(query_ids)

    # 1. Fuse dense signals
    dense = (
        (1.0 - eta) * minilm_scores.astype(np.float32)
        + eta * specter_scores.astype(np.float32)
    )

    # 2. Metadata match matrices — all vectorised, shape (nq, nd)
    domain_match = (query_domains[:, None] == corpus_domains[None, :]).astype(np.float32)
    venue_match = (
        (query_venues[:, None] == corpus_venues[None, :]) & (query_venues[:, None] != "")
    ).astype(np.float32)
    qy = query_years[:, None].astype(np.int64)
    cy = corpus_years[None, :].astype(np.int64)
    both_valid = (qy >= 0) & (cy >= 0)
    year_match = (both_valid & (qy == cy)).astype(np.float32)
    year_close = (both_valid & (np.abs(qy - cy) == 1)).astype(np.float32)

    # 3. Boost raw scores before normalisation
    boost = (
        domain_boost * domain_match
        + venue_boost * venue_match
        + year_boost * year_match
        + year_close_boost * year_close
    )
    dense_aug = dense + boost
    sparse_aug = sparse_scores.astype(np.float32) + boost

    # 4. Per-query adaptive alpha
    # Lean more sparse for queries from rare domains (count ≤ 4 in this query set)
    q_counts: Dict[str, int] = {}
    for d in query_domains.tolist():
        key = str(d)
        q_counts[key] = q_counts.get(key, 0) + 1
    rare = np.array(
        [1.0 if q_counts.get(str(d), 0) <= 4 else 0.0 for d in query_domains],
        dtype=np.float32,
    )[:, None]  # (nq, 1)
    alpha_q = np.clip(alpha - alpha_domain_boost * rare, 0.20, 0.90)

    if alpha_gap_delta > 0.0:
        # Sparse confidence: gap between top-1 and top-2 augmented sparse score.
        # High gap (clear winner) → more confident in sparse → lower alpha.
        ord2 = np.argsort(-sparse_aug, axis=1)[:, :2]
        top1 = sparse_aug[np.arange(nq), ord2[:, 0]]
        top2 = sparse_aug[np.arange(nq), ord2[:, 1]]
        gaps = top1 - top2
        thresh = float(np.quantile(gaps, conf_quantile))
        sharp = (gaps >= thresh).astype(np.float32)[:, None]
        alpha_q = np.clip(
            alpha_q - alpha_gap_delta * sharp + alpha_gap_delta * (1.0 - sharp),
            0.20,
            0.90,
        )

    # 5. Charlotte blend
    charlotte = (
        alpha_q * minmax_per_query(dense_aug)
        + (1.0 - alpha_q) * minmax_per_query(sparse_aug)
    )

    # 6. RRF second stage
    final_scores = (1.0 - gamma_rrf) * charlotte + gamma_rrf * rrf_norm

    # 7. Rank and apply optional hard_domain_first; remove self-citation
    submission: Dict[str, List[str]] = {}
    for i, qid in enumerate(query_ids):
        order = np.argsort(-final_scores[i], kind="stable")

        if hard_domain_first and str(query_domains[i]) != "":
            # Same-domain docs first (score order preserved within each group)
            dm = domain_match[i].astype(bool)
            order = np.concatenate([order[dm[order]], order[~dm[order]]])

        ranked = [str(corpus_ids[j]) for j in order if str(corpus_ids[j]) != qid]

        if len(ranked) < TOP_K:
            # Pad from full score order (self-citation already excluded)
            seen = set(ranked)
            extras = [
                str(corpus_ids[j])
                for j in np.argsort(-final_scores[i], kind="stable")
                if str(corpus_ids[j]) not in seen and str(corpus_ids[j]) != qid
            ]
            ranked.extend(extras[: TOP_K - len(ranked)])

        submission[qid] = ranked[:TOP_K]
    return submission


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_stage1(
    qrels: Dict[str, List[str]],
    public_query_ids: List[str],
    corpus_ids: np.ndarray,
    pub_minilm_scores: np.ndarray,
    pub_specter_scores: np.ndarray,
    pub_ft_scores: np.ndarray,
    pub_ta_scores: np.ndarray,
    query_domains: np.ndarray,
    corpus_domains: np.ndarray,
    query_venues: np.ndarray,
    corpus_venues: np.ndarray,
    query_years: np.ndarray,
    corpus_years: np.ndarray,
    rrf_norm: np.ndarray,         # precomputed — ignored here since gamma=0
) -> Tuple[dict, float]:
    """Joint grid search with gamma_rrf=0 fixed."""
    best_cfg: dict | None = None
    best_ndcg = -1.0

    for eta in [0.32]:
      for alpha in [0.29, 0.30, 0.31]:
        for alpha_domain_boost in [0.01]:
          for alpha_gap_delta in [0.005]:
            for conf_quantile in [0.62, 0.65, 0.68, 0.70]:
              for domain_boost in [0.08]:
                for venue_boost in [0.005]:
                  for year_boost in [0.00]:
                    for year_close_boost in [0.0, 0.002, 0.004, 0.006]:
                      for sparse_beta in [0.99]:
                        for hard_domain_first in [False, True]:
                            sparse_mix = (
                                sparse_beta * pub_ft_scores
                                + (1.0 - sparse_beta) * pub_ta_scores
                            )
                            sub = build_submission(
                                public_query_ids,
                                corpus_ids,
                                pub_minilm_scores,
                                pub_specter_scores,
                                sparse_mix,
                                query_domains,
                                corpus_domains,
                                query_venues,
                                corpus_venues,
                                query_years,
                                corpus_years,
                                eta=eta,
                                alpha=alpha,
                                alpha_domain_boost=alpha_domain_boost,
                                alpha_gap_delta=alpha_gap_delta,
                                conf_quantile=conf_quantile,
                                domain_boost=domain_boost,
                                venue_boost=venue_boost,
                                year_boost=year_boost,
                                year_close_boost=year_close_boost,
                                gamma_rrf=0.0,
                                rrf_norm=rrf_norm,
                                hard_domain_first=hard_domain_first,
                            )
                            score = evaluate_ndcg10(sub, qrels)
                            if score > best_ndcg:
                                best_ndcg = score
                                best_cfg = {
                                    "eta": eta,
                                    "alpha": alpha,
                                    "alpha_domain_boost": alpha_domain_boost,
                                    "alpha_gap_delta": alpha_gap_delta,
                                    "conf_quantile": conf_quantile,
                                    "domain_boost": domain_boost,
                                    "venue_boost": venue_boost,
                                    "year_boost": year_boost,
                                    "year_close_boost": year_close_boost,
                                    "sparse_beta": sparse_beta,
                                    "hard_domain_first": hard_domain_first,
                                }
    assert best_cfg is not None
    return best_cfg, best_ndcg


def tune_stage2_rrf(
    qrels: Dict[str, List[str]],
    public_query_ids: List[str],
    corpus_ids: np.ndarray,
    pub_minilm_scores: np.ndarray,
    pub_specter_scores: np.ndarray,
    pub_ft_scores: np.ndarray,
    pub_ta_scores: np.ndarray,
    query_domains: np.ndarray,
    corpus_domains: np.ndarray,
    query_venues: np.ndarray,
    corpus_venues: np.ndarray,
    query_years: np.ndarray,
    corpus_years: np.ndarray,
    base: dict,
) -> Tuple[dict, float]:
    """Tune eta_rrf (dense mix for the RRF signal) and gamma_rrf (RRF blend weight)."""
    best = {**base, "gamma_rrf": 0.0, "eta_rrf": base["eta"]}
    best_ndcg = -1.0

    sparse_mix = (
        base["sparse_beta"] * pub_ft_scores
        + (1.0 - base["sparse_beta"]) * pub_ta_scores
    )
    eta = base["eta"]

    for eta_rrf in [max(0.0, eta - 0.04), eta, min(1.0, eta + 0.04)]:
        # Recompute RRF with this eta blend for the dense ranking signal
        dense_rrf = (1.0 - eta_rrf) * pub_minilm_scores + eta_rrf * pub_specter_scores
        rrf_norm = rrf_from_full_scores(dense_rrf, sparse_mix)

        for gamma_rrf in [0.0, 0.001, 0.002, 0.004]:
            sub = build_submission(
                public_query_ids,
                corpus_ids,
                pub_minilm_scores,
                pub_specter_scores,
                sparse_mix,
                query_domains,
                corpus_domains,
                query_venues,
                corpus_venues,
                query_years,
                corpus_years,
                eta=base["eta"],
                alpha=base["alpha"],
                alpha_domain_boost=base["alpha_domain_boost"],
                alpha_gap_delta=base["alpha_gap_delta"],
                conf_quantile=base["conf_quantile"],
                domain_boost=base["domain_boost"],
                venue_boost=base["venue_boost"],
                year_boost=base["year_boost"],
                year_close_boost=base["year_close_boost"],
                gamma_rrf=gamma_rrf,
                rrf_norm=rrf_norm,
                hard_domain_first=base["hard_domain_first"],
            )
            score = evaluate_ndcg10(sub, qrels)
            if score > best_ndcg:
                best_ndcg = score
                best = {**base, "gamma_rrf": gamma_rrf, "eta_rrf": eta_rrf}

    return best, best_ndcg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    root = challenge_dir()
    data_dir = root / "data"
    minilm_dir = data_dir / "embeddings" / MINILM_EMB_SUBDIR
    specter_dir = data_dir / "embeddings" / SPECTER_EMB_SUBDIR

    dp = data_paths()
    if not dp["using_held_out_queries"]:
        raise RuntimeError(
            "held_out_queries.parquet not found. "
            "Check the locations searched by submission_utils.data_paths()."
        )
    paths = {**dp, **iteration_submission_paths(ITERATION_NAME)}
    out_dir: Path = paths["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load tabular data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading data …")
    queries, corpus = load_queries_corpus(
        data_dir / "queries.parquet", data_dir / "corpus.parquet"
    )
    heldout = pd.read_parquet(dp["queries_path"])
    qrels: Dict[str, List[str]] = load_json(data_dir / "qrels.json")

    public_query_ids = [str(x) for x in queries["doc_id"]]
    heldout_query_ids = [str(x) for x in heldout["doc_id"]]
    print(f"  Public queries : {len(public_query_ids)}")
    print(f"  Held-out queries: {len(heldout_query_ids)}")
    print(f"  Corpus docs    : {len(corpus)}")

    # ------------------------------------------------------------------
    # 2. Validate corpus ordering against embedding cache
    # ------------------------------------------------------------------
    corpus_ids_m = np.array(load_json(minilm_dir / "corpus_ids.json"), dtype=str)

    # If SPECTER2 precomputed corpus exists, verify its ID list matches MiniLM's
    specter_ids_path = specter_dir / "corpus_ids.json"
    if specter_ids_path.exists():
        corpus_ids_s = np.array(load_json(specter_ids_path), dtype=str)
        if not np.array_equal(corpus_ids_m, corpus_ids_s):
            raise ValueError(
                "MiniLM and SPECTER2 corpus_ids.json files differ — cannot fuse embeddings. "
                "Re-generate one of the caches so both use the same document ordering."
            )

    corpus_ids = corpus_ids_m
    n_corpus = len(corpus_ids)

    # Reorder corpus rows to match the embedding cache order
    corpus_ordered = corpus_df_ordered(corpus, corpus_ids)

    corpus_domains = corpus_ordered["domain"].fillna("").to_numpy()
    corpus_venues = corpus_ordered["venue"].fillna("").to_numpy()
    corpus_years = years_as_int64(corpus_ordered["year"])

    # ------------------------------------------------------------------
    # 3. Sparse retrieval — two TF-IDF channels
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"Building sparse full-text index (TF-IDF, ≤ {SPARSE_MAX_CHARS} chars/doc) …")
    vec_ft = TfidfVectorizer(
        sublinear_tf=True, min_df=2, max_df=0.95,
        ngram_range=(1, 2), stop_words="english",
    )
    corpus_ft_texts = [format_sparse_full(r) for _, r in corpus_ordered.iterrows()]
    pub_ft_texts = [format_sparse_full(r) for _, r in queries.iterrows()]
    hold_ft_texts = [format_sparse_full(r) for _, r in heldout.iterrows()]

    corpus_ft_mat = vec_ft.fit_transform(corpus_ft_texts)
    pub_ft_scores = cosine_similarity(
        vec_ft.transform(pub_ft_texts), corpus_ft_mat
    ).astype(np.float32)
    hold_ft_scores = cosine_similarity(
        vec_ft.transform(hold_ft_texts), corpus_ft_mat
    ).astype(np.float32)
    print(f"  Vocabulary size (FT): {len(vec_ft.vocabulary_)}")

    print("Building sparse title+abstract index (TF-IDF) …")
    vec_ta = TfidfVectorizer(
        sublinear_tf=True, min_df=2, max_df=0.95,
        ngram_range=(1, 2), stop_words="english",
    )
    corpus_ta_texts = [format_sparse_ta(r) for _, r in corpus_ordered.iterrows()]
    pub_ta_texts = [format_sparse_ta(r) for _, r in queries.iterrows()]
    hold_ta_texts = [format_sparse_ta(r) for _, r in heldout.iterrows()]

    corpus_ta_mat = vec_ta.fit_transform(corpus_ta_texts)
    pub_ta_scores = cosine_similarity(
        vec_ta.transform(pub_ta_texts), corpus_ta_mat
    ).astype(np.float32)
    hold_ta_scores = cosine_similarity(
        vec_ta.transform(hold_ta_texts), corpus_ta_mat
    ).astype(np.float32)
    print(f"  Vocabulary size (TA): {len(vec_ta.vocabulary_)}")

    # ------------------------------------------------------------------
    # 4. Dense retrieval — load corpus embeddings, encode queries
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading precomputed MiniLM corpus embeddings …")
    corpus_minilm = np.load(minilm_dir / "corpus_embeddings.npy").astype(np.float32)
    if corpus_minilm.shape[0] != n_corpus:
        raise ValueError(
            f"MiniLM corpus embedding rows ({corpus_minilm.shape[0]}) != "
            f"corpus_ids count ({n_corpus}). Delete stale cache and re-run."
        )
    print(f"  Loaded: {corpus_minilm.shape}")

    # Load SPECTER2 model (needed for query encoding regardless)
    from sentence_transformers import SentenceTransformer

    print("Loading SPECTER2 model …")
    specter_model = SentenceTransformer(SPECTER_MODEL)
    sep = specter_model.tokenizer.sep_token or "[SEP]"

    # SPECTER2 corpus: precomputed if present, otherwise encode and cache
    specter_corpus_emb_path = specter_dir / "corpus_embeddings.npy"
    if specter_corpus_emb_path.exists():
        print("Loading precomputed SPECTER2 corpus embeddings …")
        corpus_specter = np.load(specter_corpus_emb_path).astype(np.float32)
        if corpus_specter.shape[0] != n_corpus:
            raise ValueError(
                f"SPECTER2 corpus embedding rows ({corpus_specter.shape[0]}) != "
                f"corpus_ids count ({n_corpus}). Delete stale cache and re-run."
            )
        print(f"  Loaded: {corpus_specter.shape}")
    else:
        print(
            f"SPECTER2 corpus cache not found at {specter_corpus_emb_path}.\n"
            f"Encoding {n_corpus} corpus documents — this runs once and is cached …"
        )
        specter_dir.mkdir(parents=True, exist_ok=True)
        # Encode using the same sep-token format as queries for consistent embedding space
        corpus_specter_texts = [
            format_specter_query(r, sep) for _, r in corpus_ordered.iterrows()
        ]
        corpus_specter = _encode_specter(specter_model, corpus_specter_texts)
        np.save(specter_corpus_emb_path, corpus_specter)
        # Persist corpus_ids.json so future cross-checks pass
        with (specter_dir / "corpus_ids.json").open("w", encoding="utf-8") as f:
            json.dump(corpus_ids.tolist(), f)
        print(f"  Saved SPECTER2 corpus embeddings to {specter_corpus_emb_path}")

    # Query embeddings: always encode with full_text, cached per iteration
    print("Loading / encoding query embeddings (MiniLM full-text) …")
    pub_minilm = load_or_encode_minilm(
        out_dir / "pub_minilm_ft.npy",
        [format_minilm_query(r) for _, r in queries.iterrows()],
        len(public_query_ids),
    )
    hold_minilm = load_or_encode_minilm(
        out_dir / "hold_minilm_ft.npy",
        [format_minilm_query(r) for _, r in heldout.iterrows()],
        len(heldout_query_ids),
    )

    print("Loading / encoding query embeddings (SPECTER2 full-text) …")
    pub_specter = load_or_encode_specter(
        out_dir / "pub_specter_ft.npy",
        specter_model,
        [format_specter_query(r, sep) for _, r in queries.iterrows()],
        len(public_query_ids),
    )
    hold_specter = load_or_encode_specter(
        out_dir / "hold_specter_ft.npy",
        specter_model,
        [format_specter_query(r, sep) for _, r in heldout.iterrows()],
        len(heldout_query_ids),
    )

    # ------------------------------------------------------------------
    # 5. Score matrices — dot product = cosine (L2-normalised embeddings)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Computing dense score matrices …")
    pub_minilm_scores = (pub_minilm @ corpus_minilm.T).astype(np.float32)
    hold_minilm_scores = (hold_minilm @ corpus_minilm.T).astype(np.float32)
    pub_specter_scores = (pub_specter @ corpus_specter.T).astype(np.float32)
    hold_specter_scores = (hold_specter @ corpus_specter.T).astype(np.float32)
    print(f"  Score matrix shape: {pub_minilm_scores.shape}")

    # ------------------------------------------------------------------
    # 6. Query metadata arrays
    # ------------------------------------------------------------------
    qmeta_pub = queries.set_index(queries["doc_id"].astype(str), drop=False).loc[
        public_query_ids
    ]
    qmeta_hold = heldout.set_index(heldout["doc_id"].astype(str), drop=False).loc[
        heldout_query_ids
    ]
    qd_pub = qmeta_pub["domain"].fillna("").to_numpy()
    qv_pub = qmeta_pub["venue"].fillna("").to_numpy()
    qy_pub = years_as_int64(qmeta_pub["year"])
    qd_hold = qmeta_hold["domain"].fillna("").to_numpy()
    qv_hold = qmeta_hold["venue"].fillna("").to_numpy()
    qy_hold = years_as_int64(qmeta_hold["year"])

    # ------------------------------------------------------------------
    # 7. Stage 1: joint grid search (gamma_rrf = 0)
    # ------------------------------------------------------------------
    print("=" * 60)
    # Precompute a reference RRF for stage 1 (unused since gamma=0, but
    # required by build_submission signature — pass correct shape)
    _dense_ref = 0.68 * pub_minilm_scores + 0.32 * pub_specter_scores
    _sparse_ref = 0.99 * pub_ft_scores + 0.01 * pub_ta_scores
    rrf_ref = rrf_from_full_scores(_dense_ref, _sparse_ref)

    print("Tuning stage 1 (joint grid, gamma_rrf=0 fixed) …")
    cfg1, ndcg1 = tune_stage1(
        qrels=qrels,
        public_query_ids=public_query_ids,
        corpus_ids=corpus_ids,
        pub_minilm_scores=pub_minilm_scores,
        pub_specter_scores=pub_specter_scores,
        pub_ft_scores=pub_ft_scores,
        pub_ta_scores=pub_ta_scores,
        query_domains=qd_pub,
        corpus_domains=corpus_domains,
        query_venues=qv_pub,
        corpus_venues=corpus_venues,
        query_years=qy_pub,
        corpus_years=corpus_years,
        rrf_norm=rrf_ref,
    )
    print(f"\nStage-1 best public NDCG@10: {ndcg1:.5f}")
    print(f"Stage-1 best config: {cfg1}")

    # ------------------------------------------------------------------
    # 8. Stage 2: tune eta_rrf and gamma_rrf
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Tuning stage 2 (eta_rrf + gamma_rrf blend) …")
    cfg2, ndcg2 = tune_stage2_rrf(
        qrels=qrels,
        public_query_ids=public_query_ids,
        corpus_ids=corpus_ids,
        pub_minilm_scores=pub_minilm_scores,
        pub_specter_scores=pub_specter_scores,
        pub_ft_scores=pub_ft_scores,
        pub_ta_scores=pub_ta_scores,
        query_domains=qd_pub,
        corpus_domains=corpus_domains,
        query_venues=qv_pub,
        corpus_venues=corpus_venues,
        query_years=qy_pub,
        corpus_years=corpus_years,
        base=cfg1,
    )
    print(f"\nStage-2 best public NDCG@10: {ndcg2:.5f}")
    print(f"Stage-2 best config: {cfg2}")

    # ------------------------------------------------------------------
    # 9. Generate held-out submission with best hyperparameters
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Generating held-out submission …")
    sparse_mix_hold = (
        cfg2["sparse_beta"] * hold_ft_scores
        + (1.0 - cfg2["sparse_beta"]) * hold_ta_scores
    )
    dense_hold_rrf = (
        (1.0 - cfg2["eta_rrf"]) * hold_minilm_scores
        + cfg2["eta_rrf"] * hold_specter_scores
    )
    rrf_hold = rrf_from_full_scores(dense_hold_rrf, sparse_mix_hold)

    heldout_sub = build_submission(
        query_ids=heldout_query_ids,
        corpus_ids=corpus_ids,
        minilm_scores=hold_minilm_scores,
        specter_scores=hold_specter_scores,
        sparse_scores=sparse_mix_hold,
        query_domains=qd_hold,
        corpus_domains=corpus_domains,
        query_venues=qv_hold,
        corpus_venues=corpus_venues,
        query_years=qy_hold,
        corpus_years=corpus_years,
        eta=cfg2["eta"],
        alpha=cfg2["alpha"],
        alpha_domain_boost=cfg2["alpha_domain_boost"],
        alpha_gap_delta=cfg2["alpha_gap_delta"],
        conf_quantile=cfg2["conf_quantile"],
        domain_boost=cfg2["domain_boost"],
        venue_boost=cfg2["venue_boost"],
        year_boost=cfg2["year_boost"],
        year_close_boost=cfg2["year_close_boost"],
        gamma_rrf=cfg2["gamma_rrf"],
        rrf_norm=rrf_hold,
        hard_domain_first=cfg2["hard_domain_first"],
    )

    # ------------------------------------------------------------------
    # 10. Validate and save
    # ------------------------------------------------------------------
    validate_submission(heldout_sub, expected_query_ids=heldout_query_ids, top_k=TOP_K)
    validate_doc_ids_in_corpus(heldout_sub, corpus_doc_ids=corpus_ids.tolist())
    save_submission(heldout_sub, output_file=paths["output_file"])
    create_submission_zip(paths["output_file"], zip_file=paths["zip_file"])

    print("=" * 60)
    print(f"Iteration '{ITERATION_NAME}' complete.")
    print(f"Best public NDCG@10 : {ndcg2:.5f}")
    print(f"Submission saved to : {paths['zip_file']}")


if __name__ == "__main__":
    main()
