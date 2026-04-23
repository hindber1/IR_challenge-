# Scientific Information Retrieval Challenge
## Adaptive Hybrid Retrieval Pipeline — NDCG@10: 0.72

---

## Overview

This project implements an adaptive hybrid retrieval pipeline for the **Scientific Article IR Challenge** hosted on Codabench.

**Task:** Given a query paper (title + abstract), retrieve the 100 most relevant documents from a corpus of 20,000 scientific papers. Relevance is defined by citation — a document is relevant if the query paper cites it.

**Best score achieved:** `0.72 NDCG@10` on Codabench (held-out queries).

---

## Pipeline

```
Query Paper (title + abstract + full_text)
        │
        ├──► TF-IDF (full text, ≤ 80k chars)  ──┐
        │                                         │
        ├──► TF-IDF (title + abstract)           ─┼──► Sparse mix (β=0.99)
        │                                         │          │
        ├──► MiniLM dense retrieval              ─┼──► Dense mix (η)
        │                                         │          │
        └──► SPECTER2 dense retrieval            ─┘          │
                                                             │
                                          Metadata boosts (domain, venue, year)
                                                             │
                                          Adaptive alpha blend (per query)
                                                             │
                                          RRF second stage (γ_rrf)
                                                             │
                                                        Top 100
```

Five signals are combined through a two-stage adaptive fusion:

1. **TF-IDF (Full Text)** — sparse retrieval on title + abstract + body (≤ 80k chars). Bigram, sublinear TF, min_df=2, max_df=0.95.
2. **TF-IDF (Title + Abstract)** — sparse retrieval on title + abstract only. Same vectorizer settings.
3. **MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`) — dense retrieval; queries encoded with full text.
4. **SPECTER2** (`allenai/specter2_base`) — dense retrieval using a transformer trained on scientific citation graphs.
5. **Metadata boosts** — domain, venue, and year match signals applied before normalisation.

### Fusion stages

**Stage 1 — Charlotte blend:**
```
dense     = (1 - η) * MiniLM  +  η * SPECTER2
sparse    = β * TF-IDF(FT)  +  (1 - β) * TF-IDF(TA)

boost     = domain_boost * domain_match
          + venue_boost  * venue_match
          + year_boost   * year_match
          + year_close_boost * year_adjacent_match

charlotte = α_q * norm(dense + boost)  +  (1 - α_q) * norm(sparse + boost)
```

**Adaptive alpha (per query):**
```
α_q = clip(α - α_domain_boost * rare_domain_flag, 0.20, 0.90)
    ± α_gap_delta  (adjusted by sparse confidence / score gap)
```

**Stage 2 — RRF blend:**
```
rrf_score(doc) = 1/(60 + rank_dense) + 1/(60 + rank_sparse)
final          = (1 - γ_rrf) * charlotte  +  γ_rrf * norm(rrf)
```

Hyperparameters are tuned in two stages on the public queries + qrels (NDCG@10).

---

## Results

### Local evaluation (public queries + qrels)

| System | NDCG@10 | NDCG@100 | Recall@100 | MAP |
|---|---|---|---|---|
| BM25 (TA) | 0.4639 | 0.5189 | 0.6865 | 0.3464 |
| SPECTER2 dense | 0.4722 | 0.5539 | 0.7753 | 0.3765 |
| RRF (BM25 + SPECTER2) | 0.4973 | 0.5780 | 0.7927 | 0.3944 |
| Triple RRF | 0.5124 | 0.5899 | 0.8078 | 0.4074 |
| **Adaptive hybrid (best)** | **0.5300+** | — | — | — |

### Codabench (held-out queries)

| Submission | NDCG@10 |
|---|---|
| TF-IDF baseline | 0.45 |
| MiniLM dense | 0.48 |
| SPECTER2 dense | 0.49 |
| Triple RRF | 0.57 |
| **Adaptive hybrid (iteration_next_candidate)** | **0.72** |

---

## Project Structure

```
ir_challenge/
├── data/
│   ├── queries.parquet              # 100 public queries with ground truth
│   ├── corpus.parquet               # 20,000 candidate documents
│   ├── qrels.json                   # Ground truth: {query_id: [doc_id, ...]}
│   ├── held_out_queries.parquet     # Test queries for Codabench submission
│   ├── sample_submission.json       # Example submission format
│   └── embeddings/
│       ├── sentence-transformers_all-MiniLM-L6-v2/
│       │   ├── query_embeddings.npy     # shape (100, 384)
│       │   ├── corpus_embeddings.npy    # shape (20000, 384)
│       │   ├── query_ids.json
│       │   └── corpus_ids.json
│       └── allenai_specter2_base/
│           ├── corpus_embeddings.npy    # encoded and cached on first run
│           └── corpus_ids.json
├── notebooks/
│   ├── challenge.ipynb              # Original starter kit notebook
│   └── challenge_57.ipynb           # Earlier triple-RRF pipeline (0.57)
├── iterations/
│   └── iteration_next_candidate/    # Current best pipeline (0.72)
│       └── iteration_next_candidate.py
├── submissions/
│   └── submission_data.json         # Final submission file
├── scripts/
│   ├── embed.py                     # Script to re-encode with any model
│   └── submission_utils.py          # Shared I/O, evaluation, and validation helpers
└── requirements.txt
```

---

## Setup

### 1. Create virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install sentence-transformers torch transformers scikit-learn
```

### 3. Run the pipeline

```bash
python iterations/iteration_next_candidate/iteration_next_candidate.py
```

The script will:
- Build TF-IDF indexes for both full-text and title+abstract channels
- Load precomputed MiniLM corpus embeddings
- Encode or load cached SPECTER2 corpus embeddings (first run encodes and caches automatically)
- Encode query embeddings at runtime
- Run two-stage hyperparameter tuning on public qrels
- Generate and validate the held-out submission

---

## How to Submit

```bash
# The script creates submission_data.json and zips it automatically.
# The zip is saved to:
#   submissions/iteration_next_candidate/submission.zip
```

Then upload `submission.zip` on the Codabench competition page.

**Important:** The pipeline uses `held_out_queries.parquet` for the final submission and `queries.parquet` for tuning. These files use different ID namespaces — never mix them.

---

## Key Design Decisions

### Why five signals instead of three?
The jump from 0.57 to 0.72 came from two additions: (1) a second TF-IDF channel over full text vs. title+abstract only, and (2) metadata boosts for domain/venue/year. Each signal captures a different aspect of relevance; fusion consistently outperforms any single signal.

### Why adaptive alpha?
A single global α treats all queries the same. Queries from rare domains benefit from leaning more on sparse signals (lower α), while queries with a strong sparse top-1 candidate also warrant a shift. Per-query α narrows the gap between easy and hard queries.

### Why SPECTER2 over MiniLM for the SPECTER slot?
SPECTER2 is trained on scientific citation pairs — paper A cites paper B, so their embeddings should be close. This is exactly how relevance is defined in this challenge. MiniLM is kept as a complementary signal because it encodes full text and captures different vocabulary patterns.

### Why RRF and not score interpolation?
BM25/TF-IDF scores and dense scores live on different scales. RRF uses only rank positions, which are always comparable across systems, and the lightweight second-stage blend (γ_rrf ≥ 0) adds a small but consistent improvement.

### Why two-stage tuning?
Stage 1 searches the main fusion parameters with γ_rrf=0 fixed, dramatically reducing the search space. Stage 2 then tunes only the RRF blend weight and its dense-mix ratio, which are near-orthogonal to the stage-1 parameters.

---

## Error Analysis

Queries scoring NDCG@10 = 0 share common failure patterns:

- **Vocabulary gap** — query and gold document use different terminology; both TF-IDF and SPECTER2 fail to bridge it.
- **Gold doc outside top-100** — system finds it at rank ~120 but submission is capped at 100 results.
- **Niche domains** — Philosophy and some interdisciplinary fields have very few relevant papers in the 20,000-doc corpus.
- **Cross-domain citations** — a CS paper citing a biology paper; domain-specific embeddings do not generalise well.
- **Missing abstracts** — some corpus documents have no abstract, leading to poor dense embeddings.

---

## Known Bug (fixed)

Early experiments scored 0 locally. Root cause: the notebook was loading `held_out_queries.parquet` instead of `queries.parquet` for local evaluation. These files use completely different `doc_id` namespaces, so no query ID matched the `qrels` keys.

**Fix:** Always verify that submission keys match qrels keys before debugging:
```python
assert len(set(submission.keys()) & set(qrels.keys())) == 100
```

---

## Future Work

| Priority | Method | Expected gain |
|---|---|---|
| High | Cross-encoder reranker (top-50) | +3 to 7 NDCG pts |
| High | Replace MiniLM with BGE-large-en-v1.5 | +2 to 4 NDCG pts |
| Medium | Tune RRF constant k ∈ {30, 45, 60, 100} | +0.5 to 1 NDCG pt |
| Medium | Query expansion via Pseudo-Relevance Feedback | +1 to 2 NDCG pts |
| Exploratory | Chunk-level embedding with max pooling | unknown |
| Exploratory | LLM-based query reformulation | unknown |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `scikit-learn` | latest | TF-IDF indexing and cosine similarity |
| `sentence-transformers` | ≥ 2.7.0 | MiniLM and SPECTER2 encoding |
| `torch` | latest | Backend for transformers |
| `numpy` | latest | Matrix operations |
| `pandas` | latest | Data loading |
| `tqdm` | latest | Progress bars |

---

## References

- Robertson, S. & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.*
- Cormack, G., Clarke, C., & Buettcher, S. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.*
- Specter2: *SPECTER2: Scientific Paper Representations using Citation-Informed Transformers.* Allen Institute for AI.
- MTEB Leaderboard: https://huggingface.co/spaces/mteb/leaderboard
