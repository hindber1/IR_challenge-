# Scientific Information Retrieval Challenge
## Hybrid Retrieval Pipeline — NDCG@10: 0.57

---

## Overview

This project implements a hybrid retrieval pipeline for the **Scientific Article IR Challenge** hosted on Codabench.

**Task:** Given a query paper (title + abstract), retrieve the 100 most relevant documents from a corpus of 20,000 scientific papers. Relevance is defined by citation — a document is relevant if the query paper cites it.

**Best score achieved:** `0.57 NDCG@10` on Codabench (held-out queries).

---

## Pipeline

```
Query Paper (title + abstract)
        │
        ├──► BM25 (title + abstract)     ──┐
        │                                  │
        ├──► BM25 (full text)             ─┼──► RRF Fusion (k=60) ──► Top 100
        │                                  │
        └──► SPECTER2 dense retrieval    ──┘
```

Three retrievers are combined using **Reciprocal Rank Fusion (RRF)**:

1. **BM25 (TA)** — sparse retrieval on title + abstract. Good for exact keyword matching.
2. **BM25 (Full Text)** — sparse retrieval on title + abstract + body. Recovers documents whose relevance only appears in the body, not the abstract.
3. **SPECTER2** (`allenai/specter2_base`) — dense retrieval using a transformer trained specifically on scientific citation graphs. Understands semantic similarity even when papers use different words.

**RRF formula:**
```
score(doc) = Σ  1 / (60 + rank_i(doc))
```
No score normalization needed — only ranks are combined.

---

## Results

### Local evaluation (public queries + qrels)

| System | NDCG@10 | NDCG@100 | Recall@100 | MAP |
|---|---|---|---|---|
| BM25 (TA) | 0.4639 | 0.5189 | 0.6865 | 0.3464 |
| SPECTER2 dense | 0.4722 | 0.5539 | 0.7753 | 0.3765 |
| RRF (BM25 + SPECTER2) | 0.4973 | 0.5780 | 0.7927 | 0.3944 |
| **Triple RRF (best)** | **0.5124** | **0.5899** | **0.8078** | **0.4074** |

### Codabench (held-out queries)

| Submission | NDCG@10 |
|---|---|
| TF-IDF baseline | 0.45 |
| MiniLM dense | 0.48 |
| SPECTER2 dense | 0.49 |
| **Triple RRF** | **0.57** |

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
│       └── sentence-transformers_all-MiniLM-L6-v2/
│           ├── query_embeddings.npy     # shape (100, 384)
│           ├── corpus_embeddings.npy    # shape (20000, 384)
│           ├── query_ids.json
│           └── corpus_ids.json
├── notebooks/
│   ├── challenge.ipynb              # Original starter kit notebook
│   └── challenge_57.ipynb           # Our best pipeline (0.57 on Codabench)
├── submissions/
│   └── submission_data.json         # Final submission file
├── scripts/
│   └── embed.py                     # Script to re-encode with any model
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
pip install rank_bm25 sentence-transformers torch transformers
```

### 3. Run the pipeline

Open `notebooks/challenge_57.ipynb` and run all cells in order.

---

## How to Submit

The notebook generates a `submission_data.json` file. To submit on Codabench:

```bash
# The notebook creates this automatically
# Just zip it:
zip submission.zip submission_data.json
```

Then upload `submission.zip` on the Codabench competition page.

**Important:** The submission must use `held_out_queries.parquet` as input, not `queries.parquet`. These two files use different ID namespaces.

---

## Key Design Decisions

### Why SPECTER2?
SPECTER2 is trained on scientific citation pairs — paper A cites paper B, so their embeddings should be close. This is exactly how relevance is defined in this challenge. Generic models like MiniLM are not optimized for this.

### Why RRF and not score interpolation?
BM25 scores and SPECTER2 scores are on completely different scales. Adding them directly would give one system too much weight. RRF only uses rank positions, which are always comparable across systems.

### Why BM25 full-text?
Some papers only share relevant vocabulary in their body sections, not their abstracts. Adding full-text BM25 to the fusion gave +0.08 NDCG@10 on Codabench — the biggest single gain in our ablation.

---

## Error Analysis

10 queries still score NDCG@10 = 0. Main failure reasons:

- **Vocabulary gap** — query and gold document use different terminology. Both BM25 and SPECTER2 fail.
- **Gold doc outside top-100** — system finds it at rank 120 but we only submit 100 results.
- **Niche domains** — Philosophy has very few relevant papers in the 20,000 corpus.
- **Cross-domain citations** — a CS paper citing a biology paper; models don't handle this well.
- **Missing abstracts** — some corpus documents have no abstract, leading to poor embeddings.

---

## Known Bug

TF-IDF and early BM25 experiments scored 0 locally. Root cause: the notebook was loading `held_out_queries.parquet` instead of `queries.parquet`. These files use completely different `doc_id` namespaces, so no query ID matched the `qrels` keys.

**Fix:** Always verify that submission keys match qrels keys before debugging:
```python
assert len(set(submission.keys()) & set(qrels.keys())) == 100
```

---

## Future Work

| Priority | Method | Expected gain |
|---|---|---|
| High | Cross-encoder reranker (top-50) | +3 to 7 NDCG pts |
| High | Replace SPECTER2 with BGE-large-en-v1.5 | +2 to 4 NDCG pts |
| Medium | Tune RRF constant k ∈ {30, 45, 60, 100} | +0.5 to 1 NDCG pt |
| Medium | Query expansion via Pseudo-Relevance Feedback | +1 to 2 NDCG pts |
| Exploratory | Chunk-level embedding with max pooling | unknown |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `rank_bm25` | latest | BM25 indexing and retrieval |
| `sentence-transformers` | ≥ 2.7.0 | SPECTER2 encoding |
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
