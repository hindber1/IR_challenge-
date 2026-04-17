"""
Multi-model hybrid experiment.
Models: MiniLM (pre-computed) + SciNCL + BGE-base + SPECTER2_base + BM25-FT8k
Combines all with RRF.
"""
import sys, json, math, re
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

DATA_DIR  = Path("../data")
EMB_BASE  = DATA_DIR / "embeddings"

queries   = pd.read_parquet(DATA_DIR / "queries.parquet")
corpus    = pd.read_parquet(DATA_DIR / "corpus.parquet")
with open(DATA_DIR / "qrels.json", encoding="utf-8") as f:
    qrels = json.load(f)
query_ids  = queries["doc_id"].tolist()
corpus_ids = corpus["doc_id"].tolist()

def fmt_ta(row):
    t = str(row.get("title","") or "").strip()
    a = str(row.get("abstract","") or "").strip()
    return (t + " " + a).strip()
def fmt_full(row, cap=8000):
    return (fmt_ta(row) + " " + str(row.get("full_text","") or "").strip()[:cap]).strip()

# ── Metrics ───────────────────────────────────────────────────────────────────
def evaluate(sub):
    r10, r100, n10, n100, mrr, AP = [], [], [], [], [], []
    for qid, ranked in sub.items():
        rel = set(qrels.get(qid, []))
        if not rel: continue
        r10.append(len(set(ranked[:10])  & rel) / len(rel))
        r100.append(len(set(ranked[:100]) & rel) / len(rel))
        dcg  = sum(1./math.log2(i+2) for i,d in enumerate(ranked[:10]) if d in rel)
        idcg = sum(1./math.log2(i+2) for i in range(min(len(rel),10)))
        n10.append(dcg/idcg if idcg else 0.)
        dcg2 = sum(1./math.log2(i+2) for i,d in enumerate(ranked[:100]) if d in rel)
        idcg2= sum(1./math.log2(i+2) for i in range(min(len(rel),100)))
        n100.append(dcg2/idcg2 if idcg2 else 0.)
        for i,d in enumerate(ranked[:10]):
            if d in rel: mrr.append(1./(i+1)); break
        else: mrr.append(0.)
        h, s = 0, 0.
        for i,d in enumerate(ranked):
            if d in rel: h+=1; s+=h/(i+1)
        AP.append(s/len(rel))
    return {k: float(np.mean(v)) for k,v in zip(
        ["r@10","r@100","n@10","n@100","mrr","map"], [r10,r100,n10,n100,mrr,AP])}

def show(label, sub):
    r = evaluate(sub)
    star = " *** >0.60 ***" if r["r@10"] >= 0.60 else ""
    print(f"  {label:<50}  r@10={r['r@10']:.4f}  r@100={r['r@100']:.4f}  "
          f"n@10={r['n@10']:.4f}  map={r['map']:.4f}{star}")

def rrf(*subs, k=60):
    result = {}
    for qid in query_ids:
        sc = {}
        for sub in subs:
            for rank, d in enumerate(sub.get(qid, [])):
                sc[d] = sc.get(d,0.) + 1./(k+rank+1)
        result[qid] = sorted(sc, key=sc.__getitem__, reverse=True)[:100]
    return result


# ── Helper: encode with a SentenceTransformer, cache to disk ─────────────────
def load_or_encode(model_id, corpus_texts, query_texts, cache_dir):
    """Load cached embeddings or encode and save them."""
    cache_dir = Path(cache_dir)
    c_path = cache_dir / "corpus_embeddings.npy"
    q_path = cache_dir / "query_embeddings.npy"

    if c_path.exists() and q_path.exists():
        print(f"  Loading cached embeddings from {cache_dir.name}/")
        c_embs = np.load(c_path).astype(np.float32)
        q_embs = np.load(q_path).astype(np.float32)
        return q_embs, c_embs

    print(f"  Loading model {model_id}...")
    model = SentenceTransformer(model_id, trust_remote_code=True)
    print(f"  Encoding corpus ({len(corpus_texts)} docs)...")
    c_embs = model.encode(corpus_texts, batch_size=128, show_progress_bar=True,
                          normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    print(f"  Encoding queries ({len(query_texts)} docs)...")
    q_embs = model.encode(query_texts, batch_size=128, show_progress_bar=True,
                          normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(c_path, c_embs)
    np.save(q_path, q_embs)
    with open(cache_dir / "corpus_ids.json", "w") as f: json.dump(corpus_ids, f)
    with open(cache_dir / "query_ids.json",  "w") as f: json.dump(query_ids, f)
    with open(cache_dir / "model_info.txt",  "w") as f: f.write(f"model: {model_id}\n")
    print(f"  Saved to {cache_dir.name}/")
    return q_embs, c_embs


def dense_retrieve(q_embs, c_embs, q_ids=None, c_ids=None, top_n=100):
    if q_ids is None: q_ids = query_ids
    if c_ids is None: c_ids = corpus_ids
    sim = q_embs @ c_embs.T
    top = np.argsort(-sim, axis=1)[:, :top_n]
    return {qid: [c_ids[j] for j in top[i]] for i, qid in enumerate(q_ids)}


# ─────────────────────────────────────────────────────────────────────────────
# 1. MiniLM (pre-computed)
# ─────────────────────────────────────────────────────────────────────────────
print("=== Loading MiniLM ===")
EMB_DIR = EMB_BASE / "sentence-transformers_all-MiniLM-L6-v2"
q_ml = np.load(EMB_DIR / "query_embeddings.npy").astype(np.float32)
c_ml = np.load(EMB_DIR / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR / "query_ids.json")  as f: eq = json.load(f)
with open(EMB_DIR / "corpus_ids.json") as f: ec = json.load(f)
sub_minilm = dense_retrieve(q_ml, c_ml, eq, ec)
show("A. MiniLM-L6-v2 (384-dim, pre-computed)", sub_minilm)

# ─────────────────────────────────────────────────────────────────────────────
# 2. SciNCL (scientific citations, 768-dim)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== SciNCL ===")
corpus_ta = [fmt_ta(r) for _, r in corpus.iterrows()]
query_ta  = [fmt_ta(r) for _, r in queries.iterrows()]
q_scincl, c_scincl = load_or_encode(
    "malteos/scincl", corpus_ta, query_ta,
    EMB_BASE / "malteos_scincl"
)
sub_scincl = dense_retrieve(q_scincl, c_scincl)
show("B. SciNCL (768-dim, scientific citations)", sub_scincl)

# ─────────────────────────────────────────────────────────────────────────────
# 3. BGE-base (768-dim, strong general retrieval)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== BGE-base-en-v1.5 ===")
# BGE requires "Represent this sentence for searching relevant passages: " prefix for queries
query_ta_bge = ["Represent this sentence for searching relevant passages: " + t for t in query_ta]
q_bge, c_bge = load_or_encode(
    "BAAI/bge-base-en-v1.5", corpus_ta, query_ta_bge,
    EMB_BASE / "BAAI_bge-base-en-v1.5"
)
sub_bge = dense_retrieve(q_bge, c_bge)
show("C. BGE-base-en-v1.5 (768-dim)", sub_bge)

# ─────────────────────────────────────────────────────────────────────────────
# 4. SPECTER2_base (768-dim, scientific domain pre-training)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== SPECTER2_base ===")
q_sp2, c_sp2 = load_or_encode(
    "allenai/specter2_base", corpus_ta, query_ta,
    EMB_BASE / "allenai_specter2_base"
)
sub_sp2 = dense_retrieve(q_sp2, c_sp2)
show("D. SPECTER2_base (768-dim, scientific)", sub_sp2)

# ─────────────────────────────────────────────────────────────────────────────
# 5. BM25 FT-8k (best sparse retriever)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== BM25 FT-8k ===")
corpus_ft8 = [fmt_full(r, 8000) for _, r in corpus.iterrows()]
corpus_ft8_tok = [re.findall(r"[a-z0-9]+", t.lower()) for t in corpus_ft8]
bm25 = BM25Okapi(corpus_ft8_tok)
query_ta_tok = [re.findall(r"[a-z0-9]+", t.lower()) for t in query_ta]
sub_bm25 = {}
for i, qid in enumerate(query_ids):
    sc = bm25.get_scores(query_ta_tok[i])
    sub_bm25[qid] = [corpus_ids[j] for j in np.argsort(-sc)[:100]]
show("E. BM25 FT-8k / TA-query", sub_bm25)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Pairwise combinations
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Pairwise RRF ===")
show("F1.  RRF(MiniLM + BM25)",                rrf(sub_minilm, sub_bm25))
show("F2.  RRF(SciNCL + BM25)",                rrf(sub_scincl, sub_bm25))
show("F3.  RRF(BGE + BM25)",                   rrf(sub_bge,    sub_bm25))
show("F4.  RRF(SPEC2_base + BM25)",            rrf(sub_sp2,    sub_bm25))
show("F5.  RRF(SciNCL + MiniLM)",              rrf(sub_scincl, sub_minilm))
show("F6.  RRF(BGE + MiniLM)",                 rrf(sub_bge,    sub_minilm))
show("F7.  RRF(SPEC2_base + MiniLM)",          rrf(sub_sp2,    sub_minilm))
show("F8.  RRF(SciNCL + BGE)",                 rrf(sub_scincl, sub_bge))
show("F9.  RRF(SciNCL + SPEC2_base)",          rrf(sub_scincl, sub_sp2))
show("F10. RRF(BGE + SPEC2_base)",             rrf(sub_bge,    sub_sp2))

# ─────────────────────────────────────────────────────────────────────────────
# 7. Triple/quad combinations
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Triple/Quad RRF ===")
show("G1. RRF(SciNCL + BGE + BM25)",             rrf(sub_scincl, sub_bge, sub_bm25))
show("G2. RRF(SciNCL + SPEC2_base + BM25)",      rrf(sub_scincl, sub_sp2, sub_bm25))
show("G3. RRF(BGE + SPEC2_base + BM25)",         rrf(sub_bge, sub_sp2, sub_bm25))
show("G4. RRF(SciNCL + BGE + SPEC2_base)",       rrf(sub_scincl, sub_bge, sub_sp2))
show("G5. RRF(MiniLM + SciNCL + BM25)",          rrf(sub_minilm, sub_scincl, sub_bm25))
show("G6. RRF(MiniLM + BGE + BM25)",             rrf(sub_minilm, sub_bge, sub_bm25))
show("G7. RRF(MiniLM + SciNCL + BGE + BM25)",    rrf(sub_minilm, sub_scincl, sub_bge, sub_bm25))
show("G8. RRF(MiniLM + SciNCL + SPEC2_base + BM25)", rrf(sub_minilm, sub_scincl, sub_sp2, sub_bm25))
show("G9. RRF(MiniLM + BGE + SPEC2_base + BM25)",    rrf(sub_minilm, sub_bge, sub_sp2, sub_bm25))

# ─────────────────────────────────────────────────────────────────────────────
# 8. Full ensemble
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Full Ensemble ===")
show("H1. RRF(SciNCL + BGE + SPEC2_base + MiniLM + BM25)",
     rrf(sub_scincl, sub_bge, sub_sp2, sub_minilm, sub_bm25))
show("H2. RRF(SciNCL + BGE + SPEC2_base + BM25)  [no MiniLM]",
     rrf(sub_scincl, sub_bge, sub_sp2, sub_bm25))

print("\nDone.")
