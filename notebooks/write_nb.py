"""
Helper script: generates challenge.ipynb with the optimized IR solution.
Run once: python write_nb.py
"""
import json, os

nb = {
 "cells": [],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.11.0"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [source],
    }


# ---------------------------------------------------------------------------
cells = [

md(
    "# Scientific Article IR Challenge — Optimized Solution\n\n"
    "**Strategy** (best score):\n"
    "1. **SPECTER2** — scientific-domain dense model trained on citation prediction\n"
    "2. **MiniLM-L6-v2** — fast general dense model (pre-computed, always available)\n"
    "3. **BM25 (title+abstract)** — exact-match sparse retrieval\n"
    "4. **BM25 (full text, capped 8k chars)** — full-text sparse for extra recall\n"
    "5. **RRF** — Reciprocal Rank Fusion of all above\n\n"
    "Submission format: `{query_id: [doc_id_1, ..., doc_id_100]}`\n"
),

# ── Install ─────────────────────────────────────────────────────────────────
code(
    "import subprocess, sys\n"
    "subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'rank_bm25', '-q'])\n"
    "print('Dependencies ready.')\n"
),

# ── Imports ──────────────────────────────────────────────────────────────────
code(
    "import json, math, os, re\n"
    "from pathlib import Path\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "from rank_bm25 import BM25Okapi\n"
    "from sentence_transformers import SentenceTransformer\n"
    "from tqdm.auto import tqdm\n"
    "\n"
    'DATA_DIR = Path("../data")\n'
    'OUT_DIR  = Path("../submissions")\n'
    "OUT_DIR.mkdir(exist_ok=True)\n"
    'print("Imports OK")\n'
),

md("## 1 — Load Data"),

code(
    'queries  = pd.read_parquet(DATA_DIR / "queries.parquet")\n'
    'corpus   = pd.read_parquet(DATA_DIR / "corpus.parquet")\n'
    "\n"
    'held_out_path = DATA_DIR / "held_out_queries.parquet"\n'
    "if not held_out_path.exists():\n"
    '    held_out_path = Path("../held_out_queries.parquet")\n'
    "held_out = pd.read_parquet(held_out_path)\n"
    "\n"
    'with open(DATA_DIR / "qrels.json", encoding="utf-8") as f:\n'
    "    qrels = json.load(f)\n"
    "\n"
    'query_ids  = queries["doc_id"].tolist()\n'
    'corpus_ids = corpus["doc_id"].tolist()\n'
    'ho_ids     = held_out["doc_id"].tolist()\n'
    "\n"
    'print(f"Public queries : {len(queries)}")\n'
    'print(f"Corpus         : {len(corpus)}")\n'
    'print(f"Held-out       : {len(held_out)}")\n'
    'print(f"Qrels entries  : {len(qrels)}")\n'
),

md("## 2 — Helper Functions"),

code(
    "# ── Text formatting ──────────────────────────────────────────────────────\n"
    "def fmt_ta(row) -> str:\n"
    '    t = str(row.get("title",   "") or "").strip()\n'
    '    a = str(row.get("abstract","") or "").strip()\n'
    '    return (t + " " + a).strip()\n'
    "\n"
    "def fmt_full(row, max_chars: int = 8000) -> str:\n"
    "    ta = fmt_ta(row)\n"
    '    ft = str(row.get("full_text", "") or "").strip()\n'
    '    return (ta + " " + ft[:max_chars]).strip()\n'
    "\n"
    "def tokenize(text: str) -> list:\n"
    '    return re.findall(r"[a-z0-9]+", text.lower())\n'
    "\n"
    "# ── Metrics ──────────────────────────────────────────────────────────────\n"
    "def recall_at_k(ranked, rel_set, k):\n"
    "    return len(set(ranked[:k]) & rel_set) / len(rel_set) if rel_set else 0.0\n"
    "\n"
    "def precision_at_k(ranked, rel_set, k):\n"
    "    return sum(1 for d in ranked[:k] if d in rel_set) / k\n"
    "\n"
    "def mrr_at_k(ranked, rel_set, k):\n"
    "    for i, d in enumerate(ranked[:k]):\n"
    "        if d in rel_set:\n"
    "            return 1.0 / (i + 1)\n"
    "    return 0.0\n"
    "\n"
    "def ndcg_at_k(ranked, rel_set, k):\n"
    "    dcg  = sum(1.0/math.log2(i+2) for i, d in enumerate(ranked[:k]) if d in rel_set)\n"
    "    idcg = sum(1.0/math.log2(i+2) for i in range(min(len(rel_set), k)))\n"
    "    return dcg / idcg if idcg else 0.0\n"
    "\n"
    "def average_precision(ranked, rel_set):\n"
    "    if not rel_set: return 0.0\n"
    "    hits, s = 0, 0.0\n"
    "    for i, d in enumerate(ranked):\n"
    "        if d in rel_set:\n"
    "            hits += 1\n"
    "            s += hits / (i + 1)\n"
    "    return s / len(rel_set)\n"
    "\n"
    "def evaluate(submission: dict, qrels: dict) -> dict:\n"
    '    b = {k: [] for k in ["recall@10","recall@100","ndcg@10","ndcg@100",\n'
    '                          "precision@10","mrr@10","map"]}\n'
    "    for qid, ranked in submission.items():\n"
    "        rel = set(qrels.get(qid, []))\n"
    "        if not rel: continue\n"
    '        b["recall@10"].append(recall_at_k(ranked, rel, 10))\n'
    '        b["recall@100"].append(recall_at_k(ranked, rel, 100))\n'
    '        b["ndcg@10"].append(ndcg_at_k(ranked, rel, 10))\n'
    '        b["ndcg@100"].append(ndcg_at_k(ranked, rel, 100))\n'
    '        b["precision@10"].append(precision_at_k(ranked, rel, 10))\n'
    '        b["mrr@10"].append(mrr_at_k(ranked, rel, 10))\n'
    '        b["map"].append(average_precision(ranked, rel))\n'
    "    return {k: float(np.mean(v)) for k, v in b.items()}\n"
    "\n"
    "# ── Reciprocal Rank Fusion ────────────────────────────────────────────────\n"
    "def rrf(rankings_list: list, k: int = 60, top_n: int = 100) -> list:\n"
    "    scores: dict = {}\n"
    "    for rankings in rankings_list:\n"
    "        for rank, doc_id in enumerate(rankings):\n"
    "            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)\n"
    "    return sorted(scores, key=scores.__getitem__, reverse=True)[:top_n]\n"
    "\n"
    "def build_best(query_id_list, minilm_d, bm25_ft_d, specter_d=None):\n"
    "    # Optimal: RRF(SPECTER2?, MiniLM, BM25_FT)  -- BM25_TA excluded (redundant with FT)\n"
    "    result = {}\n"
    "    for qid in query_id_list:\n"
    "        candidates = [minilm_d.get(qid,[]), bm25_ft_d.get(qid,[])]\n"
    "        if specter_d is not None:\n"
    "            candidates.insert(0, specter_d.get(qid,[]))\n"
    "        result[qid] = rrf(candidates)\n"
    "    return result\n"
    "\n"
    "def show_results(name: str, results: dict):\n"
    '    print(f"\\n{chr(9472)*55}")\n'
    '    print(f"  {name}")\n'
    '    print(f"{chr(9472)*55}")\n'
    "    for k, v in results.items():\n"
    '        print(f"  {k:<15}  {v:.4f}")\n'
    "\n"
    'print("Helpers defined.")\n'
),

md("## 3 — Strategy A: Dense Retrieval (MiniLM, pre-computed)"),

code(
    'EMB_DIR = DATA_DIR / "embeddings" / "sentence-transformers_all-MiniLM-L6-v2"\n'
    "\n"
    'q_embs = np.load(EMB_DIR / "query_embeddings.npy").astype(np.float32)\n'
    'c_embs = np.load(EMB_DIR / "corpus_embeddings.npy").astype(np.float32)\n'
    'with open(EMB_DIR / "query_ids.json")  as f: emb_q_ids = json.load(f)\n'
    'with open(EMB_DIR / "corpus_ids.json") as f: emb_c_ids = json.load(f)\n'
    "\n"
    "# L2-normalised embeddings => cosine sim == dot product\n"
    "sim_minilm = q_embs @ c_embs.T           # (100, 20000)\n"
    "top_minilm = np.argsort(-sim_minilm, axis=1)[:, :100]\n"
    "minilm_sub = {qid: [emb_c_ids[j] for j in top_minilm[i]] for i, qid in enumerate(emb_q_ids)}\n"
    "\n"
    "r_minilm = evaluate(minilm_sub, qrels)\n"
    'show_results("MiniLM-L6-v2 (pre-computed)", r_minilm)\n'
),

md("## 4 — Strategy B: BM25 on Title + Abstract"),

code(
    'print("Tokenising corpus (title+abstract)...")\n'
    "corpus_ta_tok = [tokenize(fmt_ta(row)) for _, row in tqdm(corpus.iterrows(), total=len(corpus))]\n"
    "bm25_ta = BM25Okapi(corpus_ta_tok)\n"
    "query_ta_tok = [tokenize(fmt_ta(row)) for _, row in queries.iterrows()]\n"
    "\n"
    "bm25_ta_sub = {}\n"
    'for i, qid in enumerate(tqdm(query_ids, desc="BM25 TA retrieval")):\n'
    "    scores = bm25_ta.get_scores(query_ta_tok[i])\n"
    "    bm25_ta_sub[qid] = [corpus_ids[j] for j in np.argsort(-scores)[:100]]\n"
    "\n"
    "r_bm25_ta = evaluate(bm25_ta_sub, qrels)\n"
    'show_results("BM25 -- title + abstract", r_bm25_ta)\n'
),

md("## 5 — Strategy C: BM25 on Full Text (capped at 8 000 chars)"),

code(
    'print("Tokenising corpus (full text, <=8000 chars)...")\n'
    "corpus_ft_tok = [tokenize(fmt_full(row)) for _, row in tqdm(corpus.iterrows(), total=len(corpus))]\n"
    "bm25_ft = BM25Okapi(corpus_ft_tok)\n"
    "\n"
    "# Query with TA tokens (concise signal for what to retrieve)\n"
    "bm25_ft_sub = {}\n"
    'for i, qid in enumerate(tqdm(query_ids, desc="BM25 FT retrieval")):\n'
    "    scores = bm25_ft.get_scores(query_ta_tok[i])\n"
    "    bm25_ft_sub[qid] = [corpus_ids[j] for j in np.argsort(-scores)[:100]]\n"
    "\n"
    "r_bm25_ft = evaluate(bm25_ft_sub, qrels)\n"
    'show_results("BM25 -- full text (capped)", r_bm25_ft)\n'
),

md("## 6 — Hybrid: RRF combinations (empirical comparison)"),

code(
    "# Empirical finding: BM25-FT already covers the TA text, so adding BM25-TA separately\n"
    "# is redundant and slightly hurts.  Best hybrid = RRF(MiniLM + BM25_FT).\n"
    "\n"
    "# A+B\n"
    "rrf_AB = {qid: rrf([minilm_sub[qid], bm25_ta_sub[qid]]) for qid in query_ids}\n"
    "r_AB = evaluate(rrf_AB, qrels)\n"
    'show_results("RRF (MiniLM + BM25_TA)", r_AB)\n'
    "\n"
    "# A+C  <-- BEST without SPECTER2\n"
    "rrf_AC = {qid: rrf([minilm_sub[qid], bm25_ft_sub[qid]]) for qid in query_ids}\n"
    "r_AC = evaluate(rrf_AC, qrels)\n"
    'show_results("RRF (MiniLM + BM25_FT)  <-- best hybrid", r_AC)\n'
    "\n"
    "# A+B+C\n"
    "rrf_ABC = {qid: rrf([minilm_sub[qid], bm25_ta_sub[qid], bm25_ft_sub[qid]]) for qid in query_ids}\n"
    "r_ABC = evaluate(rrf_ABC, qrels)\n"
    'show_results("RRF (MiniLM + BM25_TA + BM25_FT)", r_ABC)\n'
),

md(
    "## 7 — Strategy D: SPECTER2 (best model for scientific citation retrieval)\n\n"
    "SPECTER2 is trained so that **papers that cite each other** are close in embedding space — "
    "exactly this task.  \n"
    "Embeddings are **cached** after the first run so re-running is instant.\n"
),

code(
    'SPECTER_CACHE   = DATA_DIR / "embeddings" / "specter2"\n'
    "specter_available = False\n"
    "specter_model     = None\n"
    "\n"
    "if (SPECTER_CACHE / \"corpus_embeddings.npy\").exists():\n"
    '    print("Loading cached SPECTER2 embeddings...")\n'
    '    s_c_embs = np.load(SPECTER_CACHE / "corpus_embeddings.npy").astype(np.float32)\n'
    '    s_q_embs = np.load(SPECTER_CACHE / "query_embeddings.npy").astype(np.float32)\n'
    "    specter_available = True\n"
    "    print(f\"  corpus: {s_c_embs.shape}  queries: {s_q_embs.shape}\")\n"
    "\n"
    "else:\n"
    "    # Priority: best scientific model first, then strong general models\n"
    "    for model_id in [\n"
    '        "allenai/specter2",          # SPECTER2 - trained on scientific citation prediction\n'
    '        "malteos/scincl",            # SciNCL   - another strong citation model\n'
    '        "BAAI/bge-base-en-v1.5",     # BGE-base - top general retrieval (MTEB)\n'
    '        "intfloat/e5-base-v2",       # E5-base  - strong general retrieval\n'
    "    ]:\n"
    "        try:\n"
    "            print(f\"Trying {model_id}...\")\n"
    "            specter_model = SentenceTransformer(model_id, trust_remote_code=True)\n"
    "            print(f\"  Loaded! dim={specter_model.get_sentence_embedding_dimension()}\")\n"
    "\n"
    "            corpus_texts = [fmt_ta(row) for _, row in corpus.iterrows()]\n"
    "            query_texts  = [fmt_ta(row) for _, row in queries.iterrows()]\n"
    "\n"
    '            print("  Encoding corpus (may take a few minutes on CPU)...")\n'
    "            s_c_embs = specter_model.encode(\n"
    "                corpus_texts, batch_size=128,\n"
    "                show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True,\n"
    "            ).astype(np.float32)\n"
    "\n"
    '            print("  Encoding queries...")\n'
    "            s_q_embs = specter_model.encode(\n"
    "                query_texts, batch_size=128,\n"
    "                show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True,\n"
    "            ).astype(np.float32)\n"
    "\n"
    "            SPECTER_CACHE.mkdir(parents=True, exist_ok=True)\n"
    '            np.save(SPECTER_CACHE / "corpus_embeddings.npy", s_c_embs)\n'
    '            np.save(SPECTER_CACHE / "query_embeddings.npy",  s_q_embs)\n'
    '            with open(SPECTER_CACHE / "corpus_ids.json", "w") as f: json.dump(corpus_ids, f)\n'
    '            with open(SPECTER_CACHE / "query_ids.json",  "w") as f: json.dump(query_ids, f)\n'
    '            with open(SPECTER_CACHE / "model_info.txt",  "w") as f: f.write(f"model: {model_id}\\n")\n'
    "\n"
    "            specter_available = True\n"
    "            print(f\"  Saved to {SPECTER_CACHE}/\")\n"
    "            break\n"
    "\n"
    "        except Exception as exc:\n"
    "            print(f\"  Failed ({type(exc).__name__}): {exc}\")\n"
    "\n"
    "if not specter_available:\n"
    '    print("No scientific/BGE model available -- continuing with MiniLM + BM25.")\n'
),

code(
    "specter_sub = None\n"
    "r_specter   = None\n"
    "\n"
    "if specter_available:\n"
    "    sim_specter = s_q_embs @ s_c_embs.T\n"
    "    top_specter = np.argsort(-sim_specter, axis=1)[:, :100]\n"
    "    specter_sub = {qid: [corpus_ids[j] for j in top_specter[i]] for i, qid in enumerate(query_ids)}\n"
    "    r_specter = evaluate(specter_sub, qrels)\n"
    '    show_results("SPECTER2 / SciNCL / BGE (dense)", r_specter)\n'
    "else:\n"
    '    print("Skipped -- model not available.")\n'
),

md("## 8 — Best Combination (RRF of all available models)"),

code(
    "# Optimal combo (empirically validated):\n"
    "#   WITHOUT SPECTER2: RRF(MiniLM, BM25_FT)\n"
    "#   WITH    SPECTER2: RRF(SPECTER2, MiniLM, BM25_FT)\n"
    "# BM25_TA is intentionally excluded — it is a strict subset of BM25_FT\n"
    "# and adding it only dilutes the signal.\n"
    "\n"
    "if specter_available:\n"
    "    best_sub = {\n"
    "        qid: rrf([specter_sub[qid], minilm_sub[qid], bm25_ft_sub[qid]])\n"
    "        for qid in query_ids\n"
    "    }\n"
    "    label = \"RRF (SPECTER2 + MiniLM + BM25_FT)\"\n"
    "else:\n"
    "    best_sub = rrf_AC\n"
    "    label    = \"RRF (MiniLM + BM25_FT)\"\n"
    "\n"
    "r_best = evaluate(best_sub, qrels)\n"
    "show_results(f\"BEST: {label}\", r_best)\n"
),

md("## 9 — Results Summary"),

code(
    "all_results = {\n"
    '    "A  MiniLM-L6-v2":       r_minilm,\n'
    '    "B  BM25 (TA)":          r_bm25_ta,\n'
    '    "C  BM25 (full text)":   r_bm25_ft,\n'
    '    "D  RRF (A+B)":          r_AB,\n'
    '    "E* RRF (A+C) [BEST]":   r_AC,\n'
    '    "F  RRF (A+B+C)":        r_ABC,\n'
    "}\n"
    "if specter_available and r_specter:\n"
    '    all_results["G  SPECTER2/BGE"]              = r_specter\n'
    '    all_results["H* RRF (G+A+C) [BEST+SPEC]"]  = r_best\n'
    "\n"
    'cols = ["recall@10","recall@100","ndcg@10","ndcg@100","precision@10","mrr@10","map"]\n'
    "df_compare = pd.DataFrame(all_results).T[cols]\n"
    "try:\n"
    '    display(df_compare.style.format("{:.4f}").highlight_max(axis=0, color="#c8e6c9"))\n'
    "except NameError:\n"
    '    print(df_compare.to_string(float_format=lambda x: f"{x:.4f}"))\n'
),

md(
    "## 10 — Held-Out Query Submission\n\n"
    "Held-out queries have no pre-computed embeddings — encode on the fly with the same pipeline.\n"
),

code(
    "ho_ta_texts = [fmt_ta(row) for _, row in held_out.iterrows()]\n"
    "ho_ta_tok   = [tokenize(t) for t in ho_ta_texts]\n"
    "\n"
    "# Dense: MiniLM\n"
    'print("Loading MiniLM for held-out encoding...")\n'
    'minilm_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")\n'
    "ho_minilm_embs = minilm_model.encode(\n"
    "    ho_ta_texts, batch_size=64, show_progress_bar=True,\n"
    "    normalize_embeddings=True, convert_to_numpy=True,\n"
    ").astype(np.float32)\n"
    "ho_sim = ho_minilm_embs @ c_embs.T\n"
    "ho_top = np.argsort(-ho_sim, axis=1)[:, :100]\n"
    "ho_minilm_sub = {qid: [corpus_ids[j] for j in ho_top[i]] for i, qid in enumerate(ho_ids)}\n"
    'print(f"MiniLM held-out: {len(ho_minilm_sub)} queries done.")\n'
),

code(
    "# BM25 full-text held-out (reuse the index built above; BM25_TA excluded from final combo)\n"
    "ho_bm25_ft_sub = {}\n"
    'for i, qid in enumerate(tqdm(ho_ids, desc="BM25 FT held-out")):\n'
    "    scores = bm25_ft.get_scores(ho_ta_tok[i])\n"
    "    ho_bm25_ft_sub[qid] = [corpus_ids[j] for j in np.argsort(-scores)[:100]]\n"
),

code(
    "# SPECTER2 / BGE held-out\n"
    "ho_specter_sub = None\n"
    "\n"
    "if specter_available:\n"
    "    if specter_model is None:\n"
    '        model_info = (SPECTER_CACHE / "model_info.txt").read_text().strip()\n'
    '        model_id   = model_info.split(": ", 1)[1] if ": " in model_info else "allenai/specter2"\n'
    '        print(f"Reloading {model_id}...")\n'
    "        specter_model = SentenceTransformer(model_id, trust_remote_code=True)\n"
    "\n"
    "    ho_s_embs = specter_model.encode(\n"
    "        ho_ta_texts, batch_size=128, show_progress_bar=True,\n"
    "        normalize_embeddings=True, convert_to_numpy=True,\n"
    "    ).astype(np.float32)\n"
    "    ho_s_sim = ho_s_embs @ s_c_embs.T\n"
    "    ho_s_top = np.argsort(-ho_s_sim, axis=1)[:, :100]\n"
    "    ho_specter_sub = {qid: [corpus_ids[j] for j in ho_s_top[i]] for i, qid in enumerate(ho_ids)}\n"
    '    print(f"SPECTER2 held-out: {len(ho_specter_sub)} queries done.")\n'
    "else:\n"
    '    print("SPECTER2 not available -- held-out uses MiniLM + BM25 only.")\n'
),

code(
    "held_out_final = build_best(ho_ids, ho_minilm_sub, ho_bm25_ft_sub,\n"
    "                            specter_d=ho_specter_sub)\n"
    "\n"
    'print(f"Held-out: {len(held_out_final)} queries x 100 results")\n'
    'assert all(len(v) == 100 for v in held_out_final.values()), "Some queries have != 100 results!"\n'
    'print("Sanity check passed.")\n'
),

md("## 11 — Save Submissions"),

code(
    "assert all(len(v) == 100 for v in best_sub.values())\n"
    "\n"
    'with open(OUT_DIR / "best_public.json", "w") as f:\n'
    "    json.dump(best_sub, f)\n"
    'print(f"Saved  {OUT_DIR}/best_public.json   ({len(best_sub)} queries)")\n'
    "\n"
    'with open(OUT_DIR / "held_out.json", "w") as f:\n'
    "    json.dump(held_out_final, f)\n"
    'print(f"Saved  {OUT_DIR}/held_out.json      ({len(held_out_final)} queries)")\n'
    "\n"
    "full_submission = {**best_sub, **held_out_final}\n"
    'with open(OUT_DIR / "final_submission.json", "w") as f:\n'
    "    json.dump(full_submission, f)\n"
    'print(f"Saved  {OUT_DIR}/final_submission.json  ({len(full_submission)} queries total)")\n'
    "\n"
    'print("\\n=== FINAL PUBLIC SCORES ===")\n'
    "for k, v in r_best.items():\n"
    '    print(f"  {k:<15}  {v:.4f}")\n'
),

]  # end cells

nb["cells"] = cells

out = os.path.join(os.path.dirname(__file__), "challenge.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# verify
with open(out, encoding="utf-8") as f:
    nb2 = json.load(f)
print(f"Written {len(nb2['cells'])} cells to {out}")
