"""
submission_utils.py — Shared helpers for IR challenge submissions.

Provides: path resolution, data loading, validation, JSON export, zip.
Used by iteration scripts under iterations/. The challenge root is derived
from this file's location: scripts/ -> parent = challenge root.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

DEFAULT_TOP_K = 100


def challenge_dir() -> Path:
    """Return the challenge root directory (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def data_paths() -> dict:
    """
    Resolve all relevant data paths.

    Searches for held_out_queries.parquet in order:
      1. data/held_out_queries.parquet
      2. <root>/held_out_queries.parquet   ← typical location in this project
      3. <root>/starter_kit/held_out_queries.parquet

    If found, queries_path points to the held-out file and
    using_held_out_queries is True. Otherwise falls back to
    data/queries.parquet (public queries).
    """
    root = challenge_dir()
    data = root / "data"
    candidates = [
        data / "held_out_queries.parquet",
        root / "held_out_queries.parquet",
        root / "starter_kit" / "held_out_queries.parquet",
    ]
    held_out = next((p for p in candidates if p.exists()), None)
    queries = held_out if held_out is not None else data / "queries.parquet"
    return {
        "challenge_dir": root,
        "data_dir": data,
        "queries_path": queries,
        "queries_source": str(queries.relative_to(root)) if queries.exists() else str(queries),
        "using_held_out_queries": held_out is not None,
        "corpus_path": data / "corpus.parquet",
        "qrels_path": data / "qrels.json",
        "sample_submission_path": data / "sample_submission.json",
    }


def iteration_submission_paths(iteration_name: str) -> dict:
    """
    Return output paths for a named iteration.

    Creates: submissions/<iteration_name>/submission_data.json
             submissions/<iteration_name>/<iteration_name>.zip
    """
    root = challenge_dir()
    out = root / "submissions" / iteration_name
    return {
        "output_dir": out,
        "output_file": out / "submission_data.json",
        "zip_file": out / f"{iteration_name}.zip",
    }


def load_queries_corpus(
    queries_path: Path, corpus_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load public queries and corpus from parquet files."""
    queries = pd.read_parquet(queries_path)
    corpus = pd.read_parquet(corpus_path)
    return queries, corpus


def validate_submission(
    submission: Dict[str, List[str]],
    expected_query_ids: List[str],
    top_k: int = DEFAULT_TOP_K,
) -> None:
    """
    Raise ValueError if the submission format is invalid:
      - query IDs must match expected_query_ids exactly
      - each query must have exactly top_k results
      - all result doc_ids must be strings
      - no duplicates within a query's ranked list
    """
    submission_ids = set(submission.keys())
    expected_ids = set(expected_query_ids)
    if submission_ids != expected_ids:
        missing = expected_ids - submission_ids
        extra = submission_ids - expected_ids
        raise ValueError(
            f"Submission query ID mismatch — missing: {len(missing)}, extra: {len(extra)}"
        )
    for qid in expected_query_ids:
        ranked = submission[qid]
        if len(ranked) != top_k:
            raise ValueError(
                f"Query {qid}: expected {top_k} docs, got {len(ranked)}"
            )
        if not all(isinstance(d, str) for d in ranked):
            raise ValueError(f"Query {qid}: all doc_ids must be strings")
        if len(set(ranked)) != top_k:
            raise ValueError(f"Query {qid}: duplicate doc_ids in ranked list")


def validate_doc_ids_in_corpus(
    submission: Dict[str, List[str]], corpus_doc_ids: List[str]
) -> None:
    """Raise ValueError if any ranked doc_id is not present in the corpus."""
    corpus_set = set(corpus_doc_ids)
    invalid = sum(
        1
        for ranked in submission.values()
        for doc_id in ranked
        if doc_id not in corpus_set
    )
    if invalid > 0:
        raise ValueError(
            f"Submission contains {invalid} doc_ids not found in the corpus"
        )


def save_submission(submission: Dict[str, List[str]], output_file: Path) -> None:
    """Write submission dict to JSON."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)
    print(f"Submission saved to {output_file}")


def create_submission_zip(
    submission_file: Path,
    zip_file: Path,
    arcname: str = "submission_data.json",
) -> None:
    """Zip the submission JSON for Codabench upload."""
    zip_file.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_file, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission_file, arcname=arcname)
    print(f"Submission zipped  to {zip_file}")
