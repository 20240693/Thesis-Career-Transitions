"""
thesis_utility.py

Small, shared utilities used across notebooks for:
- Building SBERT training examples (positive CV–JD pairs)
- Building an InformationRetrievalEvaluator (CV -> job retrieval)
- Training SBERT with a provided DataLoader

Designed for Google Colab + Drive paths.
"""

from __future__ import annotations

from typing import Optional, Set, Tuple, List, Dict

import math
from datetime import datetime

import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator

import os
import json
from typing import Any, Optional, Mapping

import re


def build_train_examples_with_meta(
    train_df: pd.DataFrame,
    max_samples: Optional[int] = None,
) -> tuple[list[InputExample], list[str], list[str]]:
    """
    Build Sentence-Transformers InputExample list + aligned metadata.

    Returns:
        examples: List[InputExample] with texts=[cv_text, jd_text]
        groups:   List[str] of iscoGroup (string)
        occs:     List[str] of occupationUri

    Notes:
    - Drops rows with missing/empty cv_text or jd_text
    - Drops rows with missing iscoGroup or occupationUri
    - Deduplicates exact (cv_text, jd_text) pairs globally
    """
    if max_samples is not None:
        train_df = train_df.head(max_samples)

    examples: list[InputExample] = []
    groups: list[str] = []
    occs: list[str] = []
    seen: Set[Tuple[str, str]] = set()

    for _, row in train_df.iterrows():
        cv = row.get("cv_text")
        jd = row.get("jd_text")
        grp = row.get("iscoGroup")
        occ = row.get("occupationUri")

        if not (isinstance(cv, str) and cv.strip()):
            continue
        if not (isinstance(jd, str) and jd.strip()):
            continue
        if grp is None:
            continue

        grp_str = str(grp).strip()
        if not grp_str:
            continue

        if not (isinstance(occ, str) and occ.strip()):
            continue

        cv = cv.strip()
        jd = jd.strip()
        occ = occ.strip()

        key = (cv, jd)
        if key in seen:
            continue
        seen.add(key)

        examples.append(InputExample(texts=[cv, jd]))
        groups.append(grp_str)
        occs.append(occ)

    return examples, groups, occs


def build_ir_eval_unique_jobs_df(
    df: pd.DataFrame,
    name: str = "ir-eval-unique-jobs",
    precision_recall_at_k: list[int] = [1, 3, 5, 10, 20],
    mrr_at_k: list[int] = [10],
    ndcg_at_k: list[int] = [10],
    map_at_k: list[int] = [10],
) -> InformationRetrievalEvaluator:
    """
    Build InformationRetrievalEvaluator for CV -> JD retrieval.

    Corpus:
      - occupationUri -> jd_text
      - deduplicated by occupationUri to avoid duplicate positives

    Queries:
      - one query per row with non-empty cv_text and a valid occupationUri in corpus
      - relevant_docs[qid] = {occupationUri}
    """
    corpus: Dict[str, str] = (
        df[["occupationUri", "jd_text"]]
        .dropna()
        .drop_duplicates(subset=["occupationUri"])
        .set_index("occupationUri")["jd_text"]
        .to_dict()
    )

    queries: Dict[str, str] = {}
    relevant_docs: Dict[str, Set[str]] = {}

    q_idx = 0
    for _, row in df.iterrows():
        cv = row.get("cv_text")
        occ = row.get("occupationUri")

        if not (isinstance(cv, str) and cv.strip()):
            continue
        if not (isinstance(occ, str) and occ in corpus):
            continue

        qid = f"q{q_idx}"
        queries[qid] = cv.strip()
        relevant_docs[qid] = {occ}
        q_idx += 1

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        precision_recall_at_k=precision_recall_at_k,
        mrr_at_k=mrr_at_k,
        ndcg_at_k=ndcg_at_k,
        map_at_k=map_at_k,
    )


def train_sbert_with_dataloader(
    train_dataloader,
    ir_evaluator,
    run_name: str,
    *,
    model: Optional[SentenceTransformer] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    epochs: int = 1,
    evaluation_steps: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    out_dir: str = "./Models",
) -> str:
    """
    Train SBERT with MultipleNegativesRankingLoss using a provided DataLoader.

    Args:
        train_dataloader: DataLoader yielding InputExample batches.
        ir_evaluator: Sentence-Transformers evaluator (e.g., InformationRetrievalEvaluator).
        run_name: Name prefix for the run folder.
        model: Existing SentenceTransformer model (optional).
        model_name: HF model name if model is None.
        epochs: Training epochs.
        evaluation_steps: Steps between evals (default: max(1000, steps_per_epoch//2)).
        steps_per_epoch: Defaults to len(train_dataloader).
        out_dir: Base output directory (created if missing).

    Returns:
        output_path: Path where the model was saved.
    """
    if model is None:
        model = SentenceTransformer(model_name)

    train_loss = losses.MultipleNegativesRankingLoss(model)

    if steps_per_epoch is None:
        steps_per_epoch = len(train_dataloader)

    warmup_steps = math.ceil(steps_per_epoch * epochs * 0.1)

    if evaluation_steps is None:
        evaluation_steps = max(1000, steps_per_epoch // 2)

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = os.path.join(out_dir, f"{run_name}_{timestamp}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=ir_evaluator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_steps=warmup_steps,
        evaluation_steps=evaluation_steps,
        output_path=output_path,
        show_progress_bar=True,
    )

    return output_path


def save_json(obj: Any, filename: str, out_dir: str = ".", indent: int = 2) -> str:
    """
    Save a Python object as JSON to out_dir/filename.

    Args:
        obj: JSON-serializable object.
        filename: e.g., "metrics.json"
        out_dir: output directory (created if missing).
        indent: JSON indentation.

    Returns:
        Full output path.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)
    print("Saved:", path)
    return path


def metrics_dict_to_series(metrics_dict: Mapping[str, Any]) -> pd.Series:
    """
    Convert Sentence-Transformers evaluator metrics dict into a tidy pandas Series.

    Example:
      "test-ir_cosine_recall@10" -> "recall@10"

    Keeps:
      recall@K, precision@K, accuracy@K, ndcg@K, mrr@K, map@K
    """
    clean = {}
    pattern = re.compile(r"(recall|precision|accuracy|ndcg|mrr|map)@(\d+)", re.IGNORECASE)

    for k, v in metrics_dict.items():
        m = pattern.search(str(k))
        if not m:
            continue
        metric = m.group(1).lower()
        K = m.group(2)
        clean[f"{metric}@{K}"] = float(v)

    # Optional: return sorted keys for consistent tables
    return pd.Series(clean)