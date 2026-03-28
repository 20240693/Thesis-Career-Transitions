# Shared helper functions used across thesis notebooks

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple
from datetime import datetime
import json
import math
import os
import re

import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# Build positive SBERT training examples from CV–job pairs
def build_train_examples_with_meta(
    train_df: pd.DataFrame,
    max_samples: Optional[int] = None,
) -> tuple[list[InputExample], list[str], list[str]]:

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

        # Skip rows with missing or empty inputs
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

        # Remove duplicate CV–JD pairs
        key = (cv, jd)
        if key in seen:
            continue
        seen.add(key)

        examples.append(InputExample(texts=[cv, jd]))
        groups.append(grp_str)
        occs.append(occ)

    return examples, groups, occs

# Build retrieval evaluator for CV -> job description matching
def build_ir_eval_unique_jobs_df(
    df: pd.DataFrame,
    name: str = "ir-eval-unique-jobs",
    precision_recall_at_k: list[int] = [1, 3, 5, 10, 20],
    mrr_at_k: list[int] = [10],
    ndcg_at_k: list[int] = [10],
    map_at_k: list[int] = [10],
) -> InformationRetrievalEvaluator:

    # Build corpus with one job description per occupation
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

        # Keep only valid CV queries with a target occupation in the corpus
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

# Train SBERT with MultipleNegativesRankingLoss using a provided DataLoader
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

    # Initialize model if not provided
    if model is None:
        model = SentenceTransformer(model_name)

    # Use in-batch negatives through MultipleNegativesRankingLoss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    if steps_per_epoch is None:
        steps_per_epoch = len(train_dataloader)

    # Warm up learning rate over first 10% of total training steps
    warmup_steps = math.ceil(steps_per_epoch * epochs * 0.1)

    # Evaluate periodically during training
    if evaluation_steps is None:
        evaluation_steps = max(1000, steps_per_epoch // 2)

    # Create output directory and timestamped run folder
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = os.path.join(out_dir, f"{run_name}_{timestamp}")

    # Train and save model
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

# Save a Python object as JSON
def save_json(obj: Any, filename: str, out_dir: str = ".", indent: int = 2) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)
    print("Saved:", path)
    return path

# Convert evaluator metric names to a clean pandas Series
# Example: "test-ir_cosine_recall@10" -> "recall@10"
def metrics_dict_to_series(metrics_dict):    
    clean = {}
    for k, v in metrics_dict.items():
        m = re.search(r"(recall|precision|accuracy|ndcg|mrr|map)@(\d+)", k, re.IGNORECASE)
        if m:
            metric = m.group(1).lower()
            K = m.group(2)
            clean[f"{metric}@{K}"] = float(v)

    return pd.Series(clean)