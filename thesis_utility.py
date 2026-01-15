# thesis_utils.py
# ============================================================
# Shared utilities for SBERT baseline + ontology-aware training
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import random
import math
from datetime import datetime

from torch.utils.data import Sampler

# ----------------------------
# Similarity helpers
# ----------------------------

def jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ----------------------------
# Ontology index container
# ----------------------------

@dataclass(frozen=True)
class OntologyIndex:
    """Holds precomputed ontology indices (train-only)."""
    occ_to_isco: Dict[str, str]
    isco_to_occs: Dict[str, List[str]]
    occ_to_skills: Dict[str, Set[str]]
    occ_to_label: Optional[Dict[str, str]] = None


def build_ontology_index(
    train_df,
    occ_skill_df,
    relation_type: str = "essential",
    label_col: str = "preferredLabel",
) -> OntologyIndex:
    """
    Build ontology indices restricted to train occupations.
    Uses ESCO occupationSkillRelations_en.csv (occ_skill_df).
    """
    train_occs = set(train_df["occupationUri"].dropna().unique())

    occ_to_isco = (
        train_df[["occupationUri", "iscoGroup"]]
        .dropna()
        .drop_duplicates()
        .set_index("occupationUri")["iscoGroup"]
        .to_dict()
    )

    isco_to_occs = defaultdict(list)
    for occ, grp in occ_to_isco.items():
        isco_to_occs[grp].append(occ)

    # essential skills only
    occ_skill_essential = occ_skill_df[
        occ_skill_df["relationType"].str.lower() == relation_type.lower()
    ]

    occ_to_skills = defaultdict(set)
    for occ, skill in zip(occ_skill_essential["occupationUri"], occ_skill_essential["skillUri"]):
        if occ in train_occs:
            occ_to_skills[occ].add(skill)

    occ_to_label = None
    if label_col in train_df.columns:
        occ_to_label = (
            train_df[["occupationUri", label_col]]
            .dropna()
            .drop_duplicates()
            .set_index("occupationUri")[label_col]
            .to_dict()
        )

    return OntologyIndex(
        occ_to_isco=occ_to_isco,
        isco_to_occs=dict(isco_to_occs),
        occ_to_skills=dict(occ_to_skills),
        occ_to_label=occ_to_label,
    )


# ----------------------------
# Dataset builders
# ----------------------------

def build_train_examples_with_meta(train_df, max_samples: Optional[int] = None):
    """
    Build Sentence-Transformers InputExample list + aligned metadata.
    Returns: (examples, isco_groups, occupation_uris)

    Includes global deduplication of exact (cv_text, jd_text) pairs.
    """
    from sentence_transformers import InputExample

    if max_samples is not None:
        train_df = train_df.head(max_samples)

    examples: List[InputExample] = []
    groups: List[str] = []
    occs: List[str] = []
    seen: Set[Tuple[str, str]] = set()

    for _, row in train_df.iterrows():
        cv = row.get("cv_text")
        jd = row.get("jd_text")
        grp = row.get("iscoGroup")
        occ = row.get("occupationUri")

        # must have non-empty texts
        if not (isinstance(cv, str) and isinstance(jd, str) and cv.strip() and jd.strip()):
            continue

        # iscoGroup can be int -> convert to string
        if grp is None:
            continue
        grp = str(grp).strip()
        if not grp:
            continue

        # must have occupationUri
        if not (isinstance(occ, str) and occ.strip()):
            continue

        cv = cv.strip()
        jd = jd.strip()

        key = (cv, jd)
        if key in seen:
            continue
        seen.add(key)

        examples.append(InputExample(texts=[cv, jd]))
        groups.append(grp)          # now always a string
        occs.append(occ.strip())

    return examples, groups, occs


# ----------------------------
# Retrieval evaluator builder
# ----------------------------

def build_ir_eval_unique_jobs_df(df, name: str = "ir-eval-unique-jobs"):
    """
    Build InformationRetrievalEvaluator for CV -> JD retrieval.
    Corpus is deduplicated by occupationUri to prevent duplicate positives.
    """
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    corpus = (
        df[["occupationUri", "jd_text"]]
        .dropna()
        .drop_duplicates(subset=["occupationUri"])
        .set_index("occupationUri")["jd_text"]
        .to_dict()
    )

    queries = {}
    relevant_docs = {}

    q_idx = 0
    for _, row in df.iterrows():
        cv = row.get("cv_text")
        occ = row.get("occupationUri")

        if not isinstance(cv, str) or not cv.strip():
            continue
        if not isinstance(occ, str) or occ not in corpus:
            continue

        qid = f"q{q_idx}"
        queries[qid] = cv
        relevant_docs[qid] = {occ}
        q_idx += 1

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        precision_recall_at_k=[1, 3, 5, 10, 20],
        mrr_at_k=[10],
        ndcg_at_k=[10],
        map_at_k=[10],
    )


# ----------------------------
# Batch samplers (MNLR hard batching)
# ----------------------------

class ISCOBatchSampler(Sampler):
    """
    ISCO-hard batching: each batch drawn from one iscoGroup.
    Includes per-batch duplicate-pair protection.

    Cautions:
    - MNLR requires batch_size >= 2
    - Groups smaller than 2 are ignored
    """
    def __init__(self, group_to_indices, examples, batch_size: int, drop_last: bool = True, seed: int = 42):
        assert batch_size >= 2, "MNLR needs batch_size >= 2"
        self.group_to_indices = {g: inds[:] for g, inds in group_to_indices.items()}
        self.examples = examples
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = random.Random(seed)
        self.groups = [g for g, inds in self.group_to_indices.items() if len(inds) >= 2]

    def __iter__(self):
        for g in self.groups:
            self.rng.shuffle(self.group_to_indices[g])

        group_order = self.groups[:]
        self.rng.shuffle(group_order)

        for g in group_order:
            inds = self.group_to_indices[g]
            batch = []
            used_pairs = set()

            for idx in inds:
                ex = self.examples[idx]
                pair_key = (ex.texts[0].strip(), ex.texts[1].strip())
                if pair_key in used_pairs:
                    continue
                used_pairs.add(pair_key)
                batch.append(idx)

                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                    used_pairs = set()

            if batch and (not self.drop_last) and len(batch) >= 2:
                yield batch

    def __len__(self):
        return sum(len(self.group_to_indices[g]) // self.batch_size for g in self.groups)


def top_hybrid_neighbor_occs(pos_occ: str, index: OntologyIndex, top_m: int = 50) -> List[str]:
    """Return top-M neighbor occupations within same ISCO group ranked by skill overlap."""
    grp = index.occ_to_isco.get(pos_occ)
    if grp is None:
        return []
    candidates = [o for o in index.isco_to_occs.get(grp, []) if o != pos_occ]
    pos_sk = index.occ_to_skills.get(pos_occ, set())

    scored = [(o, jaccard(pos_sk, index.occ_to_skills.get(o, set()))) for o in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [o for o, _ in scored[:top_m]]


class HybridBatchSampler(Sampler):
    """
    Hybrid-hard batching: seed an example, then fill the batch using examples whose
    occupations are among the seed occupation's top hybrid neighbors.

    Safety fixes:
    - No global "used" constraint (avoids end-of-epoch infinite loops).
    - Keeps per-batch duplicate-pair protection.

    Notes:
    - With MNLR, batch_size must be >= 2.
    - Examples can appear in multiple batches within an epoch (this is OK / typical).
    """

    def __init__(
        self,
        examples,
        occs_by_example: List[str],
        occ_to_neighbors: Dict[str, List[str]],
        occ_to_example_indices: Dict[str, List[int]],
        batch_size: int,
        seed: int = 42,
        max_tries: int = 50,
    ):
        assert batch_size >= 2, "MNLR needs batch_size >= 2"
        self.examples = examples
        self.occs_by_example = occs_by_example
        self.occ_to_neighbors = occ_to_neighbors
        self.occ_to_example_indices = occ_to_example_indices
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.max_tries = max_tries
        self.all_indices = list(range(len(examples)))

    def __iter__(self):
        seeds = self.all_indices[:]
        self.rng.shuffle(seeds)

        for seed_idx in seeds:
            batch: List[int] = []
            used_pairs = set()

            def try_add(idx: int) -> bool:
                ex = self.examples[idx]
                key = (ex.texts[0].strip(), ex.texts[1].strip())
                if key in used_pairs:
                    return False
                used_pairs.add(key)
                batch.append(idx)
                return True

            # 1) Add seed
            try_add(seed_idx)

            # 2) Fill from hybrid neighbors
            seed_occ = self.occs_by_example[seed_idx]
            neighbor_occs = self.occ_to_neighbors.get(seed_occ, [])

            tries = 0
            while len(batch) < self.batch_size and tries < self.max_tries and neighbor_occs:
                tries += 1
                occ = self.rng.choice(neighbor_occs)
                cand_list = self.occ_to_example_indices.get(occ, [])
                if not cand_list:
                    continue
                cand_idx = self.rng.choice(cand_list)
                try_add(cand_idx)

            # 3) Fallback: fill with random examples (bounded attempts to avoid rare stalls)
            attempts = 0
            max_attempts = 20 * self.batch_size
            while len(batch) < self.batch_size and attempts < max_attempts:
                attempts += 1
                cand_idx = self.rng.choice(self.all_indices)
                try_add(cand_idx)

            # If we still couldn't fill due to extreme duplicate-pair filtering, just skip
            if len(batch) < 2:
                continue

            # If we couldn't fully fill, pad by relaxing duplicate protection (very rare).
            # (MNLR requires batch_size>=2; full batch preferred.)
            while len(batch) < self.batch_size:
                batch.append(self.rng.choice(self.all_indices))

            yield batch

    def __len__(self):
        # Approximate number of batches (same convention as before)
        return len(self.examples) // self.batch_size

class HybridOrderSampler(Sampler[int]):
    """
    Hybrid-hard batching but implemented as a *flat* sampler:
    - constructs batches internally
    - yields indices in batch order (flattened)
    - DataLoader(batch_size=...) will create the actual batches

    This keeps DataLoader.batch_size != None, which avoids the NoneType * int error.
    """

    def __init__(
        self,
        examples,
        occs_by_example: List[str],
        occ_to_neighbors: Dict[str, List[str]],
        occ_to_example_indices: Dict[str, List[int]],
        batch_size: int,
        seed: int = 42,
        max_tries: int = 50,
        drop_last: bool = True,
    ):
        assert batch_size >= 2, "MNLR needs batch_size >= 2"
        self.examples = examples
        self.occs_by_example = occs_by_example
        self.occ_to_neighbors = occ_to_neighbors
        self.occ_to_example_indices = occ_to_example_indices
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.max_tries = max_tries
        self.drop_last = drop_last
        self.all_indices = list(range(len(examples)))

    def __iter__(self):
        seeds = self.all_indices[:]
        self.rng.shuffle(seeds)

        for seed_idx in seeds:
            batch: List[int] = []
            used_pairs = set()

            def try_add(idx: int) -> bool:
                if idx < 0 or idx >= len(self.examples):
                    return False
                ex = self.examples[idx]
                key = (ex.texts[0].strip(), ex.texts[1].strip())
                if key in used_pairs:
                    return False
                used_pairs.add(key)
                batch.append(idx)
                return True

            # 1) seed
            try_add(seed_idx)

            # 2) neighbors
            seed_occ = self.occs_by_example[seed_idx]
            neighbor_occs = self.occ_to_neighbors.get(seed_occ, [])

            tries = 0
            while len(batch) < self.batch_size and tries < self.max_tries and neighbor_occs:
                tries += 1
                occ = self.rng.choice(neighbor_occs)
                cand_list = self.occ_to_example_indices.get(occ, [])
                if not cand_list:
                    continue
                cand_idx = self.rng.choice(cand_list)
                try_add(cand_idx)

            # 3) fallback random (bounded)
            attempts = 0
            max_attempts = 50 * self.batch_size
            while len(batch) < self.batch_size and attempts < max_attempts:
                attempts += 1
                cand_idx = self.rng.choice(self.all_indices)
                try_add(cand_idx)

            # drop incomplete batch if requested
            if len(batch) < self.batch_size:
                if self.drop_last:
                    continue
                if len(batch) < 2:
                    continue

            # yield flattened indices
            for idx in batch:
                yield idx

    def __len__(self):
        # approximate number of samples yielded
        if self.drop_last:
            return (len(self.examples) // self.batch_size) * self.batch_size
        return len(self.examples)

# ----------------------------
# Training helper
# ----------------------------

def train_sbert_with_dataloader(train_dataloader, ir_evaluator, run_name: str,
                               model=None,
                               model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                               epochs: int = 1,
                               evaluation_steps: int | None = None,
                               steps_per_epoch: int | None = None):
    """
    Train SBERT with MultipleNegativesRankingLoss using a provided dataloader.

    IMPORTANT:
    - Do NOT pass batch_size into SentenceTransformer.fit() (it's not a valid kwarg).
    - When using batch_sampler, SentenceTransformers may not infer batch_size, so provide
      steps_per_epoch explicitly (recommended: len(train_dataloader)).
    """
    import math
    from datetime import datetime
    from sentence_transformers import SentenceTransformer, losses

    if model is None:
        model = SentenceTransformer(model_name)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    if steps_per_epoch is None:
        # Works for both normal DataLoader and batch_sampler DataLoader
        steps_per_epoch = len(train_dataloader)

    warmup_steps = math.ceil(steps_per_epoch * epochs * 0.1)

    if evaluation_steps is None:
        evaluation_steps = max(1000, steps_per_epoch // 2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = f"/content/drive/MyDrive/NOVA IMS/Year 2/Thesis/Models/{run_name}_{timestamp}"

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=ir_evaluator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,   # key line
        warmup_steps=warmup_steps,
        evaluation_steps=evaluation_steps,
        output_path=output_path,
        show_progress_bar=True,
    )

    return output_path