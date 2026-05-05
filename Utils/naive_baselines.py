import numpy as np
import pandas as pd

def build_candidate_universe(*dfs):
    occs = []
    for df in dfs:
        occs.extend(df["occupationUri"].dropna().astype(str).unique().tolist())
    return sorted(set(occs))

def random_ranking(df_split, candidate_uris, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    candidates = np.array(candidate_uris, dtype=object)

    for _, q in df_split.iterrows():
        shuffled = candidates.copy()
        rng.shuffle(shuffled)

        for rank, cand in enumerate(shuffled, start=1):
            rows.append({
                "person_id": q["person_id"],
                "t": q["t"],
                "true_occ_uri": str(q["occupationUri"]),
                "candidate_occ_uri": str(cand),
                "rank": rank,
                "score": -rank,
            })

    return pd.DataFrame(rows)

def most_frequent_destination_ranking(df_split, train_df, candidate_uris):
    freq = train_df["occupationUri"].astype(str).value_counts()
    default = 0

    ranked_candidates = sorted(
        map(str, candidate_uris),
        key=lambda occ: (freq.get(occ, default), occ),
        reverse=True,
    )

    rows = []
    for _, q in df_split.iterrows():
        for rank, cand in enumerate(ranked_candidates, start=1):
            rows.append({
                "person_id": q["person_id"],
                "t": q["t"],
                "true_occ_uri": str(q["occupationUri"]),
                "candidate_occ_uri": str(cand),
                "rank": rank,
                "score": float(freq.get(cand, default)),
            })

    return pd.DataFrame(rows)