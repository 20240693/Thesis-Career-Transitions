import math
import pandas as pd

def eval_ranking_df(df, query_cols=("person_id", "t"), rank_col="rank",
                    candidate_col="candidate_occ_uri", true_col="true_occ_uri",
                    k_list=(1, 3, 5, 10, 20)):
    df = df.copy()
    df["is_true"] = (df[candidate_col].astype(str) == df[true_col].astype(str)).astype(int)

    n_queries = df.groupby(list(query_cols)).ngroups
    true_ranks = df[df["is_true"] == 1].groupby(list(query_cols))[rank_col].min()

    out = {}
    for k in k_list:
        out[f"recall@{k}"] = float((true_ranks <= k).sum() / n_queries)
        out[f"mrr@{k}"] = float(((1.0 / true_ranks[true_ranks <= k]).sum()) / n_queries)
        out[f"ndcg@{k}"] = float(sum(
            1.0 / math.log2(r + 1) for r in true_ranks if r <= k
        ) / n_queries)

    return out