import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize


def recall_key(k) -> str:
    """Return the dict key used for R@k (e.g. 10.0 → 'R@10', 10.5 → 'R@10.5')."""
    return "R@" + (str(int(k)) if k == int(k) else str(k))


def get_retrieval_metrics(modality1_emb, modality2_emb, normalized=False, k=100):
    """Compute R@k and Median Rank. Returns a dict; no per-row DataFrame saved."""
    if not normalized:
        modality1_emb_mean = normalize(modality1_emb, p=2, dim=1)
        modality2_emb_mean = normalize(modality2_emb, p=2, dim=1)
    else:
        modality1_emb_mean = modality1_emb
        modality2_emb_mean = modality2_emb

    cos_sim = torch.matmul(modality1_emb_mean, modality2_emb_mean.t()).detach().cpu().numpy()
    distance_matrix = cos_sim

    K = distance_matrix.shape[0]
    results = []
    for i in range(K):
        tmpdf = pd.DataFrame({"dist": distance_matrix[i, :]})
        tmpdf["rank"] = tmpdf.dist.rank(ascending=False)
        results.append({"rank": tmpdf.iloc[i]["rank"]})
    df = pd.DataFrame(results)
    metrics = {
        recall_key(k): (df["rank"] < k).mean(),
        "Median Rank": df["rank"].median(),
    }
    return metrics


# Backward-compatible alias
get_retrevial_metrics = get_retrieval_metrics


def get_retrieval(modality1_emb, modality2_emb, keys, normalized=False, k=100, save_top=5):
    if not normalized:
        modality1_emb_mean = normalize(modality1_emb, p=2, dim=1)
        modality2_emb_mean = normalize(modality2_emb, p=2, dim=1)
    else:
        modality1_emb_mean = modality1_emb
        modality2_emb_mean = modality2_emb

    cos_sim = torch.matmul(modality1_emb_mean, modality2_emb_mean.t()).detach().cpu().numpy()
    distance_matrix = cos_sim
    K = distance_matrix.shape[0]

    results = []
    df_final = pd.DataFrame(columns=["key", "top_keys"])
    df_final["key"] = keys

    results_keys = []
    for i in range(K):
        row_similarity = list(distance_matrix[i, :])
        top_indices = np.array(row_similarity).argsort()[-save_top:][::-1]
        top_keys = [keys[idx] for idx in top_indices]
        results_keys.append(top_keys)
        tmpdf = pd.DataFrame({"dist": distance_matrix[i, :]})
        tmpdf["rank"] = tmpdf.dist.rank(ascending=False)
        results.append({"rank": tmpdf.iloc[i]["rank"]})
    df = pd.DataFrame(results)
    metrics = {
        recall_key(k): (df["rank"] < k).mean(),
        "Median Rank": df["rank"].median(),
    }
    df_final["top_keys"] = results_keys
    return metrics, df_final


# Backward-compatible alias
get_retrevial = get_retrieval
