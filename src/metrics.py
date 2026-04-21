import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize
def get_retrevial_metrics(modality1_emb, modality2_emb, normalized=False,k=100): #Used during training where we don't need dataframe of retrevial saved.
    if not normalized:
        # Normalize embeddings using L2 normalization
        modality1_emb_mean = normalize(modality1_emb, p=2, dim=1)
        modality2_emb_mean = normalize(modality2_emb, p=2, dim=1)
    else:
        modality1_emb_mean = modality1_emb
        modality2_emb_mean = modality2_emb

    cos_sim = torch.matmul(modality1_emb_mean, modality2_emb_mean.t()).detach().cpu().numpy() 
    distance_matrix = cos_sim
    
    K = distance_matrix.shape[0]
    # Evaluate Img2Sound
    results = []
    for i in list(range(K)):
        tmpdf = pd.DataFrame(dict(
            dist = distance_matrix[i,:]
        ))

        tmpdf['rank'] = tmpdf.dist.rank(ascending=False)
        res = dict(
            rank=tmpdf.iloc[i]['rank']
        )
        results.append(res)
    df = pd.DataFrame(results)
    topk_str =str(1*k) 
    i2s_metrics = {
        'R@'+topk_str: (df['rank'] < k).mean(),
        'Median Rank': df['rank'].median(),
    }

    return i2s_metrics


def get_retrevial(modality1_emb, modality2_emb, keys,normalized=False,k=100,save_top=5):
    
    if not normalized:
        # Normalize embeddings using L2 normalization
        modality1_emb_mean = normalize(modality1_emb, p=2, dim=1)
        modality2_emb_mean = normalize(modality2_emb, p=2, dim=1)
    else:
        modality1_emb_mean = modality1_emb
        modality2_emb_mean = modality2_emb

    cos_sim = torch.matmul(modality1_emb_mean, modality2_emb_mean.t()).detach().cpu().numpy() 
    distance_matrix = cos_sim
    K = distance_matrix.shape[0]
    
    # Evaluate Img2Sound
    results = []
    df_final = pd.DataFrame(columns=['key','top_keys'])
    df_final['key'] = keys
    
    
    results_keys = []
    for i in list(range(K)):
        top_keys = []
        tmpdf = pd.DataFrame(dict(
            dist = distance_matrix[i,:]
        ))
        row_similarity = list(distance_matrix[i, :])
        top_indices = np.array(row_similarity).argsort()[-save_top:][::-1]
        top_keys = [keys[indice] for indice in top_indices]
        results_keys.append(top_keys)
        tmpdf['rank'] = tmpdf.dist.rank(ascending=False)
        res = dict(
            rank=tmpdf.iloc[i]['rank']
        )
        results.append(res)
    df = pd.DataFrame(results)
    topk_str =str(1*k)
    i2s_metrics = {
        'R@'+topk_str: (df['rank'] < k).mean(),
        'Median Rank': df['rank'].median(),
    }      
    df_final['top_keys'] = results_keys
    return i2s_metrics, df_final
