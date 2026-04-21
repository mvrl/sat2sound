# Taken from https://github.com/bair-climate-initiative/scale-mae/blob/main/mae/util/pos_embed.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
# --------------------------------------------------------
# 2D sine-cosine position embedding with per-image resolution.
# References:
#   Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
#   MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import torch


def get_2d_sincos_pos_embed_with_resolution(embed_dim, grid_size, res, cls_token=False, device="cpu"):
    """2D sine-cosine positional embedding with per-image resolution.

    Args:
        embed_dim: output dim of the embedding.
        grid_size: side length of the (square) grid.
        res: Tensor of shape ``(n,)`` giving the per-image pixel resolution (e.g. meters).
        cls_token: if True, prepend a zero row for a [CLS] token.

    Returns:
        Tensor of shape ``(n, grid_size*grid_size, embed_dim)`` (or with +1 if ``cls_token``).
    """
    res = res.to(device)
    grid_h = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # here h goes first (reversed vs numpy)
    grid = torch.stack(grid, dim=0)  # 2 x h x w
    grid = torch.einsum("chw,n->cnhw", grid, res)  # 2 x n x h x w
    _, n, h, w = grid.shape
    pos_embed = _get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid)  # (n*H*W, D)
    pos_embed = pos_embed.reshape(n, h * w, embed_dim)
    if cls_token:
        cls_row = torch.zeros([n, 1, embed_dim], dtype=torch.float32, device=pos_embed.device)
        pos_embed = torch.cat([cls_row, pos_embed], dim=1)
    return pos_embed


def _get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = _get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[1])  # (H*W, D/2)
    return torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)


def _get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)
