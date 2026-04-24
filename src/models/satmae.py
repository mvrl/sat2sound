# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
from functools import partial

import pytorch_lightning as pl
import timm.models.vision_transformer
import torch
import torch.nn as nn

from .pos_embed import get_2d_sincos_pos_embed_with_resolution

image_gsd = {'sentinel':10, 'bingmap':0.6, 'googleEarth':0.2}
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------



class VisionTransformer_w_cls(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer_w_cls, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm
            del self.cls_token
            self.class_token_flag = False
        else:
            self.class_token_flag = True

        del self.pos_embed
        del self.head
        
    def forward_features(self, x,input_res=None):
        B, _, h, w = x.shape
        input_res = input_res.cpu()
        x = self.patch_embed(x)
        num_patches = int(
            (h * w) / (self.patch_embed.patch_size[0] * self.patch_embed.patch_size[1])
        )
        pos_embed = get_2d_sincos_pos_embed_with_resolution(
            x.shape[-1],
            int(num_patches**0.5),
            input_res,
            cls_token=self.class_token_flag,
            device=x.device,
        )
        
        if self.class_token_flag:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + pos_embed

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x.mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:,0] #return cls token
        
        return outcome
    
    def forward(self, x, input_res=None):
        x = self.forward_features(x, input_res)
        return x



class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.class_token_flag = False
        del self.pos_embed
        del self.head
        del self.norm  
        del self.cls_token

    def forward_features(self, x,input_res=None):
        B, _, h, w = x.shape
        input_res = input_res.cpu()
        x = self.patch_embed(x)
        num_patches = int(
            (h * w) / (self.patch_embed.patch_size[0] * self.patch_embed.patch_size[1])
        )
        pos_embed = get_2d_sincos_pos_embed_with_resolution(
            x.shape[-1],
            int(num_patches**0.5),
            input_res,
            cls_token=self.class_token_flag,
            device=x.device,
        )
        
        x = x + pos_embed

        for blk in self.blocks:
            x = blk(x)
        
        return x
    
    def forward(self, x, input_res=None):
        x = self.forward_features(x, input_res)
        return x


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch16_w_cls(**kwargs):
    model = VisionTransformer_w_cls(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# def vit_large_patch16(**kwargs):
#     model = VisionTransformer(
#         embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def vit_huge_patch14(**kwargs):
#     model = VisionTransformer(
#         embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


def get_SatMAE_model(ckpt_path, device, global_pool=False, expr_type="main"):

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if expr_type == "main":
        model = vit_base_patch16(global_pool=global_pool)
    elif expr_type == "baseline":
        model = vit_base_patch16_w_cls(global_pool=global_pool)

    state_dict = model.state_dict()
    checkpoint_model = checkpoint['model']
    
    # for k in ['patch_embed.proj.weight', 'patch_embed.proj.bias']:
    #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #         print(f"Removing key {k} from pretrained checkpoint")
    #         del checkpoint_model[k]

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(set(msg.missing_keys))
    return model

class Projector(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Projector, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.linear1(x))
        x = self.linear2(x)
        return x

################################################################################################################################

class SatMAE_backbone(pl.LightningModule):
    def __init__(self, pretrained_model_path,device, feat_dim=768,fc_dim = 512,global_pool=False,expr_type="main"):
        super().__init__()
        self.backbone = get_SatMAE_model(ckpt_path=pretrained_model_path,device=device,global_pool=global_pool,expr_type=expr_type)
        self.projector = Projector(input_size=feat_dim,hidden_size=fc_dim)  
        
    def forward(self,x,zoom_level,sat_type="bingmap"):
        input_res = torch.tensor([1.0*z*image_gsd[sat_type] for z in zoom_level], dtype=x.dtype)
        x = self.backbone(x,input_res=input_res)
        sat_embeddings = self.projector(x)
        return sat_embeddings

