#Adapted from: https://github.com/yuxiaochen1103/FDT/blob/main/prototype/model/clip_fdt.py
import math

import torch
from torch import nn

from .sparsemax import Sparsemax


#---- attention models for FDT
class Query_model(nn.Module):
    def __init__(self, ft_dim, sd_dim, temperature=1, att_func_type='softmax', pool_type='max', q_map=True):
        '''
        ft_dim: feature dim of image patch or text token
        sd_dim: dim of FDT
        temperature: temperature for softmax or sparsemax
        att_func_type: attention normlization function type
        pool_type: pooling type for attention weights
        '''

        super().__init__()

        #activation 
        assert att_func_type in ['softmax', 'sigmoid', 'sparsemax']
        self.att_func_type = att_func_type

        assert pool_type in ['mean', 'max', 'sum']
        self.pool_type = pool_type

        if self.att_func_type == 'softmax':
            self.att_activation = nn.Softmax(dim=-1)
        elif self.att_func_type == 'sparsemax':
            self.att_activation = Sparsemax(dim=-1)
        else:
            self.att_activation = nn.Sigmoid()

        self.att_dim = sd_dim
        self.temperature = temperature
        
        #map patch/text tokens to codebook (query) spaces
        #---note that we donot use mapping for FDT
        if q_map:
            self.q_map = nn.Sequential(
                nn.LayerNorm(ft_dim),
                nn.Linear(ft_dim, sd_dim),
                nn.GELU(),
                nn.LayerNorm(sd_dim),
                nn.Linear(sd_dim, sd_dim)
            )
        else:
            self.q_map = nn.Identity()

            # Optionally, add an assert to ensure dimensions match
            assert ft_dim == sd_dim, f"Input dimension ({ft_dim}) must match output dimension ({sd_dim}) for identity function"


    def forward(self, ft, sd, mask=None, return_token_att=False):


        '''
        Args:
            ft: [batch, token_num, ft_dim]
            sd: [FDT_num, sd_dim]
            mask: [batch, token_num]: mask for padded tokens.
            return_token_att: flag for returning attention weights before nomalization.
            used for visualizing FDT.
        Returns:

        '''

        #map image/text token to query space
        q = self.q_map(ft) #batch, token_num, dim

        k = sd #code_num, sd_dim
        k = k.unsqueeze(0) #[1, code_num, sd_dim]
        k = k.transpose(2, 1) #[1,sd_dim, sd_num]
        
        #-----calculate inner dot
        inner_dot = torch.matmul(q, k) #[batch, token_num, code_num]

        if return_token_att: #cosine sim
            token_att = inner_dot

        inner_dot = inner_dot / math.sqrt(self.att_dim) #scale dot norm

        if mask is not None: # mask paded tokens
            
            assert mask.shape == q.shape[:2]
            mask = (mask == 0) * 1 #0 --> 1, inf --> 0

            inner_dot = inner_dot * mask.unsqueeze(-1) #sigmod(-inf) = 0, softmax(-inf) = 0

            if return_token_att: #if has pad, return maksed
                token_att = inner_dot


        # temptural norm
        inner_dot = inner_dot / self.temperature #[batch, token_num, code_num]

        #pooling
        if self.pool_type == 'sum':
            inner_dot = inner_dot.sum(1) #mean poolings
        elif self.pool_type == 'mean':
            inner_dot = inner_dot.mean(1)
        else:
            inner_dot = inner_dot.max(1)[0]

        #----get attention weights
        att_weight = self.att_activation(inner_dot) #normaliztion

        #----calculate weighted sum of v
        #v = self.ln_v(ft) #map to v_space
        
        att_ft = att_weight @ sd  #[batch, dictory_size] * [dictory_size, dim]  ---> [batch, sd_num, dim]

        if self.att_func_type == 'sigmoid':
            att_ft = att_ft / att_weight.sum(dim=-1, keepdim=True)
        
        if return_token_att:
            return token_att, att_ft, sd
        return att_weight, att_ft, sd


class FDT(nn.Module):
    def __init__(self, sd_num=16384, sd_dim=1024, raw_img_ft_dim=512, raw_audio_ft_dim=768, raw_txt_ft_dim=1024, att_func_type='sparsemax', pool_type='max', sd_temperature=1000, text_qmap=True, shared_codebook=False):
        super().__init__()
        '''
        Args:
            sd_num: number of FDT
            sd_dim: dimension of FDT
            raw_img_ft_dim: dimension of patch features
            raw_txt_ft_dim: dimension of text token features
            att_func_type: attention function type
            pool_type: pooling type of FDT attention weights
            sd_temperature: temperature for FDT attention
        '''
        #learnable FDT
        self.space_dict = nn.Parameter(torch.randn(sd_num, sd_dim))

        #query mapping
        self.img_query_model = Query_model(ft_dim=raw_img_ft_dim, sd_dim=sd_dim, temperature=sd_temperature, att_func_type=att_func_type, pool_type=pool_type)
        self.txt_query_model = Query_model(ft_dim=raw_txt_ft_dim, sd_dim=sd_dim, temperature=sd_temperature, att_func_type=att_func_type, pool_type=pool_type, q_map=text_qmap)
        if shared_codebook:
            self.audio_query_model = Query_model(ft_dim=raw_audio_ft_dim, sd_dim=sd_dim, temperature=sd_temperature, att_func_type=att_func_type, pool_type=pool_type)


    def extract_img_sd_ft(self, images_patch_ft, return_token_att=False):

        att_weight, sd_img_ft, sd = self.img_query_model(images_patch_ft, self.space_dict, return_token_att=return_token_att)

        return att_weight, sd_img_ft, sd


    def extract_txt_sd_ft(self, texts_word_ft, pad_mask=None, return_token_att=False):

        att_weight , sd_txt_ft, sd = self.txt_query_model(texts_word_ft, self.space_dict, mask=pad_mask, return_token_att=return_token_att)

        return att_weight , sd_txt_ft, sd
    
    def extract_audio_sd_ft(self, audio_token_ft, pad_mask=None, return_token_att=False):

        att_weight , sd_audio_ft, sd = self.audio_query_model(audio_token_ft, self.space_dict, mask=pad_mask, return_token_att=return_token_att)

        return att_weight , sd_audio_ft, sd
