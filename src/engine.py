# This is the main training engine of the Sat2Sound framework.

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.config import cfg
from src.dataloader import Dataset_soundscape, collate_batch
from src.loss import compute_loss
from src.metrics import get_retrieval_metrics, recall_key
from src.models.FDT.fdt_model import FDT
from src.models.audio_encoder import MGACLAP_audiomodel
from src.models.sat_encoder import SatMAE_backbone, SatMetaEncoder_early
from src.models.text_encoder import TextEncoder, get_flant5_embeds


def l2normalize(batch_embeddings):
    return batch_embeddings / (batch_embeddings.norm(p=2, dim=-1, keepdim=True) + 1e-8)

def prepare_flant5_text_embeds(text_input, device):
    text_patch_embeds, text_boolean_mask = get_flant5_embeds(text_input)
    text_patch_embeds = text_patch_embeds.to(device)
    text_boolean_mask = text_boolean_mask.to(device)
    return text_patch_embeds, text_boolean_mask

class sat2soundModel(pl.LightningModule):
    def __init__(self, hparams):

        #save parameters
        super().__init__()
        #save initialized hyperparameters
        self.save_hyperparameters(hparams)
        #set path attributes
        self.valid_end_list =[]
        satmae_ckpt_path = self.hparams.satmae_ckpt_path
        self.satmae_backbone = SatMAE_backbone(satmae_ckpt_path,device=self.device, fc_dim =self.hparams.fc_dim, global_pool=False)

        self.sat_encoder = SatMetaEncoder_early(d_model=self.hparams.fc_dim, metadata_type=self.hparams.metadata_type, meta_droprate=self.hparams.meta_droprate)
        self.audio_encoder = MGACLAP_audiomodel(yaml_path=cfg.mgaclap_yml_path, d_model=self.hparams.fc_dim, embed_type="hidden_states", device=self.device)
        raw_audio_ft_dim = 768
        raw_txt_ft_dim = 1024

        self.text_encoder = TextEncoder(input_dim=raw_txt_ft_dim, out_dim=self.hparams.fc_dim, num_heads=8, num_proj_layers=3, text_encoder_type="flant5")

        # Shared-codebook FDT with text query-map. Keep kwargs explicit to match ckpt submodule names.
        self.fdt = FDT(sd_num=self.hparams.codebook_size, sd_dim=self.hparams.codebook_dim, raw_img_ft_dim=self.hparams.fc_dim, raw_audio_ft_dim=raw_audio_ft_dim, raw_txt_ft_dim=raw_txt_ft_dim, text_qmap=True, shared_codebook=True).to(self.device)

        self.logit_scale_at = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_it = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_ia = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_fdt = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Preserved for checkpoint compatibility: old runs with combined_modality_loss=1
        # persist both logit scales even though only logit_scale_i_at is read by the loss.
        if self.hparams.combined_modality_loss:
            self.logit_scale_i_at = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.logit_scale_it_a = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.automatic_optimization = False

    def prepare_batch(self,batch):
        #For some reason input is not automatically casted into the cuda device, so this hack for now.
        ignore_keys = ['key','audio_caption','llava_caption','sat_zoom_level']
        for k in batch.keys():
            if k not in  ignore_keys:
                item = batch[k]
                if isinstance(item, dict):
                    for i in item.keys():
                        batch[k][i] = batch[k][i].to(self.device)
                else:
                    if item is not None:
                        batch[k] = batch[k].to(self.device)

        audio_caption_patch_embeds, audio_caption_boolean_mask = prepare_flant5_text_embeds(text_input=batch['audio_caption_input'], device=self.device)
        batch['audio_caption_input'] = {'patch_embeds':audio_caption_patch_embeds,'boolean_mask':audio_caption_boolean_mask}
        llava_caption_patch_embeds, llava_caption_boolean_mask = prepare_flant5_text_embeds(text_input=batch['llava_caption_input'], device=self.device)
        batch['llava_caption_input'] = {'patch_embeds':llava_caption_patch_embeds,'boolean_mask':llava_caption_boolean_mask}

        return batch


    def get_embeds(self,batch):

        sat_token_embeddings = self.satmae_backbone(batch['sat'],zoom_level=batch['sat_zoom_level'], sat_type=self.hparams.sat_type)

        # Early metadata fusion: fuse metadata into sat tokens before FDT.
        if self.hparams.metadata_type != 'none':
            sat_token_embeddings_before_fdt = self.sat_encoder(sat_embeddings=sat_token_embeddings, audio_source=batch['audio_source'], caption_source=batch['caption_source'],
                                                        latlong=batch['latlong'], time=batch['time'], month=batch['month'],
                                                        time_valid=batch['time_valid'], month_valid=batch['month_valid'])
        else:
            sat_token_embeddings_before_fdt = sat_token_embeddings

        _, sd_img_ft, _ = self.fdt.extract_img_sd_ft(sat_token_embeddings_before_fdt, return_token_att=False)
        sat_embeds_dict = {'ctotal': sd_img_ft}

        #encode audio
        audio_token_embeds = self.audio_encoder(batch['audio'])
        _, sd_audio_ft, _ = self.fdt.extract_audio_sd_ft(audio_token_embeds, return_token_att=False)

        #encode audio caption
        audio_text_patch_embeds, audio_text_boolean_mask = self.text_encoder(batch['audio_caption_input'], embed_type="hidden_states")
        pad_mask = torch.where(audio_text_boolean_mask==1, torch.tensor(0.0).to(self.device), torch.tensor(float('-inf')).to(self.device))
        _, sd_audiotxt_ft, _ = self.fdt.extract_txt_sd_ft(audio_text_patch_embeds, pad_mask=pad_mask, return_token_att=False)

        #encode image caption
        text_patch_embeds, text_boolean_mask = self.text_encoder(batch['llava_caption_input'], embed_type="hidden_states")
        pad_mask = torch.where(text_boolean_mask==1, torch.tensor(0.0).to(self.device), torch.tensor(float('-inf')).to(self.device))
        _, sd_txt_ft, _ = self.fdt.extract_txt_sd_ft(text_patch_embeds, pad_mask=pad_mask, return_token_att=False)

        return {'sat_embeds_dict':sat_embeds_dict,
                'audio_embeds': sd_audio_ft,
                'audio_caption_embeds':sd_audiotxt_ft,
                'fdt_sat_embeds':sd_img_ft,
                'fdt_txt_embeds':sd_txt_ft,
                }

    def forward(self, batch):
        embeds = self.get_embeds(batch)
        return embeds

    def shared_step(self, batch,train=True):
        embeds = self(batch)
        if train:
            loss_dict = self.get_loss(embeds)
            outputs = {'loss_dict':loss_dict}
            return outputs
        else:
            loss_dict = self.get_loss(embeds)
            outputs = {'embeds':embeds,'loss_dict':loss_dict}
            return outputs

    def get_loss(self,embeds):
        # Pairwise contrastive losses (image↔caption, audio↔caption, image↔audio)
        #   loss = (1/3)·(L_ia + L_it + L_at) + L_comb + fdt_weight·L_fdt
        # where L_comb = L(i, a+c) if combined_modality_loss=1 else 0.

        logit_scale_it = self.logit_scale_it.exp()
        logits_per_it = torch.matmul(l2normalize(embeds['sat_embeds_dict']['ctotal']),l2normalize(embeds['audio_caption_embeds']).t())*logit_scale_it
        loss_it = compute_loss(similarity=logits_per_it,pseudo_match_alpha=self.hparams.pseudo_match_alpha)

        logit_scale_at = self.logit_scale_at.exp()
        logits_per_at = torch.matmul(l2normalize(embeds['audio_embeds']),l2normalize(embeds['audio_caption_embeds']).t())*logit_scale_at
        loss_at = compute_loss(similarity=logits_per_at,pseudo_match_alpha=self.hparams.pseudo_match_alpha)

        logit_scale_ia = self.logit_scale_ia.exp()
        logits_per_ia = torch.matmul(l2normalize(embeds['sat_embeds_dict']['ctotal']),l2normalize(embeds['audio_embeds']).t())*logit_scale_ia
        loss_ia = compute_loss(similarity=logits_per_ia,pseudo_match_alpha=self.hparams.pseudo_match_alpha)

        combined_modality_loss = 0.0
        if self.hparams.combined_modality_loss:
            # L(i, a+c): image vs. (audio + audio-caption) embedding sum
            logit_scale_i_at = self.logit_scale_i_at.exp()
            logits_per_i_at = torch.matmul(l2normalize(embeds['sat_embeds_dict']['ctotal']),l2normalize(embeds['audio_embeds']+embeds['audio_caption_embeds']).t())*logit_scale_i_at
            combined_modality_loss = compute_loss(similarity=logits_per_i_at,pseudo_match_alpha=self.hparams.pseudo_match_alpha)

        multimodal_loss = (loss_ia + loss_it + loss_at)/3 + combined_modality_loss

        if self.hparams.fdt_weight != 0.0:
            logit_scale_fdt = self.logit_scale_fdt.exp()
            logits_per_fdt = torch.matmul(l2normalize(embeds['fdt_sat_embeds']),l2normalize(embeds['fdt_txt_embeds']).t())*logit_scale_fdt
            loss_fdt = compute_loss(similarity=logits_per_fdt,pseudo_match_alpha=self.hparams.pseudo_match_alpha)
        else:
            loss_fdt = torch.tensor(0.0).to(self.device)

        loss = multimodal_loss + self.hparams.fdt_weight*loss_fdt
        loss_dict = {'loss':loss,'multimodal_loss':multimodal_loss, 'fdt_loss':loss_fdt}
        return loss_dict

    def training_step(self, batch):
        batch = self.prepare_batch(batch)

        optimizer = self.optimizers()
        optimizer.zero_grad()

        outputs = self.shared_step(batch, train=True)
        loss = outputs['loss_dict']['loss']
        self.manual_backward(loss)
        self.clip_gradients(optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        optimizer.step()

        self.scheduler.step()
        self.log('train_loss', outputs['loss_dict']['loss'], sync_dist=True, batch_size=self.hparams.batch_size, prog_bar=True)
        self.log('train_loss_multimodal', outputs['loss_dict']['multimodal_loss'], sync_dist=True, batch_size=self.hparams.batch_size, prog_bar=True)
        self.log('train_loss_fdt', outputs['loss_dict']['fdt_loss'], sync_dist=True, batch_size=self.hparams.batch_size, prog_bar=True)
        return outputs['loss_dict']['loss']

    def validation_step(self, batch, batch_idx):
        batch = self.prepare_batch(batch)
        outputs = self.shared_step(batch,train=False)
        val_loss = outputs['loss_dict']
        self.log('val_loss', val_loss['loss'].detach(), sync_dist=True, batch_size=self.hparams.batch_size, prog_bar=True)
        self.log('val_loss_multimodal', val_loss['multimodal_loss'].detach(), sync_dist=True, batch_size=self.hparams.batch_size, prog_bar=True)
        self.log('val_loss_fdt', val_loss['fdt_loss'].detach(), sync_dist=True, batch_size=self.hparams.batch_size, prog_bar=True)

        for k in outputs['embeds']['sat_embeds_dict'].keys():
            outputs['embeds']['sat_embeds_dict'][k] = outputs['embeds']['sat_embeds_dict'][k].detach().cpu().to(torch.float32)

        outputs['embeds']['audio_embeds'] = outputs['embeds']['audio_embeds'].detach().cpu().to(torch.float32)
        outputs['embeds']['audio_caption_embeds'] = outputs['embeds']['audio_caption_embeds'].detach().cpu().to(torch.float32)
        outputs['embeds']['fdt_sat_embeds'] = outputs['embeds']['fdt_sat_embeds'].detach().cpu().to(torch.float32)
        outputs['embeds']['fdt_txt_embeds'] = outputs['embeds']['fdt_txt_embeds'].detach().cpu().to(torch.float32)

        self.valid_end_list.append(outputs)
        return outputs

    #compute retrieval metrics for a random batch of validation
    def on_validation_epoch_end(self):
        outputs = self.valid_end_list
        sat_embeddings = []
        audio_embeddings = []
        text_embeddings = []

        for i in range(len(outputs)):
            sat_embeddings.append(outputs[i]['embeds']['sat_embeds_dict']['ctotal'])
            audio_embeddings.append(outputs[i]['embeds']['audio_embeds'])
            text_embeddings.append(outputs[i]['embeds']['audio_caption_embeds'])

        sat_embeddings = l2normalize(torch.cat(sat_embeddings,axis=0))
        audio_embeddings = l2normalize(torch.cat(audio_embeddings,axis=0))
        text_embeddings = l2normalize(torch.cat(text_embeddings,axis=0))

        R_k = self.hparams.recall_at/100*sat_embeddings.shape[0] # Validation with Recall@

        retrieval_results_I2S = get_retrieval_metrics(modality1_emb=sat_embeddings, modality2_emb=audio_embeddings, normalized=True,k=R_k)
        retrieval_results_S2I = get_retrieval_metrics(modality1_emb=audio_embeddings, modality2_emb=sat_embeddings, normalized=True,k=R_k)

        self.log(f'I2S_Recall', retrieval_results_I2S[recall_key(R_k)])
        self.log(f'I2S_Median_Rank', retrieval_results_I2S['Median Rank'])

        self.log(f'S2I_Recall', retrieval_results_S2I[recall_key(R_k)])
        self.log(f'S2I_Median_Rank', retrieval_results_S2I['Median Rank'])


        #composed Image-to-Sound retrieval:
        retrieval_results_I2St = get_retrieval_metrics(modality1_emb=sat_embeddings, modality2_emb=l2normalize(audio_embeddings+text_embeddings), normalized=True,k=R_k)
        retrieval_results_St2I = get_retrieval_metrics(modality1_emb=l2normalize(audio_embeddings+text_embeddings), modality2_emb=sat_embeddings, normalized=True,k=R_k)

        self.log(f'I2St_Recall', retrieval_results_I2St[recall_key(R_k)])
        self.log(f'I2St_Median_Rank', retrieval_results_I2St['Median Rank'])

        self.log(f'St2I_Recall', retrieval_results_St2I[recall_key(R_k)])
        self.log(f'St2I_Median_Rank', retrieval_results_St2I['Median Rank'])
        self.valid_end_list = []
        return retrieval_results_I2S, retrieval_results_S2I


    def train_dataloader(self):
        dset = Dataset_soundscape(self.hparams, split="train")
        loader = torch.utils.data.DataLoader(dset,batch_size=self.hparams.batch_size,
                    shuffle=True, pin_memory=False, persistent_workers=False,num_workers=self.hparams.num_workers,
                    collate_fn=lambda batch:collate_batch(batch, metadata_type=self.hparams.metadata_type))
        return loader

    def val_dataloader(self):
        dset = Dataset_soundscape(self.hparams, split="val")
        loader = torch.utils.data.DataLoader(dset,batch_size=self.hparams.batch_size,
                    shuffle=False, pin_memory=False, persistent_workers=False,num_workers=self.hparams.num_workers,
                    collate_fn=lambda batch:collate_batch(batch, metadata_type=self.hparams.metadata_type))
        return loader

    def configure_optimizers(self):
        print(f'Initializing Learning rate {self.hparams.learning_rate}')

        params = self.parameters()
        self.optim = torch.optim.AdamW(params=params,
                    lr=self.hparams.learning_rate,
                    weight_decay=self.hparams.weight_decay,
                    betas=(0.9,0.98),
                    eps=1e-8
                    )

        self.warm_up_iterations = self.hparams.warm_up_iterations
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=self.optim,
            T_0=self.warm_up_iterations
        )

        return {'optimizer': self.optim,
        'lr_scheduler': {
            'name':'train/lr',
            'scheduler': self.scheduler,
            'interval': 'step',
            'frequency': 1
        }
        }
