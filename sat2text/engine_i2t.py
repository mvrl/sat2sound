#This experiment is for only image to text training on satellite image and LlaVA caption
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel

from src.loss import compute_loss
from src.metrics import get_retrieval_metrics, recall_key
from src.models.FDT.fdt_model import FDT
from src.models.sat_encoder import SatMAE_backbone

from .dataloader_i2t import Dataset_soundscape

def l2normalize(batch_embeddings):
    return batch_embeddings / (batch_embeddings.norm(p=2, dim=-1, keepdim=True) + 1e-8)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_tokenizer = None
_text_encoder = None


def _get_text_encoder():
    global _tokenizer, _text_encoder
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        _text_encoder = T5EncoderModel.from_pretrained("google/flan-t5-large").to(device)
    return _tokenizer, _text_encoder

def encode_text(prompt):
    tokenizer, text_encoder = _get_text_encoder()
    device = text_encoder.device
    batch = tokenizer(
        prompt, max_length=tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
    )
    input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

    with torch.no_grad():
        encoder_hidden_states = text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]
   
    boolean_encoder_mask = (attention_mask == 1).to(device)
    return encoder_hidden_states, boolean_encoder_mask

def get_text_embeds(prompts):
    prompt_embeds, boolean_prompt_mask = encode_text(prompts)
    return prompt_embeds, boolean_prompt_mask


class sat2textModel(pl.LightningModule):
    def __init__(self, hparams):

        #save parameters
        super().__init__()
        #save initialized hyperparameters
        self.save_hyperparameters(hparams)
        #set path attributes
        self.valid_end_list =[]
        satmae_ckpt_path = self.hparams.satmae_ckpt_path
        self.satmae_backbone = SatMAE_backbone(satmae_ckpt_path,device=self.device, fc_dim =self.hparams.fc_dim, global_pool=False)
        
        raw_audio_ft_dim = 768
        raw_txt_ft_dim=1024
       
        self.fdt = FDT(sd_num=self.hparams.codebook_size, sd_dim=self.hparams.codebook_dim, raw_img_ft_dim=self.hparams.fc_dim, raw_audio_ft_dim=raw_audio_ft_dim, raw_txt_ft_dim=raw_txt_ft_dim, text_qmap=True, shared_codebook=False).to(self.device)

        self.logit_scale_fdt = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.automatic_optimization = False

    def prepare_batch(self,batch):
        #For some reason input is not automatically casted into the cuda device, so this hack for now.
        ignore_keys = ['key','llava_caption','sat_zoom_level']
        for k in batch.keys():
            if k not in  ignore_keys:
                item = batch[k]
                if isinstance(item, dict):
                    for i in item.keys():
                        batch[k][i] = batch[k][i].to(self.device)
                else:
                    if item is not None:
                        batch[k] = batch[k].to(self.device)

        #llava_caption_input
        llava_caption_patch_embeds, llava_caption_boolean_mask = get_text_embeds(batch['llava_caption'])
        batch['llava_caption_input'] = {'patch_embeds':llava_caption_patch_embeds,'boolean_mask':llava_caption_boolean_mask}

        return batch


    def get_embeds(self,batch):

        sat_token_embeddings = self.satmae_backbone(batch['sat'],zoom_level=batch['sat_zoom_level'], sat_type=self.hparams.sat_type)
            
        _, sd_img_ft, _ = self.fdt.extract_img_sd_ft(sat_token_embeddings,return_token_att=False)

        text_patch_embeds, text_boolean_mask = batch['llava_caption_input']['patch_embeds'], batch['llava_caption_input']['boolean_mask']
        pad_mask = torch.where(text_boolean_mask==1, torch.tensor(0.0).to(self.device), torch.tensor(float('-inf')).to(self.device))
        _, sd_txt_ft, _ = self.fdt.extract_txt_sd_ft(text_patch_embeds,pad_mask=pad_mask,return_token_att=False)

                 
        return {
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
       #Loss= L(i,t)  
        logit_scale_fdt = self.logit_scale_fdt.exp()
        logits_per_fdt = torch.matmul(l2normalize(embeds['fdt_sat_embeds']),l2normalize(embeds['fdt_txt_embeds']).t())*logit_scale_fdt #(B,1024), (B,1024)
        loss_fdt = compute_loss(similarity=logits_per_fdt,pseudo_match_alpha=self.hparams.pseudo_match_alpha)
       
        loss_dict = {'loss':loss_fdt}
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
        return outputs['loss_dict']['loss']
        
    def validation_step(self, batch, batch_idx):
        batch = self.prepare_batch(batch)
        outputs = self.shared_step(batch,train=False)
        val_loss = outputs['loss_dict']
        self.log('val_loss', val_loss['loss'].detach(), sync_dist=True, batch_size=self.hparams.batch_size, prog_bar=True)
        
        
        outputs['embeds']['fdt_sat_embeds'] = outputs['embeds']['fdt_sat_embeds'].detach().cpu().to(torch.float32)
        outputs['embeds']['fdt_txt_embeds'] = outputs['embeds']['fdt_txt_embeds'].detach().cpu().to(torch.float32)

        self.valid_end_list.append(outputs)
        return outputs

    #compute retrieval metrics for a random batch of validation 
    def on_validation_epoch_end(self):
        outputs = self.valid_end_list
        sat_embeddings = []
        text_embeddings = []
        
        for i in range(len(outputs)):
            sat_embeddings.append(outputs[i]['embeds']['fdt_sat_embeds'])
            text_embeddings.append(outputs[i]['embeds']['fdt_txt_embeds'])
            
        sat_embeddings = l2normalize(torch.cat(sat_embeddings,axis=0))
        text_embeddings = l2normalize(torch.cat(text_embeddings,axis=0))
        
        R_k = self.hparams.recall_at/100*sat_embeddings.shape[0] # Validation with Recall@
        
        retrieval_results_I2T = get_retrieval_metrics(modality1_emb=sat_embeddings, modality2_emb=text_embeddings, normalized=True,k=R_k)
        retrieval_results_T2I = get_retrieval_metrics(modality1_emb=text_embeddings, modality2_emb=sat_embeddings, normalized=True,k=R_k)
        
        self.log(f'I2T_Recall', retrieval_results_I2T[recall_key(R_k)])
        self.log(f'I2T_Median_Rank', retrieval_results_I2T['Median Rank'])
        
        self.log(f'T2I_Recall', retrieval_results_T2I[recall_key(R_k)])
        self.log(f'T2I_Median_Rank', retrieval_results_T2I['Median Rank'])
        self.valid_end_list = []
        return retrieval_results_I2T, retrieval_results_T2I
       

    def train_dataloader(self):
        dset = Dataset_soundscape(
                                split="train",
                                sat_input_size=self.hparams.sat_input_size,
                                test_zoom_level=None,
                                dataset_type=self.hparams.dataset_type)
        loader = torch.utils.data.DataLoader(dset,batch_size=self.hparams.batch_size,
                    shuffle=True, pin_memory=False, persistent_workers=False,num_workers=self.hparams.num_workers,
                    )
        return loader

    def val_dataloader(self):
        dset = Dataset_soundscape(
                                split="val",
                                sat_input_size=self.hparams.sat_input_size,
                                test_zoom_level=None,
                                dataset_type=self.hparams.dataset_type)
        loader = torch.utils.data.DataLoader(dset,batch_size=self.hparams.batch_size,
                    shuffle=False, pin_memory=False, persistent_workers=False,num_workers=self.hparams.num_workers,
                    )
        return loader

    # def on_before_optimizer_step(self, optimizer):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)

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