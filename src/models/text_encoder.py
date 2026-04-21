import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import ClapTextModelWithProjection, T5EncoderModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flant5text_encoder = T5EncoderModel.from_pretrained("google/flan-t5-large").to(device)

def encode_text(text_input):
    with torch.no_grad():
        encoder_hidden_states = flant5text_encoder(input_ids=text_input['input_ids'].to(device), attention_mask=text_input['attention_mask'].to(device))[0]
   
    boolean_encoder_mask = (text_input['attention_mask'] == 1).to(device)
    return encoder_hidden_states, boolean_encoder_mask

def get_flant5_embeds(text_input,precision="full"):
    prompt_embeds, boolean_prompt_mask = encode_text(text_input)
    if precision == "full":
        return prompt_embeds.to(device), boolean_prompt_mask
    else:
        return prompt_embeds.to(torch.float16).to(device), boolean_prompt_mask


class CLAP_textmodel_withProjection(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-fused")

    def forward(self,text_input,embed_type="pooled"):
        if embed_type == "pooled":    
            batch_embeddings_text = self.model(**text_input)['text_embeds']
            return batch_embeddings_text
        elif embed_type == "hidden_states":
            raise NotImplementedError(f"embed_type:{self.embed_type} Not implemented for CLAP_textmodel_withProjection")


class TextEncoder(pl.LightningModule):
    def __init__(self,shared_codebook=True, input_dim=1024, out_dim=512, num_heads=8, num_proj_layers=3,text_encoder_type="flant5", precision="full",expr_type="main"):
        super(TextEncoder, self).__init__()
        self.shared_codebook = shared_codebook
        self.precision = precision
        self.text_encoder_type = text_encoder_type
        self.expr_type = expr_type
        if self.text_encoder_type == "clap":
            self.backbone = CLAP_textmodel_withProjection()
        
        elif self.text_encoder_type == "flant5":
            if shared_codebook == False: #if codebook is shared aggregation happens through codebook itself. No need of special aggregator
                self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
                self.embedding = nn.Linear(input_dim, input_dim)
                self.transformer_encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(input_dim, num_heads,batch_first=True),
                    num_proj_layers
                )
                self.fc = nn.Linear(input_dim, out_dim)  # Final projection
        
        if self.expr_type == "baseline" and text_encoder_type=="flant5":
            # self.pooler = pooler_w_cls(fc_dim=input_dim)
            self.fc = nn.Linear(input_dim, out_dim)
        elif self.expr_type == "baseline" and text_encoder_type=="clap":
            self.fc = nn.Linear(input_dim, out_dim)
        

    def forward(self, text_input, embed_type="hidden_states"):
        if embed_type == "hidden_states": #We want token level embeddings to be used for FDT training
            if self.text_encoder_type == "clap":
                prompt_embeds, boolean_prompt_mask = self.backbone(text_input=text_input,embed_type="hidden_states")
            
            elif self.text_encoder_type == "flant5":
                prompt_embeds, boolean_prompt_mask = text_input['patch_embeds'],text_input['boolean_mask']
            return prompt_embeds, boolean_prompt_mask

        elif embed_type == "pooled":  #We want global embeddings to be used for regular geoclap like training
            if self.text_encoder_type == "flant5" and self.expr_type == "baseline":
                # prompt_embeds, boolean_prompt_mask = text_input['patch_embeds'],text_input['boolean_mask']
                # embed = self.fc(self.pooler(prompt_embeds,boolean_prompt_mask))
                prompt_embeds = text_input['patch_embeds']
                embed = self.fc(prompt_embeds)
                return embed
            else:
                embed = self.backbone(text_input,embed_type="pooled")
                return self.fc(embed)
