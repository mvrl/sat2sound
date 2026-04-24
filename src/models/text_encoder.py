import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import ClapTextModelWithProjection, T5EncoderModel

_flant5_encoder: "T5EncoderModel | None" = None


def _get_flant5_encoder() -> "T5EncoderModel":
    """Lazily load the Flan-T5 encoder on first call (avoids loading at import time)."""
    global _flant5_encoder
    if _flant5_encoder is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _flant5_encoder = T5EncoderModel.from_pretrained("google/flan-t5-large").to(device)
    return _flant5_encoder


def encode_text(text_input):
    encoder = _get_flant5_encoder()
    device = next(encoder.parameters()).device
    with torch.no_grad():
        encoder_hidden_states = encoder(
            input_ids=text_input['input_ids'].to(device),
            attention_mask=text_input['attention_mask'].to(device),
        )[0]
    boolean_encoder_mask = (text_input['attention_mask'] == 1).to(device)
    return encoder_hidden_states, boolean_encoder_mask

def get_flant5_embeds(text_input, precision="full"):
    prompt_embeds, boolean_prompt_mask = encode_text(text_input)
    if precision == "full":
        return prompt_embeds, boolean_prompt_mask
    else:
        return prompt_embeds.to(torch.float16), boolean_prompt_mask


class CLAP_textmodel_withProjection(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-fused")

    def forward(self, text_input, embed_type="pooled"):
        if embed_type == "pooled":
            batch_embeddings_text = self.model(**text_input)['text_embeds']
            return batch_embeddings_text
        elif embed_type == "hidden_states":
            raise NotImplementedError("embed_type='hidden_states' not implemented for CLAP_textmodel_withProjection")


class TextEncoder(pl.LightningModule):
    def __init__(self, shared_codebook=True, input_dim=1024, out_dim=512, num_heads=8, num_proj_layers=3, text_encoder_type="flant5", precision="full", expr_type="main"):
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
                    nn.TransformerEncoderLayer(input_dim, num_heads, batch_first=True),
                    num_proj_layers
                )
                self.fc = nn.Linear(input_dim, out_dim)
        
        if self.expr_type == "baseline" and text_encoder_type == "flant5":
            self.fc = nn.Linear(input_dim, out_dim)
        elif self.expr_type == "baseline" and text_encoder_type == "clap":
            self.fc = nn.Linear(input_dim, out_dim)
        

    def forward(self, text_input, embed_type="hidden_states"):
        if embed_type == "hidden_states": #We want token level embeddings to be used for FDT training
            if self.text_encoder_type == "clap":
                prompt_embeds, boolean_prompt_mask = self.backbone(text_input=text_input, embed_type="hidden_states")
            
            elif self.text_encoder_type == "flant5":
                prompt_embeds, boolean_prompt_mask = text_input['patch_embeds'], text_input['boolean_mask']
            return prompt_embeds, boolean_prompt_mask

        elif embed_type == "pooled":  #We want global embeddings to be used for regular geoclap like training
            if self.text_encoder_type == "flant5" and self.expr_type == "baseline":
                prompt_embeds = text_input['patch_embeds']  # (B, T, D)
                boolean_mask = text_input['boolean_mask']   # (B, T) bool
                # Masked mean-pool over the token dimension before projecting.
                mask_f = boolean_mask.unsqueeze(-1).float()  # (B, T, 1)
                pooled = (prompt_embeds * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-8)
                return self.fc(pooled)
            elif self.text_encoder_type == "flant5":
                raise NotImplementedError(
                    "Pooled Flan-T5 embeddings are only supported with expr_type='baseline'. "
                    "The main sat2sound pipeline uses embed_type='hidden_states'."
                )
            elif self.expr_type == "baseline":
                embed = self.backbone(text_input, embed_type="pooled")
                return self.fc(embed)
            else:
                raise NotImplementedError(
                    "Pooled CLAP embeddings are only supported with expr_type='baseline'."
                )
