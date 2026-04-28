import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

AUDIO_SOURCES = 4
CAPTION_SOURCES = 3 #Qwen, Pengi, metadata

meta_dim = {'latlong':4, 'month':2, 'time':2}

class source_encoding(nn.Module):
    def __init__(self,sources=4,fc_dim=512):
        super().__init__()
        self.source_embed = nn.Embedding(sources,fc_dim)

    def forward(self,x):
        x = self.source_embed(x)
        return x

class metaNet(nn.Module):
    def __init__(self,metadata_type='latlong',fc_dim = 512):
        super().__init__()
        self.fc1 = nn.Linear(meta_dim[metadata_type], 64)
        self.fc2 = nn.Linear(64,fc_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        output = self.fc2(x)
        return output

def create_mask(original_mask, dropout_rate):
    # Convert the original mask tensor to boolean type
    original_mask = original_mask.bool()

    # Calculate the number of items to be masked out
    num_items_to_mask = int(dropout_rate * original_mask.size(0))

    # Count the number of already masked items
    num_items_already_mask = original_mask.size(0) - torch.sum(original_mask)

    if num_items_already_mask >= num_items_to_mask:
        return original_mask
    else:
        # Get indices of already masked items
        already_masked_indices = torch.nonzero(~original_mask).squeeze(dim=1)

        # Get indices of items open to be masked
        open_to_be_masked_indices = torch.nonzero(original_mask).squeeze(dim=1).tolist()

        # Calculate the number of additional items to be masked
        num_items_to_mask -= len(already_masked_indices)

        # Shuffle open indices with torch.randperm so that dropout respects
        # the per-worker torch seed (set by DataLoader worker_init_fn / PL seeding).
        perm = torch.randperm(len(open_to_be_masked_indices))
        shuffled = [open_to_be_masked_indices[i] for i in perm.tolist()]
        indices_to_mask = shuffled[:num_items_to_mask]

        # Update the mask tensor by setting the selected indices to False
        updated_mask = original_mask.clone()  # Create a copy of the original mask
        updated_mask[indices_to_mask] = False  # Set the selected indices to False

        return updated_mask

def update_meta_mask(meta_mask,eval_meta): #used only for ablation to see which metadata component contributes the most
    #meta_mask_shape is (B, M)
    meta_order = {'asource':0,'tsource':1,"latlong":2,"month":3,"time":4}
    if 'asource' not in eval_meta:
        meta_mask[:,meta_order['asource']] = 0
    if 'tsource' not in eval_meta:
        meta_mask[:,meta_order['tsource']] = 0
    if 'latlong' not in eval_meta:
        meta_mask[:,meta_order['latlong']] = 0
    if 'month' not in eval_meta:
        meta_mask[:,meta_order['month']] = 0
    if 'time' not in eval_meta:
        meta_mask[:,meta_order['time']] = 0
    return meta_mask

class metadata_projector(pl.LightningModule):
    def __init__(self,metadata_type,meta_droprate=0.5, fc_dim=512):
        super().__init__()
        self.metadata_type = metadata_type
        self.meta_droprate = meta_droprate
        if 'asource' in self.metadata_type:
            self.audio_source_fc = source_encoding(sources=AUDIO_SOURCES,fc_dim=fc_dim)
        if 'tsource' in self.metadata_type:
            self.caption_source_fc = source_encoding(sources=CAPTION_SOURCES,fc_dim=fc_dim)
        if 'latlong' in self.metadata_type:
            self.location_fc = metaNet(metadata_type="latlong",fc_dim = fc_dim)
        if 'month' in self.metadata_type:
            self.month_fc = metaNet(metadata_type="month",fc_dim = fc_dim)
        if 'time' in self.metadata_type:
            self.time_fc = metaNet(metadata_type="time",fc_dim = fc_dim)

    def forward(self,audio_source=None,caption_source=None,latlong=None,month=None, time=None,month_valid=None,time_valid=None,eval_meta=None):
        meta_embeddings = []
        meta_masks = []
        if 'asource' in self.metadata_type:
            audio_source_valid = torch.ones(audio_source.shape[0]).bool() #all valid
            audio_source_embeddings = self.audio_source_fc(audio_source)
            audio_source_mask = create_mask(original_mask=audio_source_valid, dropout_rate=self.meta_droprate).to(audio_source_embeddings.device)
            audio_source_embeddings = audio_source_embeddings*audio_source_mask.unsqueeze(1)
            meta_embeddings.append(audio_source_embeddings.unsqueeze(1))
            meta_masks.append(audio_source_mask.unsqueeze(1))
        
        if 'tsource' in self.metadata_type:
            caption_source_valid = torch.ones(caption_source.shape[0]).bool() #all valid
            caption_source_embeddings = self.caption_source_fc(caption_source)
            caption_source_mask = create_mask(original_mask=caption_source_valid, dropout_rate=self.meta_droprate).to(caption_source_embeddings.device)
            caption_source_embeddings = caption_source_embeddings*caption_source_mask.unsqueeze(1)
            meta_embeddings.append(caption_source_embeddings.unsqueeze(1))
            meta_masks.append(caption_source_mask.unsqueeze(1))

        if 'latlong' in self.metadata_type:
            location_valid = torch.ones(latlong.shape[0]).bool() #all valid
            location_embeddings = self.location_fc(latlong)
            location_mask = create_mask(original_mask=location_valid, dropout_rate=self.meta_droprate).to(location_embeddings.device)
            location_embeddings = location_embeddings*location_mask.unsqueeze(1)
            meta_embeddings.append(location_embeddings.unsqueeze(1))
            meta_masks.append(location_mask.unsqueeze(1))

        if 'month' in self.metadata_type:
            month_embeddings = self.month_fc(month)
            if month_valid is None:
                month_valid = torch.ones(month.shape[0]).bool() #all valid
            month_mask = create_mask(original_mask=month_valid, dropout_rate=self.meta_droprate).to(month_embeddings.device)
            month_embeddings = month_embeddings*month_mask.unsqueeze(1)
            meta_embeddings.append(month_embeddings.unsqueeze(1))
            meta_masks.append(month_mask.unsqueeze(1))
            
        if 'time' in self.metadata_type:
            time_embeddings = self.time_fc(time)
            if time_valid is None:
                time_valid = torch.ones(time.shape[0]).bool() #all valid
            time_mask = create_mask(original_mask=time_valid, dropout_rate=self.meta_droprate).to(time_embeddings.device)
            time_embeddings = time_embeddings*time_mask.unsqueeze(1)
            meta_embeddings.append(time_embeddings.unsqueeze(1))
            meta_masks.append(time_mask.unsqueeze(1))
        
        meta_embeddings = torch.cat(meta_embeddings,dim=1)
        meta_masks = torch.cat(meta_masks,dim=1).to(meta_embeddings.device)
        if eval_meta is not None:
            meta_masks = update_meta_mask(meta_masks,eval_meta)
        return meta_embeddings, meta_masks #(B,M,fc_dim), (B, M) (B=batch_size, M=types of metadata)




class MetaFuser_w_cls(pl.LightningModule):
    def __init__(self, metadata_type="latlong_month_time_asource_tsource",meta_droprate=0.5,fc_dim=512, nhead=8, num_layers=3):
        super(MetaFuser_w_cls, self).__init__()
        self.transformer_encoder = TransformerEncoder(TransformerEncoderLayer(d_model=fc_dim, nhead=nhead,batch_first=True,activation="gelu"), num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, fc_dim))  # CLS token parameter
        self.meta_embed = metadata_projector(metadata_type=metadata_type,meta_droprate=meta_droprate, fc_dim=fc_dim)
        
    def forward(self, sat_embeddings, audio_source=None,caption_source=None,latlong=None,month=None, time=None,month_valid=None,time_valid=None,eval_meta=None):
        meta_embeddings, meta_masks = self.meta_embed(audio_source,caption_source,latlong,month, time,month_valid,time_valid,eval_meta)
        sat_embeddings = sat_embeddings.unsqueeze(1)
        
        combined_embeddings = torch.cat([sat_embeddings,meta_embeddings],dim=1)

        cls_token = self.cls_token.expand(combined_embeddings.size(0), -1, -1)
  
        combined_embeddings = torch.cat([cls_token,combined_embeddings], dim=1)


        meta_masks = torch.cat([torch.ones(meta_masks.size(0), 2).bool().to(meta_masks.device), meta_masks], dim=1)

        output = self.transformer_encoder(combined_embeddings, src_key_padding_mask=~meta_masks)
        

        cls_output = output[:, 0, :]  # Output corresponding to the CLS token

        return cls_output



class MetaFuser(pl.LightningModule):
    def __init__(self, metadata_type="latlong_month_time_asource_tsource",meta_droprate=0.5,fc_dim=512, nhead=8, num_layers=3):
        super(MetaFuser, self).__init__()
        self.transformer_encoder = TransformerEncoder(TransformerEncoderLayer(d_model=fc_dim, nhead=nhead,batch_first=True,activation="gelu"), num_layers=num_layers)
        self.meta_embed = metadata_projector(metadata_type=metadata_type,meta_droprate=meta_droprate, fc_dim=fc_dim)
        
    def forward(self, sat_embeddings, audio_source=None,caption_source=None,latlong=None,month=None, time=None,month_valid=None,time_valid=None,eval_meta=None):
        meta_embeddings, meta_masks = self.meta_embed(audio_source,caption_source,latlong,month, time,month_valid,time_valid,eval_meta)
        combined_embeddings = torch.cat([sat_embeddings,meta_embeddings],dim=1)
        combined_masks = torch.cat([torch.ones(sat_embeddings.size(0), sat_embeddings.size(1)).bool().to(sat_embeddings.device), meta_masks], dim=1)
        output = self.transformer_encoder(combined_embeddings, src_key_padding_mask=~combined_masks)
        return output
