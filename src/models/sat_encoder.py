import pytorch_lightning as pl
import torch.nn as nn

from .meta_encoder import MetaFuser, MetaFuser_w_cls


class SatMetaEncoder_early(pl.LightningModule):
    def __init__(self, d_model=512,metadata_type="latlong_month_time_asource_tsource", meta_droprate=0.5):
        super(SatMetaEncoder_early, self).__init__()

        self.metadata_type = metadata_type
        if self.metadata_type != "none":
            #Metadata Fusing module:
            self.meta_fuser = MetaFuser(metadata_type=self.metadata_type,meta_droprate=meta_droprate,fc_dim=d_model, nhead=8, num_layers=3)


    def forward(self, sat_embeddings, 
                    audio_source=None, caption_source=None, latlong=None, time= None, month = None, 
                    time_valid=None, month_valid=None, eval_meta=None):
        
        token_embeddings = sat_embeddings

        if self.metadata_type == "none":
            return token_embeddings
        else:
            # Fuse Metadata
            metadata_fused_sat_embeddings = self.meta_fuser(sat_embeddings=token_embeddings,audio_source=audio_source,caption_source=caption_source,
                                                            latlong=latlong, month=month, time=time, time_valid=time_valid, month_valid=month_valid,eval_meta=eval_meta)
        
            return metadata_fused_sat_embeddings


class SatMetaEncoder(pl.LightningModule):
    def __init__(self, d_model=512,metadata_type="latlong_month_time_asource_tsource", meta_droprate=0.5):
        super(SatMetaEncoder, self).__init__()

        self.metadata_type = metadata_type
        self.projector = nn.Linear(1024,d_model)
        #Metadata Fusing module:
        if self.metadata_type != "none":
            self.meta_fuser = MetaFuser_w_cls(metadata_type=self.metadata_type,meta_droprate=meta_droprate,fc_dim=d_model, nhead=8, num_layers=3)

       
    def forward(self, sat_embeddings, 
                    audio_source=None, caption_source=None, latlong=None, time= None, month = None, 
                    time_valid=None, month_valid=None, eval_meta=None):
        
        token_embeddings = self.projector(sat_embeddings) ## Assume shape: (B, d)
        
        if self.metadata_type == "none":
            out_dict = {'sat_feats':token_embeddings}

        else:
            token_embeddings = self.meta_fuser(sat_embeddings=token_embeddings, audio_source=audio_source,caption_source=caption_source,latlong=latlong,
                                                month=month, time=time,month_valid=month_valid,time_valid=time_valid,eval_meta=eval_meta)
            out_dict = {'sat_feats':token_embeddings}
        
        
        return out_dict


# Example usage
