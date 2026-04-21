import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torchaudio
import yaml
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer, ClapProcessor

from src.config import cfg
from src.models.MGACLAP.feature_extractor import AudioFeature
from utilities.audio_features import get_audio_feat_mgaclap
from utilities.utils import get_clean_date, sat_transform

audio_source_map = {'yfcc':0,'iNat':1, 'aporee':2,'freesound':3}
caption_source_map = {'meta':0,"qwen":1,"pengi":2}

sources = ['iNat', 'yfcc', 'aporee', 'freesound']
meta_columns = ['sample_id','date', 'latitude','longitude', 'description', 'tags', 
                'title', 'scientific_name', 'common_name', 'sound_format', 'text',
                'address', 'original_sampling_rate', 'bin_id']

clap_score_df = pd.read_csv(os.path.join(cfg.metafiles_path,"GeoSound/clap_score_geosound.csv"))
pengi_caption = pd.read_json(os.path.join(cfg.metafiles_path,"GeoSound/geosound_audio_caption_pengi.json"),lines=True)
qwen_caption = pd.read_json(os.path.join(cfg.metafiles_path,"GeoSound/geosound_audio_caption_qwen.json"),lines=True)
llava_caption_bingmap = pd.read_json(os.path.join(cfg.metafiles_path,"GeoSound/llava_caption_for_bingmap.json"),lines=True)
llava_caption_sentinel = pd.read_json(os.path.join(cfg.metafiles_path,"GeoSound/llava_caption_for_sentinel.json"),lines=True)
llava_caption_SoundingEarth = pd.read_json(os.path.join(cfg.metafiles_path,"SoundingEarth/SoundingEarth_llava_caption_for_googleEarth_zl_1.json"),lines=True)
llava_caption = {'sentinel':llava_caption_sentinel, "bingmap":llava_caption_bingmap, "googleEarth":llava_caption_SoundingEarth}

try:
    clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
    flant5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
except:
    clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused",token=True)
    flant5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large",token=True)

with open(cfg.mgaclap_yml_path, "r") as f:
    mgaclap_config = yaml.safe_load(f)

mgaclap_feature_extractor = AudioFeature(mgaclap_config['audio_args'])

def collate_batch(batch,caption_type="audio_image",audio_encoder_type="mgaclap",text_encoder_type="flant5", metadata_type="latlong_month_time_asource_tsource"):
    out_dict = {}
    keys = batch[0].keys()
    #['audio', 'sat', 'audio_caption', 'llava_caption', 'key', 'audio_source', 'caption_source', 'latlong', 'time', 'time_valid', 'month', 'month_valid']
    #['input_features', 'is_longer']
    #['input_ids','attention_mask']
    out_dict['key'] = [item['key'] for item in batch]
    out_dict['sat_zoom_level'] =[item['sat_zoom_level'] for item in batch]
    out_dict['sat'] = torch.cat([item['sat'].unsqueeze(0) for item in batch])
    if audio_encoder_type != "mgaclap":
        raise NotImplementedError(
            f"The public repo only supports audio_encoder_type='mgaclap'; got "
            f"{audio_encoder_type!r}."
        )
    if "waveform" in batch[0]['audio'].keys():
        out_dict['audio'] = {
            'input_features': mgaclap_feature_extractor(
                torch.cat([item['audio']['waveform'] for item in batch])
            )
        }
    else:
        out_dict['audio'] = {
            'input_features': torch.cat(
                [item['audio']['input_features'].unsqueeze(0) for item in batch]
            )
        }
    
    audio_captions = [item['audio_caption'] for item in batch]
    llava_captions = [item['llava_caption'] for item in batch]

    if 'audio' in caption_type:
        if text_encoder_type == "clap":
            audio_caption_input = clap_processor(text=audio_captions,return_tensors="pt",padding="max_length",truncation=True,max_length=128)
        elif text_encoder_type == "flant5":
            audio_caption_input = flant5_tokenizer(audio_captions, max_length=flant5_tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt")
        out_dict['audio_caption_input'] = audio_caption_input 
        out_dict['audio_caption'] = audio_captions
        
    if 'image' in caption_type:
        if text_encoder_type == "clap":
            llava_caption_input = clap_processor(text=llava_captions,return_tensors="pt",padding="max_length",truncation=True,max_length=128)
        elif text_encoder_type == "flant5":
            llava_caption_input = flant5_tokenizer(llava_captions, max_length=flant5_tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt")
        out_dict['llava_caption_input'] = llava_caption_input
        out_dict['llava_caption'] = llava_captions
    
    #Metadata
    if 'asource' not in metadata_type:
        out_dict['audio_source'] = None
    else:
        out_dict['audio_source'] =  torch.cat([item['audio_source'].unsqueeze(0) for item in batch])
    
    if 'tsource' not in metadata_type:
        out_dict["caption_source"] =  None
    else:
        out_dict['caption_source'] = torch.cat([item['caption_source'].unsqueeze(0) for item in batch])

    if 'latlong' not in metadata_type:
        out_dict['latlong'] = None
    else:
        out_dict['latlong'] = torch.cat([item['latlong'].unsqueeze(0) for item in batch])

    if 'time' not in metadata_type:
        out_dict['time'] = None
        out_dict['time_valid'] = None
    else:
        out_dict['time'] = torch.cat([item['time'].unsqueeze(0) for item in batch])
        out_dict['time_valid'] = torch.cat([item['time_valid'].unsqueeze(0) for item in batch])

    if 'month' not in metadata_type:
        out_dict['month'] = None
        out_dict['month_valid'] = None
    else:
        out_dict['month'] = torch.cat([item['month'].unsqueeze(0) for item in batch])
        out_dict['month_valid'] = torch.cat([item['month_valid'].unsqueeze(0) for item in batch])
    
    return out_dict


class Dataset_soundscape(Dataset):
    def __init__(self,    
                 args,                        
                 split="test",
                 test_zoom_level=None,
                 test_mel_index=None):
        
        self.args = args
        self.split = split
        self.ignore_ids = list(pd.read_csv(cfg.ignore_ids_geosound)['sample_id'])
        if self.args.dataset_type == "GeoSound":
            if self.split == "train": 
                self.meta_df = pd.read_csv(os.path.join(cfg.data_path,"metafiles/GeoSound/train_metadata.csv"))
            if self.split == "val": 
                self.meta_df = pd.read_csv(os.path.join(cfg.data_path,"metafiles/GeoSound/val_metadata.csv"))
            if self.split == "test": 
                self.meta_df = pd.read_csv(os.path.join(cfg.data_path,"metafiles/GeoSound/test_metadata.csv"))
                valid_ids = pd.read_csv(os.path.join(cfg.data_path,"metafiles/GeoSound/test_ids_geosound.csv"))
                self.meta_df = self.meta_df[self.meta_df['sample_id'].isin(list(valid_ids['sample_id']))]

            self.meta_df = self.meta_df[meta_columns]
            self.meta_df = self.meta_df[~self.meta_df['sample_id'].isin(self.ignore_ids)]
            
        elif self.args.dataset_type == "SoundingEarth":
            self.valid_ids  = list(pd.read_csv(cfg.valid_ids_SoundingEarth)['sample_id'])
            if self.split == "train": 
                self.meta_df = pd.read_csv(os.path.join(cfg.data_path,"metafiles/SoundingEarth/aporee_train_fairsplit_10km.csv"))
            if self.split == "val": 
                self.meta_df = pd.read_csv(os.path.join(cfg.data_path,"metafiles/SoundingEarth/aporee_val_fairsplit_10km.csv"))
            if self.split == "test": 
                self.meta_df = pd.read_csv(os.path.join(cfg.data_path,"metafiles/SoundingEarth/aporee_test_fairsplit_10km.csv"))
                valid_ids = list(pd.read_csv(os.path.join(cfg.data_path,"metafiles/SoundingEarth/test_ids_soundingEarth.csv"))['sample_id'])
                valid_ids = [i.replace("aporee-",'') for i in valid_ids]
                self.meta_df = self.meta_df[self.meta_df['sample_id'].isin(valid_ids)]

            self.meta_df = self.meta_df[self.meta_df['sample_id'].isin(self.valid_ids)]
        
      
        self.aporee_meta = pd.read_csv(os.path.join(cfg.data_path,"aporee/final_metadata_with_captions.csv"))
        if self.args.dataset_type == "SoundingEarth":
            self.args.sat_type = "googleEarth" # Experiment with SoundingEarth contains only googleEarth imagery.
        self.llava_caption = llava_caption[self.args.sat_type]
        if bool(self.args.precomputed_mel):
            self.mel_feats_path = os.path.join(cfg.mel_feats_path,self.args.audio_encoder_type)
        self.test_zoom_level = test_zoom_level
        self.test_mel_index = test_mel_index
      
    def __len__(self):
        return len(self.meta_df) 
    def __getitem__(self,idx):
        out_dict = {}
        if self.args.dataset_type == "GeoSound":
            zoom_level = random.choice([1,3,5])
        elif self.args.dataset_type == "SoundingEarth":
                zoom_level = 1

        ##For training
        if self.args.img_caption_zl == "all":
            llava_caption_zl = zoom_level
        else:
            llava_caption_zl = self.args.img_caption_zl

         ##For evaluating
        if self.test_zoom_level != None:
            zoom_level = self.test_zoom_level

            if self.args.img_caption_zl == "all":
                llava_caption_zl = zoom_level
            else:
                llava_caption_zl = self.args.img_caption_zl
        
        sample = dict(self.meta_df.iloc[idx])
        sample_id = sample['sample_id']
        if self.args.dataset_type == "GeoSound":
            source = sample_id.split("-")[0]
            key = sample_id.split("-")[1]
        else:
            source = "aporee"
            key = sample_id
        sound_format =  'mp3'
        
        if source == 'aporee':   
            soundname = self.aporee_meta[self.aporee_meta['long_key']==key].mp3name.item()
            audio_path = os.path.join(cfg.data_path,source,'raw_audio',str(key),soundname)
        else:
            if isinstance(key, str):
                soundname = key+"."+sound_format
            else:
                soundname = str(key)+"."+sound_format
            
            audio_path = os.path.join(cfg.data_path,source,'raw_audio',soundname)
        ################################################################################################################################################################
        #Prepare audio
        if self.args.audio_encoder_type != "mgaclap":
            raise NotImplementedError(
                f"The public repo only supports audio_encoder_type='mgaclap'; got "
                f"{self.args.audio_encoder_type!r}."
            )
        if not bool(self.args.precomputed_mel):
            # Extract mel features from raw audio on-the-fly.
            audio, original_sr = torchaudio.load(audio_path)
            audio = audio.mean(axis=0)
            mel = get_audio_feat_mgaclap(audio.unsqueeze(0), original_sr, nsamples=1)[0]
            audio_feat = {'input_features': mel}
        else:
            # Load a precomputed 5-segment stack produced by
            # data_prep.compute_mel_features_mgaclap and pick one at random.
            if self.args.dataset_type == "GeoSound":
                sample_id_mel = sample_id
            elif self.args.dataset_type == "SoundingEarth":
                sample_id_mel = "aporee-" + sample_id
            feat_path = os.path.join(self.mel_feats_path, source, f"{sample_id_mel}.pth")
            sel_index = random.choice([0, 1, 2, 3, 4])
            if self.test_mel_index is not None:
                sel_index = self.test_mel_index
            mel = torch.load(feat_path)[sel_index]
            audio_feat = {'input_features': mel}
        ################################################################################################################################################################
        out_dict['audio'] = audio_feat

        if self.args.sat_type == "sentinel":
            image_path = os.path.join(cfg.data_path,source,'images',"sentinel",str(key)+'.jpeg')
            llava_caption = self.llava_caption[self.llava_caption['sample_id']==sample_id]['captions'].item()['text'+str(llava_caption_zl)]
        if self.args.sat_type == "bingmap":
            image_path = os.path.join(cfg.data_path,source,'images',"bingmap",str(key)+'.jpeg')
            llava_caption = self.llava_caption[self.llava_caption['sample_id']==sample_id]['captions'].item()['text'+str(llava_caption_zl)]
        if self.args.dataset_type == "SoundingEarth":
            short_id = sample['key']
            image_path = os.path.join(cfg.data_path,source,'images',"googleEarth",str(short_id)+'.jpg')
            llava_caption = self.llava_caption[self.llava_caption['sample_id']==sample_id]['captions'].item()
            sample_id = "aporee-"+sample_id
        ################################################################################################################################################################
        #Prepare sat image
        image = Image.open(image_path)
        
        sat_tr = sat_transform(is_train=self.split=="train", input_size=self.args.sat_input_size,sat_type=self.args.sat_type,zoom_level=zoom_level)
        final_image = sat_tr(image)

        if self.args.precision == "half":
            out_dict['sat']= final_image.half()
        else:
            out_dict['sat']= final_image
        out_dict['sat_zoom_level'] =  zoom_level
        ################################################################################################################################################################
        #Prepare texts:
        caption_source = clap_score_df[clap_score_df["sample_id"]==sample_id]["best_caption"].item() #find which audio caption is best based on CLAP score.
        if caption_source == "pengi":
                caption = pengi_caption[pengi_caption["sample_id"]==sample_id]["pengi_caption"].item()
        elif caption_source == "qwen":
            caption = qwen_caption[qwen_caption["sample_id"]==sample_id]["qwen_caption"].item()
        else:
            if self.args.dataset_type == "GeoSound":
                caption = sample["text"]
            else:
                caption = sample['caption'].split("The location of the sound is")[0] + "."
        out_dict['audio_caption'] = caption
        out_dict['llava_caption'] = llava_caption
        ################################################################################################################################################################
        #Prepare metadata:
        out_dict['key'] = sample_id
        long = sample['longitude']
        lat = sample['latitude']
        latlong_encode = torch.tensor([np.sin(np.pi*lat/90), np.cos(np.pi*lat/90), np.sin(np.pi*long/180), np.cos(np.pi*long/180)]).float()
        
        if self.args.dataset_type == "GeoSound":
            date = get_clean_date(sample['date'])
            source = sample_id.split("-")[0]
        else:
            source = "aporee"
            date = get_clean_date(sample['date_recorded'])
        
        if source == "freesound": # No time information for freesound samples so..
            time_encode = torch.tensor([0., 0.]).float()
            time_valid = torch.tensor(False).long()
        else:
            if date is not None:
                time_encode = torch.tensor([np.sin(2*np.pi*date.hour/23), np.cos(2*np.pi*date.hour/23)]).float()
                time_valid = torch.tensor(True).long()
            else:
                time_encode = torch.tensor([0., 0.]).float()
                time_valid = torch.tensor(False).long()

        #month encoding
        if date is not None:
            month_encode = torch.tensor([np.sin(2*np.pi*date.month/12), np.cos(2*np.pi*date.month/12)]).float()
            month_valid = torch.tensor(True).long()
        else:
            month_encode = torch.tensor([0., 0.]).float()
            month_valid = torch.tensor(False).long()
        
        if 'asource' in self.args.metadata_type:
            out_dict['audio_source'] = torch.tensor(audio_source_map[source]).long()   
        if 'tsource' in self.args.metadata_type:
            out_dict["caption_source"] =  torch.tensor(caption_source_map[caption_source]).long()
        if 'latlong' in self.args.metadata_type:
            if self.args.precision == "half":
                out_dict['latlong'] = latlong_encode.half()
            else:
                out_dict['latlong'] = latlong_encode
        if 'time' in self.args.metadata_type:
            if self.args.precision == "half":
                out_dict['time'] = time_encode.half()
            else:
                out_dict['time'] = time_encode
            out_dict['time_valid'] = time_valid
        if 'month' in self.args.metadata_type:
            if self.args.precision == "half":
                out_dict['month'] = month_encode.half()
            else:
                out_dict['month'] = month_encode
            out_dict['month_valid'] = month_valid
        
        return out_dict


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description='')
    parser.add_argument('--dataset_type', type=str, default='GeoSound',choices=['GeoSound','SoundingEarth'])
    parser.add_argument('--sat_type', type=str, default='sentinel', choices=['bingmap','googleEarth'])
    parser.add_argument('--metadata_type', type=str, default='latlong_month_time_asource_tsource',choices=['none','latlong', 'month', 'time', 'asource','tsource',
                                                                                                           'latlong_month', 'latlong_time', 'latlong_month_time','latlong_month_time_asource', 'latlong_month_time_asource_tsource'])
    parser.add_argument('--text_encoder_type', type=str, default='flant5',choices=['clap', 'flant5'])
    parser.add_argument('--audio_encoder_type', type=str, default='mgaclap',choices=['mgaclap'])
    parser.add_argument('--precomputed_mel', type=int, default=0,choices=[0,1])
    args = parser.parse_args()
    args.sat_input_size = 224
    args.caption_type = "audio_image"
    args.precision = "full"
    args.precomputed_mel = bool(args.precomputed_mel)
    args.img_caption_zl = "all"

    
    print(args)
               
    dset = Dataset_soundscape(args=args,
                            split="train",
                            )

    loader = torch.utils.data.DataLoader(dset,num_workers=0, batch_size=2, shuffle=True, drop_last=False,pin_memory=True,
                                        collate_fn=lambda batch:collate_batch(batch, text_encoder_type=args.text_encoder_type, audio_encoder_type=args.audio_encoder_type, metadata_type=args.metadata_type))
    batch = next(iter(loader))
    print(batch.keys())
    print(batch['sat'].shape)                                                    #torch.Size([2, 3, 224, 224])
    for k in batch['audio_caption_input'].keys():
        print(k,batch['audio_caption_input'][k].shape)                           #input_ids torch.Size([2, 12]), attention_mask torch.Size([2, 12])
    
    for k in batch['llava_caption_input'].keys():
        print(k,batch['llava_caption_input'][k].shape)                           #input_ids torch.Size([2, 40]), attention_mask torch.Size([2, 40])

    for k in batch['audio'].keys():
        print(k,batch['audio'][k].shape)                                         #input_features[2, 1, 1001, 64]
       
    print(batch['key'])
    try:
        print("audio_source shape:",batch['audio_source'].shape)                                             #torch.Size([2])
    except:
        pass
    try:
        print("caption_source shape:",batch['caption_source'].shape)                                          #torch.Size([2])
    except:
        pass
    try:
        print("latlong shape:",batch['latlong'].shape)                                                        #torch.Size([2, 4])                                             
    except:
        pass
    try:
        print("time shape:",batch['time'].shape,batch['time_valid'].shape)                                    #torch.Size([2, 2]) torch.Size([2])
    except:
        pass
    try:
        print("month shape:",batch['month'].shape,batch['month_valid'].shape)                                #torch.Size([2, 2]) torch.Size([2])
    except:
        pass
