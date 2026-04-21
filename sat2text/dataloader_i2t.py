import os
import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.config import cfg
from utilities.utils import sat_transform


llava_caption_bingmap = pd.read_json(os.path.join(cfg.metafiles_path,"GeoSound/llava_caption_for_bingmap.json"),lines=True)
llava_caption_sentinel = pd.read_json(os.path.join(cfg.metafiles_path,"GeoSound/llava_caption_for_sentinel.json"),lines=True)
llava_caption_SoundingEarth = pd.read_json(os.path.join(cfg.metafiles_path,"SoundingEarth/SoundingEarth_llava_caption_for_googleEarth_zl_1.json"),lines=True)
llava_caption = {'sentinel':llava_caption_sentinel, "bingmap":llava_caption_bingmap, "googleEarth":llava_caption_SoundingEarth}


class Dataset_soundscape(Dataset):
    def __init__(self,                            
                 split="train",
                 sat_input_size=224,
                 sat_scale="multi",
                 test_zoom_level=None,
                 dataset_type="GeoSound", #'GeoSound_sentinel','GeoSound_bingmap', 'SoundingEarth'
                 precision="full"): 
        
        self.split = split
        self.dataset_type = dataset_type
        self.test_zoom_level = test_zoom_level
        self.sat_scale=sat_scale
        self.sat_input_size = sat_input_size
        self.ignore_ids = list(pd.read_csv(cfg.ignore_ids_geosound)['sample_id'])
        
        if "GeoSound" in dataset_type:
            self.overhead = self.dataset_type.split("_")[1]
            if self.split == "train": 
                self.meta_df = pd.read_csv(os.path.join(cfg.data_path,"metafiles/GeoSound/train_metadata.csv"))
            if self.split == "val": 
                self.meta_df = pd.read_csv(os.path.join(cfg.data_path,"metafiles/GeoSound/val_metadata.csv"))
            if self.split == "test": 
                self.meta_df = pd.read_csv(os.path.join(cfg.data_path,"metafiles/GeoSound/test_metadata.csv"))
                valid_ids = pd.read_csv(os.path.join(cfg.data_path,"metafiles/GeoSound/test_ids_geosound.csv"))
                self.meta_df = self.meta_df[self.meta_df['sample_id'].isin(list(valid_ids['sample_id']))]

            self.meta_df = self.meta_df[~self.meta_df['sample_id'].isin(self.ignore_ids)]
            
            self.llava_caption = llava_caption[self.overhead]

        elif dataset_type == "SoundingEarth":
            self.overhead = "googleEarth"
            if self.split == "train": 
                self.meta_df = pd.read_csv(os.path.join(cfg.data_path,"metafiles/SoundingEarth/aporee_train_fairsplit_10km.csv"))
            if self.split == "val": 
                self.meta_df = pd.read_csv(os.path.join(cfg.data_path,"metafiles/SoundingEarth/aporee_val_fairsplit_10km.csv"))
            if self.split == "test": 
                self.meta_df = pd.read_csv(os.path.join(cfg.data_path,"metafiles/SoundingEarth/aporee_test_fairsplit_10km.csv"))
                valid_ids = list(pd.read_csv(os.path.join(cfg.data_path,"metafiles/SoundingEarth/test_ids_soundingEarth.csv"))['sample_id'])
                valid_ids = [i.replace("aporee-",'') for i in valid_ids]
                self.meta_df = self.meta_df[self.meta_df['sample_id'].isin(valid_ids)]

        self.precision = precision
        self.aporee_meta = pd.read_csv(os.path.join(cfg.data_path,"aporee/final_metadata_with_captions.csv"))
        
       
    def __len__(self):
        return len(self.meta_df) 
    def __getitem__(self,idx):
        # idx = random.randint(0, len(self.meta_df)-1)
        if "GeoSound" in self.dataset_type:
            if self.sat_scale == "multi":
                zoom_level = random.choice([1,3,5])
            else:
                zoom_level = 1
        elif self.dataset_type == "SoundingEarth":
                zoom_level = 1
        
        if self.test_zoom_level != None:
            zoom_level = self.test_zoom_level
        
        sat_tr = sat_transform(is_train=self.split=="train", input_size=self.sat_input_size,sat_type=self.overhead,zoom_level=zoom_level)
        out_dict = {}
        out_dict['sat_zoom_level'] = zoom_level
       
        sample = dict(self.meta_df.iloc[idx])
        sample_id = sample['sample_id']
        if self.overhead == "sentinel":
            source = sample['sample_id'].split("-")[0]
            key = sample['sample_id'].split("-")[1]
            image_path = os.path.join(cfg.data_path,source,'images',"sentinel",str(key)+'.jpeg')
            llava_caption = self.llava_caption[self.llava_caption['sample_id']==sample_id]['captions'].item()[f'text{zoom_level}']
        if self.overhead == "bingmap":
            source = sample['sample_id'].split("-")[0]
            key = sample['sample_id'].split("-")[1]
            image_path = os.path.join(cfg.data_path,source,'images',"bingmap",str(key)+'.jpeg')
            llava_caption = self.llava_caption[self.llava_caption['sample_id']==sample_id]['captions'].item()[f'text{zoom_level}']
        if self.dataset_type == "SoundingEarth":
            short_id = sample['key']
            image_path = os.path.join(cfg.data_path,"aporee",'images',"googleEarth",str(short_id)+'.jpg')
            llava_caption = self.llava_caption[self.llava_caption['sample_id']==sample_id]['captions'].item()
        ################################################################################################################################################################
        #Prepare sat image
        image = Image.open(image_path)
        final_image = sat_tr(image)

        if self.precision == "half":
            out_dict['sat']= final_image.half()
        else:
            out_dict['sat']= final_image
        out_dict['llava_caption'] = llava_caption
        out_dict['key'] = sample_id

        # if torch.isnan(final_image).any():

        return out_dict
