##local imports
import json
import os
import random
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import cfg, ckpt_cfg
from src.dataloader import load_llava_caption_df
from src.metrics import get_retrieval, recall_key

from .dataloader_i2t import Dataset_soundscape
from .engine_i2t import sat2textModel

def save_dict_to_json(dictionary, output_file):
    with open(output_file, 'a') as json_file:
        json.dump(dictionary, json_file)
        json_file.write('\n')  # Add a newline character for better readability

def l2normalize(batch_embeddings):
    return batch_embeddings / (batch_embeddings.norm(p=2, dim=-1, keepdim=True) + 1e-8)

def get_captions(lcdf, topkdf, test_zoom_level=1, idx=0):
    topkdf = topkdf.copy()
    topkdf['top1_key'] = topkdf['top_keys'].apply(lambda x: x[idx])

    # Merge lcdf with topkdf for both ground truth (gt) and predicted (top1) keys
    # We need to merge on 'sample_id' to retrieve the captions for both ground truth and predicted keys
    gt_df = pd.merge(topkdf[['key', 'top1_key']], lcdf[['sample_id', 'captions']], 
                     left_on='key', right_on='sample_id', how='left')

    pred_df = pd.merge(topkdf[['key', 'top1_key']], lcdf[['sample_id', 'captions']], 
                       left_on='top1_key', right_on='sample_id', how='left')

    # Extract the relevant 'text' field for ground truth and predicted captions
    gt_df['gt_caption'] = gt_df['captions'].apply(
        lambda x: x.get('text' + str(test_zoom_level), x.get('text1', '')) if isinstance(x, dict) else '')
    pred_df['top1_caption'] = pred_df['captions'].apply(
        lambda x: x.get('text' + str(test_zoom_level), x.get('text1', '')) if isinstance(x, dict) else '')

    # Construct the final DataFrame
    outdf = pd.DataFrame({
        'key': gt_df['key'],
        'top1key': gt_df['top1_key'],
        'gt_caption': gt_df['gt_caption'],
        'top1_caption': pred_df['top1_caption']
    })

    return outdf

    
def set_seed(seed: int = 56) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

class Evaluate(object):
    def __init__(self,split, ckpt_path,device,test_zoom_level=1, recall_at=10,dataset_type="GeoSound", sat_type="bingmap"):
        super().__init__()
        set_seed(56)
        self.split = split
        self.ckpt_path = ckpt_path
        self.device = device
        self.test_zoom_level = test_zoom_level
        self.recall_at = recall_at
        self.dataset_type = dataset_type
        self.sat_type = sat_type
        self.hparams, self.model  = self.load_model()
        self.dataloader = self.get_dataloader()
    
    def load_model(self):
        #load geoclap model from checkpoint
        pretrained_ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        hparams = pretrained_ckpt['hyper_parameters']
        assert (hparams['dataset_type'] == self.dataset_type) and  (hparams['sat_type'] == self.sat_type)#just a safety check to ensure usage of right checkpoint
        pretrained_weights = pretrained_ckpt['state_dict']
        print(hparams)    
        model = sat2textModel(Namespace(**hparams)).to(self.device)
        model.load_state_dict(pretrained_weights,strict=False)
        model = model.eval()
        #set all requires grad to false
        for params in model.parameters():
            params.requires_grad=False
        
        return Namespace(**hparams), model
    
    def get_dataloader(self):
        dset = Dataset_soundscape(
                                split=self.split,
                                sat_input_size=self.hparams.sat_input_size,
                                test_zoom_level=self.test_zoom_level,
                                dataset_type=self.hparams.dataset_type) #'GeoSound_sentinel','GeoSound_bingmap', 'SoundingEarth'

        loader = torch.utils.data.DataLoader(dset,batch_size=self.hparams.batch_size,
                    shuffle=False, pin_memory=False, persistent_workers=False,num_workers=self.hparams.num_workers,
                    )
        return loader

    def validation_step(self, batch, batch_idx):
        batch = self.model.prepare_batch(batch)
        outputs = self.model.shared_step(batch,train=False)

        outputs['embeds']['fdt_sat_embeds'] = outputs['embeds']['fdt_sat_embeds'].detach().cpu().to(torch.float32)
        outputs['embeds']['fdt_txt_embeds'] = outputs['embeds']['fdt_txt_embeds'].detach().cpu().to(torch.float32)

        return outputs['embeds']
    
    @torch.no_grad()
    def get_final_metrics(self):
        
        sat_embeddings = []
        text_embeddings = []
        gt_keys = []
        
        test_dataloader = self.dataloader
       
        for i,batch in tqdm(enumerate(test_dataloader)):
            outputs = self.validation_step(batch,i) 
            sat_embeddings.append(outputs['fdt_sat_embeds'])
            text_embeddings.append(outputs['fdt_txt_embeds'])
            gt_keys = gt_keys + list(batch['key'])

        sat_embeddings = torch.cat(sat_embeddings,axis=0)
        text_embeddings = torch.cat(text_embeddings,axis=0)
        
        text_query_embeddings = text_embeddings
        sat_query_embeddings = sat_embeddings

        text_gallery_embeddings = text_embeddings
        sat_gallery_embeddings = sat_embeddings

        R_k = self.recall_at/100*sat_gallery_embeddings.shape[0]
        print("size of gallery:",sat_gallery_embeddings.shape)
        
        retrieval_results_I2T, i2t_topkeys_df = get_retrieval(modality1_emb=l2normalize(sat_query_embeddings), modality2_emb=l2normalize(text_gallery_embeddings), normalized=True,k=R_k,keys=gt_keys,save_top=1)
        retrieval_results_T2I, _topkeys_df_t2i = get_retrieval(modality1_emb=l2normalize(text_query_embeddings), modality2_emb=l2normalize(sat_gallery_embeddings), normalized=True,k=R_k,keys=gt_keys,save_top=1)
        
        lcdf = load_llava_caption_df(self.sat_type)
        outdf =  get_captions(lcdf=lcdf,topkdf=i2t_topkeys_df,test_zoom_level=self.test_zoom_level, idx=0)

        return retrieval_results_I2T, retrieval_results_T2I, R_k, outdf


#GeoSound_infonce_sentinel
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--results_path', type=str, default=os.path.join(cfg.log_dir, 'results'))
    parser.add_argument('--test_zoom_level', type=int, default=1)
    parser.add_argument('--recall_at', type=int, default=10)
    parser.add_argument('--split', type=str, default="test") #options: val, test
    parser.add_argument('--dataset_type', type=str, default="GeoSound_bingmap",choices=["GeoSound_bingmap","GeoSound_sentinel","SoundingEarth"])
    parser.add_argument('--sat_type', type=str, default='bingmap', choices=['bingmap','googleEarth']) 
    parser.add_argument('--save_results', type=str, default='false', choices=['true','false'])
    parser.add_argument('--save_retrieved', type=str, default='false', choices=['true','false'])
    parser.add_argument('--json_name', type=str, default='image2text_baseline')
    parser.add_argument('--expr', type=str, default='bingmap_i2t_baseline') 
                                                               
    args = parser.parse_args()
    if not args.ckpt_path and args.expr not in ckpt_cfg:
        parser.error(
            f"--ckpt_path not provided and --expr={args.expr!r} is not in ckpt_cfg. "
            "Either pass --ckpt_path directly or populate ckpt_cfg in src/config.py."
        )
    
    #params
    set_seed(56)
    if args.ckpt_path != '':
        ckpt_path = args.ckpt_path
    else:
        #GeoSound_pcmepp_metadata_sentinel
        ckpt_path = ckpt_cfg[args.expr]
   
    #configure evaluation
    evaluation = Evaluate(split=args.split, ckpt_path=ckpt_path,device=device, 
                          recall_at = int(args.recall_at),
                          test_zoom_level=int(args.test_zoom_level), dataset_type=args.dataset_type, sat_type=args.sat_type)

    results_I2T, results_T2I, R_k, top_df = evaluation.get_final_metrics()
    print("IMAGE TO TEXT RETRIEVAL RESULTS:",results_I2T)
    print("TEXT TO IMAGE RETRIEVAL RESULTS:",results_T2I)
    print("##############################################################################################################")
    rk_label = recall_key(R_k)
    results_dict = {
                    'dataset_type':args.dataset_type, 'overhead_type':args.sat_type, 'loss_type':"infonce",
                    'expr':args.expr,
                    'test_zoom_level':args.test_zoom_level,
                    f'I2T_{rk_label}':results_I2T[rk_label],'I2T_median':results_I2T['Median Rank'],
                    f'T2I_{rk_label}':results_T2I[rk_label],'T2I_median':results_T2I['Median Rank'],
                    'ckpt_path':ckpt_path
                    }
    
    if args.save_retrieved == "true":
        log_path = os.path.dirname(cfg.results_json)
        top_df.to_csv(os.path.join(log_path,args.expr+"_IMAGE2TEXT_"+str(args.test_zoom_level)+".csv"))

    if args.save_results == "true":
        json_path = cfg.results_json.replace(".json","_"+args.json_name+".json")
        save_dict_to_json(results_dict, json_path) 
    