# This script evaluates the image-to-text retrieval performance of the Sat2Sound framework.
# Refer to the ArgumentParser section of the code for details on the expected inputs and outputs.

from src.metrics import get_retrevial
from src.engine import sat2soundModel
import torch
import numpy as np
import random
import os
from tqdm import tqdm
from src.dataloader import Dataset_soundscape, collate_batch, llava_caption
from argparse import Namespace, ArgumentParser, RawTextHelpFormatter
import pandas as pd
import json
from src.config import ckpt_cfg, cfg

def save_dict_to_json(dictionary, output_file):
    with open(output_file, 'a') as json_file:
        json.dump(dictionary, json_file)
        json_file.write('\n')  # Add a newline character for better readability

def l2normalize(batch_embeddings):
    return batch_embeddings/batch_embeddings.norm(p=2,dim=-1, keepdim=True)

def get_captions(lcdf, topkdf, test_zoom_level=1, idx=0):
    # Merge lcdf with topkdf to get captions for both ground truth and predicted keys
    # Create a column for top1_key from the top_keys list in topkdf
    topkdf['top1_key'] = topkdf['top_keys'].apply(lambda x: x[idx])

    # Merge lcdf with topkdf for both ground truth (gt) and predicted (top1) keys
    # We need to merge on 'sample_id' to retrieve the captions for both ground truth and predicted keys
    gt_df = pd.merge(topkdf[['key', 'top1_key']], lcdf[['sample_id', 'captions']], 
                     left_on='key', right_on='sample_id', how='left')

    pred_df = pd.merge(topkdf[['key', 'top1_key']], lcdf[['sample_id', 'captions']], 
                       left_on='top1_key', right_on='sample_id', how='left')

    # Extract the relevant 'text' field for ground truth and predicted captions
    gt_df['gt_caption'] = gt_df['captions'].apply(lambda x: x['text' + str(test_zoom_level)])
    pred_df['top1_caption'] = pred_df['captions'].apply(lambda x: x['text' + str(test_zoom_level)])

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
    def __init__(self,split, ckpt_path,device,caption_type="audio",test_zoom_level=1, test_mel_index=0, recall_at=10,dataset_type="GeoSound", sat_type="bingmap"):
        super().__init__()
        set_seed(56)
        self.split = split
        self.ckpt_path = ckpt_path
        self.device = device
        self.test_zoom_level = test_zoom_level
        self.test_mel_index = test_mel_index
        self.recall_at = recall_at
        self.dataset_type = dataset_type
        self.sat_type = sat_type
        self.caption_type = caption_type
        self.hparams, self.model  = self.load_model()
        self.dataloader = self.get_dataloader()
    
    def load_model(self):
        if ".ckpt" not in self.ckpt_path:
            all_ckpts = os.listdir(os.path.join(self.ckpt_path,"checkpoints"))
            valid_ckpts = [f.replace("-v1.ckpt",".ckpt") for f in all_ckpts if "Recall=" in f]
            recalls = [float(f.split("Recall=")[1].replace(".ckpt","")) for f in valid_ckpts]
            best_ckpt = valid_ckpts[np.array(recalls).argmax()]
            best_ckpt_path = os.path.join(self.ckpt_path,"checkpoints",best_ckpt)
            print("Best ckpt path:",best_ckpt_path)
            # import sys; sys.exit()
            pretrained_ckpt = torch.load(best_ckpt_path,map_location=self.device)
            self.ckpt_path = best_ckpt_path
            
        pretrained_ckpt = torch.load(self.ckpt_path,map_location=self.device)
        hparams = pretrained_ckpt['hyper_parameters']
        if 'jsd_weight' != hparams:
            hparams['jsd_weight'] = 0
        assert (hparams['dataset_type'] == self.dataset_type) and  (hparams['sat_type'] == self.sat_type)#just a safety check to ensure usage of right checkpoint
        pretrained_weights = pretrained_ckpt['state_dict']
        hparams['meta_droprate'] = 0.0 #all metadata will be kept during inference
        
        print(hparams)    
        model = sat2soundModel(Namespace(**hparams)).to(self.device)
        model.load_state_dict(pretrained_weights,strict=False)
        model = model.eval()
        #set all requires grad to false
        for params in model.parameters():
            params.requires_grad=False
        
        return Namespace(**hparams), model
    
    def get_dataloader(self):
        dset = Dataset_soundscape(
                                    split = self.split,
                                    args=self.hparams,
                                    test_zoom_level=self.test_zoom_level,
                                    test_mel_index=self.test_mel_index)
        loader = torch.utils.data.DataLoader(dset,batch_size=100,
                    shuffle=False, pin_memory=False, persistent_workers=False,num_workers=self.hparams.num_workers,
                    collate_fn=lambda batch:collate_batch(batch, text_encoder_type=self.hparams.text_encoder_type, metadata_type=self.hparams.metadata_type))
        return loader 
    
    def validation_step(self, batch, batch_idx):
        batch = self.model.prepare_batch(batch)
        if self.hparams.precision == 'half':
            with torch.cuda.amp.autocast():
                outputs = self.model.shared_step(batch,train=False)
        else:
            outputs = self.model.shared_step(batch,train=False)
       
        for k in outputs['embeds']['sat_embeds_dict'].keys():
            outputs['embeds']['sat_embeds_dict'][k] = outputs['embeds']['sat_embeds_dict'][k].detach().cpu().to(torch.float32)
        outputs['embeds']['audio_embeds'] = outputs['embeds']['audio_embeds'].detach().cpu().to(torch.float32)
        if 'audio' in self.hparams.caption_type:
            outputs['embeds']['audio_caption_embeds'] = outputs['embeds']['audio_caption_embeds'].detach().cpu().to(torch.float32)
        
        outputs['embeds']['fdt_sat_embeds'] = outputs['embeds']['fdt_sat_embeds'].detach().cpu().to(torch.float32)
        outputs['embeds']['fdt_txt_embeds'] = outputs['embeds']['fdt_txt_embeds'].detach().cpu().to(torch.float32)

        return outputs['embeds']
    
    @torch.no_grad()
    def get_final_metrics(self):
        
        sat_embeddings = []
        text_embeddings = []
        gt_keys = []
        
        test_dataloader = self.dataloader
        #outputs: dict_keys(['sat_embeds_dict', 'audio_embeds', 'audio_caption_embeds', 'fdt_sat_embeds', 'fdt_txt_embeds'])
        #sat_embeds_dict: dict_keys(['ctotal'])
        for i,batch in tqdm(enumerate(test_dataloader)):
            outputs = self.validation_step(batch,i) 
            gt_keys = gt_keys + list(batch['key'])
            if self.caption_type == "audio":
                sat_embeddings.append(outputs['sat_embeds_dict']['ctotal'])
                text_embeddings.append(outputs['audio_caption_embeds'])
            elif self.caption_type == "image":
                sat_embeddings.append(outputs['fdt_sat_embeds'])
                text_embeddings.append(outputs['fdt_txt_embeds'])
            

        sat_embeddings = torch.cat(sat_embeddings,axis=0)
        text_embeddings = torch.cat(text_embeddings,axis=0)

        text_query_embeddings = text_embeddings
        sat_query_embeddings = sat_embeddings

        text_gallery_embeddings = text_embeddings
        sat_gallery_embeddings = sat_embeddings

        R_k = self.recall_at/100*sat_gallery_embeddings.shape[0]
        print("size of gallery:",sat_gallery_embeddings.shape)
        retrieval_results_I2T, topkeys_df = get_retrevial(modality1_emb=l2normalize(sat_query_embeddings), modality2_emb=l2normalize(text_gallery_embeddings), normalized=True,k=R_k,keys=gt_keys,save_top=1)
        retrieval_results_T2I, topkeys_df = get_retrevial(modality1_emb=l2normalize(text_query_embeddings), modality2_emb=l2normalize(sat_gallery_embeddings), normalized=True,k=R_k,keys=gt_keys,save_top=1)
        
        lcdf = llava_caption[self.sat_type]
        outdf =  get_captions(lcdf=lcdf,topkdf=topkeys_df,test_zoom_level=self.test_zoom_level, idx=0)

        return retrieval_results_I2T, retrieval_results_T2I, R_k, outdf


#GeoSound_infonce_sentinel
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--results_path', type=str, default=os.path.join(cfg.log_dir, 'results'))
    parser.add_argument('--test_zoom_level', type=int, default=1)
    parser.add_argument('--test_mel_index', type=int, default=0,choices=[0,1,2,3,4])
    parser.add_argument('--caption_type', type=str, default="image",choices=["audio","image"])
    parser.add_argument('--recall_at', type=int, default=10)
    parser.add_argument('--split', type=str, default="test") #options: val, test
    parser.add_argument('--dataset_type', type=str, default="GeoSound",choices=["GeoSound","SoundingEarth"])
    parser.add_argument('--sat_type', type=str, default='bingmap', choices=['bingmap','googleEarth']) 
    parser.add_argument('--save_results', type=str, default='false', choices=['true','false'])
    parser.add_argument('--save_retrieved', type=str, default='false', choices=['true','false'])
    parser.add_argument('--json_name', type=str, default='image2text')
    parser.add_argument('--expr', type=str, default='bingmap_withmeta')
                                                               
    args = parser.parse_args()
    assert (len(args.expr) !=0) or (len(args.ckpt_path) !=0) 
    
    #params
    set_seed(56)
    
    ckpt_path = ckpt_cfg[args.expr]
   
    #configure evaluation
    evaluation = Evaluate(split=args.split, ckpt_path=ckpt_path,device=device, 
                          recall_at = int(args.recall_at),caption_type=args.caption_type,
                          test_zoom_level=int(args.test_zoom_level), dataset_type=args.dataset_type, sat_type=args.sat_type)

    results_I2T, results_T2I, R_k, top_df = evaluation.get_final_metrics()
    print("IMAGE TO TEXT RETREVIAL RESULTS:",results_I2T)
    print("TEXT TO IMAGE RETREVIAL RESULTS:",results_T2I)
    print("##############################################################################################################")
    results_dict = {'index':args.test_mel_index,
                    'dataset_type':args.dataset_type, 'overhead_type':args.sat_type, 'loss_type':"infonce",
                    'expr':args.expr,
                    'test_zoom_level':args.test_zoom_level, 'test_mel_index':args.test_mel_index,
                    'I2T_R@10':results_I2T['R@'+str(R_k)],'I2T_median':results_I2T['Median Rank'],
                    'T2I_R@10':results_T2I['R@'+str(R_k)],'T2I_median':results_T2I['Median Rank'],
                    'ckpt_path':ckpt_path
                    }
    if args.save_retrieved:
        log_path = os.path.dirname(cfg.results_json)
        print("Saved to:",os.path.join(log_path,args.expr+"_IMAGE2TEXT_"+str(args.test_zoom_level)+".csv"))
        top_df.to_csv(os.path.join(log_path,args.expr+"_IMAGE2TEXT_"+str(args.test_zoom_level)+".csv"))

    if args.save_results == "true":
        json_path = cfg.results_json.replace(".json","_"+args.json_name+".json")
        print("Saved to:",args.json_name)
        save_dict_to_json(results_dict, json_path)
    