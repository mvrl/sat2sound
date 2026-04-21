# This script evaluates the image-to-audio retrieval performance of the Sat2Sound framework.
# Refer to the ArgumentParser section of the code for details on the expected inputs and outputs.

import json
import os
import random
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter

import numpy as np
import torch
from tqdm import tqdm

from src.config import cfg, ckpt_cfg
from src.dataloader import Dataset_soundscape, collate_batch
from src.engine import sat2soundModel
from src.metrics import get_retrevial_metrics

def save_dict_to_json(dictionary, output_file):
    with open(output_file, 'a') as json_file:
        json.dump(dictionary, json_file)
        json_file.write('\n')  # Add a newline character for better readability

def l2normalize(batch_embeddings):
    return batch_embeddings/batch_embeddings.norm(p=2,dim=-1, keepdim=True)

    
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
    def __init__(self,split, ckpt_path,device,test_zoom_level=1, test_mel_index=0, recall_at=10,
                dataset_type="GeoSound", sat_type="bingmap", metadata_type="latlong_month_time_asource_tsource", expr=""):
        super().__init__()
        set_seed(56)
        self.split = split
        self.ckpt_path = ckpt_path
        self.expr = expr
        self.device = device
        self.test_zoom_level = test_zoom_level
        self.test_mel_index = test_mel_index
        self.recall_at = recall_at
        self.dataset_type = dataset_type
        self.metadata_type = metadata_type
        self.sat_type = sat_type
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
        assert (hparams['dataset_type'] == self.dataset_type) and  (hparams['sat_type'] == self.sat_type) and (hparams['run_name']==self.expr)#just a safety check to ensure usage of right checkpoint
        pretrained_weights = pretrained_ckpt['state_dict']
        hparams['meta_droprate'] = 0.0 #all metadata will be kept during inference
        hparams['metadata_type'] = self.metadata_type
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

        if (self.hparams.combined_modality_loss) and (self.hparams.shared_codebook):
                if self.hparams.use_combined_projectors:
                    outputs['embeds']['audiocaption_combined_embeds'] = outputs['embeds']['audiocaption_combined_embeds'].detach().cpu().to(torch.float32)
                    if self.hparams.combine_image_text:
                        outputs['embeds']['imagecaption_combined_embeds'] = outputs['embeds']['imagecaption_combined_embeds'].detach().cpu().to(torch.float32)
        return outputs['embeds']
    
    @torch.no_grad()
    def get_final_metrics(self):
        results = {}
        sat_embeddings = []
        audio_embeddings = []
        text_embeddings = []
        image_text_embeddings = []
        audiocaption_combined_embeds = []
        imagecaption_combined_embeds = []

        test_dataloader = self.dataloader
        #outputs: dict_keys(['sat_embeds_dict', 'audio_embeds', 'audio_caption_embeds', 'fdt_sat_embeds', 'fdt_txt_embeds'])
        #sat_embeds_dict: dict_keys(['ctotal'])
        for i,batch in tqdm(enumerate(test_dataloader)):
            outputs = self.validation_step(batch,i) 
            sat_embeddings.append(outputs['sat_embeds_dict']['ctotal'].detach().cpu())
            audio_embeddings.append(outputs['audio_embeds'].detach().cpu())
            text_embeddings.append(outputs['audio_caption_embeds'].detach().cpu())
            image_text_embeddings.append(outputs['fdt_txt_embeds'].detach().cpu())
            if self.hparams.use_combined_projectors:
                audiocaption_combined_embeds.append(outputs['audiocaption_combined_embeds'].detach().cpu())
                if self.hparams.combine_image_text:
                    imagecaption_combined_embeds.append(outputs['imagecaption_combined_embeds'].detach().cpu())

        
        sat_embeddings = torch.cat(sat_embeddings,axis=0)
        audio_embeddings = torch.cat(audio_embeddings,axis=0)
        text_embeddings = torch.cat(text_embeddings,axis=0)
        image_text_embeddings = torch.cat(image_text_embeddings,axis=0)

        if self.hparams.use_combined_projectors:
            audiocaption_combined_embeds = torch.cat(audiocaption_combined_embeds,axis=0)
            if self.hparams.combine_image_text:
                imagecaption_combined_embeds = torch.cat(imagecaption_combined_embeds,axis=0)

        ## Don't add text
        audio_query_embeddings = audio_embeddings
        audio_gallery_embeddings = audio_query_embeddings
        sat_query_embeddings = sat_embeddings
        sat_gallery_embeddings = sat_query_embeddings
        print("size of gallery:",audio_gallery_embeddings.shape[0])
        R_k = self.recall_at/100*audio_gallery_embeddings.shape[0]
        retrieval_results_I2S = get_retrevial_metrics(modality1_emb=l2normalize(sat_query_embeddings), modality2_emb=l2normalize(audio_gallery_embeddings), normalized=True,k=R_k)
        retrieval_results_S2I = get_retrevial_metrics(modality1_emb=l2normalize(audio_query_embeddings), modality2_emb=l2normalize(sat_gallery_embeddings), normalized=True,k=R_k)
        results['addtextto_none'] = {'retrieval_results_I2S':retrieval_results_I2S,'retrieval_results_S2I':retrieval_results_S2I}
       
        
        #Add to audio
        audio_query_embeddings = l2normalize(audio_embeddings)+l2normalize(text_embeddings)
        audio_gallery_embeddings = audio_query_embeddings
        sat_query_embeddings = sat_embeddings
        sat_gallery_embeddings = sat_query_embeddings
        retrieval_results_I2S = get_retrevial_metrics(modality1_emb=l2normalize(sat_query_embeddings), modality2_emb=l2normalize(audio_gallery_embeddings), normalized=True,k=R_k)
        retrieval_results_S2I = get_retrevial_metrics(modality1_emb=l2normalize(audio_query_embeddings), modality2_emb=l2normalize(sat_gallery_embeddings), normalized=True,k=R_k)
        results['addtextto_audio'] = {'retrieval_results_I2S':retrieval_results_I2S,'retrieval_results_S2I':retrieval_results_S2I}

        #Add to query
        audio_query_embeddings = l2normalize(audio_embeddings)+l2normalize(text_embeddings)
        sat_query_embeddings = l2normalize(sat_embeddings)+l2normalize(text_embeddings)
        audio_gallery_embeddings = audio_embeddings
        sat_gallery_embeddings = sat_embeddings
        retrieval_results_I2S = get_retrevial_metrics(modality1_emb=l2normalize(sat_query_embeddings), modality2_emb=l2normalize(audio_gallery_embeddings), normalized=True,k=R_k)
        retrieval_results_S2I = get_retrevial_metrics(modality1_emb=l2normalize(audio_query_embeddings), modality2_emb=l2normalize(sat_gallery_embeddings), normalized=True,k=R_k)
        results['addtextto_query'] = {'retrieval_results_I2S':retrieval_results_I2S,'retrieval_results_S2I':retrieval_results_S2I}

        if self.hparams.use_combined_projectors:
            #Use projected audio+caption instead of audio
            audio_query_embeddings = audiocaption_combined_embeds
            audio_gallery_embeddings = audio_query_embeddings
            sat_query_embeddings = sat_embeddings
            sat_gallery_embeddings = sat_query_embeddings
            retrieval_results_I2S = get_retrevial_metrics(modality1_emb=l2normalize(sat_query_embeddings), modality2_emb=l2normalize(audio_gallery_embeddings), normalized=True,k=R_k)
            retrieval_results_S2I = get_retrevial_metrics(modality1_emb=l2normalize(audio_query_embeddings), modality2_emb=l2normalize(sat_gallery_embeddings), normalized=True,k=R_k)
            results['addtextto_combinedaudiotext'] = {'retrieval_results_I2S':retrieval_results_I2S,'retrieval_results_S2I':retrieval_results_S2I}

            if self.hparams.combine_image_text:
                # Use projected image+caption instead of audio
                sat_query_embeddings = imagecaption_combined_embeds
                sat_gallery_embeddings = sat_query_embeddings
                audio_query_embeddings = audio_embeddings
                audio_gallery_embeddings = audio_query_embeddings
                retrieval_results_I2S = get_retrevial_metrics(modality1_emb=l2normalize(sat_query_embeddings), modality2_emb=l2normalize(audio_gallery_embeddings), normalized=True,k=R_k)
                retrieval_results_S2I = get_retrevial_metrics(modality1_emb=l2normalize(audio_query_embeddings), modality2_emb=l2normalize(sat_gallery_embeddings), normalized=True,k=R_k)
                results['addtextto_combinedimagetext'] = {'retrieval_results_I2S':retrieval_results_I2S,'retrieval_results_S2I':retrieval_results_S2I}
           
        return results, R_k

def parse_results(results_dict,R_k,args,ckpt_path):
        dicts = []
        common_fields = {'test_mel_index':args.test_mel_index, 'test_zoom_level':args.test_zoom_level, 'dataset_type':args.dataset_type, 
                        'overhead_type':args.sat_type, 'metadata_type':args.metadata_type, 'expr':args.expr, 'ckpt_path':ckpt_path}
        for key in results_dict.keys():
            result = results_dict[key]
            update_dict = {}
            update_dict['addtextto'] = key.split("_")[1]
            update_dict['I2S_Recall'] =  round(result['retrieval_results_I2S']['R@'+str(R_k)],3)
            update_dict['I2S_Median_Rank'] = int(result['retrieval_results_I2S']['Median Rank'])
            update_dict['S2I_Recall'] =  round(result['retrieval_results_S2I']['R@'+str(R_k)],3)
            update_dict['S2I_Median_Rank'] = int(result['retrieval_results_S2I']['Median Rank'])
            merged_dict = {**common_fields, **update_dict}
            dicts.append(merged_dict)
        
        return dicts

#GeoSound_infonce_sentinel
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    # parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--results_path', type=str, default=os.path.join(cfg.log_dir, 'results'))
    parser.add_argument('--test_zoom_level', type=int, default=1)
    parser.add_argument('--test_mel_index', type=int, default=0,choices=[0,1,2,3,4])
    parser.add_argument('--recall_at', type=int, default=10)
    parser.add_argument('--split', type=str, default="test") #options: val, test
    parser.add_argument('--dataset_type', type=str, default="GeoSound",choices=["GeoSound","SoundingEarth"])
    parser.add_argument('--metadata_type', type=str, default='latlong_month_time_asource_tsource')
    parser.add_argument('--sat_type', type=str, default='bingmap', choices=['bingmap','sentinel','googleEarth']) 
    parser.add_argument('--save_results', type=str, default='false', choices=['true','false'])
    parser.add_argument('--json_name', type=str, default='main', help="'main','ablation','SoundingEarth','test")
    parser.add_argument('--expr', type=str, default='bingmap_withmeta') 
                                                               
    args = parser.parse_args()
    #params
    set_seed(56)
    ckpt_path = ckpt_cfg[args.expr]
   
    #configure evaluation
    evaluation = Evaluate(split=args.split, ckpt_path=ckpt_path,device=device, 
                         recall_at = int(args.recall_at),
                          test_zoom_level=int(args.test_zoom_level), dataset_type=args.dataset_type, sat_type=args.sat_type, metadata_type=args.metadata_type,expr=args.expr)

    results_dict, R_k = evaluation.get_final_metrics()
    print(results_dict)
    # print("##############################################################################################################")
    dicts =  parse_results(results_dict,R_k,args,evaluation.ckpt_path)
    if args.save_results == "true":
        json_path = cfg.results_json.replace(".json","_"+args.json_name+".json")
        for d in dicts:
            save_dict_to_json(d, json_path)
    