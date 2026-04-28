"""LLaVA soundscape captions for GeoSound bingmap/sentinel tiles; see --help for options."""

import json
import os
import time

import pandas as pd
import torch
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration


tile_size = {'sentinel':256, 'bingmap':300, 'googleEarth':256} 

def central_crop_bbox(image_width, image_height, crop_width, crop_height):
    """Return (left, upper, right, lower) bbox for a centered crop."""
    left = (image_width - crop_width) // 2
    upper = (image_height - crop_height) // 2
    right = left + crop_width
    lower = upper + crop_height
    return (left, upper, right, lower)

def get_image(image_path,zoom_level=1,sat_type="sentinel"):
    crop_size = zoom_level*tile_size[sat_type]
    image = Image.open(image_path)
    bbox = central_crop_bbox(image_width=image.size[0], image_height=image.size[1], crop_width=crop_size, crop_height=crop_size)
    image = image.crop(bbox)
    return image

def save_dict_to_json(dictionary, output_file):
    with open(output_file, 'a') as json_file:
        json.dump(dictionary, json_file)
        json_file.write('\n')  # Add a newline character for better readability

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import cfg as _cfg

    parser = ArgumentParser(description='')
    parser.add_argument('--overhead', type=str, default="sentinel", choices=["sentinel", "bingmap"])
    parser.add_argument('--data_path', type=str, default=_cfg.data_path,
                        help='Root of the GeoSound dataset (overrides SAT2SOUND_DATA_PATH).')
    parser.add_argument('--metafiles_path', type=str,
                        default=os.path.join(_cfg.data_path, "metafiles", "GeoSound"),
                        help='Path to the GeoSound metafiles directory.')
    args = parser.parse_args()

    data_path = args.data_path
    metafiles_geosound = args.metafiles_path
    train_df = pd.read_csv(os.path.join(metafiles_geosound, "train_metadata.csv"))
    val_df = pd.read_csv(os.path.join(metafiles_geosound, "val_metadata.csv"))
    test_df = pd.read_csv(os.path.join(metafiles_geosound, "test_metadata.csv"))

    df_final = pd.concat([train_df,val_df, test_df])
    if args.overhead == "bingmap":
       device = "cuda:0"
    elif args.overhead == "sentinel":
       device = "cuda:1"
    
    print("running device:",device)
    model_id = "llava-hf/llava-1.5-7b-hf"
    prompt_text = "What types of sounds can we expect to hear from the location captured by this aerial view image? Describe in up to two sentences."
    prompt = "USER: <image>\n"+prompt_text+"\nASSISTANT:"
    model = LlavaForConditionalGeneration.from_pretrained(
                                                          model_id, 
                                                          torch_dtype=torch.float16, 
                                                          low_cpu_mem_usage=False, 
                                                          ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    def get_caption(image_path):
      captions = {"text1":"",
                  "text3":"",
                  "text5":""}
      zoom_levels = [1,3,5]
      for z in zoom_levels:
        image = get_image(image_path=image_path,zoom_level=z,sat_type=args.overhead)
        try:
          inputs = processor(prompt, image, return_tensors='pt').to(device, torch.float16)
          output = model.generate(**inputs, max_new_tokens=77, do_sample=False)
          caption = processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT: ")[1]
        except Exception:
          caption = "This is a sound of some place."
        captions["text"+str(z)] = caption
      return captions

    output_json_file = os.path.join(
        metafiles_geosound, "llava_caption_for_" + str(args.overhead) + ".json"
    )
    
    for i in tqdm(range(len(df_final))):
      sample = df_final.iloc[i]
      sample_id = sample['sample_id']
      source = sample_id.split("-")[0]
      key = sample_id.split("-")[1]
      image_path = os.path.join(data_path,source,'images',args.overhead,str(key)+'.jpeg')
      captions = get_caption(image_path)
      output_dict = {'sample_id':sample_id,"captions":captions}
      save_dict_to_json(dictionary=output_dict, output_file=output_json_file)
      