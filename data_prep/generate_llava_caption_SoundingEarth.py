"""LLaVA soundscape captions for SoundingEarth GoogleEarth tiles; see --help for options."""


import json
import os

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

def get_image(image_path,zoom_level=1,sat_type="googleEarth"):
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
    parser.add_argument('--overhead', type=str, default="googleEarth", choices=["sentinel", "bingmap", "googleEarth"])
    parser.add_argument('--zoom_level', type=int, default=1)
    parser.add_argument('--data_path', type=str,
                        default=os.path.join(_cfg.data_path, "aporee"),
                        help='Path to the Aporee audio directory (overrides SAT2SOUND_DATA_PATH).')
    parser.add_argument('--metafiles_path', type=str,
                        default=os.path.join(_cfg.data_path, "metafiles", "SoundingEarth"),
                        help='Path to the SoundingEarth metafiles directory.')
    args = parser.parse_args()

    data_path = args.data_path
    metafiles_se = args.metafiles_path
    train_df = pd.read_csv(os.path.join(metafiles_se, "aporee_train_fairsplit_10km.csv"))
    val_df =  pd.read_csv(os.path.join(metafiles_se, "aporee_val_fairsplit_10km.csv"))
    test_df = pd.read_csv(os.path.join(metafiles_se, "aporee_test_fairsplit_10km.csv"))

    df_final = pd.concat([train_df,val_df, test_df])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
      image = get_image(image_path,zoom_level=args.zoom_level,sat_type=args.overhead)
      try:
        inputs = processor(prompt, image, return_tensors='pt').to(device, torch.float16)
        output = model.generate(**inputs, max_new_tokens=77, do_sample=False)
        caption = processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT: ")[1]
      except Exception:
        caption = "This is a sound of some place."
      return caption

    output_json_file = os.path.join(
        metafiles_se,
        "SoundingEarth_llava_caption_for_" + str(args.overhead) + "_zl_" + str(args.zoom_level) + ".json"
    )
    
    for i in tqdm(range(len(df_final))):
      sample = df_final.iloc[i]
      mp3name = sample['mp3name']
      short_id = sample['key']
      long_id = sample['long_key']
      
      if args.overhead == 'googleEarth':
          image_path = os.path.join(data_path,'images','googleEarth',str(short_id)+'.jpg')
      elif args.overhead == 'sentinel':
          image_path = os.path.join(data_path,'images','sentinel_geoclap',str(short_id)+'.jpeg')
      elif args.overhead == 'bingmap':
          image_path = os.path.join(data_path,'images','bingmap_geoclap',str(long_id)+'.jpg')
      else:
          raise NotImplementedError("supported satellite image types are:[googleEarth, sentinel, bingmap]")
      
      captions = get_caption(image_path)
      output_dict = {'sample_id':long_id,"captions":captions}
      save_dict_to_json(dictionary=output_dict, output_file=output_json_file)
      