"""Generate sound-related captions for satellite imagery using LLaVA-1.5-7B.

Supports both GeoSound (zoom levels 1/3/5 for a single sat_type) and
SoundingEarth (a single zoom level, with googleEarth/sentinel/bingmap imagery).

The model runs locally (no API calls); you need a GPU with ~16 GB of VRAM for
fp16 inference.

Example
-------
GeoSound bingmap (captions at zoom 1, 3, 5):

    python -m data_prep.generate_llava_captions \
        --dataset GeoSound \
        --sat_type bingmap \
        --data_path /path/to/data \
        --metadata_csv /path/to/GeoSound/train_metadata.csv \
        --metadata_csv /path/to/GeoSound/val_metadata.csv \
        --metadata_csv /path/to/GeoSound/test_metadata.csv \
        --out_json /path/to/llava_caption_for_bingmap.json

SoundingEarth googleEarth at zoom 1:

    python -m data_prep.generate_llava_captions \
        --dataset SoundingEarth \
        --sat_type googleEarth \
        --zoom_level 1 \
        --data_path /path/to/aporee \
        --metadata_csv /path/to/aporee_train_fairsplit_10km.csv \
        --metadata_csv /path/to/aporee_val_fairsplit_10km.csv \
        --metadata_csv /path/to/aporee_test_fairsplit_10km.csv \
        --out_json /path/to/SoundingEarth_llava_caption_googleEarth_zl1.json
"""

import json
import os
from argparse import ArgumentParser

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from utilities.utils import get_image


MODEL_ID = "llava-hf/llava-1.5-7b-hf"
PROMPT_TEXT = (
    "What types of sounds can we expect to hear from the location captured by "
    "this aerial view image? Describe in up to two sentences."
)
PROMPT = f"USER: <image>\n{PROMPT_TEXT}\nASSISTANT:"


def save_dict_to_json(dictionary, output_file):
    with open(output_file, "a") as json_file:
        json.dump(dictionary, json_file)
        json_file.write("\n")


def build_model(device):
    model = (
        LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
        )
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def caption_image(model, processor, image, device):
    try:
        inputs = processor(PROMPT, image, return_tensors="pt").to(device, torch.float16)
        output = model.generate(**inputs, max_new_tokens=77, do_sample=False)
        return processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT: ")[1]
    except Exception:
        return "This is a sound of some place."


def caption_at_zooms(model, processor, image_path, sat_type, zoom_levels, device):
    captions = {}
    for z in zoom_levels:
        image = get_image(image_path, zoom_level=z, sat_type=sat_type)
        captions[f"text{z}"] = caption_image(model, processor, image, device)
    return captions


def resolve_geosound_image_path(data_path, sample_id, sat_type):
    source, key = sample_id.split("-", 1)
    return os.path.join(data_path, source, "images", sat_type, f"{key}.jpeg")


def resolve_soundingearth_image_path(data_path, row, sat_type):
    short_id = row["key"]
    long_id = row["long_key"]
    if sat_type == "googleEarth":
        return long_id, os.path.join(data_path, "images", "googleEarth", f"{short_id}.jpg")
    if sat_type == "sentinel":
        return long_id, os.path.join(data_path, "images", "sentinel_geoclap", f"{short_id}.jpeg")
    if sat_type == "bingmap":
        return long_id, os.path.join(data_path, "images", "bingmap_geoclap", f"{long_id}.jpg")
    raise ValueError(f"Unsupported sat_type for SoundingEarth: {sat_type}")


def main():
    parser = ArgumentParser(description="Generate LLaVA sound-related captions for satellite images.")
    parser.add_argument("--dataset", choices=["GeoSound", "SoundingEarth"], required=True)
    parser.add_argument(
        "--sat_type",
        choices=["sentinel", "bingmap", "googleEarth"],
        required=True,
        help="Satellite imagery type. GeoSound: sentinel|bingmap. SoundingEarth: googleEarth|sentinel|bingmap.",
    )
    parser.add_argument(
        "--zoom_level",
        type=int,
        default=None,
        help="SoundingEarth: single zoom level (default 1). GeoSound: ignored (captions are written for zl 1, 3, 5).",
    )
    parser.add_argument("--data_path", required=True, help="Root directory containing image subdirectories.")
    parser.add_argument(
        "--metadata_csv",
        action="append",
        required=True,
        help="Path(s) to metadata CSV(s). Repeat the flag for train/val/test splits.",
    )
    parser.add_argument("--out_json", required=True, help="Output JSONL file. One record per line.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda:0 or cpu.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    df = pd.concat([pd.read_csv(p) for p in args.metadata_csv], ignore_index=True)
    print(f"Loaded {len(df)} records from {len(args.metadata_csv)} CSV(s).")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)

    model, processor = build_model(device)

    if args.dataset == "GeoSound":
        zoom_levels = [1, 3, 5]
        for i in tqdm(range(len(df))):
            sample_id = df.iloc[i]["sample_id"]
            image_path = resolve_geosound_image_path(args.data_path, sample_id, args.sat_type)
            captions = caption_at_zooms(model, processor, image_path, args.sat_type, zoom_levels, device)
            save_dict_to_json({"sample_id": sample_id, "captions": captions}, args.out_json)
    else:  # SoundingEarth
        zl = args.zoom_level if args.zoom_level is not None else 1
        for i in tqdm(range(len(df))):
            long_id, image_path = resolve_soundingearth_image_path(args.data_path, df.iloc[i], args.sat_type)
            captions = caption_at_zooms(model, processor, image_path, args.sat_type, [zl], device)
            save_dict_to_json({"sample_id": long_id, "captions": captions}, args.out_json)


if __name__ == "__main__":
    main()
