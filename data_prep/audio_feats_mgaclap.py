"""Offline MGACLAP mel extraction for GeoSound raw audio; see --help for options."""


import os
import random
import sys

import pandas as pd
import torch
import torchaudio
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_REPO_ROOT, "src", "models", "MGACLAP"))
from feature_extractor import AudioFeature

yaml_path = os.path.join(_REPO_ROOT, "src", "models", "MGACLAP", "inference_example.yaml")

with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)


feature_extractor = AudioFeature(config['audio_args'])

SAMPLE_RATE = 32000

_DEFAULT_GEOSOUND_ROOT = os.environ.get(
    "SAT2SOUND_DATA_PATH",
    os.path.join(_REPO_ROOT, "data", "GeoSound"),
)
_DEFAULT_OUT_DIR = os.environ.get(
    "SAT2SOUND_MEL_FEATS_PATH",
    os.path.join(_REPO_ROOT, "data", "GeoSound_audio_mel_feats"),
)

GEOSOUND_ROOT: str = _DEFAULT_GEOSOUND_ROOT
out_dir: str = os.path.join(_DEFAULT_OUT_DIR, "mgaclap")
data_path: str = _DEFAULT_GEOSOUND_ROOT
meta_columns = ['sample_id', 'date', 'latitude', 'longitude', 'description', 'tags', 
                'title', 'scientific_name', 'common_name', 'sound_format', 'text',
                'address', 'original_sampling_rate', 'bin_id']


def sample_10s_audio(audio, original_sr, sr=SAMPLE_RATE,T=10,rand=True):
    
    # Resample if necessary
    if original_sr != sr:
        audio = torchaudio.transforms.Resample(original_sr, sr)(audio)
    
    # Define the target length for 10 seconds
    target_length = sr * T  # 10 seconds of audio at the given sample rate

    # If the audio is already 10 seconds or longer, randomly sample a 10s segment
    if audio.shape[1] >= target_length:
        if rand:
            start = random.randint(0, audio.shape[1] - target_length)
        else:
            start = 0 #For demo purposes sample first 10s
        audio_sample = audio[:, start:start + target_length]  
    else:
        # If the audio is shorter than 10 seconds, pad with zeros
        padding = target_length - audio.shape[1]
        audio_sample = torch.nn.functional.pad(audio, (0, padding))

    return audio_sample

# Helper function to get audio features
def get_audio_feat_mgaclap(audio,original_sr,nsamples=5,rand=True):
    # Extract audio multiple times
    audios = [sample_10s_audio(audio, original_sr, sr=SAMPLE_RATE,rand=rand) for _ in range(nsamples)]
    stacked_audios = torch.cat(audios, dim=0)
    output = feature_extractor(stacked_audios)
    return output

# Function to process a single sample (for parallel execution)
def process_sample(sample):
    sample_id = sample['sample_id']
    parts = sample_id.split("-", 1)
    if len(parts) != 2:
        print(f"[warn] Skipping malformed sample_id (no '-' separator): {sample_id!r}")
        return
    source, key = parts
    sound_format = 'mp3'
    
    outpath = os.path.join(out_dir, source, f"{sample_id}.pth")

    if not os.path.exists(outpath):

        if source == 'aporee':
            matches = aporee_meta[aporee_meta['long_key'] == key]
            if len(matches) != 1:
                print(f"[warn] Skipping {sample_id!r}: expected 1 aporee row, got {len(matches)}")
                return
            soundname = matches.mp3name.item()
            audio_path = os.path.join(data_path, source, 'raw_audio', str(key), soundname)
        else:
            soundname = f"{key}.{sound_format}" if isinstance(key, str) else f"{str(key)}.{sound_format}"
            audio_path = os.path.join(data_path, source, 'raw_audio', soundname)

        # Load the audio
        try:
            audio, original_sr = torchaudio.load(audio_path)
        except Exception as exc:
            print(f"[warn] Skipping {sample_id!r}: could not load audio at {audio_path!r}: {exc}")
            return
        audio =  audio.mean(dim=0).unsqueeze(0)
        audio_mel = get_audio_feat_mgaclap(audio,original_sr)
        # Save output
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        torch.save(audio_mel, outpath)

# Function to split data into chunks and run parallel processing
def get_feats(split="train"):
    metafiles_geosound = os.path.join(GEOSOUND_ROOT, "metafiles", "GeoSound")
    if split == "train":
        meta_df = pd.read_csv(os.path.join(metafiles_geosound, "train_metadata.csv"))
    elif split == "val":
        meta_df = pd.read_csv(os.path.join(metafiles_geosound, "val_metadata.csv"))
    elif split == "test":
        meta_df = pd.read_csv(os.path.join(metafiles_geosound, "test_metadata.csv"))
        valid_ids = pd.read_csv(os.path.join(metafiles_geosound, "test_ids_geosound.csv"))
        meta_df = meta_df[meta_df['sample_id'].isin(list(valid_ids['sample_id']))]
    print(len(meta_df))
    global aporee_meta
    aporee_meta = pd.read_csv(
        os.path.join(GEOSOUND_ROOT, "metafiles", "SoundingEarth", "final_metadata_with_captions.csv")
    )
    
    # Convert DataFrame rows to dictionaries for easy passing to the worker function
    samples = [dict(meta_df.iloc[idx][meta_columns]) for idx in range(len(meta_df))]

    for sample in tqdm(samples):
        process_sample(sample)


if __name__ == '__main__':
    parser = ArgumentParser(description='Compute MGACLAP mel features for GeoSound audio files.')
    parser.add_argument('--data_path', type=str, default=_DEFAULT_GEOSOUND_ROOT,
                        help='Root of the GeoSound dataset (overrides SAT2SOUND_DATA_PATH env var).')
    parser.add_argument('--out_dir', type=str, default=os.path.join(_DEFAULT_OUT_DIR, "mgaclap"),
                        help='Output directory for .pth mel feature files (overrides SAT2SOUND_MEL_FEATS_PATH).')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'val', 'test', 'all'],
                        help='Which dataset split to process. Defaults to all three.')
    cli_args = parser.parse_args()
    GEOSOUND_ROOT = cli_args.data_path
    data_path = cli_args.data_path
    out_dir = cli_args.out_dir
    splits = ['train', 'val', 'test'] if cli_args.split == 'all' else [cli_args.split]
    for split in splits:
        get_feats(split=split)
