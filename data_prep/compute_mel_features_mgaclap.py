"""Pre-compute MGACLAP mel-spectrogram features and cache them to disk.

Runs over a metadata CSV (with a ``sample_id`` column of the form
``<source>-<key>``), loads each raw audio file from ``<data_path>/<source>/raw_audio/``,
extracts 5 randomly-sampled 10s mel segments (stacked), and saves the resulting
tensor as ``<out_dir>/<source>/<sample_id>.pth``. The dataloader then selects
one of the five at training time.

Use this when training with ``--precomputed_mel 1`` (faster training, no
per-step audio decoding). Skip it and use ``--precomputed_mel 0`` to extract
features on-the-fly at the cost of slower training steps.

Example
-------

    python -m data_prep.compute_mel_features_mgaclap \\
        --data_path ./data \\
        --metadata_csv ./data/metafiles/GeoSound/train_metadata.csv \\
        --metadata_csv ./data/metafiles/GeoSound/val_metadata.csv \\
        --metadata_csv ./data/metafiles/GeoSound/test_metadata.csv \\
        --out_dir ./data/GeoSound_audio_mel_feats/mgaclap \\
        --aporee_metadata ./data/aporee/final_metadata_with_captions.csv
"""

import os
from argparse import ArgumentParser

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from utilities.audio_features import get_audio_feat_mgaclap


def resolve_audio_path(data_path, sample_id, aporee_meta=None, sound_format="mp3"):
    """Resolve ``sample_id`` (``<source>-<key>``) to an absolute audio path."""
    source, key = sample_id.split("-", 1)
    if source == "aporee":
        if aporee_meta is None:
            raise ValueError(
                "aporee samples require --aporee_metadata (maps long_key -> mp3name)."
            )
        soundname = aporee_meta[aporee_meta["long_key"] == key].mp3name.item()
        return os.path.join(data_path, source, "raw_audio", str(key), soundname)
    soundname = f"{key}.{sound_format}"
    return os.path.join(data_path, source, "raw_audio", soundname)


def process_sample(sample_id, data_path, out_dir, aporee_meta=None):
    source = sample_id.split("-")[0]
    out_path = os.path.join(out_dir, source, f"{sample_id}.pth")
    if os.path.exists(out_path):
        return

    audio_path = resolve_audio_path(data_path, sample_id, aporee_meta=aporee_meta)
    audio, original_sr = torchaudio.load(audio_path)
    audio = audio.mean(axis=0).unsqueeze(0)
    audio_mel = get_audio_feat_mgaclap(audio, original_sr)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(audio_mel, out_path)


def main():
    parser = ArgumentParser(description="Pre-compute MGACLAP mel features to disk.")
    parser.add_argument("--data_path", required=True, help="Root dir containing <source>/raw_audio/<...>")
    parser.add_argument(
        "--metadata_csv",
        action="append",
        required=True,
        help="Metadata CSV with a sample_id column. Repeat for train/val/test.",
    )
    parser.add_argument("--out_dir", required=True, help="Output directory for <source>/<sample_id>.pth")
    parser.add_argument(
        "--aporee_metadata",
        default=None,
        help="CSV mapping aporee long_key -> mp3name. Required if aporee samples are present.",
    )
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(p) for p in args.metadata_csv], ignore_index=True)
    print(f"Processing {len(df)} samples from {len(args.metadata_csv)} CSV(s).")

    aporee_meta = pd.read_csv(args.aporee_metadata) if args.aporee_metadata else None

    for sample_id in tqdm(df["sample_id"].tolist()):
        try:
            process_sample(sample_id, args.data_path, args.out_dir, aporee_meta=aporee_meta)
        except Exception as exc:
            print(f"[warn] skipping {sample_id}: {exc}")


if __name__ == "__main__":
    main()
