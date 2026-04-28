"""Download GeoSound and SoundingEarth training columns (no raw ``audio``) to
``SAT2SOUND_LOCAL_DATA``, and cache SatMAE + MGACLAP backbones from HF.
Eval can keep streaming from HF without running this script.
"""

import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from datasets import load_dataset  # noqa: E402

from src.config import cfg  # noqa: E402
from src.hub import resolve_hf_ckpt  # noqa: E402

# ---------------------------------------------------------------------------
# Destination directory
# ---------------------------------------------------------------------------

LOCAL_DATA_PATH: str = os.environ.get(
    "SAT2SOUND_LOCAL_DATA",
    os.path.join(_REPO_ROOT, "data", "local_datasets"),
)

# Training columns — imagery, captions, mel_features, metadata; no raw audio.
GEOSOUND_TRAIN_COLS = [
    "sample_id",
    "source",
    "bingmap_image",
    "sentinel_image",
    "audio_caption",
    "audio_caption_source",
    "mel_features",
    "llava_caption_bingmap_zl1",
    "llava_caption_bingmap_zl3",
    "llava_caption_bingmap_zl5",
    "llava_caption_sentinel_zl1",
    "llava_caption_sentinel_zl3",
    "llava_caption_sentinel_zl5",
    "latitude",
    "longitude",
    "date",
]

SE_TRAIN_COLS = [
    "sample_id",
    "googleearth_image",
    "audio_caption",
    "audio_caption_source",
    "mel_features",
    "llava_caption_googleearth_zl1",
    "latitude",
    "longitude",
    "date_recorded",
]

# HF uses "validation" as the canonical name for the val split.
SPLITS = ["train", "validation", "test"]


def _download_split(hf_id: str, split: str, cols: list, out_path: str) -> int:
    """Skip if already on disk; else download, drop ``audio``, select columns, save to Arrow."""
    if os.path.isdir(out_path):
        # Arrow dataset_info.json present → already downloaded
        info_file = os.path.join(out_path, "dataset_info.json")
        if os.path.isfile(info_file):
            from datasets import load_from_disk
            n = len(load_from_disk(out_path))
            print(f"    [skip] already at {out_path}  ({n:,} rows)")
            return n

    t0 = time.time()
    ds = load_dataset(hf_id, split=split, streaming=False)

    # Drop the raw audio column — mel features are used instead.
    cols_to_drop = [c for c in ["audio"] if c in ds.column_names]
    if cols_to_drop:
        ds = ds.remove_columns(cols_to_drop)

    # Keep only training-needed columns (images, mel, captions, metadata).
    missing = [c for c in cols if c not in ds.column_names]
    if missing:
        print(f"    WARNING: columns not found in {hf_id}/{split}: {missing}")
    present_cols = [c for c in cols if c in ds.column_names]
    ds = ds.select_columns(present_cols)

    os.makedirs(out_path, exist_ok=True)
    ds.save_to_disk(out_path)
    elapsed = time.time() - t0
    print(f"    {len(ds):,} rows → {out_path}  ({elapsed:.0f}s)")
    return len(ds)


def download_dataset(hf_id: str, cols: list, name: str) -> None:
    print(f"\n[{name}]  ←  {hf_id}")
    for split in SPLITS:
        out_path = os.path.join(LOCAL_DATA_PATH, name, split)
        print(f"  {split} ...", end="  ", flush=True)
        _download_split(hf_id, split, cols, out_path)


def download_backbones() -> None:
    print("\n[Backbone weights]  ←  MVRL/sat2sound")
    for rel_path, label in [
        ("backbones/pretrain-vit-base-e199.pth", "SatMAE ViT-Base"),
        ("backbones/mga-clap.pt", "MGACLAP"),
    ]:
        print(f"  {label} ...", end="  ", flush=True)
        local = resolve_hf_ckpt(rel_path)
        print(f"cached at {local}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Sat2Sound — one-time training data download")
    print("=" * 60)
    print(f"Destination: {LOCAL_DATA_PATH}")

    os.makedirs(LOCAL_DATA_PATH, exist_ok=True)

    download_dataset(cfg.hf_geosound_id, GEOSOUND_TRAIN_COLS, "GeoSound")
    download_dataset(cfg.hf_soundingearth_id, SE_TRAIN_COLS, "SoundingEarth")
    download_backbones()

    print("\n" + "=" * 60)
    print("Download complete.")
    print("\nBefore launching training, export:")
    print(f"\n    export SAT2SOUND_LOCAL_DATA={LOCAL_DATA_PATH}")
    print("\nThe dataloader will read from the local Arrow datasets instead of")
    print("streaming from HuggingFace, giving much faster training iteration.")
    print("\nFor evaluation no download is needed — eval_main.sh streams from")
    print("HuggingFace by default (SAT2SOUND_HF_STREAMING=1).")
    print("=" * 60)


if __name__ == "__main__":
    main()
