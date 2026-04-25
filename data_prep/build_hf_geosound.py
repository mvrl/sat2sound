"""Build the GeoSound HuggingFace dataset from the on-disk file layout.

All precomputed artifacts are embedded directly in the dataset:

* Raw 32 kHz audio
* Bing Maps imagery (1500 × 1500) and Sentinel-2 imagery (1280 × 1280) at
  original resolution
* Audio captions (best of CLAP-scored Pengi / Qwen / metadata)
* LLaVA soundscape captions at zoom levels 1, 3, 5 for both sat types
* Precomputed MGACLAP mel spectrogram stacks (5 × 10-second segments)
* Full geo / taxonomic / provenance metadata

The dataset is stored as columnar Parquet (geoparquet style) so callers can
select only the columns they need — imagery and mel are large but optional:

    from datasets import load_dataset
    ds = load_dataset("mvrl/GeoSound", split="train",
                      columns=["sample_id", "audio", "latitude", "longitude"])

Prerequisites
-------------
Before running this builder, the following must be precomputed locally:

* **Mel features** — run ``data_prep/audio_feats_mgaclap.py`` to produce the
  5-segment MGACLAP mel stacks under ``<mel_dir>/<source>/<sample_id>.pth``.
* **LLaVA captions** — run ``data_prep/generate_llava_caption_GeoSound.py``
  (once with ``--overhead bingmap``, once with ``--overhead sentinel``) to
  produce ``llava_caption_for_bingmap.json`` and
  ``llava_caption_for_sentinel.json`` under ``<metafiles_path>/GeoSound/``.

Row schema
----------
``sample_id, source, audio, bingmap_image, sentinel_image,
audio_caption, audio_caption_source,
mel_features,
llava_caption_bingmap_zl1, llava_caption_bingmap_zl3, llava_caption_bingmap_zl5,
llava_caption_sentinel_zl1, llava_caption_sentinel_zl3, llava_caption_sentinel_zl5,
latitude, longitude, date,
description, tags, title, scientific_name, common_name,
sound_format, text, address, original_sampling_rate, bin_id``

Examples
--------
Dry-run 500 samples per split::

    python -m data_prep.build_hf_geosound \\
        --out_dir /tmp/geosound-tiny --n 500

Full build + push (requires prior ``huggingface-cli login``)::

    python -m data_prep.build_hf_geosound \\
        --out_dir /tmp/geosound \\
        --push mvrl/GeoSound --validate

Shard sizing
------------
Shards are sized by ``--max_shard_size`` (default ``"2GB"``). Keep this
well under HF Hub's ~20 GB soft / 50 GB hard per-file limit. Small splits
(val/test) naturally collapse to a single shard if they fit under the cap.
"""

import json
import os
from argparse import ArgumentParser
from typing import Dict, Optional, Tuple

import datasets as hf_datasets
import numpy as np
import pandas as pd
import torch
from datasets import Array4D, Audio, Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage
from tqdm import tqdm

from src.config import cfg
from src.dataloader import meta_columns, resolve_audio_caption


# ---------------------------------------------------------------------------
# Integrity probe (opt-in via --validate)
# ---------------------------------------------------------------------------


def _probe_readable(audio_path: str, image_paths, sample_id: str) -> bool:
    """Header-only audio probe + PIL.verify for images. Logs and returns False on failure."""
    import torchaudio

    try:
        torchaudio.info(audio_path)
    except Exception as exc:
        print(f"[validate] {sample_id}: audio unreadable ({exc}); skipping")
        return False
    for p in image_paths:
        try:
            with PILImage.open(p) as img:
                img.verify()
        except Exception as exc:
            print(f"[validate] {sample_id}: image unreadable ({p}): {exc}; skipping")
            return False
    return True


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def resolve_audio_path(data_path, sample_id, aporee_meta=None, sound_format="mp3"):
    """Resolve ``<source>-<key>`` sample_id to an absolute audio path."""
    source, key = sample_id.split("-", 1)
    if source == "aporee":
        if aporee_meta is None:
            raise ValueError("aporee samples require aporee_meta DataFrame.")
        soundname = aporee_meta[aporee_meta["long_key"] == key].mp3name.item()
        return os.path.join(data_path, source, "raw_audio", str(key), soundname)
    return os.path.join(data_path, source, "raw_audio", f"{key}.{sound_format}")


# ---------------------------------------------------------------------------
# LLaVA caption loader
# ---------------------------------------------------------------------------


def _load_llava_jsonl(path: str) -> Dict[str, Dict[str, str]]:
    """Read a GeoSound LLaVA JSONL → ``{sample_id: {text1, text3, text5}}``.

    Each line must be ``{"sample_id": "...", "captions": {"text1": ..., "text3": ..., "text5": ...}}``.
    Missing file → empty dict (all captions will be empty strings).
    """
    lookup: Dict[str, Dict[str, str]] = {}
    if not os.path.exists(path):
        print(f"[llava] JSONL not found at {path!r}; all captions will be empty strings")
        return lookup
    total = sum(1 for _ in open(path))
    with open(path) as fh:
        for line in tqdm(fh, total=total, desc=f"  {os.path.basename(path)}", unit="lines"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = str(obj["sample_id"])
            caps = obj.get("captions", {})
            if not isinstance(caps, dict):
                caps = {}
            lookup[sid] = {
                "text1": str(caps.get("text1", "")),
                "text3": str(caps.get("text3", "")),
                "text5": str(caps.get("text5", "")),
            }
    return lookup


# ---------------------------------------------------------------------------
# Mel feature helpers
# ---------------------------------------------------------------------------


def _infer_mel_shape(mel_dir: str) -> Tuple[int, int, int]:
    """Load one .pth file from ``mel_dir`` to determine ``(nsamples, n_mels, T)``.

    Uses os.scandir to avoid sorting/listing entire large directories on network filesystems.
    """
    with os.scandir(mel_dir) as top:
        for src_entry in top:
            if not src_entry.is_dir():
                continue
            with os.scandir(src_entry.path) as sub:
                for entry in sub:
                    if entry.name.endswith(".pth"):
                        t = torch.load(entry.path, map_location="cpu", weights_only=True)
                        shape = tuple(int(d) for d in t.shape)
                        if len(shape) >= 3:
                            return shape
    raise RuntimeError(f"No .pth mel files found under {mel_dir!r}")


def _load_mel(
    mel_dir: str, source: str, sample_id: str, zero_shape: Tuple[int, int, int]
) -> np.ndarray:
    """Return mel tensor as float32 numpy array; zero-fill + warn if file missing."""
    pth = os.path.join(mel_dir, source, f"{sample_id}.pth")
    if not os.path.exists(pth):
        print(f"[warn] mel missing for {sample_id}; filling with zeros")
        return np.zeros(zero_shape, dtype=np.float32)
    try:
        return torch.load(pth, map_location="cpu", weights_only=True).float().numpy()
    except Exception as exc:
        print(f"[warn] mel load failed for {sample_id}: {exc}; filling with zeros")
        return np.zeros(zero_shape, dtype=np.float32)


# ---------------------------------------------------------------------------
# Metadata coercion
# ---------------------------------------------------------------------------

SAMPLE_RATE = 32000


def _str_or_empty(v):
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v)


def _int_or_zero(v):
    try:
        if pd.isna(v):
            return 0
        return int(v)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Split loading
# ---------------------------------------------------------------------------


def _load_split_df(split: str, metafiles_path: str, ignore_ids: list) -> pd.DataFrame:
    meta_path = os.path.join(metafiles_path, "GeoSound", f"{split}_metadata.csv")
    df = pd.read_csv(meta_path)
    if split == "test":
        valid_ids = pd.read_csv(
            os.path.join(metafiles_path, "GeoSound", "test_ids_geosound.csv")
        )
        df = df[df["sample_id"].isin(list(valid_ids["sample_id"]))]
    df = df[meta_columns]
    df = df[~df["sample_id"].isin(ignore_ids)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Row generator
# ---------------------------------------------------------------------------


def _row_iter(
    df,
    clap_df,
    pengi_df,
    qwen_df,
    aporee_meta,
    data_path: str,
    mel_dir: str,
    mel_shape: Tuple[int, int, int],
    llava_bing: Dict[str, Dict[str, str]],
    llava_sentinel: Dict[str, Dict[str, str]],
    validate: bool = False,
):
    _empty_caps = {"text1": "", "text3": "", "text5": ""}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  rows", unit="rows"):
        sample = dict(row)
        sample_id = sample["sample_id"]
        source = sample_id.split("-")[0]
        key = sample_id.split("-", 1)[1]

        try:
            caption, caption_source = resolve_audio_caption(
                sample_id, sample, "GeoSound", clap_df, pengi_df, qwen_df
            )
        except Exception as exc:
            print(f"[warn] {sample_id}: caption lookup failed ({exc}); skipping")
            continue

        try:
            audio_path = resolve_audio_path(data_path, sample_id, aporee_meta=aporee_meta)
        except Exception as exc:
            print(f"[warn] {sample_id}: audio path failed ({exc}); skipping")
            continue

        bing_path = os.path.join(data_path, source, "images", "bingmap", f"{key}.jpeg")
        sentinel_path = os.path.join(data_path, source, "images", "sentinel", f"{key}.jpeg")
        missing = [p for p in (audio_path, bing_path, sentinel_path) if not os.path.exists(p)]
        if missing:
            print(f"[warn] {sample_id}: missing {missing[0]}; skipping")
            continue

        if validate and not _probe_readable(audio_path, [bing_path, sentinel_path], sample_id):
            continue

        mel = _load_mel(mel_dir, source, sample_id, mel_shape)
        bing_caps = llava_bing.get(sample_id, _empty_caps)
        sent_caps = llava_sentinel.get(sample_id, _empty_caps)

        yield {
            "sample_id": str(sample_id),
            "source": source,
            "audio": audio_path,
            "bingmap_image": bing_path,
            "sentinel_image": sentinel_path,
            "audio_caption": _str_or_empty(caption),
            "audio_caption_source": _str_or_empty(caption_source),
            "mel_features": mel,
            "llava_caption_bingmap_zl1": bing_caps["text1"],
            "llava_caption_bingmap_zl3": bing_caps["text3"],
            "llava_caption_bingmap_zl5": bing_caps["text5"],
            "llava_caption_sentinel_zl1": sent_caps["text1"],
            "llava_caption_sentinel_zl3": sent_caps["text3"],
            "llava_caption_sentinel_zl5": sent_caps["text5"],
            "latitude": float(sample["latitude"]),
            "longitude": float(sample["longitude"]),
            "date": _str_or_empty(sample.get("date")),
            "description": _str_or_empty(sample.get("description")),
            "tags": _str_or_empty(sample.get("tags")),
            "title": _str_or_empty(sample.get("title")),
            "scientific_name": _str_or_empty(sample.get("scientific_name")),
            "common_name": _str_or_empty(sample.get("common_name")),
            "sound_format": _str_or_empty(sample.get("sound_format")),
            "text": _str_or_empty(sample.get("text")),
            "address": _str_or_empty(sample.get("address")),
            "original_sampling_rate": _int_or_zero(sample.get("original_sampling_rate")),
            "bin_id": _str_or_empty(sample.get("bin_id")),
        }


# ---------------------------------------------------------------------------
# Public build helper
# ---------------------------------------------------------------------------


def build_split(
    split: str,
    limit: Optional[int],
    clap_df,
    pengi_df,
    qwen_df,
    aporee_meta,
    data_path: str,
    metafiles_path: str,
    ignore_ids: list,
    mel_dir: str,
    mel_shape: Tuple[int, int, int],
    llava_bing: Dict,
    llava_sentinel: Dict,
    features: Features,
    validate: bool = False,
    writer_batch_size: int = 100,
) -> Dataset:
    df = _load_split_df(split, metafiles_path, ignore_ids)
    if limit is not None:
        df = df.head(limit)
    print(f"[{split}] building {len(df)} rows (validate={validate}, writer_batch_size={writer_batch_size})")
    return Dataset.from_generator(
        lambda: _row_iter(
            df,
            clap_df,
            pengi_df,
            qwen_df,
            aporee_meta,
            data_path,
            mel_dir,
            mel_shape,
            llava_bing,
            llava_sentinel,
            validate=validate,
        ),
        features=features,
        writer_batch_size=writer_batch_size,
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main():
    parser = ArgumentParser(description="Build the GeoSound HF dataset.")
    parser.add_argument("--out_dir", required=True, help="Local dir for DatasetDict.save_to_disk.")
    parser.add_argument("--n", type=int, default=None, help="Per-split row limit (dry-run).")
    parser.add_argument("--push", default=None, help="HF repo_id to push to (e.g. mvrl/GeoSound).")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--mel_dir",
        default=None,
        help="Root of precomputed MGACLAP mel .pth cache (<source>/<sample_id>.pth). "
             "Defaults to cfg.mel_feats_path/mgaclap.",
    )
    parser.add_argument(
        "--mel_shape",
        default=None,
        nargs=4,
        type=int,
        metavar=("NSAMPLES", "C", "T", "N_MELS"),
        help="Skip mel shape inference by providing it directly, e.g. --mel_shape 5 1 1001 64. "
             "Avoids slow directory scan on large network filesystems.",
    )
    parser.add_argument(
        "--bingmap_jsonl",
        default=None,
        help="GeoSound LLaVA bingmap JSONL. "
             "Defaults to <cfg.metafiles_path>/GeoSound/llava_caption_for_bingmap.json.",
    )
    parser.add_argument(
        "--sentinel_jsonl",
        default=None,
        help="GeoSound LLaVA sentinel JSONL. "
             "Defaults to <cfg.metafiles_path>/GeoSound/llava_caption_for_sentinel.json.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Probe each sample's audio (torchaudio.info) and images (PIL.verify) "
             "before yielding; skip corrupt files. Recommended for the full build.",
    )
    parser.add_argument(
        "--max_shard_size",
        default="50GB",
        help='Target shard size for save_to_disk / push_to_hub (e.g. "500MB", "2GB", "5GB"). '
             "Must stay well under HF Hub's ~20GB soft cap / 50GB hard cap. "
             "Small splits naturally collapse to one shard if they fit.",
    )
    parser.add_argument(
        "--writer_batch_size",
        type=int,
        default=100,
        help="Rows per Arrow write batch during from_generator. Smaller values (~50-200) "
             "avoid a known pyarrow bug where Audio.embed_storage hits "
             "'Mask must be a pyarrow.Array of type boolean' on chunked arrays. "
             "Default 100 is a safe choice for audio+image datasets.",
    )
    args = parser.parse_args()

    # Belt-and-suspenders: also lower the library-wide default batch size in case
    # some internal path bypasses our writer_batch_size kwarg.
    hf_datasets.config.DEFAULT_MAX_BATCH_SIZE = min(
        getattr(hf_datasets.config, "DEFAULT_MAX_BATCH_SIZE", 1000),
        args.writer_batch_size,
    )

    mel_dir = args.mel_dir or os.path.join(cfg.mel_feats_path, "mgaclap")
    bing_jsonl = args.bingmap_jsonl or os.path.join(
        cfg.metafiles_path, "GeoSound", "llava_caption_for_bingmap.json"
    )
    sentinel_jsonl = args.sentinel_jsonl or os.path.join(
        cfg.metafiles_path, "GeoSound", "llava_caption_for_sentinel.json"
    )

    if args.mel_shape:
        mel_shape = tuple(args.mel_shape)
        print(f"Using provided mel shape: {mel_shape}")
    else:
        print(f"Inferring mel shape from {mel_dir} …")
        mel_shape = _infer_mel_shape(mel_dir)
        print(f"  → mel shape: {mel_shape}")

    print("Loading LLaVA captions …")
    llava_bing = _load_llava_jsonl(bing_jsonl)
    llava_sentinel = _load_llava_jsonl(sentinel_jsonl)
    print(f"  → bingmap: {len(llava_bing)}, sentinel: {len(llava_sentinel)} entries")

    features = Features(
        {
            "sample_id": Value("string"),
            "source": Value("string"),
            "audio": Audio(sampling_rate=SAMPLE_RATE),
            "bingmap_image": Image(),
            "sentinel_image": Image(),
            "audio_caption": Value("string"),
            "audio_caption_source": Value("string"),
            "mel_features": Array4D(shape=mel_shape, dtype="float32"),
            "llava_caption_bingmap_zl1": Value("string"),
            "llava_caption_bingmap_zl3": Value("string"),
            "llava_caption_bingmap_zl5": Value("string"),
            "llava_caption_sentinel_zl1": Value("string"),
            "llava_caption_sentinel_zl3": Value("string"),
            "llava_caption_sentinel_zl5": Value("string"),
            "latitude": Value("float32"),
            "longitude": Value("float32"),
            "date": Value("string"),
            "description": Value("string"),
            "tags": Value("string"),
            "title": Value("string"),
            "scientific_name": Value("string"),
            "common_name": Value("string"),
            "sound_format": Value("string"),
            "text": Value("string"),
            "address": Value("string"),
            "original_sampling_rate": Value("int64"),
            "bin_id": Value("string"),
        }
    )

    clap_df = pd.read_csv(os.path.join(cfg.metafiles_path, "GeoSound", "clap_score_geosound.csv"))
    pengi_df = pd.read_json(
        os.path.join(cfg.metafiles_path, "GeoSound", "geosound_audio_caption_pengi.json"),
        lines=True,
    )
    qwen_df = pd.read_json(
        os.path.join(cfg.metafiles_path, "GeoSound", "geosound_audio_caption_qwen.json"),
        lines=True,
    )
    aporee_meta = pd.read_csv(
        os.path.join(cfg.metafiles_path, "SoundingEarth", "final_metadata_with_captions.csv")
    )
    ignore_ids = list(pd.read_csv(cfg.ignore_ids_geosound)["sample_id"])

    dd = DatasetDict(
        {
            split: build_split(
                split,
                args.n,
                clap_df,
                pengi_df,
                qwen_df,
                aporee_meta,
                cfg.data_path,
                cfg.metafiles_path,
                ignore_ids,
                mel_dir,
                mel_shape,
                llava_bing,
                llava_sentinel,
                features,
                validate=args.validate,
                writer_batch_size=args.writer_batch_size,
            )
            for split in args.splits
        }
    )

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Saving DatasetDict to {args.out_dir} with max_shard_size={args.max_shard_size}")
    dd.save_to_disk(args.out_dir, max_shard_size=args.max_shard_size)
    print(dd)

    if args.push:
        print(f"Pushing to hub: {args.push} (max_shard_size={args.max_shard_size})")
        dd.push_to_hub(args.push, max_shard_size=args.max_shard_size)
        print(f"Pushed to https://huggingface.co/datasets/{args.push}")


if __name__ == "__main__":
    main()