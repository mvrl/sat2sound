"""Build the SoundingEarth HuggingFace dataset from the on-disk file layout.

Companion to :mod:`data_prep.build_hf_geosound` — produces a standalone HF
dataset for SoundingEarth (Heidler et al. 2023) as packaged for Sat2Sound.

All precomputed artifacts are embedded directly in the dataset:

* Aporee-only audio (same recordings as the Aporee subset of GeoSound; the
  ``aporee-<longkey>`` ``sample_id`` convention is shared so the MGACLAP mel
  cache, computed over GeoSound, covers SoundingEarth automatically)
* GoogleEarth satellite imagery (1024 × 1024)
* Audio captions from GeoSound's CLAP-score winner lookup
* LLaVA soundscape captions (zoom level 1 for GoogleEarth)
* Precomputed MGACLAP mel spectrogram stacks (5 × 10-second segments)

Column-selective loading is supported (geoparquet style):

    from datasets import load_dataset
    ds = load_dataset("mvrl/SoundingEarth", split="test",
                      columns=["sample_id", "audio", "latitude", "longitude"])

Prerequisites
-------------
Before running this builder, the following must be precomputed locally:

* **Mel features** — run ``data_prep/audio_feats_mgaclap.py`` on the GeoSound
  dataset (SoundingEarth is a subset of GeoSound's Aporee audio, so the same
  cache covers both). Mel stacks live under
  ``<mel_dir>/aporee/<sample_id>.pth``.
* **LLaVA captions** — run ``data_prep/generate_llava_caption_SoundingEarth.py``
  with ``--overhead googleEarth --zoom_level 1`` to produce
  ``SoundingEarth_llava_caption_for_googleEarth_zl_1.json`` under
  ``<metafiles_path>/SoundingEarth/``.

Row schema
----------
``sample_id, short_id, audio, googleearth_image,
audio_caption, audio_caption_source,
mel_features, llava_caption_googleearth_zl1,
latitude, longitude, date_recorded``

Examples
--------
Dry-run 500 samples per split::

    python -m data_prep.build_hf_soundingearth --out_dir /tmp/se-tiny --n 500

Full build + push::

    python -m data_prep.build_hf_soundingearth \\
        --out_dir /tmp/soundingearth --push mvrl/SoundingEarth

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
from src.dataloader import resolve_audio_caption


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
# LLaVA caption loader
# ---------------------------------------------------------------------------


def _load_se_llava_jsonl(path: str) -> Dict[str, str]:
    """Read the SoundingEarth GoogleEarth LLaVA JSONL → ``{aporee-<longkey>: text}``.

    Each line is ``{"sample_id": <longkey>, "captions": <string or dict>}``.
    Missing file → empty dict (captions will be empty strings).
    """
    lookup: Dict[str, str] = {}
    if not os.path.exists(path):
        print(f"[llava] JSONL not found at {path!r}; captions will be empty strings")
        return lookup
    total = sum(1 for _ in open(path))
    with open(path) as fh:
        for line in tqdm(fh, total=total, desc=f"  {os.path.basename(path)}", unit="lines"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw_id = str(obj["sample_id"])
            sample_id = raw_id if raw_id.startswith("aporee-") else f"aporee-{raw_id}"
            caps = obj.get("captions", "")
            if isinstance(caps, dict):
                text = str(caps.get("text1", ""))
            else:
                text = str(caps)
            lookup[sample_id] = text
    return lookup


# ---------------------------------------------------------------------------
# Mel feature helpers (same convention as GeoSound: mel_dir/aporee/<sample_id>.pth)
# ---------------------------------------------------------------------------


def _infer_mel_shape(mel_dir: str) -> Tuple[int, int, int]:
    """Load one .pth from ``mel_dir`` to determine ``(nsamples, n_mels, T)``.

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
    mel_dir: str, sample_id: str, zero_shape: Tuple[int, int, int]
) -> np.ndarray:
    """Return mel tensor (float32 numpy) for ``sample_id``; zero-fill + warn if missing.

    SoundingEarth samples are all Aporee, so mel files live under
    ``mel_dir/aporee/<sample_id>.pth`` (e.g. ``aporee/aporee-12345.pth``).
    """
    pth = os.path.join(mel_dir, "aporee", f"{sample_id}.pth")
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


# ---------------------------------------------------------------------------
# Split loading
# ---------------------------------------------------------------------------


def _load_split_df(split: str, metafiles_path: str, valid_ids: list) -> pd.DataFrame:
    """Load SoundingEarth split CSV; apply Heidler valid-ID filter + test narrowing."""
    fname = f"aporee_{split}_fairsplit_10km.csv"
    meta_path = os.path.join(metafiles_path, "SoundingEarth", fname)
    df = pd.read_csv(meta_path)

    if split == "test":
        test_ids = list(
            pd.read_csv(
                os.path.join(metafiles_path, "SoundingEarth", "test_ids_soundingEarth.csv")
            )["sample_id"]
        )
        # test_ids_soundingEarth.csv uses "aporee-" prefix; strip it for the CSV lookup.
        test_ids = [i.replace("aporee-", "") for i in test_ids]
        df = df[df["sample_id"].isin(test_ids)]

    df = df[df["sample_id"].isin(valid_ids)]
    return df.reset_index(drop=True)


def _audio_path(data_path: str, aporee_meta, long_key: str) -> str:
    """Resolve Aporee audio: ``<data_path>/aporee/raw_audio/<long_key>/<mp3name>``."""
    mp3name = aporee_meta[aporee_meta["long_key"] == long_key].mp3name.item()
    return os.path.join(data_path, "aporee", "raw_audio", str(long_key), mp3name)


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
    llava_dict: Dict[str, str],
    validate: bool = False,
):
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  rows", unit="rows"):
        sample = dict(row)
        longkey = sample["sample_id"]  # unprefixed in the CSV
        short_id = sample["key"]
        prefixed_id = f"aporee-{longkey}"

        try:
            caption, caption_source = resolve_audio_caption(
                prefixed_id, sample, "SoundingEarth", clap_df, pengi_df, qwen_df
            )
        except Exception as exc:
            print(f"[warn] {prefixed_id}: caption resolution failed ({exc}); skipping")
            continue

        try:
            audio_path = _audio_path(data_path, aporee_meta, longkey)
        except Exception as exc:
            print(f"[warn] {prefixed_id}: audio path failed ({exc}); skipping")
            continue

        # GoogleEarth imagery uses the short numeric key.
        ge_path = os.path.join(
            data_path, "aporee", "images", "googleEarth", f"{short_id}.jpg"
        )

        missing = [p for p in (audio_path, ge_path) if not os.path.exists(p)]
        if missing:
            print(f"[warn] {prefixed_id}: missing {missing[0]}; skipping")
            continue

        if validate and not _probe_readable(audio_path, [ge_path], prefixed_id):
            continue

        mel = _load_mel(mel_dir, prefixed_id, mel_shape)
        llava_text = llava_dict.get(prefixed_id, "")

        yield {
            "sample_id": prefixed_id,
            "short_id": str(short_id),
            "audio": audio_path,
            "googleearth_image": ge_path,
            "audio_caption": _str_or_empty(caption),
            "audio_caption_source": _str_or_empty(caption_source),
            "mel_features": mel,
            "llava_caption_googleearth_zl1": llava_text,
            "latitude": float(sample["latitude"]),
            "longitude": float(sample["longitude"]),
            "date_recorded": _str_or_empty(sample.get("date_recorded")),
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
    valid_ids: list,
    mel_dir: str,
    mel_shape: Tuple[int, int, int],
    llava_dict: Dict[str, str],
    features: Features,
    validate: bool = False,
    writer_batch_size: int = 100,
) -> Dataset:
    df = _load_split_df(split, metafiles_path, valid_ids)
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
            llava_dict,
            validate=validate,
        ),
        features=features,
        writer_batch_size=writer_batch_size,
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main():
    parser = ArgumentParser(description="Build the SoundingEarth HF dataset.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n", type=int, default=None, help="Per-split row limit (dry-run).")
    parser.add_argument("--push", default=None, help="HF repo_id to push to (e.g. mvrl/SoundingEarth).")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--mel_dir",
        default=None,
        help="Root of precomputed MGACLAP mel .pth cache (<source>/<sample_id>.pth). "
             "Defaults to cfg.mel_feats_path/mgaclap. "
             "SoundingEarth mel files live under the 'aporee/' subdirectory.",
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
        "--googleearth_jsonl",
        default=None,
        help="SoundingEarth LLaVA googleEarth JSONL (zoom level 1). "
             "Defaults to <cfg.metafiles_path>/SoundingEarth/"
             "SoundingEarth_llava_caption_for_googleEarth_zl_1.json.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Probe each sample's audio (torchaudio.info) and image (PIL.verify) "
             "before yielding; skip corrupt files. Recommended for the full build.",
    )
    parser.add_argument(
        "--max_shard_size",
        default="2GB",
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
    ge_jsonl = args.googleearth_jsonl or os.path.join(
        cfg.metafiles_path,
        "SoundingEarth",
        "SoundingEarth_llava_caption_for_googleEarth_zl_1.json",
    )

    if args.mel_shape:
        mel_shape = tuple(args.mel_shape)
        print(f"Using provided mel shape: {mel_shape}")
    else:
        print(f"Inferring mel shape from {mel_dir} …")
        mel_shape = _infer_mel_shape(mel_dir)
        print(f"  → mel shape: {mel_shape}")

    print("Loading LLaVA captions …")
    llava_dict = _load_se_llava_jsonl(ge_jsonl)
    print(f"  → googleEarth zl1: {len(llava_dict)} entries")

    features = Features(
        {
            "sample_id": Value("string"),
            "short_id": Value("string"),
            "audio": Audio(sampling_rate=SAMPLE_RATE),
            "googleearth_image": Image(),
            "audio_caption": Value("string"),
            "audio_caption_source": Value("string"),
            "mel_features": Array4D(shape=mel_shape, dtype="float32"),
            "llava_caption_googleearth_zl1": Value("string"),
            "latitude": Value("float32"),
            "longitude": Value("float32"),
            "date_recorded": Value("string"),
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
    valid_ids = list(pd.read_csv(cfg.valid_ids_SoundingEarth)["sample_id"])

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
                valid_ids,
                mel_dir,
                mel_shape,
                llava_dict,
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