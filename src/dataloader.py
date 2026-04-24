"""HuggingFace-backed sat2sound dataloader.

All training / evaluation data comes from two self-contained HuggingFace
datasets that embed every artifact needed at runtime:

* ``cfg.hf_geosound_id``      — GeoSound rows: 32 kHz audio, Bing + Sentinel
  imagery, audio caption, precomputed MGACLAP mel features, LLaVA soundscape
  captions at zoom levels 1/3/5 for each sat type, and full metadata.
* ``cfg.hf_soundingearth_id`` — SoundingEarth rows: same layout for the Aporee
  subset with GoogleEarth imagery and a single (zl=1) LLaVA caption.

Both datasets use columnar Parquet storage so callers can select only the
columns they need (e.g. skip ``mel_features`` or LLaVA columns).

Mel extraction modes (``--precomputed_mel``)
--------------------------------------------
* ``1`` (default) — read the precomputed ``mel_features`` column directly from
  the HF row (reproducibility; the column is a 5 × n_mels × T float32 stack).
* ``0`` — compute a fresh mel on-the-fly from the row's ``audio`` array using
  ``utilities.audio_features.get_audio_feat_mgaclap``. No mel storage needed;
  useful for users who want to swap the audio encoder.
"""

import random
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.config import cfg
from src.models.MGACLAP.feature_extractor import AudioFeature
from utilities.audio_features import get_audio_feat_mgaclap
from utilities.utils import get_clean_date, sat_transform


# ---------------------------------------------------------------------------
# Module-level shared state (lightweight; no data files touched)
# ---------------------------------------------------------------------------

audio_source_map = {"yfcc": 0, "iNat": 1, "aporee": 2, "freesound": 3}
caption_source_map = {"meta": 0, "qwen": 1, "pengi": 2}

# Columns builders carry from the original GeoSound metadata CSV. Exposed here
# so ``data_prep.build_hf_geosound`` can import it.
meta_columns = [
    "sample_id",
    "date",
    "latitude",
    "longitude",
    "description",
    "tags",
    "title",
    "scientific_name",
    "common_name",
    "sound_format",
    "text",
    "address",
    "original_sampling_rate",
    "bin_id",
]

try:
    flant5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
except Exception:
    flant5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large", token=True)

with open(cfg.mgaclap_yml_path, "r") as f:
    _mgaclap_config = yaml.safe_load(f)

mgaclap_feature_extractor = AudioFeature(_mgaclap_config["audio_args"])


# HuggingFace's canonical "validation" split alias (our CLI uses "val").
_SPLIT_ALIASES = {"train": "train", "val": "validation", "test": "test"}

# Primary HF dataset image column per sat_type.
_IMAGE_COL_BY_SAT = {
    "bingmap": "bingmap_image",
    "sentinel": "sentinel_image",
    "googleEarth": "googleearth_image",
}

# LLaVA column name template per sat_type and zoom level.
# GeoSound: llava_caption_{bingmap|sentinel}_zl{1|3|5}
# SoundingEarth: llava_caption_googleearth_zl1  (only zl=1 was generated)
def _llava_col(sat_type: str, zoom_level: int) -> str:
    return f"llava_caption_{sat_type.lower()}_zl{zoom_level}"


def _load_hf_split(dataset_id, split, name=None):
    """Load an HF split, accepting either 'val' or the canonical 'validation'."""
    kwargs = {"split": split}
    if name is not None:
        kwargs["name"] = name
    try:
        return load_dataset(dataset_id, **kwargs)
    except (ValueError, KeyError):
        if split in _SPLIT_ALIASES and _SPLIT_ALIASES[split] != split:
            kwargs["split"] = _SPLIT_ALIASES[split]
            return load_dataset(dataset_id, **kwargs)
        raise


# ---------------------------------------------------------------------------
# Helpers used by the HF dataset builders (data_prep.build_hf_*)
# ---------------------------------------------------------------------------


def resolve_audio_caption(
    sample_id, sample, dataset_type, clap_score_df, pengi_caption_df, qwen_caption_df
):
    """Pick the best audio caption for ``sample_id`` via precomputed CLAP scores.

    Used at *build time* by :mod:`data_prep.build_hf_geosound` and
    :mod:`data_prep.build_hf_soundingearth` — the resolved caption is stored
    directly on the HF row, so training never reruns this.

    Returns ``(caption, caption_source)``.
    """
    caption_source = clap_score_df[clap_score_df["sample_id"] == sample_id]["best_caption"].item()
    if caption_source == "pengi":
        caption = pengi_caption_df[pengi_caption_df["sample_id"] == sample_id]["pengi_caption"].item()
    elif caption_source == "qwen":
        caption = qwen_caption_df[qwen_caption_df["sample_id"] == sample_id]["qwen_caption"].item()
    else:  # "meta"
        if dataset_type == "GeoSound":
            caption = sample["text"]
        else:  # SoundingEarth
            caption = sample["caption"].split("The location of the sound is")[0] + "."
    return caption, caption_source


# ---------------------------------------------------------------------------
# LLaVA captions helper (used by evaluate_text.py)
# ---------------------------------------------------------------------------


def load_llava_caption_df(sat_type: str) -> pd.DataFrame:
    """Return a ``{sample_id, captions}`` DataFrame from the primary HF dataset.

    Loads only the llava caption columns (fast — no audio or imagery
    downloaded). Used by :mod:`src.evaluate_text` to build retrieval caption
    tables.

    ``captions`` is a ``dict`` with keys ``text1 / text3 / text5`` for
    bingmap / sentinel, or ``{text1: ...}`` for googleEarth (only zl=1
    was generated for SoundingEarth).
    """
    if sat_type == "googleEarth":
        col = "llava_caption_googleearth_zl1"
        dfs = []
        for split in ("train", "val", "test"):
            try:
                ds = _load_hf_split(cfg.hf_soundingearth_id, split)
                dfs.append(ds.to_pandas()[["sample_id", col]])
            except Exception as exc:
                warnings.warn(
                    f"load_llava_caption_df: failed to load split={split!r} from "
                    f"{cfg.hf_soundingearth_id!r}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if not dfs:
            raise RuntimeError(
                f"load_llava_caption_df: all splits failed for {cfg.hf_soundingearth_id!r}. "
                "Check HuggingFace auth and dataset availability."
            )
        df = pd.concat(dfs, ignore_index=True).drop_duplicates("sample_id")
        df["captions"] = df[col].apply(lambda t: {"text1": t})
    else:
        sat_lower = sat_type.lower()
        cols = ["sample_id"] + [f"llava_caption_{sat_lower}_zl{z}" for z in [1, 3, 5]]
        dfs = []
        for split in ("train", "val", "test"):
            try:
                ds = _load_hf_split(cfg.hf_geosound_id, split)
                dfs.append(ds.to_pandas()[cols])
            except Exception as exc:
                warnings.warn(
                    f"load_llava_caption_df: failed to load split={split!r} from "
                    f"{cfg.hf_geosound_id!r}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if not dfs:
            raise RuntimeError(
                f"load_llava_caption_df: all splits failed for {cfg.hf_geosound_id!r}. "
                "Check HuggingFace auth and dataset availability."
            )
        df = pd.concat(dfs, ignore_index=True).drop_duplicates("sample_id")
        df["captions"] = df.apply(
            lambda r: {f"text{z}": r[f"llava_caption_{sat_lower}_zl{z}"] for z in [1, 3, 5]},
            axis=1,
        )
    return df[["sample_id", "captions"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def collate_batch(batch, metadata_type="latlong_month_time_asource_tsource"):
    out_dict = {}
    out_dict["key"] = [item["key"] for item in batch]
    out_dict["sat_zoom_level"] = [item["sat_zoom_level"] for item in batch]
    out_dict["sat"] = torch.cat([item["sat"].unsqueeze(0) for item in batch])
    out_dict["audio"] = {
        "input_features": torch.cat(
            [item["audio"]["input_features"].unsqueeze(0) for item in batch]
        )
    }

    audio_captions = [item["audio_caption"] for item in batch]
    llava_captions = [item["llava_caption"] for item in batch]

    out_dict["audio_caption_input"] = flant5_tokenizer(
        audio_captions,
        max_length=flant5_tokenizer.model_max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    out_dict["audio_caption"] = audio_captions
    out_dict["llava_caption_input"] = flant5_tokenizer(
        llava_captions,
        max_length=flant5_tokenizer.model_max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    out_dict["llava_caption"] = llava_captions

    # Metadata — each component included only if mentioned in `metadata_type`.
    if "asource" not in metadata_type:
        out_dict["audio_source"] = None
    else:
        out_dict["audio_source"] = torch.cat([item["audio_source"].unsqueeze(0) for item in batch])

    if "tsource" not in metadata_type:
        out_dict["caption_source"] = None
    else:
        out_dict["caption_source"] = torch.cat(
            [item["caption_source"].unsqueeze(0) for item in batch]
        )

    if "latlong" not in metadata_type:
        out_dict["latlong"] = None
    else:
        out_dict["latlong"] = torch.cat([item["latlong"].unsqueeze(0) for item in batch])

    if "time" not in metadata_type:
        out_dict["time"] = None
        out_dict["time_valid"] = None
    else:
        out_dict["time"] = torch.cat([item["time"].unsqueeze(0) for item in batch])
        out_dict["time_valid"] = torch.cat([item["time_valid"].unsqueeze(0) for item in batch])

    if "month" not in metadata_type:
        out_dict["month"] = None
        out_dict["month_valid"] = None
    else:
        out_dict["month"] = torch.cat([item["month"].unsqueeze(0) for item in batch])
        out_dict["month_valid"] = torch.cat([item["month_valid"].unsqueeze(0) for item in batch])

    return out_dict


# ---------------------------------------------------------------------------
# Canonical dataloader
# ---------------------------------------------------------------------------


class Dataset_soundscape(Dataset):
    """Sat2Sound training dataset backed by the GeoSound or SoundingEarth HF datasets.

    Both primary datasets are self-contained: they carry audio, imagery,
    audio captions, LLaVA captions, precomputed mel features, and metadata.
    No separate HF artifacts are loaded at runtime.

    Mel extraction (``--precomputed_mel``)
    ---------------------------------------
    * ``1`` — read the ``mel_features`` column directly from the HF row
      (a 5 × n_mels × T float32 stack; one segment is sampled randomly).
    * ``0`` — compute mel on-the-fly from ``row["audio"]["array"]``.

    LLaVA captions
    --------------
    Read directly from the corresponding ``llava_caption_<sat>_zl<zoom>``
    column of the HF row — no secondary dataset or local file needed.
    """

    def __init__(self, args, split="test", test_zoom_level=None, test_mel_index=None):
        self.args = args
        self.split = split
        self.test_zoom_level = test_zoom_level
        self.test_mel_index = test_mel_index

        if args.dataset_type == "GeoSound":
            self.ds = _load_hf_split(cfg.hf_geosound_id, split)
        elif args.dataset_type == "SoundingEarth":
            # SoundingEarth uses GoogleEarth imagery; override locally without
            # mutating the caller's args object.
            import copy
            self.args = copy.copy(args)
            self.args.sat_type = "googleEarth"
            self.ds = _load_hf_split(cfg.hf_soundingearth_id, split)
        else:
            raise ValueError(
                f"Unsupported dataset_type={args.dataset_type!r}; "
                "expected 'GeoSound' or 'SoundingEarth'."
            )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        out_dict = {}
        row = self.ds[idx]
        sample_id = row["sample_id"]
        source = row["source"] if self.args.dataset_type == "GeoSound" else "aporee"

        # ── Zoom-level pick ──────────────────────────────────────────────
        if self.args.dataset_type == "GeoSound":
            zoom_level = random.choice([1, 3, 5])
        else:
            zoom_level = 1
        if self.test_zoom_level is not None:
            zoom_level = self.test_zoom_level

        # ── Audio → mel ──────────────────────────────────────────────────
        if not bool(self.args.precomputed_mel):
            audio_np = row["audio"]["array"]
            sr = row["audio"]["sampling_rate"]
            audio = torch.as_tensor(audio_np, dtype=torch.float32).unsqueeze(0)
            mel = get_audio_feat_mgaclap(audio, sr, nsamples=1)[0]
        else:
            stack = np.asarray(row["mel_features"], dtype=np.float32)  # (5, 1, T, n_mels)
            sel = random.choice([0, 1, 2, 3, 4])
            if self.test_mel_index is not None:
                sel = self.test_mel_index
            mel = torch.as_tensor(stack[sel])
        out_dict["audio"] = {"input_features": mel}

        # ── Satellite image ──────────────────────────────────────────────
        pil_image = row[_IMAGE_COL_BY_SAT[self.args.sat_type]]
        sat_tr = sat_transform(
            is_train=self.split == "train",
            input_size=self.args.sat_input_size,
            sat_type=self.args.sat_type,
            zoom_level=zoom_level,
        )
        out_dict["sat"] = sat_tr(pil_image)
        out_dict["sat_zoom_level"] = zoom_level

        # ── LLaVA image caption (from primary HF row column) ─────────────
        llava_text = row.get(_llava_col(self.args.sat_type, zoom_level), "") or ""
        out_dict["llava_caption"] = llava_text

        # ── Audio caption ────────────────────────────────────────────────
        out_dict["audio_caption"] = row["audio_caption"]

        # ── Metadata encoding ────────────────────────────────────────────
        out_dict["key"] = sample_id
        lat = row["latitude"]
        lng = row["longitude"]
        latlong_encode = torch.tensor(
            [
                np.sin(np.pi * lat / 90),
                np.cos(np.pi * lat / 90),
                np.sin(np.pi * lng / 180),
                np.cos(np.pi * lng / 180),
            ]
        ).float()

        date_str = row["date"] if self.args.dataset_type == "GeoSound" else row["date_recorded"]
        # Guard against NaN / non-string values that can appear in optional date fields.
        if not isinstance(date_str, str):
            date_str = None
        date = get_clean_date(date_str) if date_str else None

        if source == "freesound":
            time_encode = torch.tensor([0.0, 0.0]).float()
            time_valid = torch.tensor(False).long()
        else:
            if date is not None:
                time_encode = torch.tensor(
                    [np.sin(2 * np.pi * date.hour / 23), np.cos(2 * np.pi * date.hour / 23)]
                ).float()
                time_valid = torch.tensor(True).long()
            else:
                time_encode = torch.tensor([0.0, 0.0]).float()
                time_valid = torch.tensor(False).long()

        if date is not None:
            month_encode = torch.tensor(
                [np.sin(2 * np.pi * date.month / 12), np.cos(2 * np.pi * date.month / 12)]
            ).float()
            month_valid = torch.tensor(True).long()
        else:
            month_encode = torch.tensor([0.0, 0.0]).float()
            month_valid = torch.tensor(False).long()

        caption_source = row["audio_caption_source"]
        if "asource" in self.args.metadata_type:
            out_dict["audio_source"] = torch.tensor(audio_source_map[source]).long()
        if "tsource" in self.args.metadata_type:
            out_dict["caption_source"] = torch.tensor(caption_source_map[caption_source]).long()
        if "latlong" in self.args.metadata_type:
            out_dict["latlong"] = latlong_encode
        if "time" in self.args.metadata_type:
            out_dict["time"] = time_encode
            out_dict["time_valid"] = time_valid
        if "month" in self.args.metadata_type:
            out_dict["month"] = month_encode
            out_dict["month_valid"] = month_valid

        return out_dict
