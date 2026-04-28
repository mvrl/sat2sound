import os
import random
from typing import Optional

from datasets import load_dataset
from torch.utils.data import Dataset

from src.config import cfg
from utilities.utils import sat_transform

# Streaming configuration — mirrors src/dataloader priority logic.
_LOCAL_DATA_PATH: str = os.environ.get("SAT2SOUND_LOCAL_DATA", "")
_HF_STREAMING: bool = os.environ.get("SAT2SOUND_HF_STREAMING", "1") == "1"
_USE_STREAMING: bool = (not _LOCAL_DATA_PATH) and _HF_STREAMING
_SHUFFLE_BUFFER: int = int(os.environ.get("SAT2SOUND_SHUFFLE_BUFFER", "1000"))

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_DATASETS_TIMEOUT", "120")

import torch.utils.data as _tud  # noqa: E402
_DatasetBase = _tud.IterableDataset if _USE_STREAMING else _tud.Dataset


_IMAGE_COL_BY_SAT = {
    "bingmap": "bingmap_image",
    "sentinel": "sentinel_image",
    "googleEarth": "googleearth_image",
}

# Zoom levels available per sat type in the HF dataset.
_ZOOM_LEVELS_BY_SAT = {
    "bingmap": [1, 3, 5],
    "sentinel": [1, 3, 5],
    "googleEarth": [1],
}

_SPLIT_ALIASES = {"train": "train", "val": "validation", "test": "test"}

# Columns to load per sat type — skips audio and mel_features (not needed here).
_COLUMNS_BY_SAT = {
    "bingmap": [
        "sample_id", "bingmap_image",
        "llava_caption_bingmap_zl1", "llava_caption_bingmap_zl3", "llava_caption_bingmap_zl5",
    ],
    "sentinel": [
        "sample_id", "sentinel_image",
        "llava_caption_sentinel_zl1", "llava_caption_sentinel_zl3", "llava_caption_sentinel_zl5",
    ],
    "googleEarth": [
        "sample_id", "googleearth_image",
        "llava_caption_googleearth_zl1",
    ],
}


def _load_hf_split(dataset_id: str, split: str, columns: list):
    """Local Arrow first (SAT2SOUND_LOCAL_DATA), else load_dataset."""
    if _LOCAL_DATA_PATH:
        dataset_name = dataset_id.rsplit("/", 1)[-1]
        split_key = _SPLIT_ALIASES.get(split, split)
        local_path = os.path.join(_LOCAL_DATA_PATH, dataset_name, split_key)
        if os.path.isdir(local_path):
            from datasets import load_from_disk
            ds = load_from_disk(local_path)
            present = [c for c in columns if c in ds.column_names]
            return ds.select_columns(present)
    kwargs = {"split": split, "streaming": _HF_STREAMING}
    try:
        ds = load_dataset(dataset_id, **kwargs)
    except (ValueError, KeyError):
        if split in _SPLIT_ALIASES and _SPLIT_ALIASES[split] != split:
            kwargs["split"] = _SPLIT_ALIASES[split]
            ds = load_dataset(dataset_id, **kwargs)
        else:
            raise
    return ds.select_columns(columns)


class Dataset_soundscape(_DatasetBase):
    """Sat2Text dataloader backed by the GeoSound / SoundingEarth HF datasets.

    Only the satellite image and LLaVA caption columns are downloaded —
    audio and mel_features are not needed for the image-to-text task.

    Streaming mode (``SAT2SOUND_HF_STREAMING=1``)
    ----------------------------------------------
    Rows are fetched row-by-row without writing the full split to disk.
    The class becomes a ``torch.utils.data.IterableDataset`` automatically.

    ``dataset_type`` follows the sat2text convention:
      - ``'GeoSound_bingmap'`` or ``'GeoSound_sentinel'``
      - ``'SoundingEarth'``  (GoogleEarth imagery, zoom level 1 only)
    """

    def __init__(
        self,
        split: str = "train",
        sat_input_size: int = 224,
        test_zoom_level: Optional[int] = None,
        dataset_type: str = "GeoSound_bingmap",
    ):
        self.split = split
        self.dataset_type = dataset_type
        self.test_zoom_level = test_zoom_level
        self.sat_input_size = sat_input_size

        if "GeoSound" in dataset_type:
            self.overhead = dataset_type.split("_")[1]
            self.ds = _load_hf_split(
                cfg.hf_geosound_id, split, _COLUMNS_BY_SAT[self.overhead]
            )
        elif dataset_type == "SoundingEarth":
            self.overhead = "googleEarth"
            self.ds = _load_hf_split(
                cfg.hf_soundingearth_id, split, _COLUMNS_BY_SAT["googleEarth"]
            )
        else:
            raise ValueError(
                f"Unsupported dataset_type={dataset_type!r}; "
                "expected 'GeoSound_bingmap', 'GeoSound_sentinel', or 'SoundingEarth'."
            )

        if _USE_STREAMING:
            split_key = _SPLIT_ALIASES.get(split, split)
            try:
                self._len = self.ds.info.splits[split_key].num_examples
            except (TypeError, KeyError, AttributeError):
                self._len = None
            if split == "train":
                self.ds = self.ds.shuffle(buffer_size=_SHUFFLE_BUFFER)

    def __len__(self):
        if _USE_STREAMING:
            if self._len is None:
                raise TypeError("Streaming dataset size unknown (info.splits not available)")
            return self._len
        return len(self.ds)

    def _process_row(self, row):
        zoom_level = random.choice(_ZOOM_LEVELS_BY_SAT[self.overhead])
        if self.test_zoom_level is not None:
            zoom_level = self.test_zoom_level

        sat_tr = sat_transform(
            is_train=self.split == "train",
            input_size=self.sat_input_size,
            sat_type=self.overhead,
            zoom_level=zoom_level,
        )

        pil_image = row[_IMAGE_COL_BY_SAT[self.overhead]]

        if self.overhead == "googleEarth":
            llava_col = "llava_caption_googleearth_zl1"
        else:
            llava_col = f"llava_caption_{self.overhead}_zl{zoom_level}"
        llava_text = row.get(llava_col, "") or ""

        return {
            "sat": sat_tr(pil_image),
            "llava_caption": llava_text,
            "key": row["sample_id"],
            "sat_zoom_level": zoom_level,
        }

    def __getitem__(self, idx):
        return self._process_row(self.ds[idx])

    def __iter__(self):
        for row in self.ds:
            yield self._process_row(row)
