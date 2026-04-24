import random
from typing import Optional

from datasets import load_dataset
from torch.utils.data import Dataset

from src.config import cfg
from utilities.utils import sat_transform


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
    """Load an HF split, falling back from 'val' → 'validation' if needed."""
    kwargs = {"split": split, "columns": columns}
    try:
        return load_dataset(dataset_id, **kwargs)
    except (ValueError, KeyError):
        if split in _SPLIT_ALIASES and _SPLIT_ALIASES[split] != split:
            kwargs["split"] = _SPLIT_ALIASES[split]
            return load_dataset(dataset_id, **kwargs)
        raise


class Dataset_soundscape(Dataset):
    """Sat2Text dataloader backed by the GeoSound / SoundingEarth HF datasets.

    Only the satellite image and LLaVA caption columns are downloaded —
    audio and mel_features are not needed for the image-to-text task.

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

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]

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
