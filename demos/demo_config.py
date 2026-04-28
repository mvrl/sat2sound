"""Demo paths from env vars with repo-relative defaults.

Satellite tiles come from ESRI World Imagery — no API key needed.
``SAT2SOUND_CKPT``: local path or HF-relative (default ``sat2sound/bingmap_withmeta.ckpt``).
``SAT2SOUND_GALLERY``: retrieval demo gallery HDF5.
"""

import os
from types import SimpleNamespace

from src.config import REPO_ROOT


def _resolve(env_name, default_rel_path):
    val = os.environ.get(env_name)
    if val:
        return val
    return os.path.join(REPO_ROOT, default_rel_path)


sat2sound_ckpt: str = os.environ.get("SAT2SOUND_CKPT", "sat2sound/bingmap_withmeta.ckpt")

gallery_path = _resolve("SAT2SOUND_GALLERY", os.path.join("data", "demo", "GeoSound_gallery_w_bingmap.h5"))

# Where the demo saves uploaded images, logs, and flagged examples.
log_dir = _resolve("SAT2SOUND_DEMO_LOG_DIR", os.path.join("logs", "demos"))

# Default metadata presets fed into the model when the UI does not override them.
metadata_config = SimpleNamespace(
    audio_source="yfcc",
    caption_source="meta",
    time=13,
    month=5,
)
