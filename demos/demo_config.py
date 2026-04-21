"""Demo-specific configuration.

Every path is resolved from an environment variable with a repo-relative
fallback. Nothing is hardcoded to a specific machine.

Secrets (Bing Maps API key) are loaded by ``src.config`` from
``.secrets/bingmap_api.txt`` or the ``BINGMAP_API_KEY`` env var.

To run a demo, at minimum set:

    SAT2SOUND_CKPT         — path to a trained checkpoint (.ckpt)
    SAT2SOUND_GALLERY      — path to the pre-computed gallery HDF5 (retrieval demo only)
    SATMAE_CKPT_PATH       — path to the SATMAE backbone checkpoint

and put your Bing Maps key in ``.secrets/bingmap_api.txt``.
"""

import os
from types import SimpleNamespace

from src.config import REPO_ROOT


def _resolve(env_name, default_rel_path):
    val = os.environ.get(env_name)
    if val:
        return val
    return os.path.join(REPO_ROOT, default_rel_path)


# Path to the Bing Maps secret file; if you prefer, set BINGMAP_API_KEY in your
# environment and this file won't be read.
bingmap_api = _resolve("BINGMAP_API_FILE", os.path.join(".secrets", "bingmap_api.txt"))

# Model checkpoints and gallery — override with env vars.
sat2sound_ckpt = _resolve("SAT2SOUND_CKPT", os.path.join("ckpts", "sat2sound.ckpt"))
gallery_path = _resolve("SAT2SOUND_GALLERY", os.path.join("data", "demo", "GeoSound_gallery.h5"))

# Where the demo saves uploaded images, logs, and flagged examples.
log_dir = _resolve("SAT2SOUND_DEMO_LOG_DIR", os.path.join("logs", "demos"))

# Default metadata presets fed into the model when the UI does not override them.
metadata_config = SimpleNamespace(
    audio_source="yfcc",
    caption_source="meta",
    time=13,
    month=5,
)
