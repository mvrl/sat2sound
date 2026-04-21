"""Central configuration.

All paths default to repository-local directories. Override any of them by:

  1. Setting the corresponding environment variable (takes precedence), or
  2. Editing this file for your local setup.

Secrets (e.g. the Bing Maps API key) are loaded from files under ``.secrets/``
and from environment variables — never hardcoded.
"""

import os

from yacs.config import CfgNode as CN


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _env_path(var_name: str, default: str) -> str:
    """Return ``$var_name`` if set, else ``default`` resolved relative to the repo root."""
    value = os.environ.get(var_name)
    if value:
        return value
    return default if os.path.isabs(default) else os.path.join(REPO_ROOT, default)


def _load_bingmap_api_key() -> str:
    """Load the Bing Maps key from ``BINGMAP_API_KEY`` env var or ``.secrets/bingmap_api.txt``.

    Returns the empty string when neither is provided. Only scripts that actually
    download satellite tiles need a key; training/evaluation does not.
    """
    env_key = os.environ.get("BINGMAP_API_KEY", "").strip()
    if env_key:
        return env_key
    secret_path = os.path.join(REPO_ROOT, ".secrets", "bingmap_api.txt")
    try:
        with open(secret_path, "r") as fh:
            return fh.read().strip()
    except (OSError, FileNotFoundError):
        return ""


cfg = CN()

cfg.bingmap_api = _load_bingmap_api_key()

cfg.data_path = _env_path("SAT2SOUND_DATA_PATH", "data")
cfg.metafiles_path = _env_path("SAT2SOUND_METAFILES_PATH", os.path.join("data", "metafiles"))
cfg.satmae_ckpt_path = _env_path("SATMAE_CKPT_PATH", os.path.join("ckpts", "SATMAE", "pretrain-vit-base-e199.pth"))

cfg.log_dir = _env_path("SAT2SOUND_LOG_DIR", "logs")
cfg.ignore_ids_geosound = _env_path(
    "SAT2SOUND_IGNORE_IDS_GEOSOUND",
    os.path.join("data", "metafiles", "GeoSound", "ignore_ids_geosound.csv"),
)
cfg.valid_ids_SoundingEarth = _env_path(
    "SAT2SOUND_VALID_IDS_SOUNDINGEARTH",
    os.path.join("data", "metafiles", "SoundingEarth", "valid_ids_SoundingEarth.csv"),
)
cfg.mel_feats_path = _env_path(
    "SAT2SOUND_MEL_FEATS_PATH",
    os.path.join("data", "GeoSound_audio_mel_feats"),
)
cfg.mgaclap_yml_path = _env_path(
    "MGACLAP_YML_PATH",
    os.path.join("src", "models", "MGACLAP", "inference_example.yaml"),
)
cfg.mgaclap_ckpt_path = _env_path(
    "MGACLAP_CKPT_PATH",
    os.path.join("ckpts", "MGACLAP", "mga-clap.pt"),
)

cfg.results_json = _env_path("SAT2SOUND_RESULTS_JSON", os.path.join("logs", "Results_main.json"))

# Optional region-image downloads (used by map-based soundscape demos).
cfg.usa_bing_images = _env_path(
    "SAT2SOUND_USA_BING_IMAGES",
    os.path.join("data", "region_images", "USA_BINGMAP", "images"),
)
cfg.usa_bing_csv = _env_path(
    "SAT2SOUND_USA_BING_CSV",
    os.path.join("data", "region_images", "USA_BINGMAP", "USA_6KM_grid_bingmap_clean.csv"),
)
cfg.usa_sentinel_images = _env_path(
    "SAT2SOUND_USA_SENTINEL_IMAGES",
    os.path.join("data", "region_images", "USA_SENTINEL", "images"),
)
cfg.usa_sentinel_csv = _env_path(
    "SAT2SOUND_USA_SENTINEL_CSV",
    os.path.join("data", "region_images", "USA_SENTINEL", "USA_6KM_grid_sentinel_clean.csv"),
)


# ``ckpt_cfg`` is an optional map from experiment shorthand (e.g. ``bingmap_withmeta``)
# to a trained checkpoint path. The public repo ships no checkpoints; users fill
# this in locally or pass ``--ckpt_path`` directly to evaluate / demo scripts.
ckpt_cfg = {}
