"""Repo-wide paths and HF dataset/model IDs; all overridable via env vars."""

import os

from yacs.config import CfgNode as CN


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _env_path(var_name: str, default: str) -> str:
    """Return ``$var_name`` if set, else ``default`` resolved relative to the repo root."""
    value = os.environ.get(var_name)
    if value:
        return value
    return default if os.path.isabs(default) else os.path.join(REPO_ROOT, default)


cfg = CN()

# Data prep only; train/eval use HF dataset IDs directly.
_GEOSOUND_ROOT = os.environ.get(
    "SAT2SOUND_DATA_PATH",
    os.path.join(REPO_ROOT, "data", "GeoSound"),
)

cfg.data_path = _GEOSOUND_ROOT
cfg.metafiles_path = _env_path(
    "SAT2SOUND_METAFILES_PATH", os.path.join(_GEOSOUND_ROOT, "metafiles")
)
cfg.satmae_ckpt_path = _env_path(
    "SATMAE_CKPT_PATH",
    os.path.join("ckpts", "SATMAE", "pretrain-vit-base-e199.pth"),
)

cfg.log_dir = _env_path("SAT2SOUND_LOG_DIR", "logs")
cfg.ignore_ids_geosound = _env_path(
    "SAT2SOUND_IGNORE_IDS_GEOSOUND",
    os.path.join(_GEOSOUND_ROOT, "metafiles", "GeoSound", "ignore_ids_geosound.csv"),
)
cfg.valid_ids_SoundingEarth = _env_path(
    "SAT2SOUND_VALID_IDS_SOUNDINGEARTH",
    os.path.join(_GEOSOUND_ROOT, "metafiles", "SoundingEarth", "valid_ids_SoundingEarth.csv"),
)
cfg.mel_feats_path = os.environ.get(
    "SAT2SOUND_MEL_FEATS_PATH",
    os.path.join(REPO_ROOT, "data", "GeoSound_audio_mel_feats"),
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

# HF dataset IDs — override to point at a fork.
cfg.hf_geosound_id = os.environ.get("SAT2SOUND_HF_GEOSOUND_ID", "MVRL/GeoSound")
cfg.hf_soundingearth_id = os.environ.get("SAT2SOUND_HF_SOUNDINGEARTH_ID", "MVRL/SoundingEarth")
cfg.hf_sat2sound_ckpts_id = os.environ.get("SAT2SOUND_HF_CKPTS_ID", "MVRL/sat2sound")

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


# expr → ckpt path; loaded from local ckpts/ckpt_cfg.json or lazily from HF.
import json as _json

_local_ckpt_cfg = os.path.join(REPO_ROOT, "ckpts", "ckpt_cfg.json")
if os.path.isfile(_local_ckpt_cfg):
    with open(_local_ckpt_cfg) as _fh:
        ckpt_cfg: dict = _json.load(_fh)
else:
    ckpt_cfg: dict = {}
