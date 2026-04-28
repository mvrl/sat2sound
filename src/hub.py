"""Resolve Sat2Sound checkpoints from disk or ``SAT2SOUND_HF_CKPTS_ID`` (default ``MVRL/sat2sound``)."""

import json
import os

HF_CKPTS_REPO: str = os.environ.get("SAT2SOUND_HF_CKPTS_ID", "MVRL/sat2sound")
_REPO_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def resolve_hf_ckpt(path: str) -> str:
    """Return a local path for ``path``; downloads from HF if not a local file."""
    if os.path.isfile(path):
        return path
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(HF_CKPTS_REPO, path, repo_type="model")
        if cached and os.path.isfile(cached):
            return cached
    except Exception:
        pass
    from huggingface_hub import hf_hub_download

    print(f"[sat2sound] '{path}' not found locally; downloading from {HF_CKPTS_REPO} ...")
    return hf_hub_download(HF_CKPTS_REPO, path, repo_type="model")


def load_ckpt_cfg() -> dict:
    """Return expr→ckpt map from ``ckpts/ckpt_cfg.json`` or HF; ``{}`` on failure."""
    local = os.path.join(_REPO_ROOT, "ckpts", "ckpt_cfg.json")
    if os.path.isfile(local):
        with open(local) as fh:
            return json.load(fh)
    try:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(HF_CKPTS_REPO, "ckpt_cfg.json", repo_type="model")
        with open(cached) as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"[sat2sound] Warning: could not load ckpt_cfg.json from HF ({exc}).")
        return {}
