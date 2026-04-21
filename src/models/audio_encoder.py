"""MGACLAP audio encoder wrapper.

MGACLAP's internal modules use flat imports (``from cnns import ...``), so the
containing directory is added to ``sys.path`` below.
"""

import os
import sys

import pytorch_lightning as pl
import torch
import torch.nn as nn
from ruamel import yaml

# Make MGACLAP's flat imports (e.g. ``from cnns import ResNet38``) resolvable.
_MGACLAP_DIR = os.path.join(os.path.dirname(__file__), "MGACLAP")
if _MGACLAP_DIR not in sys.path:
    sys.path.append(_MGACLAP_DIR)

from MGACLAP.ase_model import ASE  # noqa: E402  (import after sys.path extension)

from src.config import cfg  # noqa: E402


def load_mgaclap(device, yaml_path=None):
    if yaml_path is None:
        yaml_path = cfg.mgaclap_yml_path
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    model = ASE(config)
    model.to(device)
    try:
        cp_path = config["eval"]["ckpt"]
        cp = torch.load(cp_path, map_location=device)
    except Exception:
        cp_path = cfg.mgaclap_ckpt_path
        cp = torch.load(cp_path, map_location=device)

    model.load_state_dict(cp["model"], strict=False)
    print("Model weights loaded from {}".format(cp_path))
    return model.audio_encoder


class MGACLAP_audiomodel(pl.LightningModule):
    def __init__(self, yaml_path=None, d_model=512, embed_type="hidden_states", device=torch.device("cpu")):
        super().__init__()
        if yaml_path is None:
            yaml_path = cfg.mgaclap_yml_path
        self.model = load_mgaclap(device, yaml_path=yaml_path)
        self.embed_type = embed_type
        if self.embed_type == "pooled":
            self.projection = nn.Linear(768, d_model)

    def forward(self, audio):
        pooled_embeddings, hidden_embeddings = self.model(audio["input_features"])
        if self.embed_type == "hidden_states":
            return hidden_embeddings
        if self.embed_type == "pooled":
            return self.projection(pooled_embeddings)
        raise ValueError(f"Unsupported embed_type: {self.embed_type!r}")
