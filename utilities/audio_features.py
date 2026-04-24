"""MGACLAP mel-feature extraction helpers.

These functions are used both at runtime (by the dataloader when
``--precomputed_mel 0``) and offline (by ``data_prep.audio_feats_mgaclap``
to cache features to disk).

The AST and CLAP feature-extraction variants that lived alongside this file in
the dev repo were not used by the final experiments and have been removed from
the public release.
"""

import random

import torch
import torchaudio
import yaml

from src.config import cfg
from src.models.MGACLAP.feature_extractor import AudioFeature


SAMPLE_RATE = 32000

with open(cfg.mgaclap_yml_path, "r") as f:
    _mgaclap_config = yaml.safe_load(f)

_feature_extractor = AudioFeature(_mgaclap_config["audio_args"])


def sample_10s_audio(audio, original_sr, sr=SAMPLE_RATE, T=10, rand=True):
    """Resample to ``sr`` and return a ``T``-second segment (cropped or zero-padded).

    ``rand=True`` picks a random start; ``rand=False`` uses the first segment
    (useful for deterministic demo / eval loads).
    """
    if original_sr != sr:
        audio = torchaudio.transforms.Resample(original_sr, sr)(audio)

    target_length = sr * T
    if audio.shape[1] >= target_length:
        start = random.randint(0, audio.shape[1] - target_length) if rand else 0
        audio_sample = audio[:, start : start + target_length]
    else:
        padding = target_length - audio.shape[1]
        audio_sample = torch.nn.functional.pad(audio, (0, padding))

    return audio_sample


def get_audio_feat_mgaclap(audio, original_sr, nsamples=5, rand=True):
    """Extract ``nsamples`` 10-second mel-spectrogram features and stack them."""
    audios = [
        sample_10s_audio(audio, original_sr, sr=SAMPLE_RATE, rand=rand)
        for _ in range(nsamples)
    ]
    stacked_audios = torch.cat(audios, dim=0)
    return _feature_extractor(stacked_audios)
