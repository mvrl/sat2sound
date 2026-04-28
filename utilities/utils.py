"""General-purpose utilities shared across the repo."""

import numpy as np
import torch
from argparse import Namespace
from dateutil import parser
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------


def clean_datetime_str(date):
    if not isinstance(date, str):
        return None
    t = date.lower().strip()
    try:
        if "am" in t or "pm" in t:
            pattern = "am" if "am" in t else "pm"
            i0 = t.find(pattern)
            date = t[: i0 + 2].strip()
        elif "gmt" in t:
            i0 = t.find("gmt")
            date = t[:i0].strip()
        else:
            date = t
    except Exception:
        date = None
    return date


def get_clean_date(original_date_str):
    date_str = clean_datetime_str(original_date_str)
    try:
        return parser.parse(date_str)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Image cropping / transforms
# ---------------------------------------------------------------------------

tile_size = {"sentinel": 256, "bingmap": 300, "googleEarth": 256, "world_sentinel": 256}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def central_crop_bbox(image_width, image_height, crop_width, crop_height):
    """Bounding box for the central crop of an image: ``(left, upper, right, lower)``."""
    left = (image_width - crop_width) // 2
    upper = (image_height - crop_height) // 2
    right = left + crop_width
    lower = upper + crop_height
    return (left, upper, right, lower)


def crop_image(image, zoom_level=1, sat_type="bingmap"):
    """Central-crop a PIL image to ``zoom_level * tile_size[sat_type]`` pixels."""
    crop_size = zoom_level * tile_size[sat_type]
    bbox = central_crop_bbox(
        image_width=image.size[0],
        image_height=image.size[1],
        crop_width=crop_size,
        crop_height=crop_size,
    )
    return image.crop(bbox)


def get_image(image_path, zoom_level=1, sat_type="bingmap"):
    """File-path variant of :func:`crop_image`. Kept for backward compatibility."""
    return crop_image(Image.open(image_path), zoom_level=zoom_level, sat_type=sat_type)


# ---------------------------------------------------------------------------
# Quick-start helpers
# ---------------------------------------------------------------------------


def load_audio_mel(path: str, device=None):
    """Load an audio file and return its MGACLAP mel-spectrogram features.

    The file is resampled to 32 kHz, converted to mono, and a single 10-second
    segment is extracted (from the start of the file).  Use this to populate the
    ``"audio"`` key of the model batch.

    Args:
        path: Path to any audio file readable by ``torchaudio`` (wav, mp3, flac …).
        device: Target device (``None`` → auto-select CUDA when available).

    Returns:
        Mel tensor of shape ``(1, 1001, 64)`` on *device*.  Pass it into the
        batch as::

            "audio": {"input_features": mel.unsqueeze(0).repeat(B, 1, 1, 1)}

    Example — dummy audio (no real file needed):

        >>> import torchaudio, torch
        >>> torchaudio.save("/tmp/demo.wav", torch.randn(1, 320_000), sample_rate=32_000)
        >>> mel = load_audio_mel("/tmp/demo.wav", device)
    """
    import torchaudio
    from utilities.audio_features import get_audio_feat_mgaclap

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    waveform, sr = torchaudio.load(path)
    waveform = waveform.mean(dim=0, keepdim=True)   # → mono (1, T)
    mel = get_audio_feat_mgaclap(waveform, sr, nsamples=1)[0]  # (1, 1001, 64)
    return mel.to(device)


def prepare_batch(
    sat,
    audio_mel,
    audio_caption,
    image_caption,
    latlong=None,
    time_enc=None,
    month_enc=None,
    sat_zoom_level: int = 1,
    device=None,
) -> dict:
    """Assemble the model input dict from pre-processed tensors.

    Args:
        sat: ImageNet-normalised satellite image tensor ``(B, 3, 224, 224)``.
        audio_mel: Mel tensor ``(1, 1001, 64)`` from :func:`load_audio_mel`.
            Tiled to batch size *B* automatically.
        audio_caption: ``(patch_embeds, boolean_mask)`` tuple from
            :func:`encode_text` for the audio/soundscape description.
        image_caption: ``(patch_embeds, boolean_mask)`` tuple from
            :func:`encode_text` for the overhead-image description.
        latlong: ``(B, 4)`` GPS tensor from :func:`encode_gps_time`, or
            ``None`` for ``*_nometa`` checkpoints.
        time_enc: ``(B, 2)`` time tensor from :func:`encode_gps_time`, or
            ``None``.
        month_enc: ``(B, 2)`` month tensor from :func:`encode_gps_time`, or
            ``None``.
        sat_zoom_level: Zoom level for the satellite tile (default 1).
        device: Target device (``None`` → inferred from *sat*).

    Returns:
        Batch dict ready for ``model.get_embeds(batch)``.
    """
    if device is None:
        device = sat.device
    B = sat.shape[0]
    audio_cap_e, audio_cap_m = audio_caption
    img_cap_e,   img_cap_m   = image_caption
    batch = {
        "sat":                 sat,
        "sat_zoom_level":      [sat_zoom_level] * B,
        "audio":               {"input_features": audio_mel.unsqueeze(0).repeat(B, 1, 1, 1)},
        "audio_caption_input": {"patch_embeds": audio_cap_e, "boolean_mask": audio_cap_m},
        "llava_caption_input": {"patch_embeds": img_cap_e,   "boolean_mask": img_cap_m},
        "latlong":        latlong,
        "audio_source":   torch.zeros(B, dtype=torch.long, device=device),
        "caption_source": torch.zeros(B, dtype=torch.long, device=device),
        "time":           time_enc,
        "time_valid":     torch.ones(B, dtype=torch.long, device=device) if time_enc is not None else None,
        "month":          month_enc,
        "month_valid":    torch.ones(B, dtype=torch.long, device=device) if month_enc is not None else None,
    }
    return batch

def load_sat2sound(ckpt_name: str = "bingmap_withmeta", device=None):
    """Load a Sat2Sound model and its text tokenizer.

    Args:
        ckpt_name: Checkpoint stem, e.g. ``"bingmap_withmeta"`` or
                   ``"sentinel_nometa"``.  The full HF path
                   ``sat2sound/{ckpt_name}.ckpt`` is resolved automatically.
        device: ``torch.device`` or ``None`` (auto-selects CUDA when available).

    Returns:
        ``(model, tokenizer)`` — model in eval mode on *device*.
    """
    from transformers import AutoTokenizer
    from src.hub import resolve_hf_ckpt
    from src.engine import sat2soundModel

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(
        resolve_hf_ckpt(f"sat2sound/{ckpt_name}.ckpt"),
        map_location=device,
        weights_only=False,
    )
    model = sat2soundModel(Namespace(**ckpt["hyper_parameters"])).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    return model, tokenizer


def encode_text(texts, tokenizer, device):
    """Tokenize *texts* and return FlanT5 patch embeddings + boolean mask.

    Args:
        texts: List of strings (length B).
        tokenizer: Tokenizer returned by :func:`load_sat2sound`.
        device: Target device.

    Returns:
        ``(patch_embeds, boolean_mask)`` tensors on *device*.
    """
    from src.engine import prepare_flant5_text_embeds

    tok = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    return prepare_flant5_text_embeds(tok, device)


def encode_gps_time(lat: float, lon: float, hour: int, month: int, B: int, device):
    """Encode GPS coordinates and capture time into model-ready tensors.

    All values are mapped to circular (sin/cos) encodings expected by the
    satellite metadata encoder.

    Args:
        lat: Latitude in degrees (−90 … 90).
        lon: Longitude in degrees (−180 … 180).
        hour: Hour of day (0 … 23).
        month: Month of year (1 … 12).
        B: Batch size — the same location/time is tiled *B* times.
        device: Target device.

    Returns:
        ``(latlong, time_enc, month_enc)`` — shapes ``(B, 4)``, ``(B, 2)``,
        ``(B, 2)``.  Pass ``None`` for all three with ``*_nometa`` checkpoints.
    """
    latlong = torch.tensor(
        [[np.sin(np.pi * lat / 90), np.cos(np.pi * lat / 90),
          np.sin(np.pi * lon / 180), np.cos(np.pi * lon / 180)]] * B,
        dtype=torch.float32,
        device=device,
    )
    time_enc = torch.tensor(
        [[np.sin(2 * np.pi * hour / 23), np.cos(2 * np.pi * hour / 23)]] * B,
        dtype=torch.float32,
        device=device,
    )
    month_enc = torch.tensor(
        [[np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)]] * B,
        dtype=torch.float32,
        device=device,
    )
    return latlong, time_enc, month_enc


def sat_transform(is_train=True, input_size=224, sat_type="sentinel", zoom_level=1):
    """Build train/eval transforms for satellite imagery."""
    interpol_mode = transforms.InterpolationMode.BICUBIC

    t = []
    # world_sentinel crops already arrive at 256x256; skip the zoom-dependent crop.
    if sat_type != "world_sentinel":
        zl_size = zoom_level * tile_size[sat_type]
        t.append(transforms.CenterCrop(zl_size))
    t.append(transforms.Resize(input_size, interpolation=interpol_mode, antialias=True))
    if is_train:
        t.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD))
    return transforms.Compose(t)
