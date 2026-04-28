# Sat2Sound

Trimodal (satellite image ↔ audio ↔ text) contrastive learning with a shared multimodal codebook.

![Framework Diagram](framework.png)

## Install

```bash
conda env create -f environment.yml
conda activate sat2sound
```

## Datasets

| | Contents |
|---|---|
| [`MVRL/GeoSound`](https://huggingface.co/datasets/MVRL/GeoSound) | Bing + Sentinel imagery, audio, precomputed mel features, LLaVA captions, metadata |
| [`MVRL/SoundingEarth`](https://huggingface.co/datasets/MVRL/SoundingEarth) | GoogleEarth imagery, audio, precomputed mel features, LLaVA captions, metadata |

Raw 32 kHz `audio` is included in both datasets. Training and evaluation use precomputed mel stacks (`mel_features`); see `data_prep/audio_feats_mgaclap.py` for extraction details.

## Evaluate

Streams data from HuggingFace — no download needed. Checkpoints auto-download from [`MVRL/sat2sound`](https://huggingface.co/MVRL/sat2sound).

```bash
bash eval_main.sh                    # image-to-audio retrieval, all 6 checkpoints
bash eval_main.sh bingmap_withmeta   # single expr

bash eval_i2t.sh                     # image-to-text retrieval
bash eval_i2t.sh bingmap_withmeta
```

## Train

**Prerequisites (once):**
```bash
export SAT2SOUND_LOCAL_DATA=/mnt/big/sat2sound_local
python scripts/download_for_training.py   # downloads datasets + backbone weights
```

```bash
python -m src.train --config configs/sat2sound/bingmap_withmeta.yaml
bash launch_exprs.sh   # all 6 sat2sound + sat2text baseline
```

Training configs: `configs/sat2sound/` (6 experiments) and `configs/sat2text/`.

## Quick-start: computing embeddings

```python
import numpy as np
import torch
from argparse import Namespace
from transformers import AutoTokenizer

from src.hub import resolve_hf_ckpt
from src.engine import sat2soundModel, l2normalize, prepare_flant5_text_embeds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B = 4

ckpt = torch.load(resolve_hf_ckpt("sat2sound/bingmap_withmeta.ckpt"), map_location=device, weights_only=False)
model = sat2soundModel(Namespace(**ckpt["hyper_parameters"])).to(device)
model.load_state_dict(ckpt["state_dict"], strict=False)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
def encode_text(texts):
    tok = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    return prepare_flant5_text_embeds(tok, device)

lat, lon, hour, month = 37.77, -122.42, 13, 5
latlong   = torch.tensor([[np.sin(np.pi*lat/90), np.cos(np.pi*lat/90),
                            np.sin(np.pi*lon/180), np.cos(np.pi*lon/180)]] * B, device=device)
time_enc  = torch.tensor([[np.sin(2*np.pi*hour/23),  np.cos(2*np.pi*hour/23)]]  * B, device=device)
month_enc = torch.tensor([[np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)]] * B, device=device)

audio_cap_e, audio_cap_m = encode_text(["Traffic noise and distant birds."] * B)
llava_cap_e, llava_cap_m = encode_text(["An urban intersection with dense buildings."] * B)

batch = {
    "sat":                 torch.randn(B, 3, 224, 224, device=device),  # ImageNet-normalised BingMap tile
    "sat_zoom_level":      [1] * B,
    "audio":               {"input_features": torch.randn(B, 1, 1001, 64, device=device)},  # MGACLAP mel
    "audio_caption_input": {"patch_embeds": audio_cap_e, "boolean_mask": audio_cap_m},
    "llava_caption_input": {"patch_embeds": llava_cap_e, "boolean_mask": llava_cap_m},
    "latlong":        latlong,
    "audio_source":   torch.zeros(B, dtype=torch.long, device=device),
    "caption_source": torch.zeros(B, dtype=torch.long, device=device),
    "time":           time_enc,  "time_valid":  torch.ones(B, dtype=torch.long, device=device),
    "month":          month_enc, "month_valid": torch.ones(B, dtype=torch.long, device=device),
}

with torch.no_grad():
    embeds = model.get_embeds(batch)

sat_emb   = l2normalize(embeds["sat_embeds_dict"]["ctotal"])  # (B, 1024)
audio_emb = l2normalize(embeds["audio_embeds"])               # (B, 1024)
text_emb  = l2normalize(embeds["fdt_txt_embeds"])             # (B, 1024)

print(sat_emb @ audio_emb.T)   # (B, B) satellite ↔ audio cosine similarity
```

Use `None` for all metadata keys with `*_nometa` checkpoints.

## Demos

Satellite tiles from **ESRI World Imagery** — no API key needed.

```bash
export SAT2SOUND_GALLERY=$(python -c "from src.hub import resolve_hf_ckpt; print(resolve_hf_ckpt('demo/GeoSound_gallery_w_bingmap.h5'))")
python -m demos.sat2sound_retrieval   # click location → retrieve soundscape
```

See [`demos/README.md`](demos/README.md).

## Checkpoints

All checkpoints and backbone weights live at [`MVRL/sat2sound`](https://huggingface.co/MVRL/sat2sound) and are auto-downloaded via `src/hub.py:resolve_hf_ckpt`.

| Path in repo | |
|---|---|
| `sat2sound/{bingmap,sentinel,SoundingEarth}_{nometa,withmeta}.ckpt` | Trained checkpoints (6) |
| `sat2text/bingmap_i2t_baseline.ckpt` | Sat2Text baseline |
| `backbones/pretrain-vit-base-e199.pth` | SatMAE backbone |
| `backbones/mga-clap.pt` | MGACLAP backbone |
| `demo/GeoSound_gallery_w_bingmap.h5` | Retrieval demo gallery |

## Env vars

| Variable | Default | Purpose |
|---|---|---|
| `SAT2SOUND_LOCAL_DATA` | — | Local Arrow dataset path (training) |
| `SAT2SOUND_HF_GEOSOUND_ID` | `MVRL/GeoSound` | GeoSound dataset ID |
| `SAT2SOUND_HF_SOUNDINGEARTH_ID` | `MVRL/SoundingEarth` | SoundingEarth dataset ID |
| `SAT2SOUND_HF_CKPTS_ID` | `MVRL/sat2sound` | Checkpoint repo ID |
| `SAT2SOUND_HF_STREAMING` | `1` | Set to `0` after training download |
| `SAT2SOUND_LOG_DIR` | `./logs` | Output directory |

## Citation

Accepted at **EarthVision 2026** (IEEE/ISPRS Workshop on Large Scale Computer Vision for Remote Sensing).

- Paper: [arxiv.org/pdf/2505.13777](https://arxiv.org/pdf/2505.13777)

```bibtex
@inproceedings{khanal2026sat2sound,
  title     = {{Sat2Sound}: A Unified Framework for Zero-Shot Soundscape Mapping},
  author    = {Khanal, Subash and Sastry, Srikumar and Dhakal, Aayush and
               Ahmad, Adeel and Stylianou, Abby and Jacobs, Nathan},
  booktitle = {IEEE/ISPRS Workshop: Large Scale Computer Vision for
               Remote Sensing (EarthVision)},
  year      = {2026},
}
```

## License

See `LICENSE`.
