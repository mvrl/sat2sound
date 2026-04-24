# Sat2Sound

A unified framework for trimodal (satellite image ↔ audio ↔ text) contrastive learning with a shared multimodal codebook. Given an overhead view of a place, Sat2Sound retrieves or generates a plausible soundscape — real audio recordings, LLaVA-written captions, or synthesized audio from those captions.

![Framework Diagram](framework.png)

## Repository layout

```
sat2sound/
├── src/              # trimodal contrastive learning core (train / eval / models)
├── sat2text/         # image ↔ text ablation (no audio branch)
├── configs/          # YAML configs for main experiments (ablations via CLI overrides)
├── utilities/        # shared helpers: sat transforms, MGACLAP mel extraction
├── data_prep/        # offline preprocessing: mel features + LLaVA captions
├── demos/            # Gradio apps: retrieval + attention heatmap
├── .secrets/         # local-only API keys (gitignored)
├── environment.yml   # conda env spec
├── pyproject.toml    # repo is pip-installable as an editable package
└── launch_exprs.sh   # full train + eval recipes for every reproducible table
```

## Install

```bash
conda env create -f environment.yml
conda activate sat2sound
pip install -e .
```

`pip install -e .` registers `src/`, `sat2text/`, `utilities/`, `data_prep/`, and `demos/` as importable packages so you can do `python -m src.train ...` from the repo root.

## Configure

Paths default to repo-relative subdirectories where possible (`./ckpts`, `./logs`); the data root defaults to the author's cluster path and **must** be overridden for a fresh install. Set `SAT2SOUND_DATA_PATH` to your GeoSound data root — see `src/config.py` for the full list of env vars:

```bash
export SAT2SOUND_DATA_PATH=/mnt/big/sat2sound_data
export SAT2SOUND_LOG_DIR=/mnt/big/sat2sound_logs
export SATMAE_CKPT_PATH=/mnt/big/ckpts/SATMAE/pretrain-vit-base-e199.pth
```

Secrets (Bing / Azure Maps API key for tile downloads) live in `./.secrets/`:

```bash
echo "YOUR_BING_MAPS_KEY" > .secrets/bingmap_api.txt
# or:
export BINGMAP_API_KEY=...
```

The `.secrets/` directory is gitignored. See `.secrets/README.md`.

## Data prep

Training and evaluation read exclusively from HuggingFace datasets — no local file layout is required at runtime. Two self-contained datasets cover everything:

| Artifact | Contents | Role |
|---|---|---|
| `mvrl/GeoSound` | 32 kHz audio, Bing + Sentinel imagery, audio caption, LLaVA soundscape captions (all zoom levels), precomputed MGACLAP mel features, metadata | **Primary** — GeoSound dataset |
| `mvrl/SoundingEarth` | 32 kHz audio, GoogleEarth imagery, audio caption, LLaVA soundscape captions (zl=1), precomputed MGACLAP mel features, metadata | **Primary** — SoundingEarth (Heidler et al. 2023 splits) |

Both IDs are overridable via env var (`SAT2SOUND_HF_GEOSOUND_ID`, `SAT2SOUND_HF_SOUNDINGEARTH_ID`) so you can point at a fork or private mirror.

**SoundingEarth ⊂ GeoSound (audio-wise).** The Aporee audio in `mvrl/SoundingEarth` is a subset of GeoSound's Aporee audio, with different (Heidler 10 km geography-aware) splits. GoogleEarth imagery is unique to SoundingEarth.

### Training

```bash
# Loads GeoSound / SoundingEarth from HF (audio, imagery, captions, mel features all included).
python -m src.train --config configs/sat2sound/bingmap_withmeta.yaml
```

**Mel features** (`--precomputed_mel`):

| Value | Behavior | Trade-off |
|---|---|---|
| `1` *(default)* | Load precomputed 5-segment mel stacks from the `mel_features` column of the primary HF dataset row. | Faster training; mel stacks are already embedded in `mvrl/GeoSound` / `mvrl/SoundingEarth`. |
| `0` | Compute mel on-the-fly from the row's `audio` array each step. | No extra storage; slower steps. Useful when swapping the audio encoder. |

### Regenerating the derivative artifacts (for research)

The Sat2Sound paper uses LLaVA-1.5-7B for soundscape captions and MGACLAP for audio features. Both offline scripts are structured so a future VLM or audio encoder can slot in — the output schemas feed the rest of the pipeline unchanged.

Typical flow when swapping the VLM or encoder:

```bash
# 1. Generate LLaVA captions for each sat type.
python -m data_prep.generate_llava_caption_GeoSound --overhead bingmap

python -m data_prep.generate_llava_caption_GeoSound --overhead sentinel

python -m data_prep.generate_llava_caption_SoundingEarth --overhead googleEarth --zoom_level 1

# 2. Pre-compute MGACLAP mel features (GeoSound only — covers SoundingEarth too).
python -m data_prep.audio_feats_mgaclap \
    --data_path ./data/GeoSound \
    --out_dir ./data/GeoSound_audio_mel_feats/mgaclap

# 3. Rebuild the primary HF datasets with the new captions/features and push.
python -m data_prep.build_hf_geosound --out_dir /tmp/geosound --push <your-namespace>/GeoSound
python -m data_prep.build_hf_soundingearth --out_dir /tmp/soundingearth --push <your-namespace>/SoundingEarth

# 4. Point training at the forks:
export SAT2SOUND_HF_GEOSOUND_ID=<your-namespace>/GeoSound
export SAT2SOUND_HF_SOUNDINGEARTH_ID=<your-namespace>/SoundingEarth
```

### Rebuilding the primary HF datasets from raw files

For repo owners regenerating `mvrl/GeoSound` / `mvrl/SoundingEarth` themselves from the source on-disk layout (raw MP3s + JPEG tiles under `./data/<source>/...`):

```bash
# GeoSound — dry-run then full build + push.
python -m data_prep.build_hf_geosound --out_dir /tmp/geosound-tiny --n 500
python -m data_prep.build_hf_geosound --out_dir /tmp/geosound --push mvrl/GeoSound

# SoundingEarth — same pattern.
python -m data_prep.build_hf_soundingearth --out_dir /tmp/se-tiny --n 500
python -m data_prep.build_hf_soundingearth --out_dir /tmp/soundingearth --push mvrl/SoundingEarth
```

Both builders require `huggingface-cli login` before `--push`. They read from `cfg.data_path` / `cfg.metafiles_path` (env-overridable via `SAT2SOUND_DATA_PATH` / `SAT2SOUND_METAFILES_PATH`).

**Note on sat2text**: the sat2text baseline ([`sat2text/`](sat2text/)) uses the same HF-backed dataloader as the main trimodal pipeline.

See [`data_prep/README.md`](data_prep/README.md) for details.

## Train

Training recipes live as YAML in [`configs/sat2sound/`](configs/sat2sound) — one per dataset × ±metadata. Pass a YAML with `--config`; any flag on the CLI overrides the YAML value (precedence is *argparse defaults ← YAML ← CLI*).

```bash
# Main experiment
python -m src.train --config configs/sat2sound/bingmap_withmeta.yaml
```

For ablations, load a main config and override only the flag(s) you want to change. Always pass a fresh `--run_name` so checkpoints and W&B don't collide:

```bash
# Loss ablation: trimodal only (disable both auxiliary losses)
python -m src.train --config configs/sat2sound/bingmap_withmeta.yaml \
  --combined_modality_loss 0 --fdt_weight 0 \
  --run_name bingmap_trimodal

# Codebook size sweep
python -m src.train --config configs/sat2sound/bingmap_withmeta.yaml \
  --codebook_size 4000 --run_name bingmap_cb4000
```

[`launch_exprs.sh`](launch_exprs.sh) has the full set of train + eval commands reproducing every table in the paper except Tables 3, 10, 11, 15, 16 (which require human ratings or external baselines).

### Raw audio vs precomputed mel

The dataloader supports both paths via `--precomputed_mel`:

- `--precomputed_mel 1` (default): reads the precomputed `mel_features` column directly from the `mvrl/GeoSound` / `mvrl/SoundingEarth` HF dataset row (a 5 × n_mels × T float32 stack; one segment is sampled randomly each step).
- `--precomputed_mel 0`: loads the raw `audio` array from the HF row and extracts mel features on-the-fly using `utilities.audio_features.get_audio_feat_mgaclap`. No extra storage needed; slower training steps.

## Evaluate

```bash
# Image-to-audio retrieval
python -m src.evaluate --expr bingmap_withmeta \
  --dataset_type GeoSound --sat_type bingmap --test_zoom_level 3 \
  --metadata_type latlong_month_time_asource_tsource \
  --save_results true --json_name main

# Image-to-text retrieval (the trimodal model's text branch)
python -m src.evaluate_text --expr bingmap_withmeta \
  --dataset_type GeoSound --sat_type bingmap --test_zoom_level 3 \
  --caption_type image --save_results true --json_name image2text
```

## sat2text (image ↔ text ablation)

A smaller variant that drops the audio branch and trains satellite-image ↔ caption alignment only, useful as a baseline. Configs in [`configs/sat2text/`](configs/sat2text).

```bash
python -m sat2text.train_i2t --config configs/sat2text/bingmap_i2t_baseline.yaml

python -m sat2text.evaluate_i2t --dataset_type GeoSound_bingmap --sat_type bingmap \
  --expr bingmap_i2t_baseline --test_zoom_level 3 \
  --save_results true --json_name image2text
```

## Demos

Two Gradio apps — see [`demos/README.md`](demos/README.md).

- **`sat2sound_retrieval.py`** — click a location; retrieve a pre-synthesized soundscape from the gallery. CPU-friendly.
- **`sat2sound_map.py`** — click a location, type or LLaVA-generate a soundscape caption, and render an attention heatmap over the satellite tile for a chosen word.

```bash
export SAT2SOUND_CKPT=/path/to/your_sat2sound.ckpt
export SAT2SOUND_GALLERY=/path/to/gallery.h5   # retrieval demo only
python demos/sat2sound_retrieval.py
python demos/sat2sound_map.py
```

## Checkpoints

The public repo ships no trained weights. Download the published checkpoints with:

```bash
pip install --user -U gdown
gdown --folder https://drive.google.com/drive/folders/1z-ASeAdj6rvKZE-IJuVbWodW9RAZt4DN
```

Point `SAT2SOUND_CKPT` (or the `--ckpt_path` flag on eval/demo scripts) at the downloaded `.ckpt`. The SATMAE backbone weights are loaded from `cfg.satmae_ckpt_path` (env var `SATMAE_CKPT_PATH`).

## License & citation

See `LICENSE`. If you build on this repo, please cite the Sat2Sound paper.
