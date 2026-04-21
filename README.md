# Sat2Sound

A unified framework for trimodal (satellite image ↔ audio ↔ text) contrastive learning with a shared multimodal codebook. Given an overhead view of a place, Sat2Sound retrieves or generates a plausible soundscape — real audio recordings, LLaVA-written captions, or synthesized audio from those captions.

![Framework Diagram](framework.png)

## Repository layout

```
sat2sound/
├── src/              # trimodal contrastive learning core (train / eval / models)
├── sat2text/         # image ↔ text ablation (no audio branch)
├── utilities/        # shared helpers: sat transforms, MGACLAP mel extraction
├── data_prep/        # offline preprocessing: mel features + LLaVA captions
├── demos/            # Gradio apps: retrieval + attention heatmap
├── .secrets/         # local-only API keys (gitignored)
├── environment.yml   # conda env spec
├── pyproject.toml    # repo is pip-installable as an editable package
└── launch_exprs.sh   # full train + eval examples
```

## Install

```bash
conda env create -f environment.yml
conda activate sat2sound
pip install -e .
```

`pip install -e .` registers `src/`, `sat2text/`, `utilities/`, `data_prep/`, and `demos/` as importable packages so you can do `python -m src.train ...` from the repo root.

## Configure

Paths default to repo-relative subdirectories (`./data`, `./ckpts`, `./logs`). Override any of them via environment variable — see `src/config.py` for the full list:

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

Both steps are optional — the training loop supports both raw-audio and precomputed-mel paths. Skip the mel-precompute step and pass `--precomputed_mel 0` to train, at the cost of slower data loading.

```bash
# 1. Pre-extract MGACLAP mel features (recommended for fast training).
python -m data_prep.compute_mel_features_mgaclap \
    --data_path ./data \
    --metadata_csv ./data/metafiles/GeoSound/train_metadata.csv \
    --metadata_csv ./data/metafiles/GeoSound/val_metadata.csv \
    --metadata_csv ./data/metafiles/GeoSound/test_metadata.csv \
    --out_dir ./data/GeoSound_audio_mel_feats/mgaclap \
    --aporee_metadata ./data/aporee/final_metadata_with_captions.csv

# 2. Generate LLaVA soundscape captions for the satellite imagery.
python -m data_prep.generate_llava_captions \
    --dataset GeoSound --sat_type bingmap \
    --data_path ./data \
    --metadata_csv ./data/metafiles/GeoSound/train_metadata.csv \
    --metadata_csv ./data/metafiles/GeoSound/val_metadata.csv \
    --metadata_csv ./data/metafiles/GeoSound/test_metadata.csv \
    --out_json ./data/metafiles/GeoSound/llava_caption_for_bingmap.json
```

See [`data_prep/README.md`](data_prep/README.md) for details.

## Train

```bash
python -m src.train \
  --max_epochs 20 --batch_size 128 --num_workers 16 \
  --dataset_type GeoSound --sat_type bingmap \
  --audio_encoder_type mgaclap --text_encoder_type flant5 \
  --precomputed_mel 1 \
  --metadata_fusion early --metadata_type latlong_month_time_asource_tsource \
  --shared_codebook 1 --fdt_weight 1 \
  --run_name bingmap_withmeta
```

`launch_exprs.sh` has the full matrix of training + evaluation commands for all sat-type / metadata / dataset variants.

### Raw audio vs precomputed mel

The dataloader supports both paths via `--precomputed_mel`:

- `--precomputed_mel 1` (default): loads pre-extracted `.pth` tensors from `cfg.mel_feats_path`. Produced by `data_prep.compute_mel_features_mgaclap`.
- `--precomputed_mel 0`: loads raw audio at each step and extracts features on-the-fly using `utilities.audio_features.get_audio_feat_mgaclap`. No preprocessing needed, slower training steps.

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

A smaller variant that drops the audio branch and trains satellite-image ↔ caption alignment only, useful as a baseline.

```bash
python -m sat2text.train_i2t --dataset_type GeoSound_bingmap --sat_type bingmap \
  --run_name bingmap_i2t_baseline --caption_type image --sat_scale multi \
  --max_epochs 20 --batch_size 128

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
