#!/bin/bash
# Reproduce every table in the Sat2Sound paper (arXiv:2505.13777v2) except
# Tables 3, 10, 11, 15, 16:
#   - Table 3:      human-study soundscape synthesis (not reproducible from code)
#   - Tables 10/11: external off-the-shelf baselines (CLIP/SigLIP, ImageBind/TaxaBind)
#   - Tables 15/16: linear-probing downstream tasks (BirdCLEF, EuroSAT, etc.)
#
# YAML configs in configs/sat2sound/ and configs/sat2text/ cover the *main*
# experiments only (±metadata for each dataset). Ablations are run by loading
# a main config and overriding specific flags on the command line — see
# examples below. Precedence: argparse defaults ← YAML ← CLI flags.
#
# Data: all training/evaluation data streams from HuggingFace (mvrl/GeoSound,
# mvrl/SoundingEarth). No local-file layout is required. Override the HF dataset
# IDs via SAT2SOUND_HF_GEOSOUND_ID / SAT2SOUND_HF_SOUNDINGEARTH_ID env vars.
#
# Paths/data: set SAT2SOUND_DATA_PATH or edit src/config.py.
#
# Prereqs:
#   conda env create -f environment.yml && conda activate sat2sound
#   pip install -e .
#   (offline mel pre-computation, if needed) python -m data_prep.audio_feats_mgaclap

set -e

############################################################
# Main training — six configs cover Table 1 + the models
# reused by Tables 2, 6, 7, 12, 13, 14.
############################################################

python -m src.train --config configs/sat2sound/bingmap_nometa.yaml
python -m src.train --config configs/sat2sound/bingmap_withmeta.yaml
python -m src.train --config configs/sat2sound/sentinel_nometa.yaml
python -m src.train --config configs/sat2sound/sentinel_withmeta.yaml
python -m src.train --config configs/sat2sound/SoundingEarth_nometa.yaml
python -m src.train --config configs/sat2sound/SoundingEarth_withmeta.yaml

# sat2text baseline for Table 2
python -m sat2text.train_i2t --config configs/sat2text/bingmap_i2t_baseline.yaml


############################################################
# Ablation training — reuse a main config and override flags
# on the CLI. Always pass a fresh --run_name so W&B / checkpoints
# don't collide with the base run.
############################################################

# ── Tables 4 & 5: loss ablation on GeoSound-Bing (full metadata) ─────────
# Paper's "trimodal" column = L_tri (Eq 8), always on.
# Paper's  L(a+c)  column ↔ code flag  --combined_modality_loss {0,1}
# Paper's  L(i,t)  column ↔ code flag  --fdt_weight {0, 1.0}
# (bingmap_withmeta above is row 4: combined_modality_loss=1, fdt_weight=1)

# Row 1: trimodal only — disable both auxiliary losses
python -m src.train --config configs/sat2sound/bingmap_withmeta.yaml \
  --combined_modality_loss 0 --fdt_weight 0 \
  --run_name bingmap_trimodal

# Row 2: trimodal + L(i,t)
python -m src.train --config configs/sat2sound/bingmap_withmeta.yaml \
  --combined_modality_loss 0 --fdt_weight 1 \
  --run_name bingmap_trimodal_fdt

# Row 3: trimodal + L(a+c)
python -m src.train --config configs/sat2sound/bingmap_withmeta.yaml \
  --combined_modality_loss 1 --fdt_weight 0 \
  --run_name bingmap_trimodal_comb

# ── Tables 8 & 9: codebook size ablation ─────────────────────────────────
# (bingmap_withmeta above covers the 16000 row via the argparse default)
for cb in 4000 8000 32000; do
  python -m src.train --config configs/sat2sound/bingmap_withmeta.yaml \
    --codebook_size $cb --run_name bingmap_cb$cb
done


############################################################
# Evaluation
#
# evaluate.py writes a JSON line per (expr, metadata_type, zoom_level, composed-setting)
# with composed-setting ∈ {none, audio, query}. Filter the JSON for the column
# you want: Table 1 / non-composed rows = "none"; Composed (audio-only) = "audio";
# Composed (query) = "query".
############################################################

# ── Tables 13 & 14: GeoSound multi-scale composed retrieval ─────────────
# (Tables 1, 13, 14 all fall out of these loops — Table 1 is the zl=1, addtextto=none rows.)
for expr in bingmap_nometa bingmap_withmeta; do
  meta=$([[ $expr == *withmeta ]] && echo latlong_month_time_asource_tsource || echo none)
  for zl in 1 3 5; do
    python -m src.evaluate --save_results true --json_name geosound_bing --expr $expr \
      --dataset_type GeoSound --sat_type bingmap --test_zoom_level $zl --metadata_type $meta
  done
done
for expr in sentinel_nometa sentinel_withmeta; do
  meta=$([[ $expr == *withmeta ]] && echo latlong_month_time_asource_tsource || echo none)
  for zl in 1 3 5; do
    python -m src.evaluate --save_results true --json_name geosound_sentinel --expr $expr \
      --dataset_type GeoSound --sat_type sentinel --test_zoom_level $zl --metadata_type $meta
  done
done

# ── Table 12: SoundingEarth composed retrieval (single scale) ───────────
for expr in SoundingEarth_nometa SoundingEarth_withmeta; do
  meta=$([[ $expr == *withmeta ]] && echo latlong_month_time_tsource || echo none)
  python -m src.evaluate --save_results true --json_name soundingearth --expr $expr \
    --dataset_type SoundingEarth --sat_type googleEarth --test_zoom_level 1 --metadata_type $meta
done

# ── Table 2: Image-Text retrieval at scales {1,3,5} on GeoSound-Bing ────
for zl in 1 3 5; do
  for expr in bingmap_nometa bingmap_withmeta; do
    python -m src.evaluate_text --save_results true --json_name image2text --expr $expr \
      --dataset_type GeoSound --sat_type bingmap --caption_type image --test_zoom_level $zl
  done
  python -m sat2text.evaluate_i2t --save_results true --json_name image2text_baseline \
    --expr bingmap_i2t_baseline --dataset_type GeoSound_bingmap --sat_type bingmap \
    --test_zoom_level $zl
done

# ── Tables 4 & 5: eval the loss-ablation ckpts trained above ────────────
# Table 4 → read addtextto=none rows; Table 5 → read addtextto=audio rows.
for expr in bingmap_trimodal bingmap_trimodal_fdt bingmap_trimodal_comb bingmap_withmeta; do
  python -m src.evaluate --save_results true --json_name loss_ablation --expr $expr \
    --dataset_type GeoSound --sat_type bingmap --test_zoom_level 1 \
    --metadata_type latlong_month_time_asource_tsource
done

# ── Table 6: single-component metadata ablation (eval-time only) ────────
# Reuses the bingmap_withmeta / sentinel_withmeta ckpts; --metadata_type is
# overridden at inference (meta_droprate is forced to 0 inside evaluate.py).
for expr in bingmap_withmeta sentinel_withmeta; do
  sat=${expr%_withmeta}
  for meta in latlong month time asource tsource; do
    python -m src.evaluate --save_results true --json_name meta_single --expr $expr \
      --dataset_type GeoSound --sat_type $sat --test_zoom_level 1 --metadata_type $meta
  done
done

# ── Table 7: cumulative metadata ablation ───────────────────────────────
# Same ckpts; substring-parsed metadata_type builds up from asource.
for expr in bingmap_withmeta sentinel_withmeta; do
  sat=${expr%_withmeta}
  for meta in asource \
             asource_time \
             asource_time_latlong \
             asource_time_latlong_month \
             asource_time_latlong_month_tsource; do
    python -m src.evaluate --save_results true --json_name meta_cumulative --expr $expr \
      --dataset_type GeoSound --sat_type $sat --test_zoom_level 1 --metadata_type $meta
  done
done

# ── Tables 8 & 9: codebook size ablation eval ───────────────────────────
# Table 9 = I2A (evaluate.py); Table 8 = I2T (evaluate_text.py).
for expr in bingmap_cb4000 bingmap_cb8000 bingmap_withmeta bingmap_cb32000; do
  python -m src.evaluate --save_results true --json_name codebook_i2a --expr $expr \
    --dataset_type GeoSound --sat_type bingmap --test_zoom_level 1 \
    --metadata_type latlong_month_time_asource_tsource
  python -m src.evaluate_text --save_results true --json_name codebook_i2t --expr $expr \
    --dataset_type GeoSound --sat_type bingmap --caption_type image --test_zoom_level 1
done
