#!/bin/bash
# Example training and evaluation commands for the public sat2sound repo.
# Paths are repo-relative; override via env vars (SAT2SOUND_DATA_PATH, etc.) or
# edit src/config.py for your own layout.
#
# Prereqs:
#   conda env create -f environment.yml && conda activate sat2sound
#   pip install -e .
#   (data_prep) python -m data_prep.compute_mel_features_mgaclap ...   [for --precomputed_mel 1]

set -e

############################################################
# Train the full trimodal model (sat + audio + text)
############################################################

TRAIN_BASE="python -m src.train \
  --max_epochs 20 --num_workers 16 --batch_size 128 \
  --val_check_interval 500 --limit_val_batches 35 --sat_input_size 224 \
  --wandb_mode online --mode train \
  --caption_type audio_image --shared_codebook 1 \
  --audio_encoder_type mgaclap --text_encoder_type flant5 \
  --img_caption_zl all --warm_up_iterations 5000 --text_qmap 1 \
  --fdt_weight 1 --precision full \
  --precomputed_mel 1 \
  --metadata_fusion early --combined_modality_loss 1 --loss_weights unequal \
  --use_combined_projectors 0 --combine_image_text 0 \
  --project_name sat2sound"

# GeoSound / BingMap / no metadata
$TRAIN_BASE --dataset_type GeoSound --sat_type bingmap --metadata_type none --run_name bingmap_nometa

# GeoSound / BingMap / full metadata (lat/long, month, time, audio source, text source)
$TRAIN_BASE --dataset_type GeoSound --sat_type bingmap \
  --metadata_type latlong_month_time_asource_tsource --run_name bingmap_withmeta

# GeoSound / Sentinel variants
$TRAIN_BASE --dataset_type GeoSound --sat_type sentinel --metadata_type none --run_name sentinel_nometa
$TRAIN_BASE --dataset_type GeoSound --sat_type sentinel \
  --metadata_type latlong_month_time_asource_tsource --run_name sentinel_withmeta

# SoundingEarth / googleEarth variants
TRAIN_BASE_SE="python -m src.train \
  --max_epochs 20 --num_workers 16 --batch_size 128 \
  --val_check_interval 500 --limit_val_batches 25 --sat_input_size 224 \
  --wandb_mode online --mode train \
  --caption_type audio_image --shared_codebook 1 \
  --audio_encoder_type mgaclap --text_encoder_type flant5 \
  --img_caption_zl all --warm_up_iterations 5000 --text_qmap 1 \
  --fdt_weight 1 --precision full \
  --precomputed_mel 1 \
  --metadata_fusion early --combined_modality_loss 1 --loss_weights unequal \
  --use_combined_projectors 0 --combine_image_text 0 \
  --project_name sat2sound"

$TRAIN_BASE_SE --dataset_type SoundingEarth --sat_type googleEarth --metadata_type none \
  --run_name SoundingEarth_nometa
$TRAIN_BASE_SE --dataset_type SoundingEarth --sat_type googleEarth \
  --metadata_type latlong_month_time_tsource --run_name SoundingEarth_withmeta


############################################################
# Evaluate image-to-audio retrieval
############################################################

for zl in 1 3 5; do
  python -m src.evaluate --save_results true --json_name main --expr bingmap_nometa \
    --dataset_type GeoSound --sat_type bingmap --test_zoom_level $zl --metadata_type none
  python -m src.evaluate --save_results true --json_name main --expr bingmap_withmeta \
    --dataset_type GeoSound --sat_type bingmap --test_zoom_level $zl \
    --metadata_type latlong_month_time_asource_tsource
done

python -m src.evaluate --save_results true --json_name main --expr SoundingEarth_nometa \
  --dataset_type SoundingEarth --sat_type googleEarth --test_zoom_level 1 --metadata_type none
python -m src.evaluate --save_results true --json_name main --expr SoundingEarth_withmeta \
  --dataset_type SoundingEarth --sat_type googleEarth --test_zoom_level 1 \
  --metadata_type latlong_month_time_tsource


############################################################
# Evaluate image-to-text retrieval (trimodal model, caption side)
############################################################

for zl in 1 3 5; do
  python -m src.evaluate_text --dataset_type GeoSound --sat_type bingmap \
    --expr bingmap_withmeta --caption_type image \
    --save_results true --json_name image2text --test_zoom_level $zl
  python -m src.evaluate_text --dataset_type GeoSound --sat_type bingmap \
    --expr bingmap_nometa --caption_type image \
    --save_results true --json_name image2text --test_zoom_level $zl
done


############################################################
# Train + evaluate the sat2text (image-text only) ablation
############################################################

python -m sat2text.train_i2t \
  --max_epochs 20 --num_workers 16 --batch_size 128 \
  --val_check_interval 500 --limit_val_batches 40 --sat_input_size 224 \
  --wandb_mode online --mode train --caption_type image \
  --dataset_type GeoSound_bingmap --sat_type bingmap \
  --run_name bingmap_i2t_baseline --sat_scale multi \
  --warm_up_iterations 5000 --text_qmap 1 --precision full

for zl in 1 3 5; do
  python -m sat2text.evaluate_i2t --dataset_type GeoSound_bingmap --sat_type bingmap \
    --expr bingmap_i2t_baseline --save_results true --json_name image2text --test_zoom_level $zl
done
