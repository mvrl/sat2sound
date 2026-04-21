# data_prep

Offline preprocessing steps. Neither is strictly required — both produce caches that speed things up but are not load-bearing for correctness.

All paths below are shown relative to the repo root. Override any of them with your own absolute paths if you prefer.

## 1. Pre-compute MGACLAP mel features

Converts raw audio files into ``.pth`` tensors of stacked 10-second mel segments, which the training dataloader consumes when ``--precomputed_mel 1`` is set. If you skip this step, pass ``--precomputed_mel 0`` to training and features will be extracted on-the-fly (slower per step).

```bash
python -m data_prep.compute_mel_features_mgaclap \
    --data_path ./data \
    --metadata_csv ./data/metafiles/GeoSound/train_metadata.csv \
    --metadata_csv ./data/metafiles/GeoSound/val_metadata.csv \
    --metadata_csv ./data/metafiles/GeoSound/test_metadata.csv \
    --out_dir ./data/GeoSound_audio_mel_feats/mgaclap \
    --aporee_metadata ./data/aporee/final_metadata_with_captions.csv
```

Each ``sample_id`` in the CSV is expected to be formatted as ``<source>-<key>``. Audio is read from ``<data_path>/<source>/raw_audio/<key>.mp3`` (or, for aporee, ``<data_path>/aporee/raw_audio/<key>/<mp3name>``).

The resulting ``.pth`` tensor for each sample has shape ``(5, n_mels, T)`` — five randomly-cropped 10-second mel segments stacked. The dataloader selects one at training time.

## 2. Generate LLaVA soundscape captions

Runs LLaVA-1.5-7B locally (fp16, GPU) over satellite imagery and produces one JSONL record per ``sample_id`` containing captions like *"What types of sounds can we expect to hear..."*.

**GeoSound** (writes captions at zoom levels 1, 3, 5 per sample):

```bash
python -m data_prep.generate_llava_captions \
    --dataset GeoSound \
    --sat_type bingmap \
    --data_path ./data \
    --metadata_csv ./data/metafiles/GeoSound/train_metadata.csv \
    --metadata_csv ./data/metafiles/GeoSound/val_metadata.csv \
    --metadata_csv ./data/metafiles/GeoSound/test_metadata.csv \
    --out_json ./data/metafiles/GeoSound/llava_caption_for_bingmap.json
```

**SoundingEarth** (single zoom level):

```bash
python -m data_prep.generate_llava_captions \
    --dataset SoundingEarth \
    --sat_type googleEarth \
    --zoom_level 1 \
    --data_path ./data/aporee \
    --metadata_csv ./data/aporee/aporee_train_fairsplit_10km.csv \
    --metadata_csv ./data/aporee/aporee_val_fairsplit_10km.csv \
    --metadata_csv ./data/aporee/aporee_test_fairsplit_10km.csv \
    --out_json ./data/metafiles/SoundingEarth/SoundingEarth_llava_caption_googleEarth_zl1.json
```

If HuggingFace has the LLaVA weights behind a gate, run ``huggingface-cli login`` first (or set ``HF_TOKEN`` in your environment). The transformers fallback ``token=True`` will use your cached credentials.
