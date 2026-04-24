# data_prep

Offline preprocessing scripts for GeoSound and SoundingEarth. Steps 1 and 2
produce caches consumed by the training dataloader; step 3 assembles the
HuggingFace datasets that are pushed to the Hub.

All data is expected under the GeoSound root:

```
/projects/bdbk/subashk/data/data_raw/GeoSound/
├── aporee/
│   ├── raw_audio/<long_key>/<mp3name>
│   └── images/{bingmap,googleEarth,sentinel,sentinel_geoclap}/
├── freesound/
│   ├── raw_audio/<key>.mp3
│   └── images/{bingmap,sentinel}/
├── iNat/
│   ├── raw_audio/<key>.mp3
│   └── images/{bingmap,sentinel}/
├── yfcc/
│   ├── raw_audio/<key>.mp3
│   └── images/{bingmap,sentinel}/
└── metafiles/
    ├── GeoSound/
    │   ├── {train,val,test}_metadata.csv
    │   ├── test_ids_geosound.csv
    │   ├── ignore_ids_geosound.csv
    │   ├── clap_score_geosound.csv
    │   ├── geosound_audio_caption_{pengi,qwen}.json
    │   ├── llava_caption_for_{bingmap,sentinel}.json   ← produced by step 2
    └── SoundingEarth/
        ├── aporee_{train,val,test}_fairsplit_10km.csv
        ├── final_metadata_with_captions.csv
        ├── valid_ids_SoundingEarth.csv
        ├── test_ids_soundingEarth.csv
        └── SoundingEarth_llava_caption_for_googleEarth_zl_1.json  ← produced by step 2
```

Precomputed mel features are written to a sibling directory:

```
/projects/bdbk/subashk/data/data_raw/GeoSound_audio_mel_feats/mgaclap/
└── <source>/<sample_id>.pth    (e.g. aporee/aporee-10001_11943.pth)
```

Path defaults are centralised in `src/config.py` and can be overridden via
environment variables (see that file for the full list).

---

## 1. Pre-compute MGACLAP mel features

`audio_feats_mgaclap.py` converts raw audio into `.pth` tensors of stacked
10-second mel segments. The training dataloader uses these when
`--precomputed_mel 1` is set; pass `--precomputed_mel 0` to extract on-the-fly
instead (slower).

```bash
python -m data_prep.audio_feats_mgaclap
```

The script processes all three splits (train / val / test) in sequence. Each
`sample_id` is formatted as `<source>-<key>`; audio is resolved as:

- Non-aporee: `<GeoSound_root>/<source>/raw_audio/<key>.mp3`
- Aporee: `<GeoSound_root>/aporee/raw_audio/<long_key>/<mp3name>` (mp3name
  looked up in `metafiles/SoundingEarth/final_metadata_with_captions.csv`)

Output `.pth` tensors have shape `(5, n_mels, T)` — five randomly-cropped
10-second mel segments stacked. The dataloader selects one segment per step.

---

## 2. Generate LLaVA soundscape captions

Both scripts run LLaVA-1.5-7B locally (fp16, GPU) over satellite imagery and
append one JSONL record per sample. The prompt is:

> *"What types of sounds can we expect to hear from the location captured by
> this aerial view image? Describe in up to two sentences."*

If the model weights are gated on HuggingFace, run `huggingface-cli login`
first (or export `HF_TOKEN`).

### GeoSound — bingmap and sentinel (zoom levels 1, 3, 5)

Run once per overhead type. Each record contains captions for three central
crops of the image (zoom level 1 = 1 tile, 3 = 3×3 tiles, 5 = 5×5 tiles).

```bash
# Bingmap imagery (cuda:0)
python -m data_prep.generate_llava_caption_GeoSound --overhead bingmap

# Sentinel imagery (cuda:1)
python -m data_prep.generate_llava_caption_GeoSound --overhead sentinel
```

Output written to `metafiles/GeoSound/llava_caption_for_{bingmap,sentinel}.json`.

### SoundingEarth — GoogleEarth (zoom level 1)

```bash
python -m data_prep.generate_llava_caption_SoundingEarth \
    --overhead googleEarth --zoom_level 1
```

Output written to
`metafiles/SoundingEarth/SoundingEarth_llava_caption_for_googleEarth_zl_1.json`.

---

## 3. Build HuggingFace datasets

Both builders embed all precomputed artifacts (audio, imagery, captions, mel
features, metadata) into columnar Parquet shards. Run steps 1 and 2 first.

### GeoSound

```bash
# Dry-run — 500 rows per split, saved locally
python -m data_prep.build_hf_geosound --out_dir /tmp/geosound-tiny --n 500

# Full build, validate every file, save locally
python -m data_prep.build_hf_geosound \
    --out_dir /tmp/geosound \
    --validate

# Full build + push to Hub (requires prior huggingface-cli login)
python -m data_prep.build_hf_geosound \
    --out_dir /tmp/geosound \
    --validate \
    --push mvrl/GeoSound
```

Row schema: `sample_id, source, audio, bingmap_image, sentinel_image,
audio_caption, audio_caption_source, mel_features,
llava_caption_bingmap_zl{1,3,5}, llava_caption_sentinel_zl{1,3,5},
latitude, longitude, date, description, tags, title,
scientific_name, common_name, sound_format, text, address,
original_sampling_rate, bin_id`

### SoundingEarth

```bash
# Dry-run
python -m data_prep.build_hf_soundingearth --out_dir /tmp/se-tiny --n 500

# Full build + push
python -m data_prep.build_hf_soundingearth \
    --out_dir /tmp/soundingearth \
    --validate \
    --push mvrl/SoundingEarth
```

Row schema: `sample_id, short_id, audio, googleearth_image,
audio_caption, audio_caption_source, mel_features,
llava_caption_googleearth_zl1, latitude, longitude, date_recorded`

### Selective column loading

Both datasets support loading only the columns you need:

```python
from datasets import load_dataset

ds = load_dataset("mvrl/GeoSound", split="train",
                  columns=["sample_id", "audio", "latitude", "longitude"])
```
