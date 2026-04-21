# Demos

Two interactive Gradio apps.

## Common setup

1. Put a Bing / Azure Maps key in `./.secrets/bingmap_api.txt` (single line, no quotes) or export `BINGMAP_API_KEY` in your shell. See `./.secrets/README.md`.
2. Provide a trained Sat2Sound checkpoint. Either place it at `./ckpts/sat2sound.ckpt` (the default) or export `SAT2SOUND_CKPT=/path/to/your.ckpt`.
3. The demos share state and wiring via `demos/demo_config.py`, which resolves every path from a corresponding env var with a repo-relative fallback.

All Gradio apps print a local URL and a `share=True` tunnelled URL on launch.

## 1. `sat2sound_retrieval.py` — retrieve synthetic sound from gallery

Click a location on the Folium world map. The app downloads a satellite tile at that lat/lon, runs the Sat2Sound model, cosine-matches the resulting embedding against pre-computed text embeddings in the gallery HDF5, and plays the pre-synthesized audio stored alongside the top-1 caption.

```bash
export SAT2SOUND_CKPT=/path/to/your_sat2sound.ckpt
export SAT2SOUND_GALLERY=/path/to/GeoSound_gallery.h5
python demos/sat2sound_retrieval.py
```

The gallery HDF5 is expected to contain: `sample_id`, `audio_raw`, `audio_caption`, `audio_embedding`, and per-zoom `llava_caption_zl{1,3,5}` + `text_embedding_zl{1,3,5}` + `synth_audio_zl{1,3,5}`.

Runs on CPU (fast retrieval, no generation at inference time).

## 2. `sat2sound_map.py` — fine-grained attention heatmap for a soundscape query

Click a location, type (or LLaVA-generate) a soundscape description, pick one word or underscore-joined phrase, and the app renders a heatmap overlay showing where in the tile the model attends for that word.

```bash
export SAT2SOUND_CKPT=/path/to/your_sat2sound.ckpt
python demos/sat2sound_map.py
```

If you check the *"Generate caption with LLaVA"* box, the app downloads and runs `llava-hf/llava-1.5-7b-hf` locally on the fly — that path needs a GPU with ~16 GB VRAM. Without it, the demo runs on CPU using whatever text you typed.

## Troubleshooting

- **`FileNotFoundError: Sat2Sound checkpoint not found`** — set `SAT2SOUND_CKPT`.
- **Tile download fails** — verify the Bing/Azure Maps key works with a manual curl, and check your quota.
- **Gated HuggingFace models** — `huggingface-cli login` so the `token=True` fallback in `transformers` can use your cached credentials.
