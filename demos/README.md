# Demos

One interactive Gradio app. Run from the repo root with `python -m`.

---

## `sat2sound_retrieval` — click a location → retrieve a matching soundscape

Satellite tiles from **ESRI World Imagery** — no API key needed.

```bash
export SAT2SOUND_GALLERY=$(python -c "from src.hub import resolve_hf_ckpt; print(resolve_hf_ckpt('demo/GeoSound_gallery_w_bingmap.h5'))")
python -m demos.sat2sound_retrieval
```

The checkpoint (`sat2sound/bingmap_withmeta.ckpt`) auto-downloads from [`MVRL/sat2sound`](https://huggingface.co/MVRL/sat2sound) on first run. Override via `SAT2SOUND_CKPT`.
