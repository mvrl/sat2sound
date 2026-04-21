# `.secrets/`

Put local credentials here. Everything in this folder is gitignored except this README.

## Files expected

### `bingmap_api.txt`

Your Bing Maps API key on a single line. Used by `demos/sat2sound_retrieval.py`, `demos/sat2sound_map.py`, and any script that downloads satellite imagery via the Bing Maps REST API.

Example content (one line, no quotes, no trailing newline required):

```
YOUR_BING_MAPS_KEY_HERE
```

Get a key at https://www.bingmapsportal.com/.

## Alternative: environment variables

Any secret that can be read from `.secrets/` can also be provided via an environment variable, which takes precedence. For Bing Maps:

```bash
export BINGMAP_API_KEY=...
```

If neither the file nor the env var is set, `cfg.bingmap_api` is left empty. Training and evaluation don't need it — only satellite-tile downloads do.
