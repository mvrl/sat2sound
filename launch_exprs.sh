#!/bin/bash
set -e

python -m src.train --config configs/sat2sound/bingmap_nometa.yaml
python -m src.train --config configs/sat2sound/bingmap_withmeta.yaml
python -m src.train --config configs/sat2sound/sentinel_nometa.yaml
python -m src.train --config configs/sat2sound/sentinel_withmeta.yaml
python -m src.train --config configs/sat2sound/SoundingEarth_nometa.yaml
python -m src.train --config configs/sat2sound/SoundingEarth_withmeta.yaml

python -m sat2text.train_i2t --config configs/sat2text/bingmap_i2t_baseline.yaml
