#!/bin/bash
# Usage: bash eval_i2t.sh [expr]   # omit expr to run all
set -e
FILTER="${1:-all}"
PY="python -m"
JSON="i2t"
JSON_BASELINE="i2t_baseline"

run_sat2sound_i2t() {
    local expr=$1; shift
    local zoom_levels=("$@")
    [[ "$FILTER" != "all" && "$FILTER" != "$expr" ]] && return
    for zl in "${zoom_levels[@]}"; do
        $PY src.evaluate_text \
            --expr "$expr" --dataset_type GeoSound --sat_type bingmap \
            --caption_type image --test_zoom_level "$zl" \
            --save_results true --json_name "$JSON"
    done
}

run_sat2text_baseline() {
    local expr=$1; shift
    local zoom_levels=("$@")
    [[ "$FILTER" != "all" && "$FILTER" != "$expr" ]] && return
    for zl in "${zoom_levels[@]}"; do
        $PY sat2text.evaluate_i2t \
            --expr "$expr" --dataset_type GeoSound_bingmap --sat_type bingmap \
            --test_zoom_level "$zl" \
            --save_results true --json_name "$JSON_BASELINE"
    done
}

run_sat2sound_i2t bingmap_nometa   1 3 5
run_sat2sound_i2t bingmap_withmeta 1 3 5
run_sat2text_baseline bingmap_i2t_baseline 1 3 5
