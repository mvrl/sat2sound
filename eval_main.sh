#!/bin/bash
# Usage: bash eval_main.sh [expr]   # omit expr to run all 6
set -e
FILTER="${1:-all}"
PY="python -m"
JSON="main"

run_eval() {
    local expr=$1 dataset=$2 sat=$3 meta=$4
    shift 4
    local zoom_levels=("$@")
    [[ "$FILTER" != "all" && "$FILTER" != "$expr" ]] && return
    for zl in "${zoom_levels[@]}"; do
        $PY src.evaluate \
            --expr "$expr" --dataset_type "$dataset" --sat_type "$sat" \
            --metadata_type "$meta" --test_zoom_level "$zl" \
            --save_results true --json_name "$JSON"
    done
}

run_eval bingmap_nometa   GeoSound bingmap none                               1
run_eval bingmap_withmeta GeoSound bingmap latlong_month_time_asource_tsource 1
run_eval sentinel_nometa   GeoSound sentinel none                               1
run_eval sentinel_withmeta GeoSound sentinel latlong_month_time_asource_tsource 1
run_eval SoundingEarth_nometa   SoundingEarth googleEarth none                       1
run_eval SoundingEarth_withmeta SoundingEarth googleEarth latlong_month_time_tsource 1
