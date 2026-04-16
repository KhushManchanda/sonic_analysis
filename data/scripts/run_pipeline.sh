#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh  —  Run the full data + modeling pipeline end-to-end
# Usage: bash data/scripts/run_pipeline.sh [--workers N]
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."   # go to sonic_analysis/

WORKERS=4
for arg in "$@"; do
  case $arg in
    --workers=*) WORKERS="${arg#*=}" ;;
    --workers)   shift; WORKERS="$1" ;;
  esac
done

echo "=============================================="
echo "  AudioMuse-AI  |  Full Data Pipeline"
echo "=============================================="

echo ""
echo ">>> Step 1/8: Download raw data"
bash data/scripts/01_download.sh

echo ""
echo ">>> Step 2/8: Clean MusicNet metadata"
python3 data/scripts/02_clean_metadata.py

echo ""
echo ">>> Step 3/8: Join with HetRec ratings"
python3 data/scripts/03_join_ratings.py

echo ""
echo ">>> Step 4/8: Train / test split"
python3 data/scripts/04_split.py

echo ""
echo ">>> Step 5/8: Export master table"
python3 data/scripts/05_export_master.py

echo ""
echo ">>> Step 6/8: Build tag features (Person 2)"
python3 data/scripts/06_build_tag_features.py

echo ""
echo ">>> Step 7/8: Audio feature extraction + map (Person 3)"
python3 data/scripts/07_embed_tracks.py --workers "$WORKERS"
python3 data/scripts/fix_audio_map.py

echo ""
echo ">>> Step 8/8: Modeling & evaluation (Person 4)"
python3 data/scripts/08_model.py

echo ""
echo ">>> Validating all outputs ..."
python3 data/scripts/validate_outputs.py

echo ""
echo "=============================================="
echo "  All done! Files in data/processed/"
ls -lh data/processed/
echo "=============================================="