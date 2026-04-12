#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh  —  Person 1: Run the full data pipeline end-to-end
# Usage: bash data/scripts/run_pipeline.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."   # go to sonic_analysis/

echo "=============================================="
echo "  AudioMuse-AI  |  Person 1 Data Pipeline"
echo "=============================================="

echo ""
echo ">>> Step 1/7: Download raw data"
bash data/scripts/01_download.sh

echo ""
echo ">>> Step 2/7: Clean MusicNet metadata"
python3 data/scripts/02_clean_metadata.py

echo ""
echo ">>> Step 3/7: Join with HetRec ratings"
python3 data/scripts/03_join_ratings.py

echo ""
echo ">>> Step 4/7: Train / test split"
python3 data/scripts/04_split.py

echo ""
echo ">>> Step 5/7: Export master table"
python3 data/scripts/05_export_master.py

echo ""
echo ">>> Step 6/7: Build tag features"
python3 data/scripts/06_build_tag_features.py

echo ""
echo ">>> Step 7/7: Extract audio features"
python3 data/scripts/07_embed_tracks.py

echo ""
echo ">>> Validating outputs ..."
python3 data/scripts/validate_outputs.py

echo ""
echo "=============================================="
echo "  All done! Files in data/processed/"
ls -lh data/processed/
echo "=============================================="