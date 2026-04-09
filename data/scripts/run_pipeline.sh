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
echo ">>> Step 1/5: Download raw data"
bash data/scripts/01_download.sh

echo ""
echo ">>> Step 2/5: Clean MusicNet metadata"
python3 data/scripts/02_clean_metadata.py

echo ""
echo ">>> Step 3/5: Join with HetRec ratings"
python3 data/scripts/03_join_ratings.py

echo ""
echo ">>> Step 4/5: Train / test split"
python3 data/scripts/04_split.py

echo ""
echo ">>> Step 5/5: Export master table"
python3 data/scripts/05_export_master.py

echo ""
echo ">>> Validating outputs ..."
python3 data/scripts/validate_outputs.py

echo ""
echo "=============================================="
echo "  All done! Files in data/processed/"
ls -lh data/processed/
echo "=============================================="
