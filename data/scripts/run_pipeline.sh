#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh  —  Person 1: Run the full data pipeline end-to-end
# Usage: bash data/scripts/run_pipeline.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."   # go to sonic_analysis/

PYTHON_BIN="python3"
if [ -x "/Users/arjunranjan/Desktop/sonic_analysis/.venv/bin/python" ]; then
  PYTHON_BIN="/Users/arjunranjan/Desktop/sonic_analysis/.venv/bin/python"
fi

WORKERS="4"
SKIP_DOWNLOAD="false"
for arg in "$@"; do
  case "$arg" in
    --skip-download)
      SKIP_DOWNLOAD="true"
      ;;
    --workers=*)
      WORKERS="${arg#*=}"
      ;;
  esac
done

echo "=============================================="
echo "  AudioMuse-AI  |  Person 1 Data Pipeline"
echo "=============================================="

echo ""
if [ "$SKIP_DOWNLOAD" = "false" ]; then
  echo ">>> Step 1/8: Download raw data"
  bash data/scripts/01_download.sh
else
  echo ">>> Step 1/8: Skipping download (--skip-download)"
fi

echo ""
echo ">>> Step 2/8: Clean MusicNet metadata"
"$PYTHON_BIN" data/scripts/02_clean_metadata.py

echo ""
echo ">>> Step 3/8: Join with HetRec ratings"
"$PYTHON_BIN" data/scripts/03_join_ratings.py

echo ""
echo ">>> Step 4/8: Train / test split"
"$PYTHON_BIN" data/scripts/04_split.py

echo ""
echo ">>> Step 5/8: Export master table"
"$PYTHON_BIN" data/scripts/05_export_master.py

echo ""
echo ">>> Step 6/8: Build tag features"
"$PYTHON_BIN" data/scripts/06_build_tag_features.py

echo ""
echo ">>> Step 7/8: Extract audio features"
"$PYTHON_BIN" data/scripts/07_embed_tracks.py --workers "$WORKERS"

echo ""
echo ">>> Step 8/8: Run experiments"
"$PYTHON_BIN" data/scripts/08_run_experiments.py

echo ""
echo ">>> Validating outputs ..."
"$PYTHON_BIN" data/scripts/validate_outputs.py

echo ""
echo "=============================================="
echo "  All done! Files in data/processed/"
ls -lh data/processed/
echo "=============================================="