#!/usr/bin/env bash
# =============================================================================
# 01_download.sh  —  Person 1: Download raw data
# Downloads:
#   1. musicnet_metadata.csv  (MusicNet track/artist catalog from Zenodo)
#   2. hetrec2011-lastfm-2k   (user-artist ratings/play-counts from GroupLens)
# Idempotent: skips files that already exist.
# =============================================================================

set -euo pipefail

RAW_DIR="$(dirname "$0")/../raw"
mkdir -p "$RAW_DIR"

echo "=== Step 1: MusicNet metadata ==="
METADATA_URL="https://zenodo.org/records/5120004/files/musicnet_metadata.csv?download=1"
METADATA_OUT="$RAW_DIR/musicnet_metadata.csv"

if [ -f "$METADATA_OUT" ]; then
    echo "  [SKIP] musicnet_metadata.csv already exists"
else
    echo "  Downloading musicnet_metadata.csv ..."
    curl -L "$METADATA_URL" -o "$METADATA_OUT"
    echo "  [DONE] Saved to $METADATA_OUT"
fi

echo ""
echo "=== Step 2: HetRec 2011 Last.fm 2K ratings dataset ==="
HETREC_URL="http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
HETREC_ZIP="$RAW_DIR/hetrec2011-lastfm-2k.zip"
HETREC_DIR="$RAW_DIR/hetrec2011-lastfm-2k"

if [ -d "$HETREC_DIR" ]; then
    echo "  [SKIP] hetrec2011-lastfm-2k/ already exists"
else
    echo "  Downloading hetrec2011-lastfm-2k.zip (~1 MB) ..."
    curl -L "$HETREC_URL" -o "$HETREC_ZIP"
    echo "  Extracting ..."
    unzip -q "$HETREC_ZIP" -d "$HETREC_DIR"
    rm "$HETREC_ZIP"
    echo "  [DONE] Extracted to $HETREC_DIR"
fi

echo ""
echo "=== All downloads complete ==="
ls -lh "$RAW_DIR"
