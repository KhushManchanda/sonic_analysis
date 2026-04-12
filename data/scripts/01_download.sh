#!/usr/bin/env bash
# =============================================================================
# 01_download.sh  —  Person 1: Download raw data
# Downloads:
#   1. musicnet_metadata.csv  (MusicNet track/artist catalog from Zenodo)
#   2. hetrec2011-lastfm-2k   (user-artist ratings/play-counts from GroupLens)
#   3. musicnet.tar.gz        (MusicNet audio, ~11 GB from Zenodo)
# Idempotent: skips files that already exist and match remote size.
# =============================================================================

set -euo pipefail

RAW_DIR="$(dirname "$0")/../raw"
mkdir -p "$RAW_DIR"

# -----------------------------------------------------------------------------
# Helper: get remote file size via HEAD request
# -----------------------------------------------------------------------------
get_remote_size() {
    curl -sIL "$1" | grep -i content-length | tail -1 | awk '{print $2}' | tr -d '\r'
}

get_file_size() {
    if stat --version &>/dev/null 2>&1; then
        stat -c%s "$1"
    else
        stat -f%z "$1"
    fi
}

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
echo "=== Step 3: MusicNet audio ==="
MUSICNET_URL="https://zenodo.org/records/5120004/files/musicnet.tar.gz"
MUSICNET_TAR="$RAW_DIR/musicnet.tar.gz"
MUSICNET_DIR="$RAW_DIR/musicnet"

NEED_DOWNLOAD=false

if [ ! -f "$MUSICNET_TAR" ]; then
    echo "  Archive not found — downloading..."
    NEED_DOWNLOAD=true
else
    echo "  Checking size against remote..."
    ACTUAL_SIZE=$(get_file_size "$MUSICNET_TAR")
    REMOTE_SIZE=$(get_remote_size "$MUSICNET_URL")
    if [ "$ACTUAL_SIZE" != "$REMOTE_SIZE" ]; then
        echo "  Size mismatch (local $ACTUAL_SIZE, remote $REMOTE_SIZE) — re-downloading..."
        NEED_DOWNLOAD=true
    else
        echo "  [SKIP] musicnet.tar.gz already exists and matches remote size"
    fi
fi

if [ "$NEED_DOWNLOAD" = true ]; then
    echo "  Downloading musicnet.tar.gz (~11 GB) — this will take a while..."
    curl -L --progress-bar "$MUSICNET_URL" -o "$MUSICNET_TAR"
    echo "  [DONE] Saved to $MUSICNET_TAR"
fi

if [ ! -d "$MUSICNET_DIR" ]; then
    echo "  Extracting musicnet.tar.gz to $RAW_DIR ..."
    tar -xzf "$MUSICNET_TAR" -C "$RAW_DIR"
    echo "  [DONE] Extracted to $MUSICNET_DIR"
else
    echo "  [SKIP] $MUSICNET_DIR already exists"
fi

echo ""
echo "=== All downloads complete ==="
ls -lh "$RAW_DIR"
