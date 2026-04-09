#!/usr/bin/env python3
"""
02_clean_metadata.py  —  Person 1: Build item catalog from HetRec 2011
=======================================================================
PRIMARY catalog: HetRec 2011 artists.dat (17,632 artists → 92,834 interactions)
  track_id = artist_id from HetRec (stable integer join key for everyone)

SECONDARY enrichment: musicnet_metadata.csv
  Maps the ~10 classical composers that overlap to MusicNet WAV files
  so Person 3 can extract audio features for that subset.

Reads:
    data/raw/hetrec2011-lastfm-2k/artists.dat
    data/raw/musicnet_metadata.csv             (optional enrichment)
Writes:
    data/processed/track_metadata.csv          (17,632 rows)
    data/processed/musicnet_audio_map.csv      (track_id ↔ musicnet_id mapping)

Output columns (track_metadata.csv):
    track_id    int   — HetRec artist_id (primary join key for everyone)
    artist      str   — normalized artist name
    url         str   — Last.fm URL
    musicnet_id int   — MusicNet ID if audio is available, else NaN
"""

import sys
import pathlib
import re

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = pathlib.Path(__file__).resolve().parents[2]
RAW        = ROOT / "data" / "raw"
HETREC_DIR = RAW / "hetrec2011-lastfm-2k"
OUT        = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

ARTISTS_PATH    = HETREC_DIR / "artists.dat"
MUSICNET_PATH   = RAW / "musicnet_metadata.csv"
OUTPUT_PATH     = OUT / "track_metadata.csv"
AUDIO_MAP_PATH  = OUT / "musicnet_audio_map.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_text(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)
         .str.replace(r"[^\x20-\x7E]", "", regex=True)
         .str.lower()
    )


def normalize_composer(name: str) -> str:
    """'Bach, Johann Sebastian' → 'johann sebastian bach'"""
    name = name.strip().lower()
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        return f"{parts[1]} {parts[0]}"
    return name


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if not ARTISTS_PATH.exists():
        print(f"[ERROR] artists.dat not found in {HETREC_DIR}")
        print("  Run data/scripts/01_download.sh first.")
        sys.exit(1)

    # ── 1. Load HetRec artists → primary catalog ──────────────────────────────
    print(f"Loading {ARTISTS_PATH} ...")
    artists = pd.read_csv(ARTISTS_PATH, sep="\t", encoding="utf-8", on_bad_lines="skip")
    artists.columns = [c.strip().lower() for c in artists.columns]
    artists = artists.rename(columns={"id": "track_id", "name": "artist", "pictureurl": "picture_url"})

    artists["track_id"] = pd.to_numeric(artists["track_id"], errors="coerce").astype("Int64")
    artists["artist"]   = normalize_text(artists["artist"])

    # Keep useful columns
    keep = ["track_id", "artist", "url"]
    artists = artists[[c for c in keep if c in artists.columns]]
    artists = artists.dropna(subset=["track_id", "artist"])
    artists = artists.drop_duplicates(subset=["track_id"])
    artists["musicnet_id"] = np.nan   # placeholder; filled below

    print(f"  HetRec catalog: {len(artists)} artists")

    # ── 2. Enrich with MusicNet where composers overlap ────────────────────────
    audio_map_rows = []
    if MUSICNET_PATH.exists():
        print(f"Loading MusicNet metadata for audio enrichment ...")
        mn = pd.read_csv(MUSICNET_PATH)
        mn.columns = [c.strip().lower() for c in mn.columns]
        mn = mn.rename(columns={"id": "musicnet_id", "composer": "composer_raw"})
        mn["composer_norm"] = mn["composer_raw"].apply(normalize_composer)
        mn["musicnet_id"]   = pd.to_numeric(mn["musicnet_id"], errors="coerce").astype("Int64")

        # For each MusicNet composer, find matching HetRec artist rows
        for _, mn_row in mn.iterrows():
            key  = mn_row["composer_norm"]
            mask = artists["artist"].str.contains(re.escape(key), na=False)
            if mask.any():
                for hetrec_id in artists.loc[mask, "track_id"]:
                    audio_map_rows.append({
                        "track_id":   int(hetrec_id),
                        "musicnet_id": int(mn_row["musicnet_id"]),
                    })
                # Mark the first matched HetRec artist with musicnet_id
                first_idx = artists[mask].index[0]
                artists.at[first_idx, "musicnet_id"] = mn_row["musicnet_id"]

        n_enriched = artists["musicnet_id"].notna().sum()
        print(f"  MusicNet enrichment: {n_enriched} HetRec artists linked to audio")
    else:
        print("  [SKIP] musicnet_metadata.csv not found — skipping audio enrichment")

    # ── 3. Save main catalog ──────────────────────────────────────────────────
    artists.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[DONE] track_metadata.csv — {len(artists)} rows → {OUTPUT_PATH}")
    print(artists.head(5).to_string(index=False))

    # ── 4. Save audio map (for Person 3) ─────────────────────────────────────
    if audio_map_rows:
        audio_map = pd.DataFrame(audio_map_rows).drop_duplicates()
        audio_map.to_csv(AUDIO_MAP_PATH, index=False)
        print(f"\n[DONE] musicnet_audio_map.csv — {len(audio_map)} rows → {AUDIO_MAP_PATH}")
    else:
        print("\n  [INFO] No audio map rows (no overlap found).")


if __name__ == "__main__":
    main()
