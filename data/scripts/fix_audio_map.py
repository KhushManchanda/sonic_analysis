#!/usr/bin/env python3
"""
fix_audio_map.py — Expand musicnet_audio_map.csv to cover all MusicNet composers
================================================================================
Matches all 10 composers in MusicNet against HetRec artist names (exact + fuzzy),
writes expanded musicnet_audio_map.csv, then builds a single audio_features.csv
in data/processed/ by averaging per-composer across all their MusicNet recordings.
"""

from __future__ import annotations
import pathlib
import re
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
EMBEDS = ROOT / "data" / "embeds"

MUSICNET_META = ROOT / "data" / "raw" / "musicnet_metadata.csv"
HETREC_META = PROCESSED / "track_metadata.csv"
AUDIO_TRAIN = EMBEDS / "audio_features_train.csv"
AUDIO_TEST = EMBEDS / "audio_features_test.csv"
MAP_OUT = PROCESSED / "musicnet_audio_map.csv"
AUDIO_OUT = PROCESSED / "audio_features.csv"


# ── Composer → HetRec artist mapping ─────────────────────────────────────────
# Manually curated best-match HetRec artist_id for each MusicNet composer.
# Using the most canonical entry per composer (avoids noise like "camper van beethoven").
COMPOSER_TO_HETREC = {
    "Bach":       [("johann sebastian bach", None), ("j..s. bach", None), ("carl philipp emanuel bach", None)],
    "Beethoven":  [("ludwig van beethoven", None)],
    "Brahms":     [("johannes brahms", None), ("brahms", None)],
    "Cambini":    [],  # Not in HetRec — will be skipped
    "Dvorak":     [],  # Check below dynamically
    "Faure":      [],  # Check below dynamically
    "Haydn":      [("franz joseph haydn", None)],
    "Mozart":     [("wolfgang amadeus mozart", None), ("wulfgang amadeus mozart", None)],
    "Ravel":      [("maurice ravel", None)],
    "Schubert":   [("franz schubert", None)],
}

# Additional fuzzy patterns per composer (lowercase substring match)
COMPOSER_FUZZY = {
    "Bach":      [" bach", "j.s. bach", "j. s. bach", "bach, j"],
    "Beethoven": ["beethoven"],
    "Brahms":    ["brahms"],
    "Dvorak":    ["antonin dvorak", "dvorak", "dvořák"],
    "Faure":     ["gabriel faure", "gabriel fauré", "faure", "fauré"],
    "Haydn":     ["haydn"],
    "Mozart":    ["mozart"],
    "Ravel":     ["maurice ravel"],
    "Schubert":  ["franz schubert"],
}

# Exact names that should NOT match (false positives from substring hits)
BLACKLIST = {
    "camper van beethoven",
    "gravel",
    "traveling wilburys",
    "blues traveler",
    "the caravelles",
    "the time travelling pirates",
    "bachman-turner overdrive",
    "sebastian bach",
    "falkenbach",
    "burt bacharach",
    "bachelor girl",
    "dan auerbach",
    "museo rosenbach",
    "boom clap bachelors",
    "bachir attar",
    "bachelors of science",
    "the flying luttenbachers",
    "laibach",
    "mozart season",
    "max corbacho",
    "max corbacho & bruno sanfilippo",
    "wulfgang amadeus mozart",  # typo duplicate of the canonical entry
}


def build_expanded_map() -> pd.DataFrame:
    musicnet_meta = pd.read_csv(MUSICNET_META, dtype={"id": str})
    hetrec = pd.read_csv(HETREC_META)
    hetrec_lower = hetrec.copy()
    hetrec_lower["artist_lower"] = hetrec_lower["artist"].str.lower().str.strip()

    rows = []
    for composer in musicnet_meta["composer"].unique():
        composer_ids = musicnet_meta.loc[musicnet_meta["composer"] == composer, "id"].tolist()
        patterns = COMPOSER_FUZZY.get(composer, [])
        if not patterns:
            print(f"  [SKIP] {composer} — no fuzzy pattern, skipping")
            continue

        matched_artists = hetrec_lower[
            hetrec_lower["artist_lower"].apply(
                lambda a: isinstance(a, str) and any(pat in a for pat in patterns) and a not in BLACKLIST
            )
        ][["artist_id", "artist"]].drop_duplicates()

        if matched_artists.empty:
            print(f"  [MISS] {composer} — no HetRec match found")
            continue

        for _, art_row in matched_artists.iterrows():
            for mid in composer_ids:
                rows.append({
                    "artist_id":    int(art_row["artist_id"]),
                    "artist":       art_row["artist"],
                    "musicnet_id":  str(mid),
                    "composer_raw": composer,
                    "match_key":    art_row["artist"].lower(),
                    "match_type":   "fuzzy",
                })
            print(f"  [MAP] {composer} → {art_row['artist']} (artist_id={art_row['artist_id']}) — {len(composer_ids)} recordings")

    df = pd.DataFrame(rows).drop_duplicates(subset=["artist_id", "musicnet_id"])
    return df


def build_audio_features(map_df: pd.DataFrame) -> pd.DataFrame:
    # Load all audio features from embeds
    parts = []
    for path in [AUDIO_TRAIN, AUDIO_TEST]:
        if path.exists():
            parts.append(pd.read_csv(path, dtype={"musicnet_id": str}))
    if not parts:
        raise FileNotFoundError("No audio feature files found in data/embeds/")
    all_audio = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["musicnet_id"])

    map_df["musicnet_id"] = map_df["musicnet_id"].astype(str)
    merged = map_df.merge(all_audio, on="musicnet_id", how="inner")
    if merged.empty:
        raise ValueError("No audio rows matched after merge with map")

    numeric_cols = [
        c for c in merged.columns
        if c not in {"artist_id", "artist", "musicnet_id", "composer_raw", "match_key", "match_type"}
        and pd.api.types.is_numeric_dtype(merged[c])
    ]

    # Average features across all recordings per artist
    grouped = (
        merged.groupby("artist_id", as_index=False)[numeric_cols]
        .mean()
        .round(6)
    )

    # Add track_id column (= artist_id for this project)
    grouped.insert(0, "track_id", grouped["artist_id"])
    grouped = grouped.drop(columns=["artist_id"])
    grouped = grouped.sort_values("track_id").reset_index(drop=True)
    return grouped


def main():
    print("=== Phase 1: Expand musicnet_audio_map.csv ===")
    map_df = build_expanded_map()
    print(f"\nTotal map rows: {len(map_df)}")
    print(f"Unique HetRec artists covered: {map_df['artist_id'].nunique()}")
    map_df.to_csv(MAP_OUT, index=False)
    print(f"Written: {MAP_OUT}")

    print("\n=== Phase 2: Build audio_features.csv ===")
    audio_df = build_audio_features(map_df)
    audio_df.to_csv(AUDIO_OUT, index=False)
    feature_cols = [c for c in audio_df.columns if c != "track_id"]
    print(f"Written: {AUDIO_OUT}")
    print(f"  Rows (artists with audio): {len(audio_df)}")
    print(f"  Feature columns: {len(feature_cols)}")
    print(audio_df[["track_id"] + feature_cols[:4]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
