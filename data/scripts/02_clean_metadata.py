#!/usr/bin/env python3
"""
02_clean_metadata.py  —  Build the artist catalog and validated MusicNet overlap map
===============================================================================

Canonical item definition:
    HetRec items are artists, not tracks.
    artist_id = HetRec artist_id.

Files retained for backward compatibility:
    - track_metadata.csv (contains artist-level rows)
    - musicnet_audio_map.csv
"""

from __future__ import annotations

import pathlib
import re
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
HETREC_DIR = RAW / "hetrec2011-lastfm-2k"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

ARTISTS_PATH = HETREC_DIR / "artists.dat"
MUSICNET_PATH = RAW / "musicnet_metadata.csv"
OUTPUT_PATH = OUT / "track_metadata.csv"
AUDIO_MAP_PATH = OUT / "musicnet_audio_map.csv"

MANUAL_COMPOSER_ALIASES: dict[str, list[str]] = {
    "johann sebastian bach": ["j s bach", "js bach", "bach"],
    "ludwig van beethoven": ["beethoven"],
    "franz schubert": ["schubert"],
    "frederic chopin": ["chopin"],
    "wolfgang amadeus mozart": ["mozart"],
    "joseph haydn": ["haydn"],
    "claude debussy": ["debussy"],
    "antonin dvorak": ["dvorak"],
    "felix mendelssohn": ["mendelssohn"],
    "robert schumann": ["schumann"],
    "sergei rachmaninoff": ["rachmaninoff", "rachmaninov"],
}


def normalize_text(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E]", "", text)
    return text


def normalize_composer(name: str) -> str:
    name = normalize_text(name)
    if "," in name:
        last, first = [part.strip() for part in name.split(",", 1)]
        return normalize_text(f"{first} {last}")
    return name


def candidate_match_keys(composer_norm: str) -> list[tuple[str, str]]:
    keys = [(composer_norm, "exact")]
    for alias in MANUAL_COMPOSER_ALIASES.get(composer_norm, []):
        keys.append((normalize_text(alias), "alias"))
    return keys


def main() -> None:
    if not ARTISTS_PATH.exists():
        print(f"[ERROR] artists.dat not found in {HETREC_DIR}")
        print("  Run data/scripts/01_download.sh first.")
        sys.exit(1)

    print(f"Loading {ARTISTS_PATH} ...")
    artists = pd.read_csv(ARTISTS_PATH, sep="\t", encoding="utf-8", on_bad_lines="skip")
    artists.columns = [c.strip().lower() for c in artists.columns]
    artists = artists.rename(columns={"id": "artist_id", "name": "artist", "pictureurl": "picture_url"})
    artists["artist_id"] = pd.to_numeric(artists["artist_id"], errors="coerce").astype("Int64")
    artists["artist"] = artists["artist"].map(normalize_text)
    artists = artists[[c for c in ["artist_id", "artist", "url"] if c in artists.columns]]
    artists = artists.dropna(subset=["artist_id", "artist"]).drop_duplicates(subset=["artist_id"])
    artists["artist_id"] = artists["artist_id"].astype(int)
    artists["musicnet_id"] = np.nan

    audio_map = pd.DataFrame(columns=["artist_id", "artist", "musicnet_id", "composer_raw", "match_key", "match_type"])

    if MUSICNET_PATH.exists():
        print("Loading MusicNet metadata for audio enrichment ...")
        musicnet = pd.read_csv(MUSICNET_PATH)
        musicnet.columns = [c.strip().lower() for c in musicnet.columns]
        musicnet = musicnet.rename(columns={"id": "musicnet_id", "composer": "composer_raw"})
        musicnet["musicnet_id"] = pd.to_numeric(musicnet["musicnet_id"], errors="coerce").astype("Int64")
        musicnet = musicnet.dropna(subset=["musicnet_id", "composer_raw"]).copy()
        musicnet["musicnet_id"] = musicnet["musicnet_id"].astype(int)
        musicnet["composer_norm"] = musicnet["composer_raw"].map(normalize_composer)

        artist_name_to_ids = artists.groupby("artist")["artist_id"].apply(list).to_dict()
        rows: list[dict[str, object]] = []

        for _, row in musicnet.iterrows():
            chosen_key = None
            chosen_type = None
            candidate_ids: list[int] = []
            for key, match_type in candidate_match_keys(row["composer_norm"]):
                ids = artist_name_to_ids.get(key, [])
                if len(ids) == 1:
                    chosen_key = key
                    chosen_type = match_type
                    candidate_ids = ids
                    break
                if len(ids) > 1:
                    candidate_ids = []
                    break
            if not candidate_ids:
                continue
            artist_id = int(candidate_ids[0])
            artist_name = artists.loc[artists["artist_id"] == artist_id, "artist"].iloc[0]
            rows.append({
                "artist_id": artist_id,
                "artist": artist_name,
                "musicnet_id": int(row["musicnet_id"]),
                "composer_raw": str(row["composer_raw"]),
                "match_key": chosen_key,
                "match_type": chosen_type,
            })

        if rows:
            audio_map = pd.DataFrame(rows).drop_duplicates(subset=["artist_id", "musicnet_id"])
            first_musicnet = audio_map.groupby("artist_id", as_index=False)["musicnet_id"].min()
            artists = artists.merge(first_musicnet, on="artist_id", how="left", suffixes=("", "_mapped"))
            artists["musicnet_id"] = artists["musicnet_id_mapped"].combine_first(artists["musicnet_id"])
            artists = artists.drop(columns=["musicnet_id_mapped"])

    artists = artists[["artist_id", "artist", "url", "musicnet_id"]].sort_values("artist_id")
    artists.to_csv(OUTPUT_PATH, index=False)
    audio_map.sort_values(["artist_id", "musicnet_id"]).to_csv(AUDIO_MAP_PATH, index=False)
    print(f"[DONE] track_metadata.csv → {OUTPUT_PATH}")
    print(f"[DONE] musicnet_audio_map.csv → {AUDIO_MAP_PATH}")


if __name__ == "__main__":
    main()
