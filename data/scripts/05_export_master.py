#!/usr/bin/env python3
"""Export master artist-level table to the legacy filename master_tracks.csv."""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

RATINGS_PATH = PROCESSED / "ratings_joined.csv"
META_PATH = PROCESSED / "track_metadata.csv"
MASTER_PATH = PROCESSED / "master_tracks.csv"


def main() -> None:
    for path in [RATINGS_PATH, META_PATH]:
        if not path.exists():
            print(f"[ERROR] Missing: {path}")
            sys.exit(1)

    ratings = pd.read_csv(RATINGS_PATH, dtype={"user_id": int, "artist_id": int})
    meta = pd.read_csv(META_PATH, dtype={"artist_id": int})
    meta_cols = [c for c in meta.columns if c != "artist"]
    master = ratings.merge(meta[meta_cols], on="artist_id", how="left")
    priority = ["user_id", "artist_id", "artist", "rating", "url", "musicnet_id"]
    rest = [c for c in master.columns if c not in priority]
    master = master[priority + rest]
    master.to_csv(MASTER_PATH, index=False)
    print(f"[DONE] master_tracks.csv → {MASTER_PATH}")


if __name__ == "__main__":
    main()
