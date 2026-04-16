#!/usr/bin/env python3
"""Build artist-level ratings table from HetRec user-artist interactions."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW_LASTFM = ROOT / "data" / "raw" / "hetrec2011-lastfm-2k"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

UA_PATH = RAW_LASTFM / "user_artists.dat"
META_PATH = PROCESSED / "track_metadata.csv"
OUTPUT_PATH = PROCESSED / "ratings_joined.csv"


def log_normalize_to_scale(weights: pd.Series, scale_min: float = 1.0, scale_max: float = 5.0) -> pd.Series:
    log_w = np.log1p(weights.clip(lower=0))
    w_min, w_max = log_w.min(), log_w.max()
    if w_max == w_min:
        return pd.Series(np.full(len(weights), (scale_min + scale_max) / 2), index=weights.index)
    normed = (log_w - w_min) / (w_max - w_min)
    return (normed * (scale_max - scale_min) + scale_min).round(4)


def main() -> None:
    for path in [UA_PATH, META_PATH]:
        if not path.exists():
            print(f"[ERROR] Missing: {path}")
            sys.exit(1)

    print(f"Loading {UA_PATH} ...")
    ua = pd.read_csv(UA_PATH, sep="\t", encoding="utf-8")
    ua.columns = [c.strip().lower() for c in ua.columns]
    ua = ua.rename(columns={"userid": "user_id", "artistid": "artist_id"})
    ua["user_id"] = ua["user_id"].astype(int)
    ua["artist_id"] = ua["artist_id"].astype(int)
    ua["rating"] = log_normalize_to_scale(ua["weight"])

    print(f"Loading {META_PATH} ...")
    meta = pd.read_csv(META_PATH, dtype={"artist_id": int})[["artist_id", "artist"]]
    merged = ua.merge(meta, on="artist_id", how="inner")

    out = merged[["user_id", "artist_id", "artist", "rating"]].copy()
    out = out.dropna(subset=["user_id", "artist_id", "rating"])
    out = out.drop_duplicates(subset=["user_id", "artist_id"])
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"[DONE] ratings_joined.csv — {len(out):,} rows")
    print(f"  Unique users   : {out['user_id'].nunique():,}")
    print(f"  Unique artists : {out['artist_id'].nunique():,}")


if __name__ == "__main__":
    main()
