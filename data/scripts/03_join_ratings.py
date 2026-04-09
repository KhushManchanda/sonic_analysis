#!/usr/bin/env python3
"""
03_join_ratings.py  —  Person 1: Build ratings table from HetRec
=================================================================
Now that track_id = HetRec artist_id, this is a direct join with no
cross-dataset mismatch. Gives 92,834 user-artist interactions across
~1,892 users and 17,632 items.

Reads:
    data/raw/hetrec2011-lastfm-2k/user_artists.dat
    data/processed/track_metadata.csv
Writes:
    data/processed/ratings_joined.csv

Output columns:
    user_id    int
    track_id   int   (= HetRec artist_id)
    artist     str
    rating     float (1.0 – 5.0, log-normalized from play count)
"""

import sys
import pathlib

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = pathlib.Path(__file__).resolve().parents[2]
RAW_LASTFM = ROOT / "data" / "raw" / "hetrec2011-lastfm-2k"
PROCESSED  = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

UA_PATH     = RAW_LASTFM / "user_artists.dat"
META_PATH   = PROCESSED  / "track_metadata.csv"
OUTPUT_PATH = PROCESSED  / "ratings_joined.csv"


def log_normalize_to_scale(weights: pd.Series,
                            scale_min: float = 1.0,
                            scale_max: float = 5.0) -> pd.Series:
    log_w = np.log1p(weights.clip(lower=0))
    w_min, w_max = log_w.min(), log_w.max()
    if w_max == w_min:
        return pd.Series(np.full(len(weights), (scale_min + scale_max) / 2),
                         index=weights.index)
    normed = (log_w - w_min) / (w_max - w_min)
    return (normed * (scale_max - scale_min) + scale_min).round(4)


def main() -> None:
    for path in [UA_PATH, META_PATH]:
        if not path.exists():
            print(f"[ERROR] Missing: {path}")
            sys.exit(1)

    # ── 1. Load user-artist play counts ───────────────────────────────────────
    print(f"Loading {UA_PATH} ...")
    ua = pd.read_csv(UA_PATH, sep="\t", encoding="utf-8")
    ua.columns = [c.strip().lower() for c in ua.columns]
    ua = ua.rename(columns={"userid": "user_id", "artistid": "track_id"})
    ua["user_id"]  = ua["user_id"].astype(int)
    ua["track_id"] = ua["track_id"].astype(int)
    print(f"  Loaded {len(ua):,} user-artist rows")

    # ── 2. Log-normalize play counts → ratings ────────────────────────────────
    ua["rating"] = log_normalize_to_scale(ua["weight"])

    # ── 3. Join artist name from catalog ──────────────────────────────────────
    print(f"Loading {META_PATH} ...")
    meta = pd.read_csv(META_PATH, dtype={"track_id": int})[["track_id", "artist"]]

    merged = ua.merge(meta, on="track_id", how="inner")

    # ── 4. Select and clean output ─────────────────────────────────────────────
    out = merged[["user_id", "track_id", "artist", "rating"]].copy()
    out = out.dropna(subset=["user_id", "track_id", "rating"])
    out = out.drop_duplicates(subset=["user_id", "track_id"])

    # ── 5. Save ───────────────────────────────────────────────────────────────
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[DONE] ratings_joined.csv — {len(out):,} rows")
    print(f"  Unique users  : {out['user_id'].nunique():,}")
    print(f"  Unique tracks : {out['track_id'].nunique():,}")
    print(f"  Rating range  : {out['rating'].min():.2f} – {out['rating'].max():.2f}")
    print(f"  Rating mean   : {out['rating'].mean():.2f}")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
