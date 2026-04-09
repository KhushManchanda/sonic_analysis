#!/usr/bin/env python3
"""
05_export_master.py  —  Person 1: Build the master table
=========================================================
Reads:
    data/processed/ratings_joined.csv
    data/processed/track_metadata.csv
Writes:
    data/processed/master_tracks.csv

This is the ONE flat file everyone can join on.  It contains every
(user, track) pair with its rating and all track metadata.
Persons 2 and 3 will add their feature files separately, joined on track_id.

Also prints a dataset summary report.
"""

import sys
import pathlib

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = pathlib.Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

RATINGS_PATH = PROCESSED / "ratings_joined.csv"
META_PATH    = PROCESSED / "track_metadata.csv"
MASTER_PATH  = PROCESSED / "master_tracks.csv"


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 58)
    print("  DATASET SUMMARY")
    print("=" * 58)
    print(f"  Total (user, track) pairs : {len(df):>10,}")
    print(f"  Unique users              : {df['user_id'].nunique():>10,}")
    print(f"  Unique tracks             : {df['track_id'].nunique():>10,}")
    print(f"  Unique artists            : {df['artist'].nunique():>10,}")
    print(f"  Rating range              : {df['rating'].min():.2f} – {df['rating'].max():.2f}")
    print(f"  Rating mean ± std         : {df['rating'].mean():.2f} ± {df['rating'].std():.2f}")

    # Sparsity
    n_users  = df["user_id"].nunique()
    n_tracks = df["track_id"].nunique()
    sparsity = 1 - (len(df) / (n_users * n_tracks))
    print(f"  Matrix sparsity           : {sparsity:.4%}")

    # Ratings per user distribution
    rpu = df.groupby("user_id")["rating"].count()
    print(f"\n  Ratings per user (median) : {rpu.median():.0f}")
    print(f"  Ratings per user (max)    : {rpu.max():.0f}")
    print(f"  Users with ≤5 ratings     : {(rpu <= 5).sum():,} ({(rpu <= 5).mean():.1%})")
    print("=" * 58)


def main() -> None:
    for path in [RATINGS_PATH, META_PATH]:
        if not path.exists():
            print(f"[ERROR] Missing: {path}")
            print("  Run 02_clean_metadata.py and 03_join_ratings.py first.")
            sys.exit(1)

    print("Loading files ...")
    ratings = pd.read_csv(RATINGS_PATH, dtype={"user_id": int, "track_id": int})
    meta    = pd.read_csv(META_PATH,    dtype={"track_id": "Int64"})
    meta["track_id"] = meta["track_id"].astype(int)

    print(f"  Ratings: {ratings.shape}  |  Metadata: {meta.shape}")

    # ── Join ──────────────────────────────────────────────────────────────────
    # Ratings already has 'artist'; drop from meta to avoid duplicate columns
    meta_cols = [c for c in meta.columns if c not in ("artist",)]
    master = ratings.merge(meta[meta_cols], on="track_id", how="left")

    # ── Column order ──────────────────────────────────────────────────────────
    priority = ["user_id", "track_id", "artist", "rating", "url", "musicnet_id"]
    rest     = [c for c in master.columns if c not in priority]
    master   = master[priority + rest]

    # ── Save ──────────────────────────────────────────────────────────────────
    master.to_csv(MASTER_PATH, index=False)
    print(f"\n[DONE] master_tracks.csv → {MASTER_PATH}")
    print_summary(master)
    print(f"\nFirst 5 rows:")
    print(master.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
