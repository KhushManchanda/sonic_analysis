#!/usr/bin/env python3
"""
04_split.py  —  Artist-level train/test split with audio-overlap subsets.

Reads:
    data/processed/ratings_joined.csv
    data/processed/musicnet_audio_map.csv (optional, for overlap subsets)

Writes:
    data/processed/ratings_train.csv
    data/processed/ratings_test.csv
    data/processed/ratings_train_overlap.csv
    data/processed/ratings_test_overlap.csv

Strategy:
  - Stratified by user: each user contributes ~80% of ratings to train.
  - Users with fewer than 2 ratings go entirely to train.
  - No user is allowed to appear only in test.
  - Overlap subsets are filtered copies of train/test restricted to artists that
    have audio coverage according to musicnet_audio_map.csv.
"""

import pathlib
import sys

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = pathlib.Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

INPUT_PATH  = PROCESSED / "ratings_joined.csv"
TRAIN_PATH  = PROCESSED / "ratings_train.csv"
TEST_PATH   = PROCESSED / "ratings_test.csv"
MAP_PATH    = PROCESSED / "musicnet_audio_map.csv"
TRAIN_OVERLAP_PATH = PROCESSED / "ratings_train_overlap.csv"
TEST_OVERLAP_PATH  = PROCESSED / "ratings_test_overlap.csv"

TEST_SIZE   = 0.20
RANDOM_SEED = 42


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"[ERROR] ratings_joined.csv not found. Run 03_join_ratings.py first.")
        sys.exit(1)

    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH, dtype={"user_id": int, "artist_id": int})
    print(f"  Total rows: {len(df)}")

    # ── Users with < 2 ratings go entirely to train ───────────────────────────
    counts          = df["user_id"].value_counts()
    sparse_users    = counts[counts < 2].index
    sparse_mask     = df["user_id"].isin(sparse_users)
    df_sparse       = df[sparse_mask].copy()
    df_splittable   = df[~sparse_mask].copy()

    print(f"  Sparse users (< 2 ratings) → train only: {sparse_mask.sum()} rows")

    # ── Stratified split on splittable rows ───────────────────────────────────
    train_rows, test_rows = [], []

    for user_id, group in df_splittable.groupby("user_id"):
        group = group.sample(frac=1, random_state=RANDOM_SEED)  # shuffle
        n_test = max(1, int(round(len(group) * TEST_SIZE)))
        test_rows.append(group.iloc[:n_test])
        train_rows.append(group.iloc[n_test:])

    train = pd.concat([df_sparse] + train_rows, ignore_index=True)
    test  = pd.concat(test_rows, ignore_index=True)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    # No user should appear ONLY in test
    test_only_users = set(test["user_id"]) - set(train["user_id"])
    if test_only_users:
        moved = test[test["user_id"].isin(test_only_users)]
        train = pd.concat([train, moved], ignore_index=True)
        test  = test[~test["user_id"].isin(test_only_users)]
        print(f"  [FIX] Moved {len(moved)} rows for {len(test_only_users)} test-only users → train")

    total = len(train) + len(test)
    print(f"\n  Train: {len(train):,} rows ({len(train)/total:.1%})")
    print(f"  Test:  {len(test):,}  rows ({len(test)/total:.1%})")
    print(f"  Train users: {train['user_id'].nunique():,}  |  Test users: {test['user_id'].nunique():,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH,   index=False)
    print(f"\n[DONE] ratings_train.csv → {TRAIN_PATH}")
    print(f"[DONE] ratings_test.csv  → {TEST_PATH}")

    overlap_ids: set[int] = set()
    if MAP_PATH.exists():
        mapping = pd.read_csv(MAP_PATH)
        if "artist_id" in mapping.columns:
            overlap_ids = set(pd.to_numeric(mapping["artist_id"], errors="coerce").dropna().astype(int))

    train_overlap = train[train["artist_id"].isin(overlap_ids)].copy()
    test_overlap = test[test["artist_id"].isin(overlap_ids)].copy()
    train_overlap.to_csv(TRAIN_OVERLAP_PATH, index=False)
    test_overlap.to_csv(TEST_OVERLAP_PATH, index=False)
    print(f"[DONE] ratings_train_overlap.csv → {TRAIN_OVERLAP_PATH} ({len(train_overlap):,} rows)")
    print(f"[DONE] ratings_test_overlap.csv  → {TEST_OVERLAP_PATH} ({len(test_overlap):,} rows)")


if __name__ == "__main__":
    main()
