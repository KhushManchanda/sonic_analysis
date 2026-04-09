#!/usr/bin/env python3
"""
04_split.py  —  Person 1: Train / test split
=============================================
Reads:  data/processed/ratings_joined.csv
Writes:
    data/processed/ratings_train.csv
    data/processed/ratings_test.csv

Strategy:
  Stratified by user — each user contributes 80% of their ratings to train
  and 20% to test.  Users with fewer than 2 ratings go entirely to train to
  prevent cold-start leakage into test.
"""

import sys
import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = pathlib.Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

INPUT_PATH  = PROCESSED / "ratings_joined.csv"
TRAIN_PATH  = PROCESSED / "ratings_train.csv"
TEST_PATH   = PROCESSED / "ratings_test.csv"

TEST_SIZE   = 0.20
RANDOM_SEED = 42


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"[ERROR] ratings_joined.csv not found. Run 03_join_ratings.py first.")
        sys.exit(1)

    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH, dtype={"user_id": int, "track_id": int})
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


if __name__ == "__main__":
    main()
