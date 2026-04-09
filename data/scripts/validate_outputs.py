#!/usr/bin/env python3
"""
validate_outputs.py  —  Person 1: Validate all pipeline outputs
===============================================================
Run after the full pipeline to confirm everything is correct
before sharing files with the team.
"""

import sys
import pathlib
import pandas as pd

ROOT      = pathlib.Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

ERRORS = []
PASSES = []


def check(label: str, condition: bool, msg: str = "") -> None:
    if condition:
        PASSES.append(f"[OK]   {label}")
    else:
        ERRORS.append(f"[FAIL] {label}" + (f"  →  {msg}" if msg else ""))


def load(name: str) -> pd.DataFrame | None:
    path = PROCESSED / name
    if not path.exists():
        ERRORS.append(f"[FAIL] {name} not found at {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        ERRORS.append(f"[FAIL] {name} could not be read: {e}")
        return None


# ── track_metadata.csv ────────────────────────────────────────────────────────
meta = load("track_metadata.csv")
if meta is not None:
    check("track_metadata has track_id",  "track_id" in meta.columns)
    check("track_metadata has artist",    "artist"   in meta.columns)
    check("track_metadata no null track_id", meta["track_id"].isna().sum() == 0,
          f"{meta['track_id'].isna().sum()} nulls")
    check("track_metadata no duplicate track_id",
          meta["track_id"].duplicated().sum() == 0,
          f"{meta['track_id'].duplicated().sum()} duplicates")
    PASSES.append(f"[OK]   track_metadata.csv  — {len(meta)} tracks")

# ── ratings_joined.csv ────────────────────────────────────────────────────────
ratings = load("ratings_joined.csv")
if ratings is not None:
    check("ratings_joined has user_id",  "user_id"  in ratings.columns)
    check("ratings_joined has track_id", "track_id" in ratings.columns)
    check("ratings_joined has rating",   "rating"   in ratings.columns)
    check("rating in [1, 5]",
          ratings["rating"].between(1.0, 5.0).all(),
          f"min={ratings['rating'].min():.2f} max={ratings['rating'].max():.2f}")
    check("no null ratings", ratings["rating"].isna().sum() == 0)
    PASSES.append(f"[OK]   ratings_joined.csv  — {len(ratings)} rows  "
                  f"| {ratings['user_id'].nunique()} users  "
                  f"| {ratings['track_id'].nunique()} tracks")

# ── train / test ──────────────────────────────────────────────────────────────
train = load("ratings_train.csv")
test  = load("ratings_test.csv")
if train is not None and test is not None and ratings is not None:
    total = len(train) + len(test)
    check("train + test == ratings_joined",
          total == len(ratings),
          f"train({len(train)}) + test({len(test)}) = {total}  ≠  {len(ratings)}")
    check("test ratio approx 20%",
          0.15 <= len(test) / total <= 0.25,
          f"actual test ratio = {len(test)/total:.1%}")
    test_only = set(test["user_id"]) - set(train["user_id"])
    check("no test-only users", len(test_only) == 0,
          f"{len(test_only)} users appear only in test")
    PASSES.append(f"[OK]   ratings_train.csv   — {len(train)} rows  ({len(train)/total:.0%})")
    PASSES.append(f"[OK]   ratings_test.csv    — {len(test)} rows  ({len(test)/total:.0%})")

# ── master_tracks.csv ─────────────────────────────────────────────────────────
master = load("master_tracks.csv")
if master is not None:
    for col in ["user_id", "track_id", "artist", "rating"]:
        check(f"master_tracks has {col}", col in master.columns)
    check("master_tracks no null track_id", master["track_id"].isna().sum() == 0)
    check("master_tracks no null rating",   master["rating"].isna().sum() == 0)
    PASSES.append(f"[OK]   master_tracks.csv   — {len(master)} rows")

# ── tag_features.csv ──────────────────────────────────────────────────────────
tag_features = load("tag_features.csv")
if tag_features is not None:
    check("tag_features has track_id", "track_id" in tag_features.columns)
    check("tag_features has tags_raw", "tags_raw" in tag_features.columns)
    check(
        "tag_features non-empty",
        len(tag_features) > 0,
        "file contains zero rows",
    )
    if "track_id" in tag_features.columns:
        check(
            "tag_features no null track_id",
            tag_features["track_id"].isna().sum() == 0,
            f"{tag_features['track_id'].isna().sum()} nulls",
        )
        check(
            "tag_features unique track_id",
            tag_features["track_id"].duplicated().sum() == 0,
            f"{tag_features['track_id'].duplicated().sum()} duplicates",
        )
    tfidf_cols = [c for c in tag_features.columns if c.startswith("tfidf_")]
    check(
        "tag_features has tfidf columns",
        len(tfidf_cols) > 0,
        "expected at least one tfidf_* column",
    )
    numeric_ok = all(pd.api.types.is_numeric_dtype(tag_features[c]) for c in tfidf_cols)
    check("tag_features tfidf columns numeric", numeric_ok)
    if meta is not None and "track_id" in tag_features.columns:
        missing_from_meta = set(tag_features["track_id"]) - set(meta["track_id"])
        check(
            "tag_features track_id joinable to metadata",
            len(missing_from_meta) == 0,
            f"{len(missing_from_meta)} track_id values missing from track_metadata",
        )
    PASSES.append(
        f"[OK]   tag_features.csv    — {len(tag_features)} rows | {len(tfidf_cols)} tfidf columns"
    )

# ── Report ────────────────────────────────────────────────────────────────────
print()
print("=" * 56)
print("  VALIDATION REPORT")
print("=" * 56)
for msg in PASSES:
    print(" ", msg)
if ERRORS:
    print()
    for msg in ERRORS:
        print(" ", msg)
    print()
    print(f"  {len(ERRORS)} check(s) FAILED. Fix before sharing with team.")
    sys.exit(1)
else:
    print()
    print(f"  All {len(PASSES)} checks passed. Ready to share!")
print("=" * 56)
