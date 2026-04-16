#!/usr/bin/env python3
"""Validate artist-level pipeline outputs, while tolerating legacy filenames."""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

ERRORS = []
PASSES = []


def check(label: str, condition: bool, msg: str = "") -> None:
    if condition:
        PASSES.append(f"[OK]   {label}")
    else:
        ERRORS.append(f"[FAIL] {label}" + (f"  →  {msg}" if msg else ""))


def load(name: str):
    path = PROCESSED / name
    if not path.exists():
        ERRORS.append(f"[FAIL] {name} not found at {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        ERRORS.append(f"[FAIL] {name} could not be read: {e}")
        return None


def canonical_id_column(df: pd.DataFrame):
    if "artist_id" in df.columns:
        return "artist_id"
    if "track_id" in df.columns:
        return "track_id"
    return None


meta = load("track_metadata.csv")
meta_ids = set()
if meta is not None:
    id_col = canonical_id_column(meta)
    check("track_metadata has id column", id_col is not None)
    check("track_metadata has artist", "artist" in meta.columns)
    if id_col:
        check("track_metadata no null ids", meta[id_col].isna().sum() == 0)
        check("track_metadata no duplicate ids", meta[id_col].duplicated().sum() == 0)
        meta_ids = set(meta[id_col].dropna().astype(int))

ratings = load("ratings_joined.csv")
if ratings is not None:
    id_col = canonical_id_column(ratings)
    check("ratings_joined has user_id", "user_id" in ratings.columns)
    check("ratings_joined has item id", id_col is not None)
    check("ratings_joined has rating", "rating" in ratings.columns)
    if id_col:
        check("ratings_joined ids join to metadata", set(ratings[id_col].dropna().astype(int)) <= meta_ids)

train = load("ratings_train.csv")
test = load("ratings_test.csv")
if train is not None and test is not None and ratings is not None:
    total = len(train) + len(test)
    check("train + test == ratings_joined", total == len(ratings))
    check("no test-only users", len(set(test["user_id"]) - set(train["user_id"])) == 0)

master = load("master_tracks.csv")
if master is not None:
    id_col = canonical_id_column(master)
    check("master_tracks has item id", id_col is not None)
    for col in ["user_id", "artist", "rating"]:
        check(f"master_tracks has {col}", col in master.columns)

tag_features = load("tag_features.csv")
if tag_features is not None:
    id_col = canonical_id_column(tag_features)
    check("tag_features has item id", id_col is not None)
    check("tag_features has tags_raw", "tags_raw" in tag_features.columns)
    tfidf_cols = [c for c in tag_features.columns if c.startswith("tfidf_")]
    check("tag_features has tfidf columns", len(tfidf_cols) > 0)
    if id_col:
        check("tag_features unique ids", tag_features[id_col].duplicated().sum() == 0)
        check("tag_features ids join to metadata", set(tag_features[id_col].dropna().astype(int)) <= meta_ids)

mapping = load("musicnet_audio_map.csv")
if mapping is not None:
    check("musicnet_audio_map has artist_id", "artist_id" in mapping.columns)
    check("musicnet_audio_map has musicnet_id", "musicnet_id" in mapping.columns)
    if {"artist_id", "musicnet_id"}.issubset(mapping.columns):
        n_artists = mapping["artist_id"].nunique()
        check("musicnet_audio_map covers multiple artists", n_artists >= 5,
              f"only {n_artists} artist(s) mapped")

# ── audio_features.csv (Person 3 processed output) ───────────────────────────
audio_feat = load("audio_features.csv")
if audio_feat is not None:
    af_id = canonical_id_column(audio_feat)
    check("audio_features has item id", af_id is not None)
    if af_id:
        check("audio_features no null ids", audio_feat[af_id].isna().sum() == 0)
        check("audio_features unique ids", audio_feat[af_id].duplicated().sum() == 0)
    mfcc_cols = [c for c in audio_feat.columns if c.startswith("mfcc_")]
    check("audio_features has mfcc columns", len(mfcc_cols) > 0,
          "expected mfcc_* columns")
    PASSES.append(f"[OK]   audio_features.csv    — {len(audio_feat)} artists | {len(mfcc_cols)} mfcc cols")

# ── evaluation_results.csv (Person 4) ────────────────────────────────────────
eval_df = load("evaluation_results.csv")
if eval_df is not None:
    for col in ["model", "rmse", "mae"]:
        check(f"evaluation_results has {col}", col in eval_df.columns)
    if "rmse" in eval_df.columns:
        check("evaluation_results rmse values valid",
              eval_df["rmse"].between(0.0, 5.0).all(),
              f"min={eval_df['rmse'].min():.3f} max={eval_df['rmse'].max():.3f}")
    PASSES.append(f"[OK]   evaluation_results.csv — {len(eval_df)} model(s) evaluated")

# ── recommendations.csv (Person 4) ───────────────────────────────────────────
recs_df = load("recommendations.csv")
if recs_df is not None:
    for col in ["user_id", "track_id", "artist", "rank", "predicted_rating"]:
        check(f"recommendations has {col}", col in recs_df.columns)
    if "rank" in recs_df.columns:
        check("recommendations rank starts at 1", recs_df["rank"].min() == 1)
    PASSES.append(f"[OK]   recommendations.csv   — {recs_df['user_id'].nunique() if 'user_id' in recs_df.columns else '?'} users, {len(recs_df)} recs")

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
