#!/usr/bin/env python3
"""Validate artist-level pipeline outputs, while tolerating legacy filenames."""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

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
        check("musicnet_audio_map unique musicnet_id", mapping["musicnet_id"].duplicated().sum() == 0, f"{mapping['musicnet_id'].duplicated().sum()} duplicates")

for name in ["audio_features_artist_train.csv", "audio_features_artist_test.csv"]:
    audio = load(name)
    if audio is not None:
        check(f"{name} has artist_id", "artist_id" in audio.columns)
        check(f"{name} has recording_count", "recording_count" in audio.columns)

results_path = RESULTS / "ablation_results.csv"
if results_path.exists():
    try:
        results_df = pd.read_csv(results_path)
        check("ablation_results has model column", "model" in results_df.columns)
        check("ablation_results has rmse column", "rmse" in results_df.columns)
        check("ablation_results has evaluation_scope column", "evaluation_scope" in results_df.columns)
    except Exception as e:
        ERRORS.append(f"[FAIL] results/ablation_results.csv could not be read: {e}")
else:
    ERRORS.append(f"[FAIL] ablation_results.csv not found at {results_path}")

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
