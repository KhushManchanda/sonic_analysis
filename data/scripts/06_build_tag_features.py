#!/usr/bin/env python3
"""Build artist-level Last.fm semantic tag features."""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time
from typing import Iterable

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer


# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
HETREC_DIR = RAW / "hetrec2011-lastfm-2k"
PROCESSED = ROOT / "data" / "processed"
CACHE_DIR = RAW / "lastfm_cache"

TRACK_METADATA_PATH = PROCESSED / "track_metadata.csv"
TAGS_PATH = HETREC_DIR / "tags.dat"
USER_TAGS_PATH = HETREC_DIR / "user_taggedartists.dat"
OUTPUT_PATH = PROCESSED / "tag_features.csv"
CACHE_PATH = CACHE_DIR / "artist_top_tags.json"


# ── Tunables ──────────────────────────────────────────────────────────────────
TOP_TAGS_RAW = 10
MAX_FEATURES = 100
MIN_DF = 2
API_TIMEOUT = 20
API_SLEEP_SECONDS = 0.25


def normalize_tag(tag: str) -> str:
    """Conservative tag normalization for interpretability."""
    if tag is None:
        return ""
    text = str(tag).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[\[\]{}()<>\"'`]+", " ", text)
    text = re.sub(r"[\/_|]+", " ", text)
    text = re.sub(r"[^a-z0-9+\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def require_inputs() -> None:
    missing = [
        path for path in (TRACK_METADATA_PATH, TAGS_PATH, USER_TAGS_PATH) if not path.exists()
    ]
    if missing:
        print("[ERROR] Missing required input(s):")
        for path in missing:
            print(f"  - {path}")
        print("Run `bash data/scripts/01_download.sh` and ensure Person 1 outputs exist.")
        sys.exit(1)


def load_cache() -> dict[str, list[str]]:
    if not CACHE_PATH.exists():
        return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_cache(cache: dict[str, list[str]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, indent=2, sort_keys=True)


def fetch_artist_top_tags(artist: str, api_key: str, cache: dict[str, list[str]]) -> list[str]:
    """
    Fallback fetch from Last.fm.

    HetRec tags are artist-centric and the shared pipeline uses artist_id,
    so artist.getTopTags is the compatible fallback when no local tags exist.
    """
    key = artist.strip().lower()
    if not key:
        return []
    if key in cache:
        return cache[key]

    params = {
        "method": "artist.getTopTags",
        "artist": artist,
        "api_key": api_key,
        "format": "json",
        "autocorrect": 1,
    }
    try:
        response = requests.get(
            "https://ws.audioscrobbler.com/2.0/",
            params=params,
            timeout=API_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        tag_nodes = payload.get("toptags", {}).get("tag", [])
        if isinstance(tag_nodes, dict):
            tag_nodes = [tag_nodes]
        tags = []
        for node in tag_nodes:
            name = normalize_tag(node.get("name", ""))
            if name:
                tags.append(name)
        tags = unique_preserve_order(tags)
    except Exception:
        tags = []

    cache[key] = tags
    time.sleep(API_SLEEP_SECONDS)
    return tags


def build_local_tag_table() -> pd.DataFrame:
    print(f"Loading {TAGS_PATH} ...")
    tags = pd.read_csv(TAGS_PATH, sep="\t", encoding="iso-8859-1", on_bad_lines="skip")
    tags.columns = [c.strip().lower() for c in tags.columns]
    tags = tags.rename(columns={"tagid": "tag_id", "tagvalue": "tag_name"})
    tags["tag_id"] = pd.to_numeric(tags["tag_id"], errors="coerce").astype("Int64")
    tags["tag_name"] = tags["tag_name"].map(normalize_tag)
    tags = tags.dropna(subset=["tag_id"])
    tags = tags[tags["tag_name"].ne("")]
    tags = tags[["tag_id", "tag_name"]].drop_duplicates()

    print(f"Loading {USER_TAGS_PATH} ...")
    user_tags = pd.read_csv(USER_TAGS_PATH, sep="\t", encoding="utf-8", on_bad_lines="skip")
    user_tags.columns = [c.strip().lower() for c in user_tags.columns]
    user_tags = user_tags.rename(columns={"artistid": "artist_id", "tagid": "tag_id"})
    user_tags["artist_id"] = pd.to_numeric(user_tags["artist_id"], errors="coerce").astype("Int64")
    user_tags["tag_id"] = pd.to_numeric(user_tags["tag_id"], errors="coerce").astype("Int64")
    user_tags = user_tags.dropna(subset=["artist_id", "tag_id"])

    merged = user_tags.merge(tags, on="tag_id", how="left")
    merged = merged.dropna(subset=["tag_name"])

    # Count distinct users per (artist, tag) to avoid overweighting repeated annotations.
    agg = (
        merged.groupby(["artist_id", "tag_name"], as_index=False)["userid"]
        .nunique()
        .rename(columns={"userid": "tag_user_count"})
        .sort_values(["artist_id", "tag_user_count", "tag_name"], ascending=[True, False, True])
    )
    agg["artist_id"] = agg["artist_id"].astype(int)
    return agg


def aggregate_tags_by_artist(tag_counts: pd.DataFrame) -> tuple[pd.DataFrame, set[int]]:
    tag_lists = (
        tag_counts.groupby("artist_id")
        .apply(
            lambda frame: pd.Series(
                {
                    "tags_all": unique_preserve_order(frame["tag_name"].tolist()),
                    "tags_raw": ",".join(unique_preserve_order(frame["tag_name"].head(TOP_TAGS_RAW).tolist())),
                }
            )
        )
        .reset_index()
    )
    local_ids = set(tag_lists["artist_id"].tolist())
    return tag_lists, local_ids


def maybe_enrich_with_api(artist_meta: pd.DataFrame, tag_lists: pd.DataFrame) -> pd.DataFrame:
    api_key = os.getenv("LASTFM_API_KEY")
    all_artists = artist_meta[["artist_id", "artist"]].copy()
    merged = all_artists.merge(tag_lists, on="artist_id", how="left")
    missing_mask = merged["tags_all"].isna() | merged["tags_all"].map(lambda x: len(x) == 0 if isinstance(x, list) else True)
    missing_rows = merged.loc[missing_mask, ["artist_id", "artist"]].drop_duplicates()

    if missing_rows.empty:
        print("No artists missing local tags.")
        return merged

    print(f"Artists missing local tags: {len(missing_rows)}")
    if not api_key:
        print("[INFO] LASTFM_API_KEY not set — skipping API fallback.")
        return merged

    cache = load_cache()
    print(f"Using Last.fm fallback for up to {len(missing_rows)} artists with cache at {CACHE_PATH}")

    fallback_rows: list[dict[str, object]] = []
    for _, row in missing_rows.iterrows():
        tags = fetch_artist_top_tags(str(row["artist"]), api_key=api_key, cache=cache)
        fallback_rows.append(
            {
                "artist_id": int(row["artist_id"]),
                "tags_all": tags,
                "tags_raw": ",".join(tags[:TOP_TAGS_RAW]),
            }
        )

    save_cache(cache)

    fallback_df = pd.DataFrame(fallback_rows)
    if fallback_df.empty:
        return merged

    merged = merged.drop(columns=["tags_all", "tags_raw"], errors="ignore")
    merged = merged.merge(tag_lists, on="artist_id", how="left")

    # Fill only where local tags are absent.
    merged = merged.merge(
        fallback_df.rename(columns={"tags_all": "tags_all_fallback", "tags_raw": "tags_raw_fallback"}),
        on="artist_id",
        how="left",
    )
    merged["tags_all"] = merged["tags_all"].where(
        merged["tags_all"].map(lambda x: isinstance(x, list) and len(x) > 0),
        merged["tags_all_fallback"],
    )
    merged["tags_raw"] = merged["tags_raw"].fillna(merged["tags_raw_fallback"])
    merged = merged.drop(columns=["tags_all_fallback", "tags_raw_fallback"])
    return merged


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    documents = df["tags_all"].map(lambda tags: " ".join(tags) if isinstance(tags, list) else "")
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b[a-z0-9][a-z0-9+\-]*\b",
        lowercase=False,
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
    )
    matrix = vectorizer.fit_transform(documents)
    feature_names = [f"tfidf_{name.replace('-', '_').replace('+', 'plus')}" for name in vectorizer.get_feature_names_out()]
    features = pd.DataFrame(matrix.toarray(), columns=feature_names, index=df.index)
    out = pd.concat([df[["artist_id", "tags_raw"]].copy(), features], axis=1)

    # Ensure rows with no tags still have usable numeric feature columns.
    out["tags_raw"] = out["tags_raw"].fillna("")
    for col in feature_names:
        out[col] = out[col].fillna(0.0).astype(float)
    return out


def validate_output(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("tag_features output is empty")
    if "artist_id" not in df.columns or "tags_raw" not in df.columns:
        raise ValueError("tag_features output missing required columns")
    if df["artist_id"].isna().any():
        raise ValueError("tag_features contains null artist_id")
    if df["artist_id"].duplicated().any():
        raise ValueError("tag_features contains duplicate artist_id")

    numeric_cols = [c for c in df.columns if c.startswith("tfidf_")]
    if not numeric_cols:
        raise ValueError("tag_features contains no numeric tfidf_* columns")
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"{col} is not numeric")


def main() -> None:
    require_inputs()
    PROCESSED.mkdir(parents=True, exist_ok=True)

    artist_meta = pd.read_csv(TRACK_METADATA_PATH, dtype={"artist_id": int})[["artist_id", "artist"]]
    artist_meta = artist_meta.drop_duplicates(subset=["artist_id"]).sort_values("artist_id").reset_index(drop=True)

    tag_counts = build_local_tag_table()
    tag_lists, _ = aggregate_tags_by_artist(tag_counts)
    enriched = maybe_enrich_with_api(artist_meta, tag_lists)
    enriched["tags_all"] = enriched["tags_all"].map(
        lambda x: unique_preserve_order([normalize_tag(v) for v in x]) if isinstance(x, list) else []
    )
    enriched["tags_raw"] = enriched["tags_raw"].fillna("")

    feature_df = build_feature_matrix(enriched)
    feature_df = feature_df.sort_values("artist_id").reset_index(drop=True)
    validate_output(feature_df)

    feature_df.to_csv(OUTPUT_PATH, index=False)

    nonempty = int(feature_df["tags_raw"].ne("").sum())
    tfidf_cols = [c for c in feature_df.columns if c.startswith("tfidf_")]
    print(f"[DONE] tag_features.csv → {OUTPUT_PATH}")
    print(f"  Rows             : {len(feature_df)}")
    print(f"  Artists with tags: {nonempty}")
    print(f"  TF-IDF columns   : {len(tfidf_cols)}")
    print(feature_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()