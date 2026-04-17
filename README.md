# 🎵 Sonic Analysis — Hybrid Music Recommendation System

> **CSE 575 — Statistical Machine Learning | Spring 2026**
> Final integrated pipeline for artist-level recommendation using collaborative filtering, semantic tags, and audio-derived content features.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Question](#2-research-question)
3. [Team & Roles](#3-team--roles)
4. [System Architecture](#4-system-architecture)
5. [Dataset](#5-dataset)
6. [Repository Structure](#6-repository-structure)
7. [Getting Started](#7-getting-started)
8. [What's Already Done](#8-whats-already-done-person-1--data-pipeline)
9. [Next Steps — Complete Roadmap](#9-next-steps--complete-roadmap)
10. [Evaluation Plan](#10-evaluation-plan)
11. [Minimum Deliverable](#11-minimum-deliverable)
12. [File Schema Reference](#12-file-schema-reference)

---

## 1. Project Overview

Traditional music recommenders rely on user–item ratings alone. Two artists can still be similar even if they do not share many listeners, because they may share sonic characteristics or semantic community tags. This failure is worst in sparse and near-cold settings.

This repository now implements a full end-to-end, **artist-level** hybrid recommender that combines:

| Signal | Source | Responsible |
|--------|--------|-------------|
| **Ratings** (collaborative filtering) | HetRec 2011 Last.fm 2K | Person 1 ✅ |
| **Semantic tags** (Last.fm community labels) | Last.fm API + HetRec local cache | Person 2 ✅|
| **Audio features** (local sonic analysis) | MusicNet WAV + Librosa | Person 3 ✅ |
| **Models + Evaluation** | NumPy + scikit-learn | Integrated |

### Final repo reality

- Canonical item key: `artist_id`
- Legacy filenames such as `track_metadata.csv` and `master_tracks.csv` are preserved for compatibility
- Audio is evaluated on an overlap subset only when mapped MusicNet coverage exists
- In the checked-in environment, audio coverage is extremely small and test-overlap may be empty

---

## 2. Research Question

> **How much do audio-derived features and Last.fm tags improve rating prediction and top-N recommendation quality, especially for sparse users/items?**

We run **ablations** comparing:
- `ratings-only` — collaborative filtering baseline
- `ratings + tags` — hybrid with semantic signals
- `ratings + audio` — hybrid with sonic signals
- `ratings + tags + audio` — full hybrid

---

## 3. Team & Roles

| Person | Name | Role | Status |
|--------|------|------|--------|
| **Person 1** | Khush Manchanda | Data + pipeline owner | ✅ **Complete** |
| **Person 2** | Arjun Ranjan | Last.fm tag pipeline | ✅ **Complete** |
| **Person 3** | Diggy | Audio feature extraction | ✅ **Complete** |
| **Person 4** | Ninjaman | Modeling + evaluation | 🔲 In progress |

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Raw Data                             │
│  HetRec 2011 Last.fm 2K    +    MusicNet (.wav files)       │
└───────────────┬─────────────────────────┬───────────────────┘
                │                         │
                ▼                         ▼
┌──────────────────────┐    ┌─────────────────────────────┐
│   Person 1 Pipeline  │    │   Person 3: Audio Features  │
│  track_metadata.csv  │    │   audio_features_train.csv  │
│  ratings_train.csv   │    │   audio_features_test.csv   │
│  ratings_test.csv    │    │   (MFCC, tempo, chroma...)  │
│  master_tracks.csv   │    └─────────────┬───────────────┘
└──────┬───────────────┘    ┌─────────────▼───────────────┐
       │                    │   Person 2: Tag Features    │
       │                    │   tag_features.csv          │
       │                    │   (TF-IDF over Last.fm tags)│
       │                    └─────────────┬───────────────┘
       │                                  │
       └──────────────┬───────────────────┘
                      ▼
       ┌──────────────────────────────────┐
       │     Person 4: Recommendation    │
       │  Baselines → Matrix Factorization│
       │  → Hybrid Model (MF + tags/audio)│
       └──────────────┬───────────────────┘
                      ▼
       ┌──────────────────────────────────┐
       │   Evaluation: RMSE, MAE,        │
       │   Precision@K, NDCG@K           │
       │   Ablation table + figures      │
       └──────────────────────────────────┘
```

---

## 5. Dataset

### Primary: HetRec 2011 Last.fm 2K
> Source: https://grouplens.org/datasets/hetrec-2011/

| Stat | Value |
|------|-------|
| Users | 1,892 |
| Artists (items) | 17,632 |
| User-artist interactions | 92,834 |
| Rating scale | 1.0 – 5.0 (log-normalized from play counts) |
| Matrix sparsity | 99.72% |
| Median ratings per user | 50 |

**Included files (already downloaded to `data/raw/hetrec2011-lastfm-2k/`):**
- `user_artists.dat` — play counts per (user, artist)
- `artists.dat` — artist names + Last.fm URLs
- `tags.dat` — tag vocabulary (Person 2 can use this directly)
- `user_taggedartists.dat` — which users tagged which artists with which tags (Person 2)
- `user_friends.dat` — social graph (optional)

### Secondary: MusicNet (audio source)
> Source: https://zenodo.org/records/5120004

330 classical music recordings (.wav + CSV note annotations). Used as the audio source for artists that overlap with HetRec. Mapping is in `data/processed/musicnet_audio_map.csv`.

---

## 6. Repository Structure

```
sonic_analysis/
│
├── README.md                        ← You are here
├── README_schema.md                 ← SHARED SCHEMA — all teammates must read
├── requirements.txt                 ← pip install -r requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                         ← gitignored — run 01_download.sh to generate
│   │   ├── musicnet_metadata.csv
│   │   ├── musicnet/                ← extracted from musicnet.tar.gz (~11 GB)
│   │   │   ├── train_data/          ← WAV files used for audio_features_train.csv
│   │   │   └── test_data/           ← WAV files used for audio_features_test.csv
│   │   └── hetrec2011-lastfm-2k/
│   │       ├── artists.dat
│   │       ├── user_artists.dat
│   │       ├── tags.dat             ← Person 2: local tag cache
│   │       ├── user_taggedartists.dat  ← Person 2: user-tag assignments
│   │       └── user_friends.dat
│   │
│   ├── processed/                   ← tracked in git — shared outputs
│   │   ├── track_metadata.csv       ✅ Person 1 — 17,632 artists
│   │   ├── ratings_joined.csv       ✅ Person 1 — 92,834 ratings
│   │   ├── ratings_train.csv        ✅ Person 1 — 74,265 rows (80%)
│   │   ├── ratings_test.csv         ✅ Person 1 — 18,569 rows (20%)
│   │   ├── master_tracks.csv        ✅ Person 1 — flat team table
│   │   ├── musicnet_audio_map.csv   ✅ Person 1 — WAV file mapping for audio
│   │   ├── tag_features.csv         🔲 Person 2 — TODO
│   │   ├── audio_features_train.csv ✅ Person 3 — Librosa features, train split
│   │   └── audio_features_test.csv  ✅ Person 3 — Librosa features, test split
│   │
│   └── scripts/                     ← reproducible pipeline
│       ├── 01_download.sh           ← also downloads + extracts musicnet.tar.gz
│       ├── 02_clean_metadata.py
│       ├── 03_join_ratings.py
│       ├── 04_split.py
│       ├── 05_export_master.py
│       ├── 06_build_tag_features.py
│       ├── 07_embed_tracks.py       ← Person 3: audio feature extraction
│       ├── validate_outputs.py
│       └── run_pipeline.sh
│
├── notebooks/                       ← EDA, experimentation, figures
│   └── (add your .ipynb files here)
│
└── models/                          ← Person 4: saved model artifacts
    └── (add trained model files here)
```

---

## 7. Getting Started

### Prerequisites
- Python 3.10+
- `curl`, `unzip` (for downloading data)

### Setup

```bash
# Clone the repo
git clone <repo-url>
cd sonic_analysis

# Create local virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Run the full pipeline without re-downloading existing raw data
bash data/scripts/run_pipeline.sh --skip-download --workers=4
```

If raw data is missing and you want a full download, run:

```bash
bash data/scripts/run_pipeline.sh --workers=12
```

After running, `data/processed/` will contain integration-ready tables and `results/` will contain final evaluation outputs.

### Validate outputs
```bash
.venv/bin/python data/scripts/validate_outputs.py
```

---

## 8. Finished Pipeline Stages

The current finished project supports these stages:

| Script | What it does | Output |
|--------|-------------|--------|
| `01_download.sh` | Downloads MusicNet + HetRec 2011 | `data/raw/` |
| `02_clean_metadata.py` | Builds 17,632-artist catalog from HetRec | `track_metadata.csv` |
| `03_join_ratings.py` | Joins play counts → 1–5 ratings | `ratings_joined.csv` |
| `04_split.py` | 80/20 stratified per-user split + audio-overlap subsets | `ratings_train/test.csv`, `ratings_*_overlap.csv` |
| `05_export_master.py` | Flat join + dataset stats | `master_tracks.csv` |
| `06_build_tag_features.py` | Builds artist-level local Last.fm TF-IDF features | `tag_features.csv` |
| `07_embed_tracks.py` | Produces/rebuilds artist-level audio tables from MusicNet embeds | `audio_features_artist_*.csv` |
| `08_run_experiments.py` | Runs baselines, MF, hybrids, metrics, slices, summaries | `results/*` |
| `validate_outputs.py` | Validates processed files and final metrics table | Pass/fail report |

**Key design decisions:**
- `artist_id` is the canonical item key
- Legacy `track_*` filenames are retained where already embedded in the repo
- Play counts are log-normalized to a 1–5 explicit rating target
- Split is stratified by user so no user exists only in test
- Audio experiments are only trusted on the overlap subset and are marked unavailable when no test overlap exists

---

## 9. Final Experiment Design

> **Rule:** Final integrated tables should use `artist_id`. Raw MusicNet embed intermediates may use `musicnet_id` and must join through `musicnet_audio_map.csv`.

---

### Audio pipeline

**Status: Complete as of April 12, 2026**

**Goal:** Produce Librosa-extracted audio features keyed by `musicnet_id`, split into train and test to match the MusicNet directory structure.

**How it works:**

`07_embed_tracks.py` reads directly from `data/raw/musicnet_metadata.csv`, detects which track IDs exist in `train_data/` vs `test_data/`, and writes one output CSV per split. Extraction is incremental — if the script is interrupted it resumes from where it left off on the next run.

```bash
# Basic run
python3 data/scripts/07_embed_tracks.py

# With parallelism (recommended — supports up to --workers 16 on most machines)
python3 data/scripts/07_embed_tracks.py --workers 12
```

**Output schema:**

```
musicnet_id       (str)   — MusicNet track ID; join to musicnet_audio_map.csv to get track_id
tempo             (float) — BPM
rms_mean          (float) — RMS energy mean
mfcc_1_mean  ...  mfcc_20_mean  (float) — MFCC coefficient means
mfcc_1_std   ...  mfcc_20_std   (float) — MFCC coefficient stds
chroma_1_mean ... chroma_12_mean (float) — per-bin chroma means
contrast_1_mean ... contrast_7_mean (float) — per-band spectral contrast means
```

Total: 61 feature columns + `musicnet_id`. To join with the rest of the team's data:

```python
import pandas as pd

audio   = pd.read_csv("data/embeds/audio_features_train.csv")
mapping = pd.read_csv("data/processed/musicnet_audio_map.csv")

# mapping has: track_id (HetRec artist_id), musicnet_id
audio = audio.merge(mapping, on="musicnet_id", how="left")
# audio now has track_id and can join with ratings, tags, etc.
```

---

### Tag pipeline

**Goal:** Produce `data/processed/tag_features.csv` — one row per artist with tag-based features.

**Important shortcut:** The HetRec dataset already contains local tag data. **Do not hit the API from scratch.** Start here:

```python
import pandas as pd

# Already downloaded — no API needed for these
tags        = pd.read_csv("data/raw/hetrec2011-lastfm-2k/tags.dat",
                          sep="\t", encoding="utf-8", on_bad_lines="skip")
user_tags   = pd.read_csv("data/raw/hetrec2011-lastfm-2k/user_taggedartists.dat",
                          sep="\t", encoding="utf-8")
catalog     = pd.read_csv("data/processed/track_metadata.csv")
# track_id = artistID in all three files
```

**Step-by-step:**
1. Aggregate `user_taggedartists.dat` → count how many users applied each tag to each artist
2. Merge `tags.dat` to get tag names
3. Normalize tag text (lowercase, strip, deduplicate)
4. Build TF-IDF matrix over tag vocabulary
5. Use the Last.fm API **only** for artists whose tag counts are 0 in the local cache
6. Save to `data/processed/tag_features.csv`

**Implementation note:** HetRec is artist-centered and Person 1 defines `track_id = artist_id`, so this repo's compatible fallback is artist top tags keyed by `track_id`, with cached API responses stored under `data/raw/lastfm_cache/`.

**Required output schema:**
```
track_id  (int)   — must match master_tracks.csv
tags_raw  (str)   — comma-separated top tags e.g. "rock,indie,alternative"
tfidf_*   (float) — one column per tag in vocabulary (sparse is fine)
```

---

### Modeling & evaluation

**Goal:** Train baselines → matrix factorization → hybrid model. Produce an evaluation table.

The current experiment runner produces:
```python
import pandas as pd
train = pd.read_csv("data/processed/ratings_train.csv")
test  = pd.read_csv("data/processed/ratings_test.csv")
# Columns: user_id, artist_id, artist, rating
```

| Model family | Implemented |
|-------------|-------------|
| Global mean | ✅ |
| User mean | ✅ |
| Item mean | ✅ |
| User-based kNN CF | ✅ |
| Matrix factorization | ✅ |
| MF + tags | ✅ |
| MF + audio | ✅ overlap-aware |
| MF + tags + audio | ✅ overlap-aware |

**Required ablation table:**

| Model | RMSE | MAE | P@10 | NDCG@10 |
|-------|------|-----|------|---------|
| Global mean | | | | |
| User mean | | | | |
| Item mean | | | | |
| Matrix Factorization (ratings only) | | | | |
| MF + tags | | | | |
| MF + audio | | | | |
| MF + tags + audio | | | | |

---

### 🔲 Person 5 — Report & Presentation

**Goal:** Write the final report and slides while results come in.

**Sections to draft now (fill numbers later):**
1. Introduction & motivation
2. Related work (collaborative filtering, content-based, hybrid)
3. Dataset description (can fill in now from this README)
4. Method: baselines → MF → hybrid
5. Results: ablation table + figures
6. Cold-start analysis
7. Qualitative examples (demo recommendations)
8. Conclusion

**Figures to prepare:**
- Rating distribution histogram
- Sparsity visualization
- RMSE comparison bar chart (ratings-only vs +tags vs +audio)
- Example recommendation output table

---

## 10. Evaluation Protocol

### Rating prediction (quantitative)
- **RMSE** and **MAE** on `ratings_test.csv`
- Lower is better

### Top-N recommendation (ranking quality)
- Binarize: rating ≥ 4.0 = "liked"
- Metrics: **Precision@K**, **Recall@K**, **NDCG@K** (K = 10)

### Sparse / low-support analysis
- Sparse user slice: users with ≤ 10 training interactions
- Low-support item slice: artists with ≤ 2 training interactions
- Saved to `results/slice_metrics.csv`

---

## 11. Final Outputs

- `data/processed/tag_features.csv`
- `data/processed/audio_features_artist_train.csv`
- `data/processed/audio_features_artist_test.csv`
- `results/ablation_results.csv`
- `results/ablation_results.md`
- `results/slice_metrics.csv`
- `results/qualitative_examples.csv`
- `results/qualitative_examples.md`
- `docs/implementation_status.md`

---

## 12. File Schema Reference

See [`README_schema.md`](README_schema.md) for the complete column-level specification.

**Golden rule: final integrated recommender tables use `artist_id` as the canonical item key.**

| File | Canonical join key | Notes |
|------|--------------------|-------|
| `track_metadata.csv` | `artist_id` | Legacy filename, artist-level rows |
| `ratings_train.csv` | `artist_id` | Main training ratings |
| `ratings_test.csv` | `artist_id` | Main test ratings |
| `ratings_train_overlap.csv` | `artist_id` | Audio-overlap train subset |
| `ratings_test_overlap.csv` | `artist_id` | Audio-overlap test subset |
| `master_tracks.csv` | `artist_id` | Legacy filename, artist-level |
| `tag_features.csv` | `artist_id` | Local HetRec tags + optional Last.fm fallback |
| `audio_features_artist_train.csv` | `artist_id` | Aggregated from MusicNet embeddings |
| `audio_features_artist_test.csv` | `artist_id` | May be empty if no overlap test coverage |
| `results/ablation_results.csv` | n/a | Final experiment summary table |

---

*Last updated: April 16, 2026 after full pipeline integration.*