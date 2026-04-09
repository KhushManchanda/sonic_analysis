# 🎵 Sonic Analysis — Music Recommendation System

> **CSE 575 — Statistical Machine Learning | Spring 2026**
> Improving music recommendation accuracy by combining collaborative filtering with audio-derived features and Last.fm semantic tags.

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

Traditional music recommenders rely on user–item ratings alone. Two tracks can be similar even if they share no listeners — because they share sonic characteristics (tempo, timbre, energy) or semantic meaning (mood, genre, style tags). This failure is worst in **cold-start** settings where rating history is sparse.

AudioMuse-AI builds a hybrid recommender that combines:

| Signal | Source | Responsible |
|--------|--------|-------------|
| **Ratings** (collaborative filtering) | HetRec 2011 Last.fm 2K | Person 1 ✅ |
| **Semantic tags** (Last.fm community labels) | Last.fm API + HetRec local cache | Person 2 ✅|
| **Audio features** (local sonic analysis) | MusicNet WAV + Librosa | Person 3 |
| **Models + Evaluation** | scikit-learn, LightFM | Person 4 |

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
| **Person 3** | Diggy | Audio feature extraction | 🔲 In progress |
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
│  track_metadata.csv  │    │   audio_features.csv        │
│  ratings_train.csv   │    │   (MFCC, tempo, chroma...)  │
│  ratings_test.csv    │    └─────────────┬───────────────┘
│  master_tracks.csv   │                  │
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
│   │   └── audio_features.csv       🔲 Person 3 — TODO
│   │
│   └── scripts/                     ← Person 1's reproducible pipeline
│       ├── 01_download.sh
│       ├── 02_clean_metadata.py
│       ├── 03_join_ratings.py
│       ├── 04_split.py
│       ├── 05_export_master.py
│       ├── 06_build_tag_features.py
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

# Install Python dependencies
pip install -r requirements.txt

# Run the data pipeline (Person 1's work — already done but reproducible)
bash data/scripts/run_pipeline.sh
```

After running, `data/processed/` will contain all output files.

### Validate outputs
```bash
python3 data/scripts/validate_outputs.py
# Expected: All 23 checks passed. Ready to share!
```

---

## 8. What's Already Done (Person 1 — Data Pipeline)

**Status: ✅ Complete as of April 9, 2026**

Person 1 has delivered the full shared data foundation:

| Script | What it does | Output |
|--------|-------------|--------|
| `01_download.sh` | Downloads MusicNet + HetRec 2011 | `data/raw/` |
| `02_clean_metadata.py` | Builds 17,632-artist catalog from HetRec | `track_metadata.csv` |
| `03_join_ratings.py` | Joins play counts → 1–5 ratings | `ratings_joined.csv` |
| `04_split.py` | 80/20 stratified per-user split | `ratings_train/test.csv` |
| `05_export_master.py` | Flat join + dataset stats | `master_tracks.csv` |
| `validate_outputs.py` | 23 automated sanity checks | Pass/fail report |

**Key design decisions:**
- `track_id` = HetRec `artist_id` — stable integer, same across all files
- Play counts are **log-normalized** to 1–5 scale (handles power-law distribution)
- Split is **stratified by user** — no user appears only in test (prevents cold-start leakage)
- Sparse users with < 2 ratings go entirely to train

---

## 9. Next Steps — Complete Roadmap

> **Rule:** Every output file must have a `track_id` column (integer) to join with Person 1's data. See `README_schema.md` for exact column specs.

---

### 🔲 Person 2 — Last.fm Tag Pipeline

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

### 🔲 Person 3 — Audio Feature Extraction

**Goal:** Produce `data/processed/audio_features.csv` — one row per artist with Librosa-extracted features.

**Start with the MusicNet overlap:**

```python
import pandas as pd

# Person 1 already mapped which track_ids have audio
audio_map = pd.read_csv("data/processed/musicnet_audio_map.csv")
# Columns: track_id, musicnet_id
# WAV files are at: data/raw/musicnet/{musicnet_id}.wav (after downloading musicnet.tar.gz)
```

**Step-by-step:**
1. Download `musicnet.tar.gz` from https://zenodo.org/records/5120004 (11 GB — start early)
2. For each WAV file in `musicnet_audio_map.csv`, extract with Librosa:
   - **MFCC** (20 coefficients): mean + std → 40 features
   - **Tempo** (BPM): 1 feature
   - **RMS Energy**: mean → 1 feature
   - **Chroma**: mean → 12 features
   - **Spectral contrast**: mean → 7 features
3. Key each row by `track_id` (from the audio map)
4. Save to `data/processed/audio_features.csv`

**Starter code:**
```python
import librosa
import numpy as np

def extract_features(wav_path: str) -> dict:
    y, sr = librosa.load(wav_path, sr=22050, mono=True, duration=60)  # first 60s
    mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma      = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast    = librosa.feature.spectral_contrast(y=y, sr=sr)
    rms         = librosa.feature.rms(y=y)
    tempo, _    = librosa.beat.beat_track(y=y, sr=sr)
    feats = {"tempo": float(tempo), "rms_mean": float(rms.mean())}
    for i, (m, s) in enumerate(zip(mfcc.mean(axis=1), mfcc.std(axis=1))):
        feats[f"mfcc_{i+1}_mean"] = m; feats[f"mfcc_{i+1}_std"] = s
    feats["chroma_mean"]   = float(chroma.mean())
    feats["contrast_mean"] = float(contrast.mean())
    return feats
```

**Required output schema:**
```
track_id         (int)   — must match master_tracks.csv
tempo            (float)
rms_mean         (float)
mfcc_1_mean ... mfcc_20_mean  (float)
chroma_mean      (float)
contrast_mean    (float)
```

---

### 🔲 Person 4 — Modeling & Evaluation

**Goal:** Train baselines → matrix factorization → hybrid model. Produce an evaluation table.

**Step 1 — Load data (start immediately, don't wait for tags/audio):**
```python
import pandas as pd
train = pd.read_csv("data/processed/ratings_train.csv")
test  = pd.read_csv("data/processed/ratings_test.csv")
# Columns: user_id, track_id, artist, rating
```

**Step 2 — Baselines (implement first, today):**

| Baseline | Formula | Expected RMSE |
|----------|---------|---------------|
| Global mean | `ŷ = mean(train.rating)` | ~0.49 |
| User mean | `ŷ = mean(user ratings)` | < global mean |
| Item mean | `ŷ = mean(item ratings)` | < global mean |
| kNN (user-based) | cosine similarity | TBD |

**Step 3 — Matrix Factorization:**
```python
# Option A: surprise library
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

# Option B: LightFM (also supports hybrid)
from lightfm import LightFM
```

**Step 4 — Hybrid (once tag_features.csv and audio_features.csv exist):**
```python
tags  = pd.read_csv("data/processed/tag_features.csv")
audio = pd.read_csv("data/processed/audio_features.csv")
# Join on track_id, build item feature matrix, pass to LightFM
```

**Step 5 — Evaluation metrics:**
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
# For top-N: binarize ratings >= 3.5 as "positive"
# Then compute Precision@K, Recall@K, NDCG@K
```

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

## 10. Evaluation Plan

### Rating prediction (quantitative)
- **RMSE** and **MAE** on `ratings_test.csv`
- Lower is better

### Top-N recommendation (ranking quality)
- Binarize: rating ≥ 3.5 = "liked"
- Metrics: **Precision@K**, **Recall@K**, **NDCG@K** (K = 10)

### Cold-start analysis
- Segment test users by training rating count:
  - **Cold**: ≤ 5 ratings in train
  - **Near-cold**: 6–20 ratings
  - **Normal**: > 20 ratings
- Compare model performance across these segments to measure whether tags/audio help sparse users

---

## 11. Minimum Deliverable

If time runs short, the project is successful if it delivers:

- [x] Ratings-only baseline (global/user/item mean)
- [ ] One hybrid enhancement (+ tags OR + audio)
- [ ] One evaluation table (RMSE + MAE at minimum)
- [ ] One demo: example top-10 recommendations for a sample user

---

## 12. File Schema Reference

See [`README_schema.md`](README_schema.md) for the complete column-level specification.

**Golden rule: every file in `data/processed/` must have a `track_id` (int) column.**

| File | track_id maps to | Owner |
|------|-----------------|-------|
| `track_metadata.csv` | HetRec `artistID` | Person 1 ✅ |
| `ratings_train.csv` | HetRec `artistID` | Person 1 ✅ |
| `ratings_test.csv` | HetRec `artistID` | Person 1 ✅ |
| `master_tracks.csv` | HetRec `artistID` | Person 1 ✅ |
| `tag_features.csv` | HetRec `artistID` | Person 2 🔲 |
| `audio_features.csv` | HetRec `artistID` | Person 3 🔲 |

---

*Last updated: April 9, 2026 by Person 1*
