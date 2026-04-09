# AudioMuse-AI — Music Recommendation System

> CSE 575 Project | Spring 2026
> Improving Recommendation Accuracy using Local Sonic Analysis and Last.fm Top Tags

## Team

| Person | Role | Owner |
|--------|------|-------|
| Person 1 | Data + Pipeline | Dataset, joins, splits, master table |
| Person 2 | Last.fm Tags | `track.getTopTags` pipeline, tag features |
| Person 3 | Audio Features | Librosa extraction, feature store |
| Person 4 | Modeling + Eval | Baselines, MF, hybrid, report |

## Quick Start (reproduce everything from scratch)

```bash
# 1. Clone the repo
git clone <repo-url>
cd sonic_analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Person 1's full data pipeline
bash data/scripts/run_pipeline.sh
```

That's it. All processed files will appear in `data/processed/`.

## Project Structure

```
sonic_analysis/
├── README.md                    ← this file
├── README_schema.md             ← IMPORTANT: shared schema for all teammates
├── requirements.txt
├── data/
│   ├── raw/                     ← gitignored — re-download with 01_download.sh
│   │   ├── musicnet_metadata.csv
│   │   └── hetrec2011-lastfm-2k/
│   │       ├── artists.dat          (17,632 artists)
│   │       ├── user_artists.dat     (92,834 play counts)
│   │       ├── tags.dat             (← Person 2: tag vocabulary already here!)
│   │       └── user_taggedartists.dat (← Person 2: pre-cached tag assignments!)
│   ├── processed/               ← tracked in git — outputs everyone reads
│   │   ├── track_metadata.csv       (Person 1)  17,632 artists
│   │   ├── ratings_joined.csv       (Person 1)  92,834 user-artist ratings
│   │   ├── ratings_train.csv        (Person 1)  74,265 rows (80%)
│   │   ├── ratings_test.csv         (Person 1)  18,569 rows (20%)
│   │   ├── master_tracks.csv        (Person 1)  flat join, main team table
│   │   ├── musicnet_audio_map.csv   (Person 1)  track_id ↔ MusicNet WAV IDs
│   │   ├── tag_features.csv         (Person 2)  TODO
│   │   └── audio_features.csv       (Person 3)  TODO
│   └── scripts/                 ← Person 1's pipeline scripts
│       ├── 01_download.sh
│       ├── 02_clean_metadata.py
│       ├── 03_join_ratings.py
│       ├── 04_split.py
│       ├── 05_export_master.py
│       ├── validate_outputs.py
│       └── run_pipeline.sh
└── notebooks/                   ← EDA and modeling notebooks
```

## Dataset

**Primary ratings source:** [HetRec 2011 Last.fm 2K](https://grouplens.org/datasets/hetrec-2011/)
- 1,892 users × 17,632 artists
- 92,834 user-artist interactions (play counts → normalized ratings 1–5)
- Contains: `tags.dat` (tag vocab), `user_taggedartists.dat` (user tag assignments)

**Audio source:** [MusicNet](https://zenodo.org/records/5120004)
- 330 classical recordings (.wav + note annotations)
- Linked to 7 HetRec artists via `musicnet_audio_map.csv`

## Schema — READ THIS FIRST

See [`README_schema.md`](README_schema.md) for the exact column names and file formats everyone must follow.

**Primary join key: `track_id` (integer) — every feature file must have this column.**

## Key Note for Person 2 (Last.fm Tags)

You don't need to hit the API from scratch! The HetRec dataset already includes:
- `data/raw/hetrec2011-lastfm-2k/tags.dat` — full tag vocabulary
- `data/raw/hetrec2011-lastfm-2k/user_taggedartists.dat` — which users tagged which artists with which tags

Use these as your **primary source** and only call `track.getTopTags` for artists missing from this local cache.

## Key Note for Person 3 (Audio Features)

Check `data/processed/musicnet_audio_map.csv` — it maps `track_id` → `musicnet_id` for the classical artists that have MusicNet WAV files. Start with those for verified audio.

## Key Note for Person 4 (Modeling)

Read from:
```python
train = pd.read_csv("data/processed/ratings_train.csv")
test  = pd.read_csv("data/processed/ratings_test.csv")
# Later merge in tag_features.csv and audio_features.csv on track_id
```

The rating value is a **continuous 1–5 float** (log-normalized from play counts).
