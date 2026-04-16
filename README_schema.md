# Shared Data Schema — AudioMuse-AI Project
> **Person 1 owns this file. Last updated: 2026-04-09**
> Everyone must follow this schema so all outputs join cleanly at integration time.

---

## Primary Join Key

**`artist_id`** — integer, HetRec `artist_id`.

This project is artist-level on the HetRec side. Older filenames may still say
"track", but the canonical recommender item is an artist.

_Every_ integration-ready file produced by Persons 1, 2, and 3 should include
`artist_id`. Person 4 should join on `artist_id`.

---

## File Registry

All processed files live in `data/processed/`. Do **not** save outputs elsewhere.

| File | Owner | Description |
|------|-------|-------------|
| `track_metadata.csv` | Person 1 | Clean HetRec artist catalog |
| `ratings_joined.csv` | Person 1 | (user_id, artist_id, rating) full set |
| `ratings_train.csv` | Person 1 | 80% train split |
| `ratings_test.csv` | Person 1 | 20% test split |
| `master_tracks.csv` | Person 1 | Flat join of ratings + metadata |
| `tag_features.csv` | Person 2 | Last.fm tags per artist |
| `audio_features_artist_train.csv` | Person 3 | Train-split audio features merged to artist_id |
| `audio_features_artist_test.csv` | Person 3 | Test-split audio features merged to artist_id |

---

## Column Schemas

### `track_metadata.csv`
| Column | Type | Notes |
|--------|------|-------|
| `artist_id` | int | HetRec `artist_id` — primary key |
| `artist` | str | Normalized artist name |
| `url` | str | Last.fm artist URL |
| `musicnet_id` | int/nullable | Linked MusicNet ID if audio overlap exists |

### `ratings_train.csv` / `ratings_test.csv`
| Column | Type | Notes |
|--------|------|-------|
| `user_id` | int | HetRec user ID |
| `artist_id` | int | HetRec `artist_id` |
| `artist` | str | Artist name (for human readability) |
| `rating` | float | 1.0–5.0 (log-normalized from play counts) |

### `master_tracks.csv`
All columns from `ratings_joined.csv` + all columns from `track_metadata.csv`.

### `tag_features.csv` **(Person 2: follow this exactly)**
| Column | Type | Notes |
|--------|------|-------|
| `artist_id` | int | **Must match HetRec `artist_id` / Person 1 artist_id** |
| `tags_raw` | str | Comma-separated raw tag string |
| `tfidf_*` OR `tag_vec` | float | TF-IDF columns or serialized vector |

> Example: `artist_id,tags_raw,tfidf_classical,tfidf_piano,...`

### `audio_features_artist_train.csv` / `audio_features_artist_test.csv`
| Column | Type | Notes |
|--------|------|-------|
| `artist_id` | int | **Must match HetRec `artist_id` / Person 1 artist_id** |
| `musicnet_ids` | str | Comma-separated MusicNet IDs aggregated into this artist row |
| `recording_count` | int | Number of MusicNet recordings aggregated |
| `tempo` | float | BPM from Librosa |
| `rms_mean` | float | RMS energy mean |
| `mfcc_1_mean` ... `mfcc_20_mean` | float | MFCC coefficient means |
| `chroma_1_mean` ... `chroma_12_mean` | float | Chroma-bin means |
| `contrast_1_mean` ... `contrast_7_mean` | float | Spectral contrast means |

---

## Naming Conventions

- Files: `snake_case.csv`
- Columns: `snake_case`
- Text values: **lowercase**, stripped, single spaces (no leading/trailing whitespace)
- IDs: **integers**, no floats, no strings with padding

---

## How to Run (Person 1's pipeline)

```bash
cd sonic_analysis/

# 1. Download raw data
bash data/scripts/01_download.sh

# 2. Build artist catalog + validated MusicNet overlap map
python3 data/scripts/02_clean_metadata.py

# 3. Join with ratings
python3 data/scripts/03_join_ratings.py

# 4. Train/test split (+ overlap subset)
python3 data/scripts/04_split.py

# 5. Export master table
python3 data/scripts/05_export_master.py

# 6. Build tag features
python3 data/scripts/06_build_tag_features.py

# 7. Build audio features and merged artist-level audio outputs
python3 data/scripts/07_embed_tracks.py

# Or run all at once:
bash data/scripts/run_pipeline.sh
```

---

## Validation Checklist (run before sharing with team)

```bash
python3 data/scripts/validate_outputs.py
```

Expected output:
```
[OK] track_metadata.csv  — NNN rows, no null artist_id
[OK] ratings_joined.csv  — NNN rows, ratings in [1.0, 5.0]
[OK] ratings_train.csv   — NNN rows
[OK] ratings_test.csv    — NNN rows
[OK] master_tracks.csv   — NNN rows, all columns present
[OK] musicnet_audio_map.csv — no duplicate musicnet_id
[OK] audio_features_artist_train.csv / test.csv — artist_id present
[OK] Train + Test sum matches ratings_joined
```
