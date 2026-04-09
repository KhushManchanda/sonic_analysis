# Shared Data Schema — AudioMuse-AI Project
> **Person 1 owns this file. Last updated: 2026-04-09**
> Everyone must follow this schema so all outputs join cleanly at integration time.

---

## Primary Join Key

**`track_id`** — integer, MusicNet numeric ID.

_Every_ file produced by Persons 1, 2, and 3 must include `track_id` as a column.
Person 4 reads all files and joins on `track_id`.

---

## File Registry

All processed files live in `data/processed/`. Do **not** save outputs elsewhere.

| File | Owner | Description |
|------|-------|-------------|
| `track_metadata.csv` | Person 1 | Clean MusicNet track/artist catalog |
| `ratings_joined.csv` | Person 1 | (user_id, track_id, rating) full set |
| `ratings_train.csv` | Person 1 | 80% train split |
| `ratings_test.csv` | Person 1 | 20% test split |
| `master_tracks.csv` | Person 1 | Flat join of ratings + metadata |
| `tag_features.csv` | Person 2 | Last.fm tags per track |
| `audio_features.csv` | Person 3 | Librosa features per track |

---

## Column Schemas

### `track_metadata.csv`
| Column | Type | Notes |
|--------|------|-------|
| `track_id` | int | MusicNet ID — primary key |
| `artist` | str | Normalized composer name (e.g. `"johann sebastian bach"`) |
| `track` | str | Normalized composition name |
| `performer` | str | Performer/soloist name |
| `ensemble` | str | Orchestra/ensemble name |
| `source` | str | Recording source |
| `license` | str | License type |

### `ratings_train.csv` / `ratings_test.csv`
| Column | Type | Notes |
|--------|------|-------|
| `user_id` | int | HetRec user ID |
| `track_id` | int | MusicNet track ID |
| `artist` | str | Artist name (for human readability) |
| `rating` | float | 1.0–5.0 (log-normalized from play counts) |

### `master_tracks.csv`
All columns from `ratings_joined.csv` + all columns from `track_metadata.csv`.

### `tag_features.csv` **(Person 2: follow this exactly)**
| Column | Type | Notes |
|--------|------|-------|
| `track_id` | int | **Must match MusicNet track_id** |
| `tags_raw` | str | Comma-separated raw tag string |
| `tfidf_*` OR `tag_vec` | float | TF-IDF columns or serialized vector |

> Example: `track_id,tags_raw,tfidf_classical,tfidf_piano,...`

### `audio_features.csv` **(Person 3: follow this exactly)**
| Column | Type | Notes |
|--------|------|-------|
| `track_id` | int | **Must match MusicNet track_id** |
| `tempo` | float | BPM from Librosa |
| `rms_mean` | float | RMS energy mean |
| `mfcc_1_mean` ... `mfcc_20_mean` | float | MFCC coefficient means |
| `chroma_mean` | float | Chroma feature mean |
| `spectral_contrast_mean` | float | Spectral contrast mean |

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

# 2. Clean MusicNet metadata
python3 data/scripts/02_clean_metadata.py

# 3. Join with ratings
python3 data/scripts/03_join_ratings.py

# 4. Train/test split
python3 data/scripts/04_split.py

# 5. Export master table
python3 data/scripts/05_export_master.py

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
[OK] track_metadata.csv  — NNN rows, no null track_id
[OK] ratings_joined.csv  — NNN rows, ratings in [1.0, 5.0]
[OK] ratings_train.csv   — NNN rows
[OK] ratings_test.csv    — NNN rows
[OK] master_tracks.csv   — NNN rows, all columns present
[OK] Train + Test sum matches ratings_joined
```
