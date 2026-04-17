# Implementation Status

## Final canonical item definition

The finished recommender is **artist-level**.

- Canonical key: `artist_id`
- Source: HetRec 2011 Last.fm 2K `artist_id`
- Backward-compatibility note: legacy filenames such as `track_metadata.csv` and `master_tracks.csv` are preserved, but their rows are artist-level.

## What was already present

- Artist metadata export in `track_metadata.csv`
- Joined ratings table and train/test split outputs
- Local HetRec tag feature generation with TF-IDF output
- Raw MusicNet-derived embed CSVs in `data/embeds/`

## What was repaired

- Fixed split script to use `artist_id` consistently instead of legacy `track_id`
- Rebuilt overlap-subset logic in the split stage
- Repaired audio stage so processed artist-level audio tables can be rebuilt from existing embed CSVs even when raw WAV directories are unavailable
- Added experiment runner for baselines, MF, hybrid ablations, metrics, and saved summaries
- Extended validator to check experiment outputs in addition to processed feature tables

## Key assumptions used

- Current repo reality is artist-level, not track-level
- Existing `data/embeds/audio_features_*.csv` files are treated as valid intermediate outputs when raw MusicNet audio is absent locally
- Audio coverage is extremely small in the checked-out environment and is therefore treated as an overlap-limited experiment; if no test overlap exists, audio rows are exported as unavailable instead of fabricating metrics