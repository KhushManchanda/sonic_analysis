#!/usr/bin/env python3

import argparse
import logging
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

METADATA_PATH = Path("data/raw/musicnet_metadata.csv")
MAP_PATH = Path("data/processed/musicnet_audio_map.csv")
OUTPUT_DIR = Path("data/embeds")
PROCESSED_DIR = Path("data/processed")
SR = 22050
DURATION = 60  # seconds

SPLITS = {
    "train": Path("data/raw/musicnet/train_data"),
    "test":  Path("data/raw/musicnet/test_data"),
}


def extract_features(args: tuple) -> tuple[str, dict | None]:
    musicnet_id, wav_path = args
    try:
        y, sr = librosa.load(wav_path, sr=SR, mono=True, duration=DURATION)
    except Exception:
        return musicnet_id, None

    feats = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i, (m, s) in enumerate(zip(mfcc.mean(axis=1), mfcc.std(axis=1)), start=1):
        feats[f"mfcc_{i}_mean"] = float(m)
        feats[f"mfcc_{i}_std"]  = float(s)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    feats["tempo"] = float(np.asarray(tempo).flat[0])

    feats["rms_mean"] = float(librosa.feature.rms(y=y).mean())

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i, v in enumerate(chroma.mean(axis=1), start=1):
        feats[f"chroma_{i}_mean"] = float(v)

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i, v in enumerate(contrast.mean(axis=1), start=1):
        feats[f"contrast_{i}_mean"] = float(v)

    return musicnet_id, feats


def load_existing_ids(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    try:
        existing = pd.read_csv(output_path, usecols=["musicnet_id"])
        ids = set(existing["musicnet_id"].astype(str))
        log.info(f"Resuming: {len(ids)} tracks already processed in {output_path.name}")
        return ids
    except Exception as e:
        log.warning(f"Could not read existing output {output_path}: {e}")
        return set()


def process_split(split_name: str, audio_dir: Path, track_ids: list[str], workers: int) -> Path:
    output_path = OUTPUT_DIR / f"audio_features_{split_name}.csv"
    already_done = load_existing_ids(output_path)

    write_header = not output_path.exists()
    write_lock = threading.Lock()

    pending = [
        (tid, audio_dir / f"{tid}.wav")
        for tid in track_ids
        if tid not in already_done
    ]

    missing = [tid for tid, p in pending if not p.exists()]
    for tid in missing:
        log.warning(f"[{split_name}] Missing file for track {tid}")
    pending = [(tid, p) for tid, p in pending if p.exists()]

    log.info(f"[{split_name}] {len(pending)} tracks to process, {len(already_done)} already done, {workers} workers")

    def write_row(musicnet_id: str, feats: dict, header_ref: list) -> None:
        feats["musicnet_id"] = musicnet_id
        row_df = pd.DataFrame([feats])
        cols = ["musicnet_id"] + [c for c in row_df.columns if c != "musicnet_id"]
        row_df = row_df[cols]
        with write_lock:
            row_df.to_csv(output_path, mode="a", header=header_ref[0], index=False)
            header_ref[0] = False

    header_ref = [write_header]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_features, args): args[0] for args in pending}
        with tqdm(total=len(futures), desc=f"{split_name}") as pbar:
            for future in as_completed(futures):
                musicnet_id, feats = future.result()
                if feats is not None:
                    write_row(musicnet_id, feats, header_ref)
                else:
                    log.warning(f"[{split_name}] Failed to extract features for track {musicnet_id}")
                pbar.update(1)

    log.info(f"[{split_name}] Done. Output: {output_path}")
    return output_path


def build_artist_level_output(split_name: str, raw_output_path: Path) -> None:
    out_path = PROCESSED_DIR / f"audio_features_artist_{split_name}.csv"

    if not raw_output_path.exists():
        log.warning(f"[{split_name}] Raw audio feature file not found: {raw_output_path}")
        pd.DataFrame(columns=["artist_id", "artist", "musicnet_ids", "recording_count"]).to_csv(out_path, index=False)
        return

    if not MAP_PATH.exists():
        log.warning("musicnet_audio_map.csv not found — skipping artist-level audio output")
        return

    mapping = pd.read_csv(MAP_PATH)
    if not {"artist_id", "musicnet_id"}.issubset(mapping.columns):
        log.warning("musicnet_audio_map.csv missing artist_id/musicnet_id — skipping artist-level audio output")
        return

    if int(mapping["musicnet_id"].duplicated().sum()) > 0:
        raise ValueError("musicnet_audio_map.csv contains duplicate musicnet_id values")

    raw = pd.read_csv(raw_output_path, dtype={"musicnet_id": str})
    mapping["musicnet_id"] = mapping["musicnet_id"].astype(str)
    merged = raw.merge(mapping, on="musicnet_id", how="inner")
    if merged.empty:
        log.warning(f"[{split_name}] No mapped audio rows found after merge")
        empty_cols = ["artist_id", "artist", "musicnet_ids", "recording_count"] + [
            col for col in raw.columns if col != "musicnet_id"
        ]
        pd.DataFrame(columns=empty_cols).to_csv(out_path, index=False)
        return

    numeric_cols = [
        col for col in merged.columns
        if col not in {"artist_id", "artist", "musicnet_id", "composer_raw", "match_key", "match_type"}
        and pd.api.types.is_numeric_dtype(merged[col])
    ]

    grouped = merged.groupby(["artist_id", "artist"], as_index=False)[numeric_cols].mean()
    musicnet_lists = (
        merged.groupby("artist_id")["musicnet_id"]
        .apply(lambda s: ",".join(sorted(set(map(str, s)))))
        .rename("musicnet_ids")
        .reset_index()
    )
    recording_counts = merged.groupby("artist_id").size().rename("recording_count").reset_index()
    artist_level = grouped.merge(musicnet_lists, on="artist_id", how="left").merge(recording_counts, on="artist_id", how="left")
    lead_cols = ["artist_id", "artist", "musicnet_ids", "recording_count"]
    rest_cols = [c for c in artist_level.columns if c not in lead_cols]
    artist_level = artist_level[lead_cols + rest_cols].sort_values("artist_id")

    artist_level.to_csv(out_path, index=False)
    log.info(f"[{split_name}] Artist-level output: {out_path}")


def maybe_build_from_existing_outputs() -> bool:
    """
    Rebuild processed artist-level audio tables from existing embed CSVs.

    This keeps the project runnable in environments where raw MusicNet WAV
    directories are not available locally, but previously extracted features are.
    """
    found_any = False
    for split_name in SPLITS:
        raw_output_path = OUTPUT_DIR / f"audio_features_{split_name}.csv"
        if raw_output_path.exists():
            build_artist_level_output(split_name, raw_output_path)
            found_any = True
    return found_any


def main():
    parser = argparse.ArgumentParser(description="Extract audio features from MusicNet WAV files.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes (default: 4)")
    parser.add_argument(
        "--rebuild-only",
        action="store_true",
        help="Skip raw WAV extraction and only rebuild artist-level processed audio outputs from existing embed CSVs.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if args.rebuild_only:
        rebuilt = maybe_build_from_existing_outputs()
        if not rebuilt:
            raise SystemExit("No existing embed CSVs found to rebuild from.")
        return

    if not METADATA_PATH.exists():
        if maybe_build_from_existing_outputs():
            log.info("MusicNet metadata missing, but artist-level outputs were rebuilt from existing embed CSVs.")
            return
        raise SystemExit(f"Missing required metadata file: {METADATA_PATH}")

    metadata = pd.read_csv(METADATA_PATH, dtype={"id": str})
    all_ids = metadata["id"].tolist()

    processed_any_split = False

    for split_name, audio_dir in SPLITS.items():
        if not audio_dir.exists():
            log.warning(f"Split directory not found, skipping: {audio_dir}")
            continue

        available_wavs = {p.stem for p in audio_dir.glob("*.wav")}
        split_ids = [tid for tid in all_ids if tid in available_wavs]

        log.info(f"[{split_name}] {len(split_ids)} tracks found in {audio_dir}")
        raw_output_path = process_split(split_name, audio_dir, split_ids, args.workers)
        build_artist_level_output(split_name, raw_output_path)
        processed_any_split = True

    if not processed_any_split:
        rebuilt = maybe_build_from_existing_outputs()
        if rebuilt:
            log.info("Raw MusicNet split directories unavailable; rebuilt processed artist-level outputs from existing embed CSVs.")
        else:
            log.warning("No raw MusicNet directories or existing embed CSVs were available. No processed audio outputs were generated.")


if __name__ == "__main__":
    main()