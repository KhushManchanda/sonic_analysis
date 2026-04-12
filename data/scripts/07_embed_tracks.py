import argparse
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import threading
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

METADATA_PATH = Path("data/raw/musicnet_metadata.csv")
OUTPUT_DIR    = Path("data/embeds")
SR            = 22050
DURATION      = 60  # seconds

SPLITS = {
    "train": Path("data/raw/musicnet/train_data"),
    "test":  Path("data/raw/musicnet/test_data"),
}


def extract_features(args: tuple) -> tuple[str, dict | None]:
    track_id, wav_path = args
    try:
        y, sr = librosa.load(wav_path, sr=SR, mono=True, duration=DURATION)
    except Exception as e:
        return track_id, None

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

    return track_id, feats


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


def process_split(split_name: str, audio_dir: Path, track_ids: list[str], workers: int) -> None:
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

    def write_row(track_id: str, feats: dict, header_ref: list) -> None:
        feats["musicnet_id"] = track_id
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
                track_id, feats = future.result()
                if feats is not None:
                    write_row(track_id, feats, header_ref)
                else:
                    log.warning(f"[{split_name}] Failed to extract features for track {track_id}")
                pbar.update(1)

    log.info(f"[{split_name}] Done. Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract audio features from MusicNet WAV files.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes (default: 4)")
    args = parser.parse_args()

    metadata = pd.read_csv(METADATA_PATH, dtype={"id": str})
    all_ids = metadata["id"].tolist()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, audio_dir in SPLITS.items():
        if not audio_dir.exists():
            log.warning(f"Split directory not found, skipping: {audio_dir}")
            continue

        available_wavs = {p.stem for p in audio_dir.glob("*.wav")}
        split_ids = [tid for tid in all_ids if tid in available_wavs]

        log.info(f"[{split_name}] {len(split_ids)} tracks found in {audio_dir}")
        process_split(split_name, audio_dir, split_ids, args.workers)


if __name__ == "__main__":
    main()