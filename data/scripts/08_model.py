#!/usr/bin/env python3
"""
08_model.py  —  Person 4: Modeling & Evaluation
================================================
Trains a full ablation study over music recommendation models:

  1. Global mean baseline
  2. User mean baseline
  3. Item (artist) mean baseline
  4. Matrix Factorization (SVD on ratings only)
  5. MF + tag features (hybrid)
  6. MF + audio features (hybrid, classical artists only)
  7. MF + tags + audio (full hybrid)

Writes:
  data/processed/evaluation_results.csv   — ablation table (RMSE, MAE, P@10, NDCG@10)
  data/processed/recommendations.csv      — top-10 recommendations for 5 sample users
  models/                                  — saved model artifacts

Usage:
  python3 data/scripts/08_model.py
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = PROCESSED / "ratings_train.csv"
TEST_PATH = PROCESSED / "ratings_test.csv"
TAG_PATH = PROCESSED / "tag_features.csv"
AUDIO_PATH = PROCESSED / "audio_features.csv"
EVAL_OUT = PROCESSED / "evaluation_results.csv"
RECS_OUT = PROCESSED / "recommendations.csv"

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_FACTORS = 50          # SVD latent factors
N_TOP = 10              # Top-N for ranking metrics
ITEM_FEAT_WEIGHT = 0.3  # How much item features shift the latent embedding


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    # Normalize: some files use artist_id instead of track_id
    for df in (train, test):
        if "artist_id" in df.columns and "track_id" not in df.columns:
            df.rename(columns={"artist_id": "track_id"}, inplace=True)
    return train, test


def load_item_features(feat_path: pathlib.Path, id_col: str = "track_id") -> Optional[pd.DataFrame]:
    if not feat_path.exists():
        return None
    df = pd.read_csv(feat_path)
    # Accept both track_id and artist_id as the primary key
    if id_col not in df.columns:
        if "artist_id" in df.columns:
            df = df.rename(columns={"artist_id": id_col})
        else:
            return None
    feature_cols = [c for c in df.columns if c != id_col and pd.api.types.is_numeric_dtype(df[c])]
    df = df[[id_col] + feature_cols].dropna()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Matrix Factorization via truncated SVD
# ══════════════════════════════════════════════════════════════════════════════

class MatrixFactorization:
    """
    Truncated SVD on the user-item rating matrix.
    Item features (tags / audio) can optionally be fused into item embeddings.
    """

    def __init__(self, n_factors: int = N_FACTORS, item_feat_weight: float = ITEM_FEAT_WEIGHT):
        self.n_factors = n_factors
        self.item_feat_weight = item_feat_weight
        self.global_mean: float = 0.0
        self.user_ids: list = []
        self.item_ids: list = []
        self.user_index: dict = {}
        self.item_index: dict = {}
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None

    def fit(
        self,
        train: pd.DataFrame,
        item_features: Optional[pd.DataFrame] = None,
    ) -> "MatrixFactorization":
        self.global_mean = float(train["rating"].mean())
        self.user_ids = sorted(train["user_id"].unique())
        self.item_ids = sorted(train["track_id"].unique())
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.item_index = {t: j for j, t in enumerate(self.item_ids)}

        n_users = len(self.user_ids)
        n_items = len(self.item_ids)

        # Build dense rating matrix (mean-centered)
        R = np.zeros((n_users, n_items), dtype=np.float32)
        for row in train.itertuples():
            u = self.user_index[row.user_id]
            i = self.item_index[row.track_id]
            R[u, i] = row.rating - self.global_mean

        # Biases
        self.user_bias = np.array([
            train.loc[train["user_id"] == uid, "rating"].mean() - self.global_mean
            for uid in self.user_ids
        ], dtype=np.float32)
        self.item_bias = np.array([
            train.loc[train["track_id"] == tid, "rating"].mean() - self.global_mean
            for tid in self.item_ids
        ], dtype=np.float32)

        # SVD
        k = min(self.n_factors, n_users - 1, n_items - 1)
        U, sigma, Vt = svds(R.astype(np.float64), k=k)
        # Sort by descending singular value (svds returns ascending)
        idx = np.argsort(sigma)[::-1]
        U, sigma, Vt = U[:, idx], sigma[idx], Vt[idx, :]

        self.user_factors = (U * sigma).astype(np.float32)   # (n_users, k)
        self.item_factors = Vt.T.astype(np.float32)          # (n_items, k)

        # Fuse item features if provided
        if item_features is not None:
            self._fuse_item_features(item_features)

        return self

    def _fuse_item_features(self, item_feats: pd.DataFrame) -> None:
        feat_cols = [c for c in item_feats.columns if c != "track_id"]
        feat_matrix = item_feats.set_index("track_id")[feat_cols]

        # Project item feature matrix into latent space (linear projection)
        feat_matrix = feat_matrix.reindex(self.item_ids).fillna(0.0).values.astype(np.float32)
        feat_norm = normalize(feat_matrix, axis=1)  # L2 normalize rows

        # PCA-like projection: use top-k singular vectors of the feature matrix
        k = self.item_factors.shape[1]
        if feat_norm.shape[1] >= k:
            try:
                _, _, Vf = svds(feat_norm.astype(np.float64), k=k)
                feat_projected = (feat_norm @ Vf.T).astype(np.float32)
            except Exception:
                feat_projected = feat_norm[:, :k]
        else:
            feat_projected = np.pad(feat_norm, ((0, 0), (0, k - feat_norm.shape[1])))

        feat_projected = normalize(feat_projected, axis=1).astype(np.float32)
        item_factors_norm = normalize(self.item_factors, axis=1)

        self.item_factors = (
            (1.0 - self.item_feat_weight) * item_factors_norm
            + self.item_feat_weight * feat_projected
        ).astype(np.float32)

    def predict_pair(self, user_id: int, track_id: int) -> float:
        if user_id not in self.user_index or track_id not in self.item_index:
            return self.global_mean
        u = self.user_index[user_id]
        i = self.item_index[track_id]
        score = (
            self.global_mean
            + self.user_bias[u]
            + self.item_bias[i]
            + float(self.user_factors[u] @ self.item_factors[i])
        )
        return float(np.clip(score, 1.0, 5.0))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_pair(r.user_id, r.track_id) for r in df.itertuples()])

    def top_n_for_user(self, user_id: int, train: pd.DataFrame, n: int = N_TOP) -> list[int]:
        if user_id not in self.user_index:
            return []
        u = self.user_index[user_id]
        seen = set(train.loc[train["user_id"] == user_id, "track_id"])
        scores = self.global_mean + self.user_bias[u] + self.item_bias + (
            self.user_factors[u] @ self.item_factors.T
        )
        ranked = sorted(
            [(self.item_ids[j], float(scores[j])) for j in range(len(self.item_ids)) if self.item_ids[j] not in seen],
            key=lambda x: -x[1],
        )
        return [tid for tid, _ in ranked[:n]]

    def save(self, path: pathlib.Path) -> None:
        np.savez_compressed(
            path,
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            user_bias=self.user_bias,
            item_bias=self.item_bias,
            global_mean=np.array([self.global_mean]),
        )
        meta = {
            "user_ids": [int(x) for x in self.user_ids],
            "item_ids": [int(x) for x in self.item_ids],
            "n_factors": int(self.n_factors),
        }
        with open(str(path).replace(".npz", "_meta.json"), "w") as fh:
            json.dump(meta, fh)


# ══════════════════════════════════════════════════════════════════════════════
# Baselines
# ══════════════════════════════════════════════════════════════════════════════

class GlobalMeanModel:
    def fit(self, train: pd.DataFrame) -> "GlobalMeanModel":
        self.mean = float(train["rating"].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.mean)

    def top_n_for_user(self, user_id: int, train: pd.DataFrame, all_items: list, n: int = N_TOP) -> list[int]:
        seen = set(train.loc[train["user_id"] == user_id, "track_id"])
        return [i for i in all_items if i not in seen][:n]


class UserMeanModel:
    def fit(self, train: pd.DataFrame) -> "UserMeanModel":
        self.global_mean = float(train["rating"].mean())
        self.user_means = train.groupby("user_id")["rating"].mean().to_dict()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.user_means.get(r.user_id, self.global_mean) for r in df.itertuples()])

    def top_n_for_user(self, user_id: int, train: pd.DataFrame, all_items: list, n: int = N_TOP) -> list[int]:
        seen = set(train.loc[train["user_id"] == user_id, "track_id"])
        return [i for i in all_items if i not in seen][:n]


class ItemMeanModel:
    def fit(self, train: pd.DataFrame) -> "ItemMeanModel":
        self.global_mean = float(train["rating"].mean())
        self.item_means = train.groupby("track_id")["rating"].mean().to_dict()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.item_means.get(r.track_id, self.global_mean) for r in df.itertuples()])

    def top_n_for_user(self, user_id: int, train: pd.DataFrame, all_items: list, n: int = N_TOP) -> list[int]:
        seen = set(train.loc[train["user_id"] == user_id, "track_id"])
        ranked = sorted(
            [(tid, score) for tid, score in self.item_means.items() if tid not in seen],
            key=lambda x: -x[1],
        )
        return [tid for tid, _ in ranked[:n]]


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation metrics
# ══════════════════════════════════════════════════════════════════════════════

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def precision_at_k(recommended: list, relevant: set, k: int = N_TOP) -> float:
    if not recommended:
        return 0.0
    hits = sum(1 for r in recommended[:k] if r in relevant)
    return hits / k


def ndcg_at_k(recommended: list, relevant: set, k: int = N_TOP) -> float:
    dcg = sum(
        (1 / math.log2(i + 2)) for i, r in enumerate(recommended[:k]) if r in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    all_items: list,
    k: int = N_TOP,
    is_mf: bool = False,
) -> tuple[float, float]:
    """Compute mean Precision@K and NDCG@K over all test users."""
    # Binarize: rating >= 3.5 is "liked"
    liked = test[test["rating"] >= 3.5].groupby("user_id")["track_id"].apply(set).to_dict()
    test_users = test["user_id"].unique()

    p_scores, n_scores = [], []
    for uid in test_users:
        relevant = liked.get(uid, set())
        if not relevant:
            continue
        if is_mf:
            recs = model.top_n_for_user(uid, train, n=k)
        else:
            recs = model.top_n_for_user(uid, train, all_items, n=k)
        p_scores.append(precision_at_k(recs, relevant, k))
        n_scores.append(ndcg_at_k(recs, relevant, k))

    return (
        float(np.mean(p_scores)) if p_scores else 0.0,
        float(np.mean(n_scores)) if n_scores else 0.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Sample recommendations
# ══════════════════════════════════════════════════════════════════════════════

def build_recommendations(
    mf_full: MatrixFactorization,
    train: pd.DataFrame,
    track_meta: pd.DataFrame,
    n_users: int = 5,
    n_recs: int = N_TOP,
) -> pd.DataFrame:
    """Generate top-N recommendations for a few sample users using the best model."""
    # Pick users with >= 10 train ratings for interesting recommendations
    active_users = (
        train.groupby("user_id").size()
        .sort_values(ascending=False)
        .head(n_users)
        .index.tolist()
    )
    artist_lookup = track_meta.set_index("track_id")["artist"].to_dict() if "track_id" in track_meta.columns else track_meta.set_index("artist_id")["artist"].to_dict()

    rows = []
    for uid in active_users:
        recs = mf_full.top_n_for_user(uid, train, n=n_recs)
        for rank, tid in enumerate(recs, start=1):
            rows.append({
                "user_id": uid,
                "rank": rank,
                "track_id": tid,
                "artist": artist_lookup.get(tid, "Unknown"),
                "predicted_rating": round(mf_full.predict_pair(uid, tid), 3),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  AudioMuse-AI  |  Person 4: Modeling & Evaluation")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    train, test = load_data()
    print(f"  Train: {len(train):,} ratings | {train['user_id'].nunique()} users | {train['track_id'].nunique()} items")
    print(f"  Test : {len(test):,} ratings  | {test['user_id'].nunique()} users  | {test['track_id'].nunique()} items")

    tags_df = load_item_features(TAG_PATH)
    audio_df = load_item_features(AUDIO_PATH)
    print(f"  Tags : {len(tags_df) if tags_df is not None else 0:,} items, {len([c for c in (tags_df.columns if tags_df is not None else []) if c != 'track_id'])} features")
    print(f"  Audio: {len(audio_df) if audio_df is not None else 0:,} items, {len([c for c in (audio_df.columns if audio_df is not None else []) if c != 'track_id'])} features")

    track_meta = pd.read_csv(PROCESSED / "track_metadata.csv")
    all_items = sorted(train["track_id"].unique().tolist())
    y_true = test["rating"].values

    results = []

    # ── Baseline 1: Global mean ───────────────────────────────────────────────
    print("\n[2/4] Training baseline models...")
    gm = GlobalMeanModel().fit(train)
    y_pred = gm.predict(test)
    p10, nd10 = evaluate_ranking(gm, train, test, all_items, is_mf=False)
    results.append({
        "model": "Global Mean",
        "rmse": round(rmse(y_true, y_pred), 4),
        "mae": round(mae(y_true, y_pred), 4),
        "precision_at_10": round(p10, 4),
        "ndcg_at_10": round(nd10, 4),
    })
    print(f"  Global Mean  — RMSE: {results[-1]['rmse']:.4f}  MAE: {results[-1]['mae']:.4f}")

    # ── Baseline 2: User mean ─────────────────────────────────────────────────
    um = UserMeanModel().fit(train)
    y_pred = um.predict(test)
    p10, nd10 = evaluate_ranking(um, train, test, all_items, is_mf=False)
    results.append({
        "model": "User Mean",
        "rmse": round(rmse(y_true, y_pred), 4),
        "mae": round(mae(y_true, y_pred), 4),
        "precision_at_10": round(p10, 4),
        "ndcg_at_10": round(nd10, 4),
    })
    print(f"  User Mean    — RMSE: {results[-1]['rmse']:.4f}  MAE: {results[-1]['mae']:.4f}")

    # ── Baseline 3: Item mean ─────────────────────────────────────────────────
    im = ItemMeanModel().fit(train)
    y_pred = im.predict(test)
    p10, nd10 = evaluate_ranking(im, train, test, all_items, is_mf=False)
    results.append({
        "model": "Item Mean",
        "rmse": round(rmse(y_true, y_pred), 4),
        "mae": round(mae(y_true, y_pred), 4),
        "precision_at_10": round(p10, 4),
        "ndcg_at_10": round(nd10, 4),
    })
    print(f"  Item Mean    — RMSE: {results[-1]['rmse']:.4f}  MAE: {results[-1]['mae']:.4f}")

    # ── MF: ratings only ──────────────────────────────────────────────────────
    print("\n[3/4] Training Matrix Factorization models...")
    print("  Fitting MF (ratings only)...")
    mf_base = MatrixFactorization(n_factors=N_FACTORS).fit(train)
    y_pred = mf_base.predict(test)
    p10, nd10 = evaluate_ranking(mf_base, train, test, all_items, is_mf=True)
    results.append({
        "model": "MF (ratings only)",
        "rmse": round(rmse(y_true, y_pred), 4),
        "mae": round(mae(y_true, y_pred), 4),
        "precision_at_10": round(p10, 4),
        "ndcg_at_10": round(nd10, 4),
    })
    mf_base.save(MODELS_DIR / "mf_base.npz")
    print(f"  MF base      — RMSE: {results[-1]['rmse']:.4f}  MAE: {results[-1]['mae']:.4f}")

    # ── MF + tags ─────────────────────────────────────────────────────────────
    if tags_df is not None:
        print("  Fitting MF + tags...")
        mf_tags = MatrixFactorization(n_factors=N_FACTORS).fit(train, item_features=tags_df)
        y_pred = mf_tags.predict(test)
        p10, nd10 = evaluate_ranking(mf_tags, train, test, all_items, is_mf=True)
        results.append({
            "model": "MF + tags",
            "rmse": round(rmse(y_true, y_pred), 4),
            "mae": round(mae(y_true, y_pred), 4),
            "precision_at_10": round(p10, 4),
            "ndcg_at_10": round(nd10, 4),
        })
        mf_tags.save(MODELS_DIR / "mf_tags.npz")
        print(f"  MF + tags    — RMSE: {results[-1]['rmse']:.4f}  MAE: {results[-1]['mae']:.4f}")

    # ── MF + audio ────────────────────────────────────────────────────────────
    if audio_df is not None:
        print("  Fitting MF + audio...")
        mf_audio = MatrixFactorization(n_factors=N_FACTORS).fit(train, item_features=audio_df)
        y_pred = mf_audio.predict(test)
        p10, nd10 = evaluate_ranking(mf_audio, train, test, all_items, is_mf=True)
        results.append({
            "model": "MF + audio",
            "rmse": round(rmse(y_true, y_pred), 4),
            "mae": round(mae(y_true, y_pred), 4),
            "precision_at_10": round(p10, 4),
            "ndcg_at_10": round(nd10, 4),
        })
        mf_audio.save(MODELS_DIR / "mf_audio.npz")
        print(f"  MF + audio   — RMSE: {results[-1]['rmse']:.4f}  MAE: {results[-1]['mae']:.4f}")

    # ── MF + tags + audio ────────────────────────────────────────────────────
    if tags_df is not None and audio_df is not None:
        print("  Fitting MF + tags + audio...")
        # Merge tag and audio features, filling audio NaN with 0
        tag_cols = [c for c in tags_df.columns if c != "track_id"]
        audio_cols = [c for c in audio_df.columns if c != "track_id"]
        combined = tags_df.merge(audio_df, on="track_id", how="left")
        for col in audio_cols:
            combined[col] = combined[col].fillna(0.0)
        mf_full = MatrixFactorization(n_factors=N_FACTORS).fit(train, item_features=combined)
        y_pred = mf_full.predict(test)
        p10, nd10 = evaluate_ranking(mf_full, train, test, all_items, is_mf=True)
        results.append({
            "model": "MF + tags + audio",
            "rmse": round(rmse(y_true, y_pred), 4),
            "mae": round(mae(y_true, y_pred), 4),
            "precision_at_10": round(p10, 4),
            "ndcg_at_10": round(nd10, 4),
        })
        mf_full.save(MODELS_DIR / "mf_full.npz")
        print(f"  MF + full    — RMSE: {results[-1]['rmse']:.4f}  MAE: {results[-1]['mae']:.4f}")
    else:
        mf_full = mf_tags if tags_df is not None else mf_base

    # ── Save evaluation results ───────────────────────────────────────────────
    print("\n[4/4] Saving results...")
    eval_df = pd.DataFrame(results)
    eval_df.to_csv(EVAL_OUT, index=False)
    print(f"\n  Evaluation results → {EVAL_OUT}")

    # ── Top-N recommendations for sample users ────────────────────────────────
    recs_df = build_recommendations(mf_full, train, track_meta)
    recs_df.to_csv(RECS_OUT, index=False)
    print(f"  Recommendations    → {RECS_OUT}")

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ABLATION TABLE")
    print("=" * 60)
    print(eval_df.to_string(index=False))
    print("=" * 60)

    best = eval_df.loc[eval_df["rmse"].idxmin()]
    print(f"\n  Best model: {best['model']}")
    print(f"  RMSE: {best['rmse']}  |  MAE: {best['mae']}")
    print(f"  P@10: {best['precision_at_10']}  |  NDCG@10: {best['ndcg_at_10']}")

    # ── Sample recommendations table ─────────────────────────────────────────
    print("\n  SAMPLE RECOMMENDATIONS (top-5 for first user)")
    first_user = recs_df["user_id"].iloc[0]
    print(recs_df[recs_df["user_id"] == first_user][["rank", "artist", "predicted_rating"]].to_string(index=False))

    print("\n[DONE] Modeling complete.")


if __name__ == "__main__":
    main()
