#!/usr/bin/env python3
"""
08_model.py  —  Person 4: Modeling & Evaluation
================================================
Full ablation study:
  1. Global mean baseline
  2. User mean baseline
  3. Item mean baseline
  4. kNN user-based collaborative filtering
  5. Matrix Factorization — SVD (ratings only)
  6. MF + tag features (hybrid)
  7. MF + audio features (hybrid)
  8. MF + tags + audio (full hybrid)

Metrics: RMSE, MAE, Precision@10, Recall@10, NDCG@10

Writes:
  data/processed/evaluation_results.csv
  data/processed/recommendations.csv
  models/mf_*.npz + *_meta.json
"""

from __future__ import annotations

import json
import math
import pathlib
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

ROOT       = pathlib.Path(__file__).resolve().parents[2]
PROCESSED  = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = PROCESSED / "ratings_train.csv"
TEST_PATH  = PROCESSED / "ratings_test.csv"
TAG_PATH   = PROCESSED / "tag_features.csv"
AUDIO_PATH = PROCESSED / "audio_features.csv"
EVAL_OUT   = PROCESSED / "evaluation_results.csv"
RECS_OUT   = PROCESSED / "recommendations.csv"

N_FACTORS        = 50
N_TOP            = 10
ITEM_FEAT_WEIGHT = 0.3
KNN_K            = 30   # number of nearest neighbours


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    for df in (train, test):
        if "artist_id" in df.columns and "track_id" not in df.columns:
            df.rename(columns={"artist_id": "track_id"}, inplace=True)
    return train, test


def load_item_features(feat_path: pathlib.Path, id_col: str = "track_id") -> Optional[pd.DataFrame]:
    if not feat_path.exists():
        return None
    df = pd.read_csv(feat_path)
    if id_col not in df.columns:
        if "artist_id" in df.columns:
            df = df.rename(columns={"artist_id": id_col})
        else:
            return None
    feature_cols = [c for c in df.columns if c != id_col and pd.api.types.is_numeric_dtype(df[c])]
    return df[[id_col] + feature_cols].dropna()


# ══════════════════════════════════════════════════════════════════════════════
# kNN user-based collaborative filter
# ══════════════════════════════════════════════════════════════════════════════

class UserKNNModel:
    """User-based kNN CF using cosine similarity on sparse rating matrix."""

    def __init__(self, k: int = KNN_K):
        self.k            = k
        self.global_mean  = 0.0
        self.user_means:  dict = {}
        self.item_means:  dict = {}
        self.user_index:  dict = {}
        self.item_index:  dict = {}
        self.user_ids:    list = []
        self.item_ids:    list = []
        self.sim_matrix:  np.ndarray | None = None   # (n_users, n_users)
        self.R_norm:      np.ndarray | None = None   # mean-centred dense matrix

    def fit(self, train: pd.DataFrame) -> "UserKNNModel":
        self.global_mean = float(train["rating"].mean())
        self.user_means  = train.groupby("user_id")["rating"].mean().to_dict()
        self.item_means  = train.groupby("track_id")["rating"].mean().to_dict()

        self.user_ids  = sorted(train["user_id"].unique())
        self.item_ids  = sorted(train["track_id"].unique())
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.item_index = {t: j for j, t in enumerate(self.item_ids)}

        n_u, n_i = len(self.user_ids), len(self.item_ids)

        # Build sparse mean-centred rating matrix
        rows, cols, vals = [], [], []
        for r in train.itertuples():
            u_idx = self.user_index[r.user_id]
            i_idx = self.item_index[r.track_id]
            rows.append(u_idx)
            cols.append(i_idx)
            vals.append(r.rating - self.user_means[r.user_id])

        R_sparse = csr_matrix((vals, (rows, cols)), shape=(n_u, n_i), dtype=np.float32)
        # Dense for similarity (1892×1892 is fine)
        self.R_norm       = R_sparse.toarray()
        self.sim_matrix   = cosine_similarity(self.R_norm)   # (n_u, n_u)
        np.fill_diagonal(self.sim_matrix, 0.0)               # exclude self
        return self

    def predict_pair(self, user_id: int, track_id: int) -> float:
        if user_id not in self.user_index:
            return self.item_means.get(track_id, self.global_mean)
        u          = self.user_index[user_id]
        u_mean     = self.user_means[user_id]
        sims       = self.sim_matrix[u]                        # (n_users,)
        top_k_idx  = np.argsort(sims)[::-1][:self.k]

        if track_id not in self.item_index:
            return u_mean

        i = self.item_index[track_id]
        numer, denom = 0.0, 0.0
        for v in top_k_idx:
            if sims[v] <= 0:
                break
            v_rating = self.R_norm[v, i]   # already mean-centred (0 = not rated)
            if v_rating != 0:
                numer += sims[v] * v_rating
                denom += abs(sims[v])

        if denom == 0:
            return u_mean
        return float(np.clip(u_mean + numer / denom, 1.0, 5.0))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_pair(r.user_id, r.track_id) for r in df.itertuples()])

    def top_n_for_user(self, user_id: int, train: pd.DataFrame, n: int = N_TOP) -> list[int]:
        if user_id not in self.user_index:
            return []
        u     = self.user_index[user_id]
        sims  = self.sim_matrix[u]
        top_k = np.argsort(sims)[::-1][:self.k]
        seen  = set(train.loc[train["user_id"] == user_id, "track_id"])

        scores: dict[int, float] = {}
        for v in top_k:
            if sims[v] <= 0:
                continue
            for j, tid in enumerate(self.item_ids):
                if tid in seen:
                    continue
                if self.R_norm[v, j] != 0:
                    scores[tid] = scores.get(tid, 0.0) + sims[v] * self.R_norm[v, j]

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [tid for tid, _ in ranked[:n]]


# ══════════════════════════════════════════════════════════════════════════════
# Matrix Factorization — truncated SVD
# ══════════════════════════════════════════════════════════════════════════════

class MatrixFactorization:
    def __init__(self, n_factors: int = N_FACTORS, item_feat_weight: float = ITEM_FEAT_WEIGHT):
        self.n_factors       = n_factors
        self.item_feat_weight = item_feat_weight
        self.global_mean:   float         = 0.0
        self.user_ids:      list          = []
        self.item_ids:      list          = []
        self.user_index:    dict          = {}
        self.item_index:    dict          = {}
        self.user_factors:  np.ndarray | None = None
        self.item_factors:  np.ndarray | None = None
        self.user_bias:     np.ndarray | None = None
        self.item_bias:     np.ndarray | None = None

    def fit(self, train: pd.DataFrame, item_features: Optional[pd.DataFrame] = None) -> "MatrixFactorization":
        self.global_mean = float(train["rating"].mean())
        self.user_ids    = sorted(train["user_id"].unique())
        self.item_ids    = sorted(train["track_id"].unique())
        self.user_index  = {u: i for i, u in enumerate(self.user_ids)}
        self.item_index  = {t: j for j, t in enumerate(self.item_ids)}

        n_users, n_items = len(self.user_ids), len(self.item_ids)

        R = np.zeros((n_users, n_items), dtype=np.float32)
        for row in train.itertuples():
            R[self.user_index[row.user_id], self.item_index[row.track_id]] = row.rating - self.global_mean

        self.user_bias = np.array([
            train.loc[train["user_id"] == uid, "rating"].mean() - self.global_mean
            for uid in self.user_ids], dtype=np.float32)
        self.item_bias = np.array([
            train.loc[train["track_id"] == tid, "rating"].mean() - self.global_mean
            for tid in self.item_ids], dtype=np.float32)

        k = min(self.n_factors, n_users - 1, n_items - 1)
        U, sigma, Vt = svds(R.astype(np.float64), k=k)
        idx = np.argsort(sigma)[::-1]
        U, sigma, Vt = U[:, idx], sigma[idx], Vt[idx, :]
        self.user_factors = (U * sigma).astype(np.float32)
        self.item_factors = Vt.T.astype(np.float32)

        if item_features is not None:
            self._fuse_item_features(item_features)
        return self

    def _fuse_item_features(self, item_feats: pd.DataFrame) -> None:
        feat_cols    = [c for c in item_feats.columns if c != "track_id"]
        feat_matrix  = item_feats.set_index("track_id")[feat_cols].reindex(self.item_ids).fillna(0.0).values.astype(np.float32)
        feat_norm    = normalize(feat_matrix, axis=1)
        k            = self.item_factors.shape[1]
        if feat_norm.shape[1] >= k:
            try:
                _, _, Vf = svds(feat_norm.astype(np.float64), k=k)
                feat_projected = normalize((feat_norm @ Vf.T).astype(np.float32), axis=1)
            except Exception:
                feat_projected = normalize(feat_norm[:, :k], axis=1)
        else:
            feat_projected = normalize(np.pad(feat_norm, ((0, 0), (0, k - feat_norm.shape[1]))), axis=1)
        self.item_factors = ((1.0 - self.item_feat_weight) * normalize(self.item_factors, axis=1)
                             + self.item_feat_weight * feat_projected).astype(np.float32)

    def predict_pair(self, user_id: int, track_id: int) -> float:
        if user_id not in self.user_index or track_id not in self.item_index:
            return self.global_mean
        u = self.user_index[user_id]
        i = self.item_index[track_id]
        return float(np.clip(
            self.global_mean + self.user_bias[u] + self.item_bias[i]
            + float(self.user_factors[u] @ self.item_factors[i]), 1.0, 5.0))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_pair(r.user_id, r.track_id) for r in df.itertuples()])

    def top_n_for_user(self, user_id: int, train: pd.DataFrame, n: int = N_TOP) -> list[int]:
        if user_id not in self.user_index:
            return []
        u    = self.user_index[user_id]
        seen = set(train.loc[train["user_id"] == user_id, "track_id"])
        scores = self.global_mean + self.user_bias[u] + self.item_bias + (self.user_factors[u] @ self.item_factors.T)
        ranked = sorted(
            [(self.item_ids[j], float(scores[j])) for j in range(len(self.item_ids)) if self.item_ids[j] not in seen],
            key=lambda x: -x[1])
        return [tid for tid, _ in ranked[:n]]

    def save(self, path: pathlib.Path) -> None:
        np.savez_compressed(path,
            user_factors=self.user_factors, item_factors=self.item_factors,
            user_bias=self.user_bias, item_bias=self.item_bias,
            global_mean=np.array([self.global_mean]))
        meta = {"user_ids": [int(x) for x in self.user_ids],
                "item_ids": [int(x) for x in self.item_ids],
                "n_factors": int(self.n_factors)}
        with open(str(path).replace(".npz", "_meta.json"), "w") as fh:
            json.dump(meta, fh)


# ══════════════════════════════════════════════════════════════════════════════
# Simple baselines
# ══════════════════════════════════════════════════════════════════════════════

class GlobalMeanModel:
    def fit(self, train: pd.DataFrame):
        self.mean = float(train["rating"].mean()); return self
    def predict(self, df): return np.full(len(df), self.mean)
    def top_n_for_user(self, uid, train, all_items, n=N_TOP):
        seen = set(train.loc[train["user_id"] == uid, "track_id"])
        return [i for i in all_items if i not in seen][:n]

class UserMeanModel:
    def fit(self, train: pd.DataFrame):
        self.global_mean = float(train["rating"].mean())
        self.user_means  = train.groupby("user_id")["rating"].mean().to_dict(); return self
    def predict(self, df):
        return np.array([self.user_means.get(r.user_id, self.global_mean) for r in df.itertuples()])
    def top_n_for_user(self, uid, train, all_items, n=N_TOP):
        seen = set(train.loc[train["user_id"] == uid, "track_id"])
        return [i for i in all_items if i not in seen][:n]

class ItemMeanModel:
    def fit(self, train: pd.DataFrame):
        self.global_mean = float(train["rating"].mean())
        self.item_means  = train.groupby("track_id")["rating"].mean().to_dict(); return self
    def predict(self, df):
        return np.array([self.item_means.get(r.track_id, self.global_mean) for r in df.itertuples()])
    def top_n_for_user(self, uid, train, all_items, n=N_TOP):
        seen   = set(train.loc[train["user_id"] == uid, "track_id"])
        ranked = sorted([(t, s) for t, s in self.item_means.items() if t not in seen], key=lambda x: -x[1])
        return [t for t, _ in ranked[:n]]


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation metrics
# ══════════════════════════════════════════════════════════════════════════════

def rmse(y_true, y_pred): return float(np.sqrt(mean_squared_error(y_true, y_pred)))
def mae(y_true, y_pred):  return float(mean_absolute_error(y_true, y_pred))

def precision_at_k(recs: list, relevant: set, k: int = N_TOP) -> float:
    if not recs: return 0.0
    return sum(1 for r in recs[:k] if r in relevant) / k

def recall_at_k(recs: list, relevant: set, k: int = N_TOP) -> float:
    if not recs or not relevant: return 0.0
    return sum(1 for r in recs[:k] if r in relevant) / len(relevant)

def ndcg_at_k(recs: list, relevant: set, k: int = N_TOP) -> float:
    dcg   = sum((1 / math.log2(i + 2)) for i, r in enumerate(recs[:k]) if r in relevant)
    idcg  = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(model, train, test, all_items, k=N_TOP, is_mf=False, is_knn=False):
    """Returns (P@k, R@k, NDCG@k) averaged over test users that have >= 1 liked item."""
    liked      = test[test["rating"] >= 3.5].groupby("user_id")["track_id"].apply(set).to_dict()
    test_users = test["user_id"].unique()
    p_s, r_s, n_s = [], [], []
    for uid in test_users:
        relevant = liked.get(uid, set())
        if not relevant:
            continue
        if is_mf or is_knn:
            recs = model.top_n_for_user(uid, train, n=k)
        else:
            recs = model.top_n_for_user(uid, train, all_items, n=k)
        p_s.append(precision_at_k(recs, relevant, k))
        r_s.append(recall_at_k(recs, relevant, k))
        n_s.append(ndcg_at_k(recs, relevant, k))
    mn = lambda lst: round(float(np.mean(lst)), 4) if lst else 0.0
    return mn(p_s), mn(r_s), mn(n_s)


def make_result(model_name, y_true, y_pred, p10, r10, nd10):
    return {
        "model":            model_name,
        "rmse":             round(rmse(y_true, y_pred), 4),
        "mae":              round(mae(y_true, y_pred),  4),
        "precision_at_10":  p10,
        "recall_at_10":     r10,
        "ndcg_at_10":       nd10,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Recommendations builder
# ══════════════════════════════════════════════════════════════════════════════

def build_recommendations(mf, train, track_meta, n_users=5, n_recs=N_TOP):
    active_users = (train.groupby("user_id").size()
                    .sort_values(ascending=False).head(n_users).index.tolist())
    id_col       = "track_id" if "track_id" in track_meta.columns else "artist_id"
    artist_lookup = track_meta.set_index(id_col)["artist"].to_dict()
    rows = []
    for uid in active_users:
        for rank, tid in enumerate(mf.top_n_for_user(uid, train, n=n_recs), start=1):
            rows.append({"user_id": uid, "rank": rank, "track_id": tid,
                         "artist": artist_lookup.get(tid, "Unknown"),
                         "predicted_rating": round(mf.predict_pair(uid, tid), 3)})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  AudioMuse-AI  |  Person 4: Modeling & Evaluation")
    print("=" * 62)

    print("\n[1/4] Loading data...")
    train, test = load_data()
    print(f"  Train: {len(train):,} | {train['user_id'].nunique()} users | {train['track_id'].nunique()} items")
    print(f"  Test : {len(test):,}  | {test['user_id'].nunique()} users  | {test['track_id'].nunique()} items")

    tags_df  = load_item_features(TAG_PATH)
    audio_df = load_item_features(AUDIO_PATH)
    print(f"  Tags : {len(tags_df) if tags_df is not None else 0:,} items × "
          f"{len([c for c in (tags_df.columns if tags_df is not None else []) if c != 'track_id'])} features")
    print(f"  Audio: {len(audio_df) if audio_df is not None else 0:,} items × "
          f"{len([c for c in (audio_df.columns if audio_df is not None else []) if c != 'track_id'])} features")

    track_meta = pd.read_csv(PROCESSED / "track_metadata.csv")
    all_items  = sorted(train["track_id"].unique().tolist())
    y_true     = test["rating"].values
    results    = []

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("\n[2/4] Training baseline models...")

    gm = GlobalMeanModel().fit(train)
    p10, r10, nd10 = evaluate_ranking(gm, train, test, all_items)
    results.append(make_result("Global Mean", y_true, gm.predict(test), p10, r10, nd10))
    print(f"  Global Mean  — RMSE {results[-1]['rmse']:.4f}  MAE {results[-1]['mae']:.4f}")

    um = UserMeanModel().fit(train)
    p10, r10, nd10 = evaluate_ranking(um, train, test, all_items)
    results.append(make_result("User Mean", y_true, um.predict(test), p10, r10, nd10))
    print(f"  User Mean    — RMSE {results[-1]['rmse']:.4f}  MAE {results[-1]['mae']:.4f}")

    im = ItemMeanModel().fit(train)
    p10, r10, nd10 = evaluate_ranking(im, train, test, all_items)
    results.append(make_result("Item Mean", y_true, im.predict(test), p10, r10, nd10))
    print(f"  Item Mean    — RMSE {results[-1]['rmse']:.4f}  MAE {results[-1]['mae']:.4f}")

    print("  Fitting kNN (k=30, user-based cosine)...")
    knn = UserKNNModel(k=KNN_K).fit(train)
    p10, r10, nd10 = evaluate_ranking(knn, train, test, all_items, is_knn=True)
    results.append(make_result("kNN (user-based)", y_true, knn.predict(test), p10, r10, nd10))
    print(f"  kNN          — RMSE {results[-1]['rmse']:.4f}  MAE {results[-1]['mae']:.4f}")

    # ── Matrix Factorization ──────────────────────────────────────────────────
    print("\n[3/4] Training Matrix Factorization models...")

    print("  MF (ratings only)...")
    mf_base = MatrixFactorization(n_factors=N_FACTORS).fit(train)
    p10, r10, nd10 = evaluate_ranking(mf_base, train, test, all_items, is_mf=True)
    results.append(make_result("MF (ratings only)", y_true, mf_base.predict(test), p10, r10, nd10))
    mf_base.save(MODELS_DIR / "mf_base.npz")
    print(f"  MF base      — RMSE {results[-1]['rmse']:.4f}  MAE {results[-1]['mae']:.4f}")

    mf_full = mf_base   # fallback

    if tags_df is not None:
        print("  MF + tags...")
        mf_tags = MatrixFactorization(n_factors=N_FACTORS).fit(train, item_features=tags_df)
        p10, r10, nd10 = evaluate_ranking(mf_tags, train, test, all_items, is_mf=True)
        results.append(make_result("MF + tags", y_true, mf_tags.predict(test), p10, r10, nd10))
        mf_tags.save(MODELS_DIR / "mf_tags.npz")
        mf_full = mf_tags
        print(f"  MF + tags    — RMSE {results[-1]['rmse']:.4f}  MAE {results[-1]['mae']:.4f}")

    if audio_df is not None:
        print("  MF + audio...")
        mf_audio = MatrixFactorization(n_factors=N_FACTORS).fit(train, item_features=audio_df)
        p10, r10, nd10 = evaluate_ranking(mf_audio, train, test, all_items, is_mf=True)
        results.append(make_result("MF + audio", y_true, mf_audio.predict(test), p10, r10, nd10))
        mf_audio.save(MODELS_DIR / "mf_audio.npz")
        print(f"  MF + audio   — RMSE {results[-1]['rmse']:.4f}  MAE {results[-1]['mae']:.4f}")

    if tags_df is not None and audio_df is not None:
        print("  MF + tags + audio...")
        audio_cols = [c for c in audio_df.columns if c != "track_id"]
        combined   = tags_df.merge(audio_df, on="track_id", how="left")
        for col in audio_cols:
            combined[col] = combined[col].fillna(0.0)
        mf_full = MatrixFactorization(n_factors=N_FACTORS).fit(train, item_features=combined)
        p10, r10, nd10 = evaluate_ranking(mf_full, train, test, all_items, is_mf=True)
        results.append(make_result("MF + tags + audio", y_true, mf_full.predict(test), p10, r10, nd10))
        mf_full.save(MODELS_DIR / "mf_full.npz")
        print(f"  MF + full    — RMSE {results[-1]['rmse']:.4f}  MAE {results[-1]['mae']:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n[4/4] Saving results...")
    eval_df = pd.DataFrame(results)
    eval_df.to_csv(EVAL_OUT, index=False)
    print(f"  Evaluation results → {EVAL_OUT}")

    recs_df = build_recommendations(mf_full, train, track_meta)
    recs_df.to_csv(RECS_OUT, index=False)
    print(f"  Recommendations    → {RECS_OUT}")

    print("\n" + "=" * 62)
    print("  ABLATION TABLE")
    print("=" * 62)
    print(eval_df.to_string(index=False))
    print("=" * 62)
    best = eval_df.loc[eval_df["rmse"].idxmin()]
    print(f"\n  Best RMSE model : {best['model']}  →  RMSE {best['rmse']}  MAE {best['mae']}")

    print("\n  SAMPLE RECOMMENDATIONS (top-10, user 1)")
    first_user = recs_df["user_id"].iloc[0]
    print(recs_df[recs_df["user_id"] == first_user][["rank", "artist", "predicted_rating"]].to_string(index=False))
    print("\n[DONE] Modeling complete.")


if __name__ == "__main__":
    main()
