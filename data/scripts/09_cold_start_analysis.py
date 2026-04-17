#!/usr/bin/env python3
"""
09_cold_start_analysis.py  —  Person 5: Cold-start & sparsity analysis
=======================================================================
Segments test users by training history depth and compares model
performance across cold / near-cold / normal user populations.

Segments (defined by # ratings in train):
  Cold      : <= 5
  Near-cold : 6 – 20
  Normal    : > 20

Writes:
  data/processed/cold_start_results.csv
"""

from __future__ import annotations

import math
import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

ROOT       = pathlib.Path(__file__).resolve().parents[2]
PROCESSED  = ROOT / "data" / "processed"

TRAIN_PATH = PROCESSED / "ratings_train.csv"
TEST_PATH  = PROCESSED / "ratings_test.csv"
TAG_PATH   = PROCESSED / "tag_features.csv"
EVAL_OUT   = PROCESSED / "evaluation_results.csv"
COLD_OUT   = PROCESSED / "cold_start_results.csv"

N_FACTORS        = 50
ITEM_FEAT_WEIGHT = 0.3
N_TOP            = 10
COLD_THRESH      = 5
NC_THRESH        = 20


# ── helpers ───────────────────────────────────────────────────────────────────

def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    for df in (train, test):
        if "artist_id" in df.columns and "track_id" not in df.columns:
            df.rename(columns={"artist_id": "track_id"}, inplace=True)
    return train, test


def load_tags():
    if not TAG_PATH.exists():
        return None
    df = pd.read_csv(TAG_PATH)
    if "artist_id" in df.columns and "track_id" not in df.columns:
        df = df.rename(columns={"artist_id": "track_id"})
    feat_cols = [c for c in df.columns if c != "track_id" and pd.api.types.is_numeric_dtype(df[c])]
    return df[["track_id"] + feat_cols].dropna()


def segment_users(train: pd.DataFrame):
    counts = train.groupby("user_id").size()
    cold   = set(counts[counts <= COLD_THRESH].index)
    nc     = set(counts[(counts > COLD_THRESH) & (counts <= NC_THRESH)].index)
    normal = set(counts[counts > NC_THRESH].index)
    return cold, nc, normal


# ── lightweight MF ────────────────────────────────────────────────────────────

def fit_mf(train: pd.DataFrame, item_features=None):
    global_mean = float(train["rating"].mean())
    user_ids    = sorted(train["user_id"].unique())
    item_ids    = sorted(train["track_id"].unique())
    user_index  = {u: i for i, u in enumerate(user_ids)}
    item_index  = {t: j for j, t in enumerate(item_ids)}

    n_u, n_i = len(user_ids), len(item_ids)
    R = np.zeros((n_u, n_i), dtype=np.float32)
    for row in train.itertuples():
        R[user_index[row.user_id], item_index[row.track_id]] = row.rating - global_mean

    user_bias = np.array([train.loc[train["user_id"] == u, "rating"].mean() - global_mean
                          for u in user_ids], dtype=np.float32)
    item_bias = np.array([train.loc[train["track_id"] == t, "rating"].mean() - global_mean
                          for t in item_ids], dtype=np.float32)

    k = min(N_FACTORS, n_u - 1, n_i - 1)
    U, sigma, Vt = svds(R.astype(np.float64), k=k)
    idx = np.argsort(sigma)[::-1]
    U, sigma, Vt = U[:, idx], sigma[idx], Vt[idx, :]
    UF = (U * sigma).astype(np.float32)
    VF = Vt.T.astype(np.float32)

    if item_features is not None:
        feat_cols   = [c for c in item_features.columns if c != "track_id"]
        feat_matrix = item_features.set_index("track_id")[feat_cols].reindex(item_ids).fillna(0.0).values.astype(np.float32)
        feat_norm   = normalize(feat_matrix, axis=1)
        try:
            _, _, Vf = svds(feat_norm.astype(np.float64), k=k)
            fp = normalize((feat_norm @ Vf.T).astype(np.float32), axis=1)
        except Exception:
            fp = normalize(feat_norm[:, :k], axis=1) if feat_norm.shape[1] >= k else normalize(np.pad(feat_norm, ((0,0),(0,k-feat_norm.shape[1]))), axis=1)
        VF = ((1 - ITEM_FEAT_WEIGHT) * normalize(VF, axis=1) + ITEM_FEAT_WEIGHT * fp).astype(np.float32)

    return dict(global_mean=global_mean, user_ids=user_ids, item_ids=item_ids,
                user_index=user_index, item_index=item_index,
                UF=UF, VF=VF, user_bias=user_bias, item_bias=item_bias)


def predict_mf(model, df: pd.DataFrame) -> np.ndarray:
    preds = []
    for r in df.itertuples():
        u = model["user_index"].get(r.user_id)
        i = model["item_index"].get(r.track_id)
        if u is None or i is None:
            preds.append(model["global_mean"])
        else:
            preds.append(float(np.clip(
                model["global_mean"] + model["user_bias"][u] + model["item_bias"][i]
                + float(model["UF"][u] @ model["VF"][i]), 1.0, 5.0)))
    return np.array(preds)


# ── metrics ───────────────────────────────────────────────────────────────────

def rmse(yt, yp): return float(np.sqrt(mean_squared_error(yt, yp)))
def mae_(yt, yp): return float(mean_absolute_error(yt, yp))

def precision_at_k(recs, relevant, k=N_TOP):
    if not recs: return 0.0
    return sum(1 for r in recs[:k] if r in relevant) / k

def recall_at_k(recs, relevant, k=N_TOP):
    if not recs or not relevant: return 0.0
    return sum(1 for r in recs[:k] if r in relevant) / len(relevant)

def ndcg_at_k(recs, relevant, k=N_TOP):
    dcg  = sum(1/math.log2(i+2) for i,r in enumerate(recs[:k]) if r in relevant)
    idcg = sum(1/math.log2(i+2) for i in range(min(len(relevant), k)))
    return dcg/idcg if idcg > 0 else 0.0

def top_n_mf(model, uid, train, n=N_TOP):
    if uid not in model["user_index"]: return []
    u    = model["user_index"][uid]
    seen = set(train.loc[train["user_id"] == uid, "track_id"])
    scores = model["global_mean"] + model["user_bias"][u] + model["item_bias"] + (model["UF"][u] @ model["VF"].T)
    ranked = sorted([(model["item_ids"][j], float(scores[j])) for j in range(len(model["item_ids"]))
                     if model["item_ids"][j] not in seen], key=lambda x: -x[1])
    return [t for t,_ in ranked[:n]]

def eval_segment(model_obj, train, seg_test, all_items, is_mf=False):
    if len(seg_test) == 0:
        return dict(n_users=0, n_ratings=0, rmse=None, mae=None,
                    precision_at_10=None, recall_at_10=None, ndcg_at_10=None)
    y_true = seg_test["rating"].values
    if is_mf:
        y_pred = predict_mf(model_obj, seg_test)
    else:
        um = model_obj
        y_pred = np.array([um.get(r.user_id, float(np.mean(list(um.values()))))
                           for r in seg_test.itertuples()])

    liked = seg_test[seg_test["rating"] >= 3.5].groupby("user_id")["track_id"].apply(set).to_dict()
    p_s, r_s, n_s = [], [], []
    for uid in seg_test["user_id"].unique():
        relevant = liked.get(uid, set())
        if not relevant: continue
        if is_mf:
            recs = top_n_mf(model_obj, uid, train, n=N_TOP)
        else:
            seen = set(train.loc[train["user_id"] == uid, "track_id"])
            recs = [i for i in all_items if i not in seen][:N_TOP]
        p_s.append(precision_at_k(recs, relevant))
        r_s.append(recall_at_k(recs, relevant))
        n_s.append(ndcg_at_k(recs, relevant))

    mn = lambda lst: round(float(np.mean(lst)), 4) if lst else 0.0
    return dict(
        n_users=seg_test["user_id"].nunique(),
        n_ratings=len(seg_test),
        rmse=round(rmse(y_true, y_pred), 4),
        mae=round(mae_(y_true, y_pred), 4),
        precision_at_10=mn(p_s),
        recall_at_10=mn(r_s),
        ndcg_at_10=mn(n_s),
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  AudioMuse-AI  |  Cold-Start & Sparsity Analysis")
    print("=" * 62)

    train, test = load_data()
    tags_df     = load_tags()
    cold_ids, nc_ids, normal_ids = segment_users(train)
    all_items = sorted(train["track_id"].unique().tolist())

    test_cold   = test[test["user_id"].isin(cold_ids)]
    test_nc     = test[test["user_id"].isin(nc_ids)]
    test_normal = test[test["user_id"].isin(normal_ids)]

    seg_labels = {
        "cold":      (test_cold,   f"<= {COLD_THRESH} train ratings"),
        "near_cold": (test_nc,     f"{COLD_THRESH+1}–{NC_THRESH} train ratings"),
        "normal":    (test_normal, f"> {NC_THRESH} train ratings"),
    }

    print(f"\n  Cold users in test    : {test['user_id'].isin(cold_ids).sum()} ratings, "
          f"{test_cold['user_id'].nunique()} users")
    print(f"  Near-cold users in test: {test['user_id'].isin(nc_ids).sum()} ratings, "
          f"{test_nc['user_id'].nunique()} users")
    print(f"  Normal users in test  : {test['user_id'].isin(normal_ids).sum()} ratings, "
          f"{test_normal['user_id'].nunique()} users")

    # Models to evaluate
    configs = [
        ("Global Mean",       None,     False),
        ("MF (ratings only)", None,     True),
        ("MF + tags",         tags_df,  True),
    ]

    rows = []
    for model_name, item_feats, is_mf in configs:
        print(f"\n  Evaluating: {model_name}")
        if is_mf:
            model = fit_mf(train, item_features=item_feats)
        else:
            global_mean = float(train["rating"].mean())
            model       = train.groupby("user_id")["rating"].mean().to_dict()

        for seg_name, (seg_test, seg_desc) in seg_labels.items():
            metrics = eval_segment(model, train, seg_test, all_items, is_mf=is_mf)
            rows.append({
                "model":       model_name,
                "segment":     seg_name,
                "description": seg_desc,
                **metrics,
            })
            n = metrics["n_users"]
            r = metrics["rmse"] if metrics["rmse"] is not None else "—"
            p = metrics["precision_at_10"] if metrics["precision_at_10"] is not None else "—"
            print(f"    {seg_name:12s}  users={n:4d}  RMSE={r}  P@10={p}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(COLD_OUT, index=False)
    print(f"\n  Written: {COLD_OUT}")

    # Summary table
    print("\n" + "=" * 62)
    print("  COLD-START SUMMARY (RMSE by model × segment)")
    print("=" * 62)
    pivot = results_df.pivot_table(index="model", columns="segment", values="rmse", aggfunc="first")
    print(pivot.to_string())
    print("=" * 62)
    print("\n[DONE]")


if __name__ == "__main__":
    main()
