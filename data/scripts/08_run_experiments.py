#!/usr/bin/env python3
"""Run rating, ranking, and hybrid ablation experiments for the finished project."""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

TRAIN_PATH = PROCESSED / "ratings_train.csv"
TEST_PATH = PROCESSED / "ratings_test.csv"
TRAIN_OVERLAP_PATH = PROCESSED / "ratings_train_overlap.csv"
TEST_OVERLAP_PATH = PROCESSED / "ratings_test_overlap.csv"
TAGS_PATH = PROCESSED / "tag_features.csv"
AUDIO_TRAIN_PATH = PROCESSED / "audio_features_artist_train.csv"
AUDIO_TEST_PATH = PROCESSED / "audio_features_artist_test.csv"
META_PATH = PROCESSED / "track_metadata.csv"

K = 10
POSITIVE_THRESHOLD = 4.0


def require(path: pathlib.Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_ratings(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"user_id": int, "artist_id": int})


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def ndcg_at_k(relevances: list[int], k: int) -> float:
    rel = np.asarray(relevances[:k], dtype=float)
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float(np.sum(rel * discounts))
    ideal = np.sort(rel)[::-1]
    idcg = float(np.sum(ideal * discounts))
    return 0.0 if idcg == 0.0 else dcg / idcg


@dataclass
class RatingContext:
    train: pd.DataFrame
    test: pd.DataFrame
    users: np.ndarray
    items: np.ndarray
    user_mean: dict[int, float]
    item_mean: dict[int, float]
    global_mean: float
    train_seen: dict[int, set[int]]
    test_truth: dict[int, dict[int, float]]


def build_context(train: pd.DataFrame, test: pd.DataFrame) -> RatingContext:
    users = np.sort(train["user_id"].unique())
    items = np.sort(train["artist_id"].unique())
    user_mean = train.groupby("user_id")["rating"].mean().to_dict()
    item_mean = train.groupby("artist_id")["rating"].mean().to_dict()
    global_mean = float(train["rating"].mean())
    train_seen = train.groupby("user_id")["artist_id"].apply(lambda s: set(s.tolist())).to_dict()
    test_truth = test.groupby("user_id").apply(
        lambda g: {int(a): float(r) for a, r in zip(g["artist_id"], g["rating"])}, include_groups=False
    ).to_dict()
    return RatingContext(train, test, users, items, user_mean, item_mean, global_mean, train_seen, test_truth)


class BaseModel:
    name = "base"

    def fit(self, train: pd.DataFrame) -> None:
        self.train = train

    def predict(self, user_id: int, artist_id: int) -> float:
        raise NotImplementedError


class GlobalMeanModel(BaseModel):
    name = "Global mean"

    def fit(self, train: pd.DataFrame) -> None:
        super().fit(train)
        self.mean_ = float(train["rating"].mean())

    def predict(self, user_id: int, artist_id: int) -> float:
        return self.mean_


class UserMeanModel(BaseModel):
    name = "User mean"

    def fit(self, train: pd.DataFrame) -> None:
        super().fit(train)
        self.global_mean_ = float(train["rating"].mean())
        self.user_mean_ = train.groupby("user_id")["rating"].mean().to_dict()

    def predict(self, user_id: int, artist_id: int) -> float:
        return float(self.user_mean_.get(user_id, self.global_mean_))


class ItemMeanModel(BaseModel):
    name = "Item mean"

    def fit(self, train: pd.DataFrame) -> None:
        super().fit(train)
        self.global_mean_ = float(train["rating"].mean())
        self.item_mean_ = train.groupby("artist_id")["rating"].mean().to_dict()

    def predict(self, user_id: int, artist_id: int) -> float:
        return float(self.item_mean_.get(artist_id, self.global_mean_))


class UserKNNModel(BaseModel):
    name = "kNN CF"

    def __init__(self, top_k: int = 20):
        self.top_k = top_k

    def fit(self, train: pd.DataFrame) -> None:
        super().fit(train)
        pivot = train.pivot_table(index="user_id", columns="artist_id", values="rating")
        self.user_ids_ = pivot.index.to_numpy()
        self.item_ids_ = pivot.columns.to_numpy()
        self.global_mean_ = float(train["rating"].mean())
        self.user_mean_ = train.groupby("user_id")["rating"].mean().to_dict()
        filled = pivot.sub(pivot.mean(axis=1), axis=0).fillna(0.0)
        self.matrix_ = filled.to_numpy(dtype=float)
        self.user_index_ = {int(u): i for i, u in enumerate(self.user_ids_)}
        self.item_index_ = {int(a): i for i, a in enumerate(self.item_ids_)}
        self.sim_ = cosine_similarity(self.matrix_)
        self.ratings_lookup_ = train.groupby("user_id").apply(
            lambda g: {int(a): float(r) for a, r in zip(g["artist_id"], g["rating"])}, include_groups=False
        ).to_dict()

    def predict(self, user_id: int, artist_id: int) -> float:
        if user_id not in self.user_index_:
            return self.global_mean_
        if artist_id not in self.item_index_:
            return self.user_mean_.get(user_id, self.global_mean_)

        uidx = self.user_index_[user_id]
        sims = self.sim_[uidx]
        neighbors = []
        for other_uid, ratings in self.ratings_lookup_.items():
            if other_uid == user_id or artist_id not in ratings or other_uid not in self.user_index_:
                continue
            s = sims[self.user_index_[other_uid]]
            if s > 0:
                neighbors.append((s, ratings[artist_id], self.user_mean_.get(other_uid, self.global_mean_)))
        if not neighbors:
            return self.user_mean_.get(user_id, self.global_mean_)
        neighbors.sort(key=lambda x: x[0], reverse=True)
        neighbors = neighbors[: self.top_k]
        numer = sum(sim * (rating - mean_other) for sim, rating, mean_other in neighbors)
        denom = sum(abs(sim) for sim, _, _ in neighbors)
        base = self.user_mean_.get(user_id, self.global_mean_)
        return float(base if denom == 0 else base + numer / denom)


class MatrixFactorizationModel(BaseModel):
    name = "MF (Ratings)"

    def __init__(self, n_factors: int = 20, n_epochs: int = 20, lr: float = 0.01, reg: float = 0.05):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg

    def fit(self, train: pd.DataFrame) -> None:
        super().fit(train)
        self.global_mean_ = float(train["rating"].mean())
        self.user_ids_ = sorted(train["user_id"].unique())
        self.item_ids_ = sorted(train["artist_id"].unique())
        self.user_index_ = {u: i for i, u in enumerate(self.user_ids_)}
        self.item_index_ = {a: i for i, a in enumerate(self.item_ids_)}

        rng = np.random.default_rng(42)
        self.user_bias_ = np.zeros(len(self.user_ids_), dtype=float)
        self.item_bias_ = np.zeros(len(self.item_ids_), dtype=float)
        self.user_factors_ = rng.normal(0, 0.1, size=(len(self.user_ids_), self.n_factors))
        self.item_factors_ = rng.normal(0, 0.1, size=(len(self.item_ids_), self.n_factors))

        rows = train[["user_id", "artist_id", "rating"]].to_numpy()
        for _ in range(self.n_epochs):
            rng.shuffle(rows)
            for user_id, artist_id, rating in rows:
                u = self.user_index_[int(user_id)]
                i = self.item_index_[int(artist_id)]
                pred = self.global_mean_ + self.user_bias_[u] + self.item_bias_[i] + float(np.dot(self.user_factors_[u], self.item_factors_[i]))
                err = float(rating) - pred
                pu = self.user_factors_[u].copy()
                qi = self.item_factors_[i].copy()
                self.user_bias_[u] += self.lr * (err - self.reg * self.user_bias_[u])
                self.item_bias_[i] += self.lr * (err - self.reg * self.item_bias_[i])
                self.user_factors_[u] += self.lr * (err * qi - self.reg * pu)
                self.item_factors_[i] += self.lr * (err * pu - self.reg * qi)

    def predict(self, user_id: int, artist_id: int) -> float:
        if user_id not in self.user_index_ and artist_id not in self.item_index_:
            return self.global_mean_
        if user_id not in self.user_index_:
            return float(self.global_mean_ + self.item_bias_[self.item_index_[artist_id]]) if artist_id in self.item_index_ else self.global_mean_
        if artist_id not in self.item_index_:
            return float(self.global_mean_ + self.user_bias_[self.user_index_[user_id]])
        u = self.user_index_[user_id]
        i = self.item_index_[artist_id]
        pred = self.global_mean_ + self.user_bias_[u] + self.item_bias_[i] + float(np.dot(self.user_factors_[u], self.item_factors_[i]))
        return float(pred)


class HybridResidualModel(BaseModel):
    def __init__(self, name: str, base_model: MatrixFactorizationModel, features: pd.DataFrame | None):
        self.name = name
        self.base_model = base_model
        self.features = features

    def fit(self, train: pd.DataFrame) -> None:
        self.base_model.fit(train)
        self.global_mean_ = self.base_model.global_mean_
        self.feature_lookup_ = {}
        self.residual_lookup_ = {}
        if self.features is None or self.features.empty:
            return

        numeric_cols = [
            c
            for c in self.features.columns
            if c not in {"artist_id", "artist", "tags_raw", "musicnet_ids", "recording_count", "audio_split_source"}
            and pd.api.types.is_numeric_dtype(self.features[c])
        ]
        features = self.features[["artist_id"] + numeric_cols].copy()
        features = features.drop_duplicates(subset=["artist_id"]).set_index("artist_id")
        imp = SimpleImputer(strategy="constant", fill_value=0.0)
        scaler = StandardScaler(with_mean=True, with_std=True)
        x = imp.fit_transform(features)
        x = scaler.fit_transform(x)

        item_residuals = []
        item_ids = []
        for artist_id, grp in train.groupby("artist_id"):
            if artist_id not in features.index:
                continue
            preds = np.array([self.base_model.predict(int(u), int(artist_id)) for u in grp["user_id"]])
            residual = float(np.mean(grp["rating"].to_numpy() - preds))
            item_ids.append(int(artist_id))
            item_residuals.append(residual)

        if not item_ids:
            return

        x_train = pd.DataFrame(x, index=features.index).loc[item_ids].to_numpy()
        y_train = np.asarray(item_residuals, dtype=float)

        if x_train.shape[1] == 0:
            return

        max_comp = min(16, x_train.shape[0], x_train.shape[1])
        if max_comp >= 1:
            svd = TruncatedSVD(n_components=max_comp, random_state=42)
            z_train = svd.fit_transform(x_train)
            z_all = svd.transform(x)
        else:
            z_train = x_train
            z_all = x

        reg = 1e-2
        xtx = z_train.T @ z_train + reg * np.eye(z_train.shape[1])
        xty = z_train.T @ y_train
        weights = np.linalg.solve(xtx, xty)
        residuals = z_all @ weights

        for artist_id, residual in zip(features.index.tolist(), residuals.tolist()):
            self.residual_lookup_[int(artist_id)] = float(residual)

    def predict(self, user_id: int, artist_id: int) -> float:
        base = self.base_model.predict(user_id, artist_id)
        return float(base + self.residual_lookup_.get(int(artist_id), 0.0))


def clip_rating(pred: float) -> float:
    return float(np.clip(pred, 1.0, 5.0))


def evaluate_model(model: BaseModel, context: RatingContext, evaluation_scope: str) -> dict[str, float | str | int]:
    preds = np.array([clip_rating(model.predict(int(u), int(a))) for u, a in zip(context.test["user_id"], context.test["artist_id"])])
    truth = context.test["rating"].to_numpy(dtype=float)
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    candidate_items = context.items.tolist()

    for user_id, truth_map in context.test_truth.items():
        positives = {item for item, rating in truth_map.items() if rating >= POSITIVE_THRESHOLD}
        if not positives:
            continue
        seen = context.train_seen.get(user_id, set())
        candidates = [item for item in candidate_items if item not in seen]
        if not candidates:
            continue
        scored = [(item, clip_rating(model.predict(int(user_id), int(item)))) for item in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_items = [item for item, _ in scored[:K]]
        hits = [1 if item in positives else 0 for item in top_items]
        precision_scores.append(sum(hits) / K)
        recall_scores.append(sum(hits) / len(positives))
        ndcg_scores.append(ndcg_at_k(hits, K))

    return {
        "rmse": rmse(truth, preds),
        "mae": mae(truth, preds),
        f"precision@{K}": float(np.mean(precision_scores)) if precision_scores else np.nan,
        f"recall@{K}": float(np.mean(recall_scores)) if recall_scores else np.nan,
        f"ndcg@{K}": float(np.mean(ndcg_scores)) if ndcg_scores else np.nan,
        "n_train": int(len(context.train)),
        "n_test": int(len(context.test)),
        "n_test_users": int(context.test["user_id"].nunique()),
        "n_test_items": int(context.test["artist_id"].nunique()),
        "evaluation_scope": evaluation_scope,
    }


def load_tag_features() -> pd.DataFrame:
    require(TAGS_PATH)
    return pd.read_csv(TAGS_PATH)


def load_audio_features() -> pd.DataFrame | None:
    if not AUDIO_TRAIN_PATH.exists() and not AUDIO_TEST_PATH.exists():
        return None
    frames = []
    for path in [AUDIO_TRAIN_PATH, AUDIO_TEST_PATH]:
        if path.exists():
            frame = pd.read_csv(path)
            frame["audio_split_source"] = path.stem
            frames.append(frame)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["artist_id", "recording_count"], ascending=[True, False])
    return combined.drop_duplicates(subset=["artist_id"]).reset_index(drop=True)


def build_slice_metrics(test: pd.DataFrame, pred_lookup: dict[tuple[int, int], float], train: pd.DataFrame) -> pd.DataFrame:
    user_counts = train.groupby("user_id").size().rename("train_user_count")
    item_counts = train.groupby("artist_id").size().rename("train_item_count")
    enriched = test.merge(user_counts, on="user_id", how="left").merge(item_counts, on="artist_id", how="left")
    enriched["pred"] = [pred_lookup[(int(u), int(a))] for u, a in zip(enriched["user_id"], enriched["artist_id"])]

    slices = []
    sparse_users = enriched[enriched["train_user_count"] <= 10]
    if not sparse_users.empty:
        slices.append({"slice": "sparse_users_train_le_10", "rmse": rmse(sparse_users["rating"], sparse_users["pred"]), "mae": mae(sparse_users["rating"], sparse_users["pred"]), "n": len(sparse_users)})
    low_items = enriched[enriched["train_item_count"] <= 2]
    if not low_items.empty:
        slices.append({"slice": "low_interaction_items_train_le_2", "rmse": rmse(low_items["rating"], low_items["pred"]), "mae": mae(low_items["rating"], low_items["pred"]), "n": len(low_items)})
    return pd.DataFrame(slices)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small markdown table without requiring optional tabulate dependency."""
    if df.empty:
        return "_No rows available._"

    safe_df = df.copy()
    for col in safe_df.columns:
        safe_df[col] = safe_df[col].map(lambda x: "" if pd.isna(x) else str(x))

    headers = list(safe_df.columns)
    rows = safe_df.values.tolist()
    widths = []
    for idx, header in enumerate(headers):
        max_cell = max(len(str(row[idx])) for row in rows) if rows else 0
        widths.append(max(len(str(header)), max_cell))

    def fmt_row(row: list[str]) -> str:
        return "| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    header_row = fmt_row(headers)
    sep_row = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    body = [fmt_row(row) for row in rows]
    return "\n".join([header_row, sep_row] + body)


def write_markdown(results_df: pd.DataFrame, slices_df: pd.DataFrame, qualitative_df: pd.DataFrame) -> None:
    lines = [
        "# Ablation Results",
        "",
        f"Top-N metrics use K={K} and explicit-positive threshold rating >= {POSITIVE_THRESHOLD}.",
        "",
        dataframe_to_markdown(results_df),
        "",
        "## Slice Analysis",
        "",
    ]
    if slices_df.empty:
        lines.append("No slice metrics were available.")
    else:
        lines.append(dataframe_to_markdown(slices_df))
    lines.extend(["", "## Qualitative Examples", "", dataframe_to_markdown(qualitative_df)])
    (RESULTS / "ablation_results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    require(TRAIN_PATH)
    require(TEST_PATH)
    require(META_PATH)

    train = load_ratings(TRAIN_PATH)
    test = load_ratings(TEST_PATH)
    train_overlap = load_ratings(TRAIN_OVERLAP_PATH) if TRAIN_OVERLAP_PATH.exists() else train.iloc[0:0].copy()
    test_overlap = load_ratings(TEST_OVERLAP_PATH) if TEST_OVERLAP_PATH.exists() else test.iloc[0:0].copy()
    tags = load_tag_features()
    audio = load_audio_features()

    full_context = build_context(train, test)
    overlap_context = build_context(train_overlap, test_overlap) if not train_overlap.empty and not test_overlap.empty else None

    models: list[tuple[str, BaseModel, RatingContext, str]] = [
        ("Global mean", GlobalMeanModel(), full_context, "full_catalog"),
        ("User mean", UserMeanModel(), full_context, "full_catalog"),
        ("Item mean", ItemMeanModel(), full_context, "full_catalog"),
        ("kNN CF", UserKNNModel(), full_context, "full_catalog"),
        ("MF (Ratings)", MatrixFactorizationModel(), full_context, "full_catalog"),
        ("MF + Tags", HybridResidualModel("MF + Tags", MatrixFactorizationModel(), tags), full_context, "full_catalog"),
    ]

    if overlap_context is not None and audio is not None:
        models.append(("MF + Audio", HybridResidualModel("MF + Audio", MatrixFactorizationModel(), audio), overlap_context, "audio_overlap"))
        tag_audio = tags.merge(audio, on="artist_id", how="inner")
        models.append(("MF + Tags + Audio", HybridResidualModel("MF + Tags + Audio", MatrixFactorizationModel(), tag_audio), overlap_context, "audio_overlap"))
    else:
        models.append(("MF + Audio", None, full_context, "audio_overlap_unavailable"))
        models.append(("MF + Tags + Audio", None, full_context, "audio_overlap_unavailable"))

    results = []
    slice_frames = []
    qualitative_rows = []

    for model_name, model, context, scope in models:
        if model is None or context.test.empty:
            results.append({
                "model": model_name,
                "rmse": np.nan,
                "mae": np.nan,
                f"precision@{K}": np.nan,
                f"recall@{K}": np.nan,
                f"ndcg@{K}": np.nan,
                "n_train": int(len(context.train)),
                "n_test": int(len(context.test)),
                "n_test_users": int(context.test["user_id"].nunique()),
                "n_test_items": int(context.test["artist_id"].nunique()),
                "evaluation_scope": scope,
                "status": "unavailable_no_audio_test_overlap",
            })
            continue

        print(f"[RUN] {model_name} ({scope})")
        model.fit(context.train)
        metrics = evaluate_model(model, context, scope)
        metrics["model"] = model_name
        metrics["status"] = "ok"
        results.append(metrics)

        pred_lookup = {(int(u), int(a)): clip_rating(model.predict(int(u), int(a))) for u, a in zip(context.test["user_id"], context.test["artist_id"])}
        slice_df = build_slice_metrics(context.test, pred_lookup, context.train)
        if not slice_df.empty:
            slice_df.insert(0, "model", model_name)
            slice_df.insert(1, "evaluation_scope", scope)
            slice_frames.append(slice_df)

        sample_users = sorted(context.test["user_id"].unique())[:5]
        for user_id in sample_users:
            seen = context.train_seen.get(int(user_id), set())
            candidates = [int(item) for item in context.items if int(item) not in seen]
            if not candidates:
                continue
            ranked = sorted(((item, clip_rating(model.predict(int(user_id), item))) for item in candidates), key=lambda x: x[1], reverse=True)[:3]
            qualitative_rows.append({
                "model": model_name,
                "user_id": int(user_id),
                "recommendations": json.dumps(ranked),
                "evaluation_scope": scope,
            })

    results_df = pd.DataFrame(results)
    order = ["Global mean", "User mean", "Item mean", "kNN CF", "MF (Ratings)", "MF + Tags", "MF + Audio", "MF + Tags + Audio"]
    results_df["model"] = pd.Categorical(results_df["model"], categories=order, ordered=True)
    results_df = results_df.sort_values("model").reset_index(drop=True)
    results_df.to_csv(RESULTS / "ablation_results.csv", index=False)

    slices_df = pd.concat(slice_frames, ignore_index=True) if slice_frames else pd.DataFrame(columns=["model", "evaluation_scope", "slice", "rmse", "mae", "n"])
    slices_df.to_csv(RESULTS / "slice_metrics.csv", index=False)

    qualitative_df = pd.DataFrame(qualitative_rows)
    qualitative_df.to_csv(RESULTS / "qualitative_examples.csv", index=False)
    qualitative_md = ["# Qualitative Examples", ""]
    if qualitative_df.empty:
        qualitative_md.append("No qualitative examples were generated.")
    else:
        qualitative_md.append(dataframe_to_markdown(qualitative_df))
    (RESULTS / "qualitative_examples.md").write_text("\n".join(qualitative_md), encoding="utf-8")

    summary = {
        "positive_threshold": POSITIVE_THRESHOLD,
        "k": K,
        "audio_train_overlap_rows": int(len(train_overlap)),
        "audio_test_overlap_rows": int(len(test_overlap)),
        "audio_train_overlap_artists": int(train_overlap["artist_id"].nunique()) if not train_overlap.empty else 0,
        "audio_test_overlap_artists": int(test_overlap["artist_id"].nunique()) if not test_overlap.empty else 0,
    }
    (RESULTS / "experiment_config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_markdown(results_df, slices_df, qualitative_df if not qualitative_df.empty else pd.DataFrame([{"model": "n/a", "user_id": "n/a", "recommendations": "none", "evaluation_scope": "n/a"}]))
    print(f"[DONE] Saved experiment outputs to {RESULTS}")


if __name__ == "__main__":
    main()