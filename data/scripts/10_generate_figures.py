#!/usr/bin/env python3
"""
10_generate_figures.py  —  Person 5: Generate all report figures
================================================================
Produces publication-quality plots saved to notebooks/figures/.

Figures:
  fig1_rating_distribution.png    — histogram of rating values
  fig2_ratings_per_user.png       — ratings-per-user distribution (log scale)
  fig3_rmse_comparison.png        — RMSE bar chart across all models
  fig4_ranking_metrics.png        — P@10, R@10, NDCG@10 grouped bar chart
  fig5_cold_start_rmse.png        — RMSE by model × user-sparsity segment
  fig6_cold_start_ranking.png     — NDCG@10 by model × user-sparsity segment
  fig7_top_tags.png               — top-20 most common TF-IDF tags
  fig8_sample_recs.png            — formatted table of sample recommendations
"""

from __future__ import annotations
import pathlib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

ROOT      = pathlib.Path(__file__).resolve().parents[2]
PROC      = ROOT / "data" / "processed"
FIG_DIR   = ROOT / "notebooks" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette ─────────────────────────────────────────────────────────────
BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"
PURPLE = "#8172B3"
TEAL   = "#64B5CD"
GRAY   = "#8c8c8c"

PALETTE = [BLUE, ORANGE, GREEN, RED, PURPLE, TEAL, GRAY, "#937860"]

FONT = {"family": "DejaVu Sans"}
plt.rc("font", **FONT)
plt.rc("axes",  titlesize=13, labelsize=11)
plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)
plt.rc("legend", fontsize=10)


def save(name: str, fig=None):
    path = FIG_DIR / name
    (fig or plt).savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"  Saved: {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Rating distribution
# ══════════════════════════════════════════════════════════════════════════════
def fig_rating_distribution():
    ratings = pd.read_csv(PROC / "ratings_joined.csv")
    if "artist_id" in ratings.columns and "track_id" not in ratings.columns:
        ratings = ratings.rename(columns={"artist_id": "track_id"})

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ratings["rating"], bins=40, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(ratings["rating"].mean(), color=RED, linestyle="--", linewidth=1.5,
               label=f"Mean = {ratings['rating'].mean():.2f}")
    ax.set_xlabel("Rating (1.0 – 5.0)")
    ax.set_ylabel("Count")
    ax.set_title("Rating Distribution  (N = {:,})".format(len(ratings)))
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save("fig1_rating_distribution.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Ratings per user (sparsity)
# ══════════════════════════════════════════════════════════════════════════════
def fig_ratings_per_user():
    train = pd.read_csv(PROC / "ratings_train.csv")
    counts = train.groupby("user_id").size()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(counts, bins=50, color=ORANGE, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(5,  color=RED,    linestyle="--", linewidth=1.2, label="Cold  (≤ 5)")
    ax.axvline(20, color=PURPLE, linestyle="--", linewidth=1.2, label="Near-cold (≤ 20)")
    ax.set_xlabel("Number of ratings per user (train split)")
    ax.set_ylabel("Number of users")
    ax.set_title("Ratings-per-User Distribution  (sparsity = 99.7 %)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save("fig2_ratings_per_user.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — RMSE comparison
# ══════════════════════════════════════════════════════════════════════════════
def fig_rmse_comparison():
    eval_df = pd.read_csv(PROC / "evaluation_results.csv")
    models  = eval_df["model"].tolist()
    rmses   = eval_df["rmse"].tolist()
    colours = [GREEN if r == min(rmses) else BLUE for r in rmses]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(models, rmses, color=colours, edgecolor="white", linewidth=0.5, height=0.6)
    for bar, val in zip(bars, rmses):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=9.5)
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_title("RMSE Comparison Across Models")
    ax.set_xlim(0, max(rmses) * 1.18)
    ax.invert_yaxis()
    best_patch = mpatches.Patch(color=GREEN, label="Best model")
    ax.legend(handles=[best_patch], loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save("fig3_rmse_comparison.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Ranking metrics (P@10, R@10, NDCG@10)
# ══════════════════════════════════════════════════════════════════════════════
def fig_ranking_metrics():
    eval_df = pd.read_csv(PROC / "evaluation_results.csv")
    # Only show models with at least one nonzero ranking metric
    ranked = eval_df[(eval_df["precision_at_10"] > 0) |
                     (eval_df["recall_at_10"] > 0) |
                     (eval_df["ndcg_at_10"] > 0)].copy()
    if ranked.empty:
        ranked = eval_df.copy()

    models  = ranked["model"].tolist()
    p10     = ranked["precision_at_10"].tolist()
    r10     = ranked["recall_at_10"].tolist()
    nd10    = ranked["ndcg_at_10"].tolist()

    x   = np.arange(len(models))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w,   p10,  w, label="Precision@10", color=BLUE,   edgecolor="white")
    ax.bar(x,       r10,  w, label="Recall@10",    color=ORANGE, edgecolor="white")
    ax.bar(x + w,   nd10, w, label="NDCG@10",      color=GREEN,  edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Ranking Metrics @ K=10")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save("fig4_ranking_metrics.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Cold-start RMSE heatmap
# ══════════════════════════════════════════════════════════════════════════════
def fig_cold_start_rmse():
    if not (PROC / "cold_start_results.csv").exists():
        print("  [SKIP] cold_start_results.csv not found")
        return
    cs = pd.read_csv(PROC / "cold_start_results.csv")
    cs = cs.dropna(subset=["rmse"])

    seg_order   = ["cold", "near_cold", "normal"]
    seg_labels  = {"cold": "Cold\n(≤5 ratings)", "near_cold": "Near-cold\n(6–20)", "normal": "Normal\n(>20)"}
    model_order = cs["model"].unique().tolist()

    x   = np.arange(len(seg_order))
    w   = 0.8 / len(model_order)
    fig, ax = plt.subplots(figsize=(9, 4.5))

    for mi, model in enumerate(model_order):
        sub    = cs[cs["model"] == model].set_index("segment")
        rmses  = [sub.loc[s, "rmse"] if s in sub.index else np.nan for s in seg_order]
        offset = (mi - len(model_order) / 2 + 0.5) * w
        ax.bar(x + offset, rmses, w * 0.9, label=model, color=PALETTE[mi % len(PALETTE)], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([seg_labels[s] for s in seg_order])
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE by User Sparsity Segment")
    ax.legend(loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save("fig5_cold_start_rmse.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 — Cold-start NDCG@10
# ══════════════════════════════════════════════════════════════════════════════
def fig_cold_start_ndcg():
    if not (PROC / "cold_start_results.csv").exists():
        print("  [SKIP] cold_start_results.csv not found")
        return
    cs = pd.read_csv(PROC / "cold_start_results.csv")
    cs = cs.dropna(subset=["ndcg_at_10"])

    seg_order  = ["cold", "near_cold", "normal"]
    seg_labels = {"cold": "Cold\n(≤5)", "near_cold": "Near-cold\n(6–20)", "normal": "Normal\n(>20)"}
    model_order = cs["model"].unique().tolist()

    x  = np.arange(len(seg_order))
    w  = 0.8 / len(model_order)
    fig, ax = plt.subplots(figsize=(9, 4.5))

    for mi, model in enumerate(model_order):
        sub   = cs[cs["model"] == model].set_index("segment")
        ndcgs = [sub.loc[s, "ndcg_at_10"] if s in sub.index else np.nan for s in seg_order]
        offset = (mi - len(model_order) / 2 + 0.5) * w
        ax.bar(x + offset, ndcgs, w * 0.9, label=model, color=PALETTE[mi % len(PALETTE)], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([seg_labels[s] for s in seg_order])
    ax.set_ylabel("NDCG@10")
    ax.set_title("NDCG@10 by User Sparsity Segment")
    ax.legend(loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save("fig6_cold_start_ndcg.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 — Top-20 tags
# ══════════════════════════════════════════════════════════════════════════════
def fig_top_tags():
    tags = pd.read_csv(PROC / "tag_features.csv")
    tfidf_cols = [c for c in tags.columns if c.startswith("tfidf_")]
    tag_sums   = tags[tfidf_cols].sum().sort_values(ascending=False).head(20)
    labels     = [c.replace("tfidf_", "").replace("_", " ") for c in tag_sums.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels[::-1], tag_sums.values[::-1], color=TEAL, edgecolor="white", alpha=0.9)
    ax.set_xlabel("Aggregate TF-IDF weight (all artists)")
    ax.set_title("Top 20 Last.fm Tags by Corpus Weight")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save("fig7_top_tags.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 8 — Sample recommendations table
# ══════════════════════════════════════════════════════════════════════════════
def fig_sample_recs():
    recs = pd.read_csv(PROC / "recommendations.csv")
    users = recs["user_id"].unique()[:3]
    sub   = recs[recs["user_id"].isin(users)][["user_id", "rank", "artist", "predicted_rating"]]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    col_labels = ["User ID", "Rank", "Artist", "Predicted Rating"]
    table_data = [[str(r.user_id), str(r.rank), r.artist.title(), f"{r.predicted_rating:.3f}"]
                  for r in sub.itertuples()]
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    # Style header row
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#4C72B0")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    # Alternating row shading
    for i in range(1, len(table_data) + 1):
        colour = "#f0f4fa" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[(i, j)].set_facecolor(colour)
    ax.set_title("Sample Top-10 Recommendations (MF + tags + audio model)", pad=12, fontsize=12)
    fig.tight_layout()
    save("fig8_sample_recs.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 56)
    print("  AudioMuse-AI  |  Figure Generation")
    print("=" * 56)
    print(f"  Output → {FIG_DIR}\n")

    fig_rating_distribution()
    fig_ratings_per_user()
    fig_rmse_comparison()
    fig_ranking_metrics()
    fig_cold_start_rmse()
    fig_cold_start_ndcg()
    fig_top_tags()
    fig_sample_recs()

    print(f"\n  All figures saved to {FIG_DIR}")
    print("[DONE]")


if __name__ == "__main__":
    main()
