#!/usr/bin/env python3
"""
Phase 1: Smart Data Prep — correlation from scenarios, 8 clusters, Sharpe winners.

Uses investment_dataset_scenarios.csv for return samples, builds 50×50 correlation,
clusters assets into exactly 8 groups via hierarchical clustering on correlation-based
distance, then picks the highest Sharpe (mean return / vol) asset per cluster.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent
SCENARIOS_PATH = ROOT / "investment_dataset_scenarios.csv"
ASSETS_PATH = ROOT / "investment_dataset_assets.csv"
OUT_DIR = ROOT / "phase1_output"


def load_scenario_returns() -> pd.DataFrame:
    df = pd.read_csv(SCENARIOS_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_asset_meta() -> pd.DataFrame:
    return pd.read_csv(ASSETS_PATH)


def correlation_distance_matrix(corr: pd.DataFrame) -> np.ndarray:
    """Metric distance from correlation: d_ij = sqrt(2 * (1 - rho_ij))."""
    r = corr.values.astype(np.float64)
    np.clip(r, -1.0, 1.0, out=r)
    d = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - r)))
    np.fill_diagonal(d, 0.0)
    return d


def cluster_hierarchical_average(corr: pd.DataFrame, n_clusters: int) -> np.ndarray:
    d = correlation_distance_matrix(corr)
    condensed = squareform(d, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return labels.astype(np.int32), Z


def cluster_kmeans_on_correlation_features(corr: pd.DataFrame, n_clusters: int, seed: int = 42) -> np.ndarray:
    """Each asset = row of correlation matrix (how it co-moves with all others)."""
    X = corr.values.astype(np.float64)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return km.fit_predict(X).astype(np.int32) + 1  # 1..K to match fcluster style


def sharpe_per_asset(returns: pd.DataFrame, vol_floor: float = 1e-8) -> pd.Series:
    mu = returns.mean()
    sigma = returns.std(ddof=1).replace(0, np.nan)
    sigma = sigma.fillna(vol_floor).clip(lower=vol_floor)
    return mu / sigma


def pick_winners(
    labels: np.ndarray,
    asset_ids: list[str],
    sharpe: pd.Series,
) -> list[dict]:
    winners = []
    for k in sorted(np.unique(labels)):
        mask = labels == k
        members = [asset_ids[i] for i in range(len(asset_ids)) if mask[i]]
        sub = sharpe.loc[members]
        best_id = sub.idxmax()
        winners.append(
            {
                "cluster": int(k),
                "selected_asset": best_id,
                "sharpe": float(sharpe[best_id]),
                "cluster_size": int(mask.sum()),
                "members": members,
            }
        )
    return winners


def plot_correlation_heatmap(
    corr: pd.DataFrame,
    labels: np.ndarray,
    asset_ids: list[str],
    path: Path,
) -> None:
    order = np.argsort(labels)
    ids_ordered = [asset_ids[i] for i in order]
    c_ordered = corr.loc[ids_ordered, ids_ordered]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        c_ordered,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0,
        cbar_kws={"label": "Correlation"},
    )
    ax.set_title("Asset correlation (rows/columns ordered by cluster)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dendrogram(Z, asset_ids: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 8))
    dendrogram(
        Z,
        labels=asset_ids,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=7,
        color_threshold=0,
    )
    ax.set_title("Hierarchical clustering (average linkage on correlation distance)")
    ax.set_ylabel("Distance sqrt(2(1−ρ))")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_sizes_bar(labels: np.ndarray, path: Path) -> None:
    uniq, cnts = np.unique(labels, return_counts=True)
    order = np.argsort(uniq)
    uniq, cnts = uniq[order], cnts[order]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"Cluster {int(k)}" for k in uniq], cnts, color="teal", edgecolor="black")
    ax.set_ylabel("Number of assets")
    ax.set_title("Hierarchical clusters — sizes (8 clusters)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_eight_winners_sharpe(winners: list[dict], path: Path) -> None:
    wsort = sorted(winners, key=lambda x: x["cluster"])
    assets = [w["selected_asset"] for w in wsort]
    sharpes = [w["sharpe"] for w in wsort]
    sectors = [w.get("sector", "")[:12] for w in wsort]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(assets))
    ax.barh(y, sharpes, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a} ({s})" for a, s in zip(assets, sectors)], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Scenario Sharpe (mean/std)")
    ax.set_title("One winner per cluster — Sharpe ratios")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mean_vs_vol_all_assets(
    mean_ret: pd.Series,
    vol_series: pd.Series,
    winner_ids: set[str],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    m = mean_ret.reindex(vol_series.index)
    v = vol_series
    ax.scatter(v, m, s=22, alpha=0.45, c="gray", edgecolors="none", label="All 50 assets")
    win_mask = [i in winner_ids for i in v.index]
    ax.scatter(
        v[win_mask],
        m[win_mask],
        s=85,
        c="crimson",
        edgecolors="black",
        linewidths=0.6,
        label="8 cluster winners",
        zorder=5,
    )
    ax.set_xlabel("Volatility σ (scenario std)")
    ax.set_ylabel("Mean scenario return")
    ax.set_title("Risk–return cloud (scenario data) + Phase-1 picks")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # Stable defaults in CI/sandboxed environments (matplotlib cache, joblib/loky).
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    returns = load_scenario_returns()
    asset_ids = list(returns.columns)
    assert len(asset_ids) == 50, f"Expected 50 assets, got {len(asset_ids)}"

    corr = returns.corr()
    vol_series = returns.std(ddof=1)
    mean_ret = returns.mean()
    sharpe = sharpe_per_asset(returns)

    meta = load_asset_meta().set_index("asset_id")

    labels_hier, Z = cluster_hierarchical_average(corr, n_clusters=8)
    labels_km = cluster_kmeans_on_correlation_features(corr, n_clusters=8)

    # Primary story: hierarchical on correlation distance (matches dendrogram).
    winners = pick_winners(labels_hier, asset_ids, sharpe)
    winners_km = pick_winners(labels_km, asset_ids, sharpe)

    summary = {
        "n_assets": len(asset_ids),
        "n_scenarios": int(len(returns)),
        "clustering_method_primary": "hierarchical_average_correlation_distance",
        "clustering_method_alternate": "kmeans_on_correlation_rows",
        "sharpe_definition": "mean(scenario_return) / std(scenario_return), std ddof=1, vol floor 1e-8",
        "selected_assets_hierarchical": [w["selected_asset"] for w in winners],
        "winners_hierarchical": winners,
        "selected_assets_kmeans": [w["selected_asset"] for w in winners_km],
        "winners_kmeans": winners_km,
    }

    for w in winners:
        aid = w["selected_asset"]
        row = meta.loc[aid]
        w["sector"] = str(row["sector"])
        r = row["rating"]
        w["rating"] = "" if pd.isna(r) else str(r)
        w["exp_return_dataset"] = float(row["exp_return"])
        w["volatility_dataset"] = float(row["volatility"])

    with open(OUT_DIR / "phase1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Deliverables for slides
    plot_correlation_heatmap(
        corr,
        labels_hier,
        asset_ids,
        path=OUT_DIR / "correlation_heatmap_by_cluster.png",
    )
    plot_dendrogram(Z, asset_ids, path=OUT_DIR / "dendrogram.png")

    plot_cluster_sizes_bar(labels_hier, OUT_DIR / "cluster_sizes_bar.png")
    plot_eight_winners_sharpe(winners, OUT_DIR / "eight_winners_sharpe_barh.png")
    plot_mean_vs_vol_all_assets(
        mean_ret,
        vol_series,
        winner_ids={w["selected_asset"] for w in winners},
        path=OUT_DIR / "scenario_mean_vs_vol_with_winners.png",
    )

    # Compact table for judges
    rows = []
    for w in winners:
        rows.append(
            {
                "cluster": w["cluster"],
                "asset": w["selected_asset"],
                "sharpe_scenarios": w["sharpe"],
                "sector": w["sector"],
                "cluster_size": w["cluster_size"],
            }
        )
    pd.DataFrame(rows).sort_values("cluster").to_csv(OUT_DIR / "eight_qubit_assets.csv", index=False)

    print("Phase 1 complete. Outputs:", OUT_DIR.resolve())
    print("\n8 assets (hierarchical / dendrogram-aligned):")
    for w in sorted(winners, key=lambda x: x["cluster"]):
        print(
            f"  Cluster {w['cluster']}: {w['selected_asset']} "
            f"(Sharpe={w['sharpe']:.4f}, sector={w['sector']}, n={w['cluster_size']})"
        )


if __name__ == "__main__":
    main()
