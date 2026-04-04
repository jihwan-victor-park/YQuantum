#!/usr/bin/env python3
"""
Phase 2: Classical Markowitz baseline on the 8 Phase-1 assets.

Mean–variance scalarization: maximize μᵀw − (λ/2) wᵀΣw subject to Σw = 1 and
dataset w_min / w_max bounds. Solved with scipy.optimize.minimize (SLSQP).
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
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
ASSETS_PATH = ROOT / "investment_dataset_assets.csv"
COV_PATH = ROOT / "investment_dataset_covariance.csv"
EIGHT_PATH = ROOT / "phase1_output" / "eight_qubit_assets.csv"
OUT_DIR = ROOT / "phase2_output"


def load_eight_assets() -> list[str]:
    df = pd.read_csv(EIGHT_PATH)
    return df.sort_values("cluster")["asset"].tolist()


def load_mu_bounds(ids: list[str]) -> tuple[np.ndarray, list[tuple[float, float]]]:
    meta = pd.read_csv(ASSETS_PATH).set_index("asset_id")
    mu = np.array([float(meta.loc[i, "exp_return"]) for i in ids], dtype=np.float64)
    bounds = [(float(meta.loc[i, "w_min"]), float(meta.loc[i, "w_max"])) for i in ids]
    return mu, bounds


def textbook_long_only_bounds(n: int) -> list[tuple[float, float]]:
    return [(0.0, 1.0)] * n


def load_cov_submatrix(ids: list[str]) -> np.ndarray:
    full = pd.read_csv(COV_PATH, index_col=0)
    full.columns = [str(c).strip() for c in full.columns]
    full.index = [str(i).strip() for i in full.index]
    sub = full.loc[ids, ids].values.astype(np.float64)
    sub = 0.5 * (sub + sub.T)
    return sub


def _feasible_start(bounds: list[tuple[float, float]], budget: float) -> np.ndarray:
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    n = len(lo)
    mid = 0.5 * (lo + hi)
    s = float(mid.sum())
    if s > 1e-12:
        w0 = mid / s * budget
    else:
        w0 = np.ones(n, dtype=np.float64) * (budget / n)
    w0 = np.clip(w0, lo, hi)
    for _ in range(200):
        if abs(w0.sum() - budget) < 1e-12:
            return w0
        w0 = w0 + (budget - w0.sum()) / n
        w0 = np.clip(w0, lo, hi)
    raise RuntimeError("Could not build feasible starting weights for given bounds and budget.")


def markowitz_max_utility(
    mu: np.ndarray,
    sigma: np.ndarray,
    bounds: list[tuple[float, float]],
    risk_aversion: float,
    *,
    budget: float | None = None,
) -> tuple[np.ndarray, object]:
    n = len(mu)

    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    sum_hi = float(hi.sum())
    sum_lo = float(lo.sum())
    if budget is None:
        # Dataset caps may make sum(w)=1 infeasible on a subset; invest up to full cap.
        budget = min(1.0, sum_hi)
    if budget < sum_lo - 1e-12 or budget > sum_hi + 1e-12:
        raise ValueError(
            f"Infeasible budget {budget:g}: need sum_lo={sum_lo:g} <= budget <= sum_hi={sum_hi:g}"
        )

    def neg_utility(w: np.ndarray) -> float:
        w = np.asarray(w, dtype=np.float64)
        port_var = float(w @ sigma @ w)
        port_ret = float(mu @ w)
        return float(-(port_ret - 0.5 * risk_aversion * port_var))

    def neg_utility_grad(w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=np.float64)
        return -mu + risk_aversion * (sigma @ w)

    def sum_constraint(w: np.ndarray) -> float:
        return float(np.sum(w) - budget)

    def sum_constraint_jac(_w: np.ndarray) -> np.ndarray:
        return np.ones((1, n), dtype=np.float64)

    w0 = _feasible_start(bounds, budget)
    w0 = lo + (w0 - lo) * 0.98  # slightly interior helps SLSQP
    for _ in range(200):
        if abs(w0.sum() - budget) < 1e-12:
            break
        w0 = w0 + (budget - w0.sum()) / n
        w0 = np.clip(w0, lo, hi)
    else:
        raise RuntimeError("Could not construct feasible start; check bounds and budget.")

    cons = {"type": "eq", "fun": sum_constraint, "jac": sum_constraint_jac}
    res = minimize(
        neg_utility,
        w0,
        jac=neg_utility_grad,
        method="SLSQP",
        bounds=list(zip(lo, hi)),
        constraints=cons,
        options={"ftol": 1e-14, "maxiter": 5000, "disp": False},
    )
    return np.asarray(res.x, dtype=np.float64), res


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    risk_aversion = float(os.environ.get("MARKOWITZ_LAMBDA", "12.0"))
    budget_env = os.environ.get("PORTFOLIO_BUDGET")
    budget = float(budget_env) if budget_env is not None else None
    bounds_mode = os.environ.get("CLASSICAL_BOUNDS", "textbook").strip().lower()

    ids = load_eight_assets()
    mu, dataset_bounds = load_mu_bounds(ids)
    sigma = load_cov_submatrix(ids)

    lo_ds = np.array([b[0] for b in dataset_bounds], dtype=np.float64)
    hi_ds = np.array([b[1] for b in dataset_bounds], dtype=np.float64)
    cap_sum = float(hi_ds.sum())

    if bounds_mode in ("dataset", "hackathon", "caps"):
        bounds = dataset_bounds
        effective_budget = min(1.0, cap_sum) if budget is None else budget
        bounds_label = "dataset_w_min_w_max"
    elif bounds_mode in ("textbook", "standard", "long_only"):
        bounds = textbook_long_only_bounds(len(ids))
        effective_budget = 1.0 if budget is None else budget
        bounds_label = "long_only_0_1"
    else:
        raise ValueError(
            "CLASSICAL_BOUNDS must be 'textbook' or 'dataset' "
            f"(got {bounds_mode!r})."
        )

    w_opt, res = markowitz_max_utility(
        mu, sigma, bounds, risk_aversion=risk_aversion, budget=effective_budget
    )

    port_ret = float(mu @ w_opt)
    port_var = float(w_opt @ sigma @ w_opt)
    port_vol = float(np.sqrt(max(port_var, 0.0)))

    meta = pd.read_csv(ASSETS_PATH).set_index("asset_id")
    sectors = [str(meta.loc[i, "sector"]) for i in ids]

    summary = {
        "assets": ids,
        "sectors": sectors,
        "risk_aversion_lambda": risk_aversion,
        "bounds_mode": bounds_label,
        "budget_sum_w": effective_budget,
        "dataset_w_max_sum": cap_sum,
        "notes": (
            "Default CLASSICAL_BOUNDS=textbook uses 0<=w<=1 and sum(w)=1 (standard long-only Markowitz). "
            "CLASSICAL_BOUNDS=dataset uses hackathon w_min/w_max; if sum(w_max)<1 on this subset, "
            "budget defaults to that sum and the feasible set can collapse to a single corner (all at caps)."
        ),
        "objective": "maximize mu.T @ w - (lambda/2) * w.T @ Sigma @ w",
        "constraints": f"sum(w)={effective_budget:g}, bounds per CLASSICAL_BOUNDS",
        "optimizer": "scipy.optimize.minimize SLSQP",
        "success": bool(res.success),
        "message": str(res.message),
        "weights": {i: float(w) for i, w in zip(ids, w_opt)},
        "portfolio_exp_return": port_ret,
        "portfolio_variance": port_var,
        "portfolio_volatility": port_vol,
        "portfolio_sharpe_like": port_ret / port_vol if port_vol > 1e-12 else None,
    }

    with open(OUT_DIR / "phase2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(
        {
            "asset": ids,
            "sector": sectors,
            "weight": w_opt,
        }
    ).to_csv(OUT_DIR / "classical_optimal_weights.csv", index=False)

    labels = [f"{a}\n({s[:10]})" if len(s) > 10 else f"{a}\n({s})" for a, s in zip(ids, sectors)]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(ids)))
    ax.bar(labels, w_opt * 100.0, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_ylabel("Weight (%)")
    ax.set_title(
        f"Classical optimal portfolio (Markowitz, λ={risk_aversion:g}, Σw={effective_budget:.2%})\n"
        f"E[R]={port_ret:.2%}, σ={port_vol:.2%}"
    )
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "classical_optimal_portfolio.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Extra charts ---
    thr = 0.005
    pie_mask = w_opt >= thr
    if pie_mask.sum() > 0:
        fig_p, ax_p = plt.subplots(figsize=(6.5, 6.5))
        w_pie = w_opt[pie_mask]
        lbl = [f"{ids[i]}\n{w_opt[i]*100:.1f}%" for i in range(len(ids)) if pie_mask[i]]
        ax_p.pie(
            w_pie,
            labels=lbl,
            autopct=lambda pct: f"{pct:.1f}%",
            startangle=90,
            counterclock=False,
        )
        ax_p.set_title("Classical optimal weights (slices ≥ 0.5%)")
        fig_p.tight_layout()
        fig_p.savefig(OUT_DIR / "classical_weights_pie.png", dpi=180, bbox_inches="tight")
        plt.close(fig_p)

    indiv_vol = np.sqrt(np.maximum(np.diag(sigma), 0.0))
    indiv_mu = mu
    fig_s, ax_s = plt.subplots(figsize=(7, 5))
    areas = np.maximum(w_opt * 2500, 15.0)
    ax_s.scatter(
        indiv_vol,
        indiv_mu,
        s=areas,
        c=range(len(ids)),
        cmap="tab10",
        alpha=0.75,
        edgecolors="black",
        linewidths=0.5,
    )
    ax_s.scatter(
        [port_vol],
        [port_ret],
        s=220,
        marker="*",
        c="gold",
        edgecolors="black",
        linewidths=1.2,
        zorder=10,
        label="Portfolio",
    )
    for i, aid in enumerate(ids):
        if w_opt[i] > 0.02:
            ax_s.annotate(aid, (indiv_vol[i], indiv_mu[i]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax_s.set_xlabel("Individual asset σ = √Σᵢᵢ")
    ax_s.set_ylabel("Expected return μᵢ")
    ax_s.set_title("Eight assets: risk–return (bubble size ∝ classical weight)")
    ax_s.legend(loc="lower right")
    ax_s.grid(True, alpha=0.3)
    fig_s.tight_layout()
    fig_s.savefig(OUT_DIR / "assets_bubble_risk_return_with_portfolio.png", dpi=180, bbox_inches="tight")
    plt.close(fig_s)

    print("Phase 2 complete:", OUT_DIR.resolve())
    print("Optimizer success:", res.success, "-", res.message)
    for a, ww in zip(ids, w_opt):
        print(f"  {a}: {ww:.4f} ({ww*100:.2f}%)")


if __name__ == "__main__":
    main()
