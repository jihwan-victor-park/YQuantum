#!/usr/bin/env python3
"""
Scenario-based benchmark: compare naive baselines vs YQuantum pipeline (cluster + Markowitz + analog bitstring).

Uses investment_dataset_scenarios.csv only (no external data). Defines synthetic "stress" as rows
where the equal-weight market proxy is in the bottom quantile (e.g. worst 10% market scenarios).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT_EARLY = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_EARLY / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = ROOT_EARLY
SCENARIOS_PATH = ROOT / "investment_dataset_scenarios.csv"
EIGHT_PATH = ROOT / "phase1_output" / "eight_qubit_assets.csv"
WEIGHTS_PATH = ROOT / "phase2_output" / "classical_optimal_weights.csv"
PHASE4_SUMMARY = ROOT / "phase4_output" / "phase4_summary.json"
OUT_DIR = ROOT / "scenario_benchmark_output"


def load_returns_matrix() -> tuple[pd.DataFrame, list[str], np.ndarray]:
    df = pd.read_csv(SCENARIOS_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    ids = list(df.columns)
    R = df.values.astype(np.float64)
    return df, ids, R


def sharpe_per_column(R: np.ndarray) -> np.ndarray:
    mu = R.mean(axis=0)
    sig = R.std(axis=0, ddof=1)
    sig = np.maximum(sig, 1e-12)
    return mu / sig


def portfolio_returns(R: np.ndarray, asset_ids: list[str], weights: dict[str, float]) -> np.ndarray:
    """R: (T, n_assets), weights only for ids that appear; others 0."""
    idx = {a: i for i, a in enumerate(asset_ids)}
    wvec = np.zeros(len(asset_ids), dtype=np.float64)
    for a, w in weights.items():
        wvec[idx[a]] = float(w)
    s = wvec.sum()
    if s > 1e-15:
        wvec /= s
    return R @ wvec


def cvar_alpha(returns: np.ndarray, alpha: float) -> float:
    """Mean of the worst alpha fraction of returns (loss tail)."""
    x = np.sort(returns)
    k = max(1, int(np.ceil(alpha * len(x))))
    return float(x[:k].mean())


def summarize_method(name: str, r: np.ndarray, stress_mask: np.ndarray) -> dict:
    st = r[stress_mask]
    return {
        "method": name,
        "n_scenarios": int(len(r)),
        "mean_all": float(np.mean(r)),
        "std_all": float(np.std(r, ddof=1)),
        "sharpe_all": float(np.mean(r) / max(np.std(r, ddof=1), 1e-12)),
        "min_all": float(np.min(r)),
        "cvar5pct_all": cvar_alpha(r, 0.05),
        "mean_stress_mkt": float(np.mean(st)) if len(st) else float("nan"),
        "std_stress_mkt": float(np.std(st, ddof=1)) if len(st) > 1 else float("nan"),
        "sharpe_stress_mkt": (
            float(np.mean(st) / max(np.std(st, ddof=1), 1e-12)) if len(st) > 1 else float("nan")
        ),
        "cvar5pct_stress_mkt": cvar_alpha(st, 0.05) if len(st) else float("nan"),
        "worst_stress_mkt": float(np.min(st)) if len(st) else float("nan"),
    }


def load_cluster8_ids() -> list[str]:
    df = pd.read_csv(EIGHT_PATH)
    return df.sort_values("cluster")["asset"].tolist()


def load_markowitz_weights(ids8: list[str]) -> dict[str, float]:
    df = pd.read_csv(WEIGHTS_PATH)
    m = dict(zip(df["asset"].astype(str), df["weight"].astype(float)))
    return {a: float(m[a]) for a in ids8}


def load_phase4_summary() -> dict:
    if not PHASE4_SUMMARY.is_file():
        raise FileNotFoundError(f"Missing {PHASE4_SUMMARY}; run phase4 first.")
    with open(PHASE4_SUMMARY, encoding="utf-8") as f:
        return json.load(f)


def bitstring_to_ew_weights(bitstring: str, order: list[str]) -> dict[str, float]:
    sel = [order[i] for i, c in enumerate(bitstring) if c == "1"]
    if not sel:
        return {order[0]: 1.0}
    w = 1.0 / len(sel)
    return {a: w for a in sel}


def markowitz_on_subset(
    selected_ids: list[str],
    all_ids: list[str],
    R: np.ndarray,
) -> dict[str, float]:
    """
    Run Markowitz max-Sharpe on a subset of assets selected by a bitstring.
    Uses annualised μ and Σ (×252) so the optimiser sees meaningful risk premiums.
    """
    from scipy.optimize import minimize as sp_minimize

    idx = [all_ids.index(a) for a in selected_ids]
    R_sub = R[:, idx]
    mu = R_sub.mean(axis=0) * 252
    cov = np.cov(R_sub, rowvar=False, ddof=1) * 252
    k = len(selected_ids)
    bounds = [(0.0, 1.0)] * k

    def neg_sharpe(w):
        ret = mu @ w
        var = w @ cov @ w
        return -(ret / max(np.sqrt(var), 1e-15))

    w0 = np.ones(k) / k
    res = sp_minimize(
        neg_sharpe, w0, method="SLSQP", bounds=bounds,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0,
                      "jac": lambda w: np.ones(k)},
        options={"maxiter": 500},
    )
    wvec = np.clip(res.x, 0, 1)
    s = wvec.sum()
    if s > 1e-15:
        wvec /= s
    return {a: float(wvec[i]) for i, a in enumerate(selected_ids)}


def ensemble_weights(all_counts: dict, order: list[str], total_shots: int) -> dict[str, float]:
    """Frequency-weighted portfolio across ALL sampled bitstrings."""
    wvec = np.zeros(len(order), dtype=np.float64)
    for bs, cnt in all_counts.items():
        prob = cnt / total_shots
        mask = np.array([1.0 if c == "1" else 0.0 for c in bs])
        n_sel = mask.sum()
        if n_sel > 0:
            wvec += prob * mask / n_sel
    s = wvec.sum()
    if s > 1e-15:
        wvec /= s
    return {a: float(wvec[i]) for i, a in enumerate(order)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stress_q = float(os.environ.get("STRESS_MARKET_QUANTILE", "0.10"))
    rng_seed = int(os.environ.get("RANDOM8_SEED", "42"))

    df, asset_ids, R = load_returns_matrix()
    n, d = R.shape
    assert d == len(asset_ids)

    mkt = R.mean(axis=1)
    thr = np.quantile(mkt, stress_q)
    stress_mask = mkt <= thr

    ids8 = load_cluster8_ids()
    w_mz = load_markowitz_weights(ids8)

    p4 = load_phase4_summary()
    order_phase4 = list(p4["assets_order"])

    def align_bs(bs: str) -> str:
        if order_phase4 != ids8:
            pos = {a: i for i, a in enumerate(order_phase4)}
            return "".join(bs[pos[a]] for a in ids8)
        return bs

    # Return-weighted bitstring
    rw_bs = align_bs(str(p4["return_weighted"]["top_portfolio"]["bitstring"]))
    w_rw_ew = bitstring_to_ew_weights(rw_bs, ids8)

    # Sharpe-weighted bitstring
    sw_bs = align_bs(str(p4["sharpe_weighted"]["top_portfolio"]["bitstring"]))
    w_sw_ew = bitstring_to_ew_weights(sw_bs, ids8)

    # Hybrid: quantum selects → classical Markowitz optimizes weights on that subset
    sw_selected = [ids8[i] for i, c in enumerate(sw_bs) if c == "1"]
    w_sw_mz = markowitz_on_subset(sw_selected, asset_ids, R)

    rw_selected = [ids8[i] for i, c in enumerate(rw_bs) if c == "1"]
    w_rw_mz = markowitz_on_subset(rw_selected, asset_ids, R)

    # Ensemble: frequency-weighted across all bitstrings
    sw_all_counts = p4["sharpe_weighted"]["all_counts"]
    sw_shots = p4["sharpe_weighted"]["shots"]
    w_sw_ensemble = ensemble_weights(sw_all_counts, order_phase4, sw_shots)

    rw_all_counts = p4["return_weighted"]["all_counts"]
    rw_shots = p4["return_weighted"]["shots"]
    w_rw_ensemble = ensemble_weights(rw_all_counts, order_phase4, rw_shots)

    # --- Build weight dicts (sparse over full 50) ---
    n50 = len(asset_ids)
    ew50 = {a: 1.0 / n50 for a in asset_ids}

    rng = np.random.default_rng(rng_seed)
    pick8 = rng.choice(asset_ids, size=8, replace=False)
    ew8rnd = {a: 1.0 / 8 for a in pick8}

    sharpe = sharpe_per_column(R)
    top8_idx = np.argsort(-sharpe)[:8]
    top8 = [asset_ids[i] for i in top8_idx]
    ew8top = {a: 1.0 / 8 for a in top8}

    ew8clust = {a: 1.0 / 8 for a in ids8}

    methods: list[tuple[str, dict[str, float]]] = [
        ("1_EW50_all_assets", ew50),
        ("2_EW8_random", ew8rnd),
        ("3_EW8_top_Sharpe_no_cluster", ew8top),
        ("4_EW8_cluster_winners_only", ew8clust),
        ("5_Markowitz_on_cluster8", w_mz),
        ("6_Analog_return_wtd_EW", w_rw_ew),
        ("7_Analog_sharpe_wtd_EW", w_sw_ew),
        ("8_Hybrid_sharpe_select_MaxSharpe", w_sw_mz),
        ("9_Hybrid_return_select_MaxSharpe", w_rw_mz),
        ("10_Ensemble_sharpe_wtd", w_sw_ensemble),
        ("11_Ensemble_return_wtd", w_rw_ensemble),
    ]

    rows_metrics = []
    series: dict[str, np.ndarray] = {}
    for name, wdict in methods:
        r = portfolio_returns(R, asset_ids, wdict)
        series[name] = r
        rows_metrics.append(summarize_method(name, r, stress_mask))

    metrics_df = pd.DataFrame(rows_metrics)
    metrics_df.to_csv(OUT_DIR / "method_metrics_full_and_stress.csv", index=False)

    wide = pd.DataFrame({"scenario_index": np.arange(n), "mkt_EW50_return": mkt, "stress_mkt_bottom10pct": stress_mask})
    for name, r in series.items():
        wide[name] = r
    wide.to_csv(OUT_DIR / "per_scenario_portfolio_returns.csv", index=False)

    # --- Deltas vs baseline EW50 ---
    base = series["1_EW50_all_assets"]
    delta_rows = []
    for name, r in series.items():
        if name == "1_EW50_all_assets":
            continue
        delta_rows.append(
            {
                "method_vs_EW50": name,
                "mean_all_delta": float(np.mean(r - base)),
                "mean_stress_delta": float(np.mean((r - base)[stress_mask])),
                "cvar5_all_delta": cvar_alpha(r, 0.05) - cvar_alpha(base, 0.05),
            }
        )
    pd.DataFrame(delta_rows).to_csv(OUT_DIR / "deltas_vs_EW50.csv", index=False)

    # --- Summary JSON ---
    summary = {
        "data": {
            "scenarios_csv": str(SCENARIOS_PATH.relative_to(ROOT)),
            "n_scenarios": n,
            "n_assets": d,
        },
        "stress_definition": {
            "market_proxy": "Equal-weight return across all 50 assets per scenario row.",
            "stress_rule": f"Stress = rows where market_proxy <= {stress_q:.0%} quantile (bad market scenarios).",
            "n_stress_rows": int(stress_mask.sum()),
            "stress_quantile": stress_q,
        },
        "methods": rows_metrics,
        "random_EW8_seed": rng_seed,
        "random_EW8_assets": list(pick8),
        "top8_Sharpe_assets": top8,
        "return_weighted_bitstring": rw_bs,
        "sharpe_weighted_bitstring": sw_bs,
        "sharpe_weighted_selected_assets": sw_selected,
        "hybrid_sharpe_markowitz_weights": w_sw_mz,
        "cluster8_assets": ids8,
    }
    with open(OUT_DIR / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # --- Plots ---
    plot_metrics_grouped(metrics_df, OUT_DIR / "metrics_mean_sharpe_cvar5.png")
    plot_stress_bar(metrics_df, OUT_DIR / "stress_scenario_mean_return_by_method.png")
    plot_delta_waterfall(delta_rows, OUT_DIR / "improvement_vs_EW50_stress_mean.png")

    write_report_md(OUT_DIR / "BENCHMARK_REPORT.md", metrics_df, delta_rows, stress_q, int(stress_mask.sum()), n)

    print("Wrote:", OUT_DIR.resolve())


def _short_label(m: str) -> str:
    return m.split("_", 1)[-1].replace("_", " ")[:30]


def plot_metrics_grouped(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    names = [_short_label(m) for m in df["method"]]
    x = np.arange(len(df))

    is_quantum = [i >= 5 for i in range(len(df))]
    colors_mean = ["coral" if q else "steelblue" for q in is_quantum]
    colors_sharpe = ["coral" if q else "seagreen" for q in is_quantum]
    colors_cvar = ["coral" if q else "indianred" for q in is_quantum]

    axes[0].bar(x, df["mean_all"], color=colors_mean, edgecolor="k")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=6, rotation=45, ha="right")
    axes[0].set_title("Mean return (all scenarios)")
    axes[0].set_ylabel("Return")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, df["sharpe_all"], color=colors_sharpe, edgecolor="k")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=6, rotation=45, ha="right")
    axes[1].set_title("Sharpe (mean/std, all scenarios)")
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(x, df["cvar5pct_all"], color=colors_cvar, edgecolor="k")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, fontsize=6, rotation=45, ha="right")
    axes[2].set_title("CVaR 5% (tail mean, all scenarios)")
    axes[2].set_ylabel("Avg of worst 5% returns")
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Portfolio methods — full sample (orange = quantum-assisted)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_stress_bar(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df))
    names = [_short_label(m) for m in df["method"]]
    is_quantum = [i >= 5 for i in range(len(df))]
    colors = ["coral" if q else "darkorange" for q in is_quantum]
    ax.bar(x, df["mean_stress_mkt"], color=colors, edgecolor="k", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Mean portfolio return")
    ax.set_title("Mean return in stress scenarios (worst market decile)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=df.loc[df["method"] == "1_EW50_all_assets", "mean_stress_mkt"].values[0], color="navy", ls="--", lw=1.2, label="EW50 stress mean")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_delta_waterfall(delta_rows: list[dict], path: Path) -> None:
    if not delta_rows:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labs = [_short_label(d["method_vs_EW50"]) for d in delta_rows]
    vals = [d["mean_stress_delta"] for d in delta_rows]
    colors = ["green" if v > 0 else "crimson" for v in vals]
    ax.barh(labs, vals, color=colors, edgecolor="k", alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Δ mean return vs EW50 (stress scenarios only)")
    ax.set_title("Who beats naive diversification in bad markets?")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_report_md(
    path: Path,
    metrics_df: pd.DataFrame,
    delta_rows: list[dict],
    stress_q: float,
    n_stress: int,
    n_all: int,
) -> None:
    lines = [
        "# 시나리오 기반 벤치마크: 일반 전략 vs YQuantum 파이프라인",
        "",
        "## 데이터",
        "",
        f"- **입력:** `investment_dataset_scenarios.csv` ({n_all}행 × 50자산 수익률).",
        "- **외부 데이터 없음** — 모든 비교는 이 시뮬레이션 표본 안에서만 정의합니다.",
        "",
        "## 스트레스(의사 침체) 정의",
        "",
        f"- **시장 프록시:** 각 행에서 50자산 **동일가중** 수익률.",
        f"- **스트레스 행:** 그 프록시가 **하위 {stress_q:.0%} 분위** 이하인 시나리오 (약 **{n_stress}행**).",
        "- 해석: “시장 전체가 크게 깨진 날/경로”에 가깝게 두고, 그 부분집합에서만 평균·CVaR을 다시 봅니다.",
        "",
        "## 비교한 방법 (10가지)",
        "",
        "### 베이스라인 (1–5)",
        "",
        "1. **EW50** — 50자산 동일가중 (가장 단순한 분산 투자).",
        "2. **EW8_random** — 무작위 8자산 동일가중 (시드 고정, `RANDOM8_SEED`).",
        "3. **EW8_top_Sharpe_no_cluster** — 시나리오 기준 Sharpe 상위 8자산 동일가중 (**클러스터 없음**).",
        "4. **EW8_cluster_winners** — Phase 1과 동일한 8종(클러스터별 우승) **동일가중** (최적화 없음).",
        "5. **Markowitz_cluster8** — 같은 8종에 대해 Phase 2 Markowitz 가중 (`classical_optimal_weights.csv`).",
        "",
        "### 양자/하이브리드 (6–10)",
        "",
        "6. **Analog_return_wtd_EW** — Phase 4 **수익 가중 detuning** 최빈 비트스트링 → 동일가중.",
        "7. **Analog_sharpe_wtd_EW** — Phase 4 **Sharpe 가중 detuning** (μ/σ) 최빈 비트스트링 → 동일가중.",
        "8. **Hybrid_sharpe_select_Markowitz** — Sharpe 가중 비트스트링 **선택** + 선택된 부분집합에 **Markowitz** 최적화.",
        "9. **Hybrid_return_select_Markowitz** — 수익 가중 비트스트링 선택 + Markowitz 최적화.",
        "10. **Analog_sharpe_ensemble** — Sharpe 가중 run의 **전체 비트스트링 분포**를 확률 가중 포트폴리오로 환산.",
        "",
        "## 산출 파일",
        "",
        "| 파일 | 내용 |",
        "| --- | --- |",
        "| `method_metrics_full_and_stress.csv` | 방법별 전체/스트레스 평균, 표준편차, Sharpe, CVaR5% 등 |",
        "| `per_scenario_portfolio_returns.csv` | 행마다 시장프록시, 스트레스 플래그, 방법별 포트 수익 |",
        "| `deltas_vs_EW50.csv` | EW50 대비 평균 수익 차이(전체·스트레스), CVaR 차이 |",
        "| `benchmark_summary.json` | 위 내용 + 사용 자산 목록·비트스트링 |",
        "| `*.png` | 막대/수평막대 요약 그래프 |",
        "",
        "## 숫자 요약 (표)",
        "",
        "```\n" + metrics_df.to_string(index=False) + "\n```",
        "",
        "## EW50 대비 스트레스 구간 평균 수익 차이",
        "",
        "```\n" + pd.DataFrame(delta_rows).to_string(index=False) + "\n```",
        "",
        "## 주의 (심사/발표용)",
        "",
        "- 시나리오가 **시간 순서**가 아니면 “낙폭(drawdown)”은 정의하지 않았고, **단면 분포** 기준 지표만 사용했습니다.",
        "- **양자(아날로그) 결과**는 에뮬레이터 샘플에서 나온 **비트스트링 선택**을 여기서는 해당 방식(동일가중/Markowitz/앙상블)으로 매핑합니다.",
        "- Bloqade 재실행 시 `phase4_summary.json`이 바뀌면 방법 6–10이 모두 갱신됩니다.",
        "",
        "---",
        "",
        "## English summary (for judges)",
        "",
        f"- **Stress set:** scenarios in the bottom **{stress_q:.0%}** of the equal-weight 50-asset market return (~**{n_stress}** of **{n_all}** rows).",
        "- **Classical baselines:** EW50, random EW8, top-Sharpe EW8, cluster winners EW8, Markowitz on cluster-8.",
        "- **Quantum-assisted (methods 6–10):**",
        "  - **Return-weighted** (raw μ) + EW on modal bitstring.",
        "  - **Sharpe-weighted** (μ/σ) + EW on modal bitstring — risk-adjusted Hamiltonian.",
        "  - **Hybrid:** quantum Sharpe/return selects assets, classical Markowitz optimizes weights on that subset.",
        "  - **Ensemble:** frequency-weighted average over the full Sharpe-run bitstring distribution.",
        "- **Key CSVs:** `method_metrics_full_and_stress.csv`, `deltas_vs_EW50.csv`, `per_scenario_portfolio_returns.csv`.",
        "",
        "**Run:** `python scenario_portfolio_benchmark.py`  (optional: `STRESS_MARKET_QUANTILE=0.1`, `RANDOM8_SEED=42`).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
