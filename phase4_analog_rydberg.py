#!/usr/bin/env python3
"""
Phase 4: Bloqade analog (Rydberg) portfolio geometry + adiabatic sweep + local emulation.

Requires Python 3.10–3.12 and: pip install bloqade-analog numpy pandas scipy
(Python 3.14+ cannot install bloqade-analog on PyPI as of 2026.)

Physics / constraint narrative (Step 4.2)
-----------------------------------------
In Rydberg neutral-atom systems, atoms within the *blockade radius* (here taken as
~7.5 µm for this hackathon story) experience a strong interaction that shifts the
doubly-excited state off resonance. Operationally, this **Rydberg blockade** makes it
very costly for neighboring atoms to *both* be in the Rydberg (|1⟩) manifold during
the pulse — a **physical** mechanism that discourages simultaneously selecting
highly correlated “risk-cluster” assets without manually tuning a QUBO penalty λ to
encode the same constraint.

Step 4.1 places pairs with moderately high return correlation closer than 7.5 µm and
keeps nearly uncorrelated pairs beyond 15 µm (targets are met in a least-squares sense).

Step 4.3 applies a simple adiabatic-style waveform: Ω ramps up, holds, ramps down;
Δ is swept from negative to positive (uniform across atoms).

Deliverable: top-5 bitstrings from 1000-shot local emulation (bloqade.python()).

Geometry scale sweep (risk–return curve)
----------------------------------------
After the base layout, optionally scale all (x, y) coordinates by factors in
``scale_factors`` (default ``[0.8, 1.0, 1.2, 1.5]``), re-run Bloqade for each scale,
take the **most frequent** bitstring, and map it to an **equal-weight classical**
sub-portfolio on the 8 assets (character ``'1'`` = selected). Expected return and
volatility use ``exp_return`` and the 8×8 covariance block from the hackathon CSVs.
Results: ``scale_sweep_risk_return.png`` and ``scale_sweep_results.csv``.

Disable with ``SCALE_SWEEP=0``. Tune shots via ``SCALE_SWEEP_SHOTS``.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
SCENARIOS_PATH = ROOT / "investment_dataset_scenarios.csv"
ASSETS_PATH = ROOT / "investment_dataset_assets.csv"
COV_PATH = ROOT / "investment_dataset_covariance.csv"
EIGHT_PATH = ROOT / "phase1_output" / "eight_qubit_assets.csv"
OUT_DIR = ROOT / "phase4_output"

# Geometry rules (µm)
BLOCKADE_NEAR = 7.5
UNCORR_FAR = 15.0
TARGET_HIGH_CORR = 6.0  # < BLOCKADE_NEAR
TARGET_LOW_CORR = 16.0  # > UNCORR_FAR

# |ρ| thresholds for “moderately high” vs “nearly uncorrelated”
RHO_HIGH = 0.30
RHO_LOW = 0.12


def load_eight_assets() -> list[str]:
    df = pd.read_csv(EIGHT_PATH)
    return df.sort_values("cluster")["asset"].tolist()


def correlation_8() -> tuple[list[str], np.ndarray]:
    ids = load_eight_assets()
    r = pd.read_csv(SCENARIOS_PATH)
    r.columns = [c.strip() for c in r.columns]
    sub = r[ids]
    corr = sub.corr().values.astype(np.float64)
    return ids, corr


def pair_target_distance(rho_ij: float) -> float:
    a = abs(float(rho_ij))
    if a >= RHO_HIGH:
        return TARGET_HIGH_CORR
    if a <= RHO_LOW:
        return TARGET_LOW_CORR
    t = (a - RHO_LOW) / (RHO_HIGH - RHO_LOW)
    return TARGET_LOW_CORR + t * (TARGET_HIGH_CORR - TARGET_LOW_CORR)


def optimize_positions_2d(corr: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimize sum_{i<j} (||r_i-r_j|| - d_ij)^2 with d_ij from |ρ_ij|.
    Returns positions shape (n,2) in µm.
    """
    n = corr.shape[0]
    d_target = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d_target[i, j] = d_target[j, i] = pair_target_distance(corr[i, j])

    def stress(flat: np.ndarray) -> float:
        xy = flat.reshape(n, 2)
        s = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(xy[i] - xy[j])
                s += (dist - d_target[i, j]) ** 2
        return s

    rng = np.random.default_rng(seed)
    best = None
    best_x = None
    for t in range(12):
        x0 = rng.normal(scale=8.0, size=2 * n)
        res = minimize(stress, x0, method="L-BFGS-B", options={"maxiter": 800})
        if best is None or res.fun < best:
            best = res.fun
            best_x = res.x
    assert best_x is not None
    xy = best_x.reshape(n, 2)
    xy -= xy.mean(axis=0, keepdims=True)
    return xy, d_target


def layout_stress(xy: np.ndarray, d_target: np.ndarray) -> float:
    n = xy.shape[0]
    s = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(xy[i] - xy[j])
            s += (dist - d_target[i, j]) ** 2
    return float(s)


def pairwise_distances(xy: np.ndarray) -> np.ndarray:
    n = xy.shape[0]
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = float(np.linalg.norm(xy[i] - xy[j]))
    return d


def load_mu_sigma_8(ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    meta = pd.read_csv(ASSETS_PATH).set_index("asset_id")
    mu = np.array([float(meta.loc[i, "exp_return"]) for i in ids], dtype=np.float64)
    full = pd.read_csv(COV_PATH, index_col=0)
    full.columns = [str(c).strip() for c in full.columns]
    full.index = [str(i).strip() for i in full.index]
    sub = full.loc[ids, ids].values.astype(np.float64)
    sub = 0.5 * (sub + sub.T)
    return mu, sub


def portfolio_mu_sigma_equal_weight(
    bitstring: str,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    selected_char: str = "1",
) -> tuple[float, float]:
    """
    Equal weight on assets where bit == selected_char; return (E[R], σ_portfolio).
    """
    if len(bitstring) != mu.shape[0]:
        raise ValueError(f"bitstring length {len(bitstring)} != n_assets {mu.shape[0]}")
    mask = np.array([1.0 if c == selected_char else 0.0 for c in bitstring], dtype=np.float64)
    k = float(mask.sum())
    if k < 1.0:
        return float("nan"), float("nan")
    w = mask / k
    pret = float(w @ mu)
    pvar = float(w @ sigma @ w)
    pvol = float(np.sqrt(max(pvar, 0.0)))
    return pret, pvol


def plot_pulse_waveforms(pulse_kw: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    dr, ds = pulse_kw["dur_ramp"], pulse_kw["dur_sweep"]
    t = np.array([0.0, dr, dr + ds, dr + ds + dr])
    om = pulse_kw["omega_peak"]
    de0, de1 = pulse_kw["delta_neg"], pulse_kw["delta_pos"]
    omega_knots = [0.0, om, om, 0.0]
    delta_knots = [de0, de0, de1, de1]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)
    ax1.plot(t, omega_knots, "o-", color="C0", lw=2, markersize=6)
    ax1.set_ylabel("Ω (rad/µs)")
    ax1.set_title("Rabi amplitude (piecewise linear)")
    ax1.grid(True, alpha=0.3)
    ax2.plot(t, delta_knots, "o-", color="C1", lw=2, markersize=6)
    ax2.set_ylabel("Δ (rad/µs)")
    ax2.set_xlabel("Time (µs)")
    ax2.set_title("Detuning (negative → positive in middle segment)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_pair_geometry_scatter(csv_path: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not csv_path.is_file():
        return
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    abs_rho = df["rho"].abs()
    near = df["within_blockade_7p5"]
    ax.scatter(abs_rho[~near], df.loc[~near, "actual_dist_um"], s=35, alpha=0.65, c="steelblue", label="Other pairs")
    ax.scatter(abs_rho[near], df.loc[near, "actual_dist_um"], s=55, c="orangered", edgecolors="k", label="< 7.5 µm (blockade scale)")
    ax.axhline(7.5, color="red", ls="--", lw=1, alpha=0.7)
    ax.axhline(15.0, color="green", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("|ρ| (scenario correlation)")
    ax.set_ylabel("Actual pairwise distance (µm)")
    ax.set_title("Layout fit: distance vs |correlation|")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_scale_sweep_top_frequency(sweep_rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    if not sweep_rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [r["scale_factor"] for r in sweep_rows]
    ys = [r["top_frequency"] for r in sweep_rows]
    ax.bar([str(s) + "×" for s in xs], ys, color="mediumpurple", edgecolor="black")
    ax.set_ylabel("Frequency of most likely bitstring")
    ax.set_xlabel("Geometry scale factor")
    ax.set_title("Scale sweep — dominance of modal outcome")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def build_analog_program(
    positions: list[tuple[float, float]],
    *,
    start_mod,
    dur_ramp: float,
    dur_sweep: float,
    omega_peak: float,
    delta_neg: float,
    delta_pos: float,
):
    durations = [dur_ramp, dur_sweep, dur_ramp]
    geometry = start_mod.add_position(positions)
    return (
        geometry.rydberg.rabi.amplitude.uniform.piecewise_linear(
            durations=durations,
            values=[0.0, omega_peak, omega_peak, 0.0],
        ).detuning.uniform.piecewise_linear(
            durations=durations,
            values=[delta_neg, delta_neg, delta_pos, delta_pos],
        )
    )


def run_scale_sweep(
    xy_base: np.ndarray,
    ids: list[str],
    *,
    start_mod,
    mu: np.ndarray,
    sigma: np.ndarray,
    scale_factors: list[float],
    sweep_shots: int,
    pulse_kw: dict,
    selected_char: str,
) -> list[dict]:
    rows: list[dict] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sf in scale_factors:
            xy_s = xy_base * float(sf)
            pos = [(float(x), float(y)) for x, y in xy_s]
            program = build_analog_program(pos, start_mod=start_mod, **pulse_kw)
            results = program.bloqade.python().run(sweep_shots, interaction_picture=True)
            counts = results.report().counts()[0]
            top_bs, top_c = max(counts.items(), key=lambda kv: kv[1])
            pret, pvol = portfolio_mu_sigma_equal_weight(top_bs, mu, sigma, selected_char=selected_char)
            rows.append(
                {
                    "scale_factor": float(sf),
                    "top_bitstring": top_bs,
                    "top_count": int(top_c),
                    "top_frequency": float(top_c) / sweep_shots,
                    "portfolio_exp_return": pret,
                    "portfolio_volatility": pvol,
                    "n_selected_assets": int(sum(1 for c in top_bs if c == selected_char)),
                }
            )
    return rows


def main() -> None:
    if sys.version_info >= (3, 13):
        print(
            "Warning: bloqade-analog may not install on Python 3.13+. "
            "Use Python 3.12, e.g. ./.venv312/bin/python phase4_analog_rydberg.py",
            file=sys.stderr,
        )

    try:
        from bloqade.analog import start
    except ImportError as e:
        raise SystemExit(
            "Missing bloqade.analog. Install with Python 3.12:\n"
            "  /opt/homebrew/bin/python3.12 -m venv .venv312\n"
            "  .venv312/bin/pip install bloqade-analog numpy pandas scipy\n"
            f"Original error: {e}"
        ) from e

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

    ids, corr = correlation_8()
    xy, d_target = optimize_positions_2d(corr, seed=42)
    dists = pairwise_distances(xy)
    stress = layout_stress(xy, d_target)

    positions = [(float(x), float(y)) for x, y in xy]
    geo_records = []
    for i, aid in enumerate(ids):
        geo_records.append(
            {
                "asset": aid,
                "x_um": positions[i][0],
                "y_um": positions[i][1],
            }
        )
    pd.DataFrame(geo_records).to_csv(OUT_DIR / "atom_positions_um.csv", index=False)

    # Pair summary for judges
    pairs = []
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(
                {
                    "i": ids[i],
                    "j": ids[j],
                    "rho": float(corr[i, j]),
                    "target_dist_um": float(d_target[i, j]),
                    "actual_dist_um": float(dists[i, j]),
                    "within_blockade_7p5": bool(dists[i, j] < BLOCKADE_NEAR),
                    "beyond_15um_lowcorr_rule": bool(dists[i, j] > UNCORR_FAR),
                }
            )
    pd.DataFrame(pairs).to_csv(OUT_DIR / "pair_geometry_vs_correlation.csv", index=False)

    # --- Adiabatic-style pulse (Step 4.3) ---
    # Rydberg blockade (Step 4.2): atoms closer than ~7.5 µm have strong van der Waals
    # interaction; both cannot simultaneously sit in |1⟩ (Rydberg) at resonance — a hardware
    # constraint on correlated “clusters” without a hand-tuned λ penalty in the QUBO.
    dur_ramp = float(os.environ.get("PULSE_RAMP_US", "0.18"))
    dur_sweep = float(os.environ.get("PULSE_SWEEP_US", "2.2"))
    omega_peak = float(os.environ.get("OMEGA_PEAK_RAD_US", "2.8"))
    delta_neg = float(os.environ.get("DELTA_NEG_MRAD", "-14.0"))  # rad/us scale
    delta_pos = float(os.environ.get("DELTA_POS_MRAD", "14.0"))

    pulse_kw = dict(
        dur_ramp=dur_ramp,
        dur_sweep=dur_sweep,
        omega_peak=omega_peak,
        delta_neg=delta_neg,
        delta_pos=delta_pos,
    )
    program = build_analog_program(positions, start_mod=start, **pulse_kw)

    shots = int(os.environ.get("SHOTS", "1000"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = program.bloqade.python().run(shots, interaction_picture=True)

    report = results.report()
    counts_list = report.counts()
    counts = counts_list[0] if isinstance(counts_list, list) else counts_list
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top5 = ranked[:5]

    lines = [
        f"Phase 4 — {shots}-shot Bloqade analog emulation (local)",
        "Bitstring order: qubit 0 = first asset in eight_qubit_assets.csv (cluster order), MSB-left convention as in report.counts().",
        "",
        "Top 5 bitstrings:",
    ]
    for k, (bs, c) in enumerate(top5, 1):
        lines.append(f"  {k}. {bs}  count={c}  freq={c/shots:.4f}")
    text_out = "\n".join(lines) + "\n"
    print(text_out)
    (OUT_DIR / "top5_bitstrings_printout.txt").write_text(text_out, encoding="utf-8")

    summary = {
        "assets_order": ids,
        "atom_positions_um": positions,
        "layout_least_squares_stress": stress,
        "layout_note": (
            "2D positions minimize sum (d_actual - d_target(|ρ|))²; global geometry may not "
            "satisfy every pairwise rule exactly—see pair_geometry_vs_correlation.csv."
        ),
        "pair_geometry_csv": "pair_geometry_vs_correlation.csv",
        "pulse": {
            "durations_us": [dur_ramp, dur_sweep, dur_ramp],
            "omega_piecewise_linear_rad_per_us": [0.0, omega_peak, omega_peak, 0.0],
            "detuning_piecewise_linear_rad_per_us": [delta_neg, delta_neg, delta_pos, delta_pos],
            "notes": "Ω ramps up / holds / ramps down; Δ sweeps negative → positive in middle segment.",
        },
        "shots": shots,
        "top5": [{"bitstring": b, "count": int(c), "frequency": c / shots} for b, c in top5],
        "all_counts": {k: int(v) for k, v in ranked},
        "blockade_narrative_um": BLOCKADE_NEAR,
    }
    # --- Scale sweep: geometry multipliers → top bitstring → classical μ, σ ---
    do_sweep = os.environ.get("SCALE_SWEEP", "1").strip().lower() not in ("0", "false", "no", "off")
    sweep_rows: list[dict] = []
    if do_sweep:
        sf_raw = os.environ.get("SCALE_FACTORS", "0.8,1.0,1.2,1.5").strip()
        scale_factors = [float(x.strip()) for x in sf_raw.split(",") if x.strip()]
        sweep_shots = int(os.environ.get("SCALE_SWEEP_SHOTS", "1000"))
        selected_char = os.environ.get("BITSTRING_ONE_MEANS_SELECTED", "1").strip() or "1"
        mu8, sigma8 = load_mu_sigma_8(ids)
        sweep_rows = run_scale_sweep(
            xy,
            ids,
            start_mod=start,
            mu=mu8,
            sigma=sigma8,
            scale_factors=scale_factors,
            sweep_shots=sweep_shots,
            pulse_kw=pulse_kw,
            selected_char=selected_char,
        )
        pd.DataFrame(sweep_rows).to_csv(OUT_DIR / "scale_sweep_results.csv", index=False)
        print(
            "Scale sweep: wrote",
            OUT_DIR / "scale_sweep_results.csv",
            "and scale_sweep_risk_return.png",
            f"({len(scale_factors)} factors × {sweep_shots} shots).",
        )
        summary["scale_sweep"] = {
            "scale_factors": scale_factors,
            "shots_per_scale": sweep_shots,
            "bitstring_convention": (
                f"'{selected_char}' = asset in equal-weight portfolio; order matches assets_order."
            ),
            "csv": "scale_sweep_results.csv",
            "rows": sweep_rows,
        }

    with open(OUT_DIR / "phase4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Optional layout figure + risk–return scatter
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=120,
            c=np.arange(n),
            cmap="tab10",
            vmin=0,
            vmax=9,
            edgecolors="k",
            zorder=3,
        )
        for i, aid in enumerate(ids):
            ax.annotate(aid, (xy[i, 0], xy[i, 1]), xytext=(4, 4), textcoords="offset points")
        for i in range(n):
            for j in range(i + 1, n):
                ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], "k-", alpha=0.15, lw=0.8)
        ax.set_aspect("equal")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_title("Risk-cluster atom layout (optimized targets from |ρ|)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "atom_layout_um.png", dpi=160)
        plt.close(fig)

        plot_pulse_waveforms(pulse_kw, OUT_DIR / "pulse_waveforms_rabi_detuning.png")
        plot_pair_geometry_scatter(
            OUT_DIR / "pair_geometry_vs_correlation.csv",
            OUT_DIR / "pair_distance_vs_correlation_scatter.png",
        )

        if sweep_rows:
            vols = [r["portfolio_volatility"] for r in sweep_rows]
            rets = [r["portfolio_exp_return"] for r in sweep_rows]
            scales = [r["scale_factor"] for r in sweep_rows]
            fig2, ax2 = plt.subplots(figsize=(6.5, 5))
            sc = ax2.scatter(vols, rets, c=scales, cmap="viridis", s=120, edgecolors="k", zorder=3)
            for r in sweep_rows:
                ax2.annotate(
                    f"{r['scale_factor']}×",
                    (r["portfolio_volatility"], r["portfolio_exp_return"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=9,
                )
            ax2.set_xlabel("Portfolio volatility σ (classical, equal-weight on |1⟩)")
            ax2.set_ylabel("Portfolio expected return E[R]")
            ax2.set_title("Scale sweep: geometry vs risk–return (top bitstring per run)")
            ax2.grid(True, alpha=0.3)
            fig2.colorbar(sc, ax=ax2, label="Coordinate scale factor")
            fig2.tight_layout()
            fig2.savefig(OUT_DIR / "scale_sweep_risk_return.png", dpi=170, bbox_inches="tight")
            plt.close(fig2)
            plot_scale_sweep_top_frequency(sweep_rows, OUT_DIR / "scale_sweep_modal_frequency.png")
    except Exception as e:
        print("Phase 4: figure export skipped:", e, file=sys.stderr)


if __name__ == "__main__":
    main()
