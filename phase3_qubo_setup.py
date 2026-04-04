#!/usr/bin/env python3
"""
Phase 3: QUBO construction and λ penalty narrative (setup for quantum story).

Step 3.1: Build 8×8 Q with expected returns on the diagonal and covariances on
the off-diagonals (per hackathon brief).

Also exports a standard *minimization* QUBO encoding and notes on how λ penalty
weights trade constraint satisfaction vs objective (README budget penalty).
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

ROOT = Path(__file__).resolve().parent
ASSETS_PATH = ROOT / "investment_dataset_assets.csv"
COV_PATH = ROOT / "investment_dataset_covariance.csv"
EIGHT_PATH = ROOT / "phase1_output" / "eight_qubit_assets.csv"
OUT_DIR = ROOT / "phase3_output"


def load_eight_assets() -> list[str]:
    df = pd.read_csv(EIGHT_PATH)
    return df.sort_values("cluster")["asset"].tolist()


def load_mu(ids: list[str]) -> np.ndarray:
    meta = pd.read_csv(ASSETS_PATH).set_index("asset_id")
    return np.array([float(meta.loc[i, "exp_return"]) for i in ids], dtype=np.float64)


def load_cov_submatrix(ids: list[str]) -> np.ndarray:
    full = pd.read_csv(COV_PATH, index_col=0)
    full.columns = [str(c).strip() for c in full.columns]
    full.index = [str(i).strip() for i in full.index]
    sub = full.loc[ids, ids].values.astype(np.float64)
    return 0.5 * (sub + sub.T)


def build_Q_returns_diag_cov_off(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Q_ii = μ_i, Q_ij = Σ_ij for i ≠ j (Σ_ii not placed on Q diagonal)."""
    n = len(mu)
    q = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(q, mu)
    mask = ~np.eye(n, dtype=bool)
    q[mask] = sigma[mask]
    return q


def qubo_energy_symmetric(x: np.ndarray, Q: np.ndarray) -> float:
    """E(x) = x^T Q x for binary x, with symmetric Q (standard QUBO)."""
    return float(x @ Q @ x)


def symmetric_Q_to_ising(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Map symmetric QUBO energy to an Ising model on spins s_i ∈ {-1, +1}.

    Substitution: x_i = (1 + s_i) / 2.

    E_QUBO(x) = x^T Q x = C_0 + sum_i h_i s_i + sum_{i<j} J_ij s_i s_j

    with (standard textbook identity for symmetric Q):
      h_i = (1/2) * sum_j Q_ij
      J_ij = Q_ij / 2   for i < j (symmetric extension J_ji = J_ij)
      C_0 = (1/2) * sum_i Q_ii + (1/2) * sum_{i<j} Q_ij
    """
    n = Q.shape[0]
    Qs = 0.5 * (Q + Q.T)
    h = 0.5 * Qs.sum(axis=1)
    J = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = J[j, i] = 0.5 * Qs[i, j]
    c0 = 0.5 * np.trace(Qs) + 0.5 * sum(float(Qs[i, j]) for i in range(n) for j in range(i + 1, n))
    return h, J, float(c0)


def ising_energy(s: np.ndarray, h: np.ndarray, J: np.ndarray, c0: float) -> float:
    """H(s) = C_0 + h^T s + sum_{i<j} J_ij s_i s_j (J symmetric, zero diagonal)."""
    n = len(s)
    e = c0 + float(h @ s)
    for i in range(n):
        for j in range(i + 1, n):
            e += float(J[i, j] * s[i] * s[j])
    return e


def _matrix_to_markdown_table(M: np.ndarray, row_labels: list[str], col_labels: list[str]) -> str:
    fmt = lambda v: f"{v:.6g}"
    header = "| | " + " | ".join(col_labels) + " |"
    sep = "|" + "|".join(["---"] * (len(col_labels) + 1)) + "|"
    lines = [header, sep]
    for i, ri in enumerate(row_labels):
        row = "| " + ri + " | " + " | ".join(fmt(M[i, j]) for j in range(M.shape[1])) + " |"
        lines.append(row)
    return "\n".join(lines)


def write_qubo_ising_representation_report(
    path: Path,
    ids: list[str],
    Q_hackathon: np.ndarray,
    Q_min_total: np.ndarray,
    *,
    lambda_risk: float,
    lambda_budget: float,
    budget_B: float,
) -> None:
    """
    Single deliverable file answering: 'Constructs a QUBO or Ising representation.'
    """
    h1, J1, c1 = symmetric_Q_to_ising(Q_hackathon)
    h2, J2, c2 = symmetric_Q_to_ising(Q_min_total)

    rng = np.random.default_rng(1)
    for Qm, h_m, J_m, c_m in (
        (Q_hackathon, h1, J1, c1),
        (Q_min_total, h2, J2, c2),
    ):
        for _ in range(20):
            x = (rng.random(len(ids)) > 0.5).astype(np.float64)
            s = 2.0 * x - 1.0
            e_q = qubo_energy_symmetric(x, Qm)
            e_i = ising_energy(s, h_m, J_m, c_m)
            if abs(e_q - e_i) > 1e-9:
                raise RuntimeError(f"QUBO/Ising mismatch: {e_q} vs {e_i}")

    lines: list[str] = []
    lines.append("# QUBO and Ising representation (8-asset portfolio)")
    lines.append("")
    lines.append(
        "This document satisfies the requirement **“Constructs a QUBO or Ising representation”**: "
        "we give explicit **QUBO** matrices \\(Q\\) and the equivalent **Ising** Hamiltonian "
        "parameters \\((h, J, C_0)\\) obtained by the change of variables \\(x_i = (1+s_i)/2\\), "
        "\\(s_i \\in \\{-1,+1\\}\\)."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. QUBO form")
    lines.append("")
    lines.append("Binary decision variables \\(x_i \\in \\{0,1\\}\\) (e.g. include asset \\(i\\) or not).")
    lines.append("")
    lines.append("For **symmetric** \\(Q\\), the QUBO energy is")
    lines.append("")
    lines.append("$$E_{\\mathrm{QUBO}}(\\mathbf{x}) = \\mathbf{x}^{\\top} Q \\mathbf{x}")
    lines.append("= \\sum_i Q_{ii} x_i + 2 \\sum_{i<j} Q_{ij} x_i x_j.$$")
    lines.append("")
    lines.append("CSV files with full numeric matrices live in this same folder (`phase3_output/`).")
    lines.append("")
    lines.append("### 1a. Hackathon-style \\(Q\\) (returns on diagonal, covariances off-diagonal)")
    lines.append("")
    lines.append("- **Construction:** \\(Q_{ii} = \\mu_i\\), \\(Q_{ij} = \\Sigma_{ij}\\) for \\(i \\neq j\\).")
    lines.append("- **File:** `Q_returns_diag_cov_off.csv`")
    lines.append("")
    lines.append(_matrix_to_markdown_table(Q_hackathon, ids, ids))
    lines.append("")
    lines.append("### 1b. Minimization QUBO (return − risk + budget penalty)")
    lines.append("")
    lines.append(
        f"- **Risk–return piece:** minimize \\(-\\boldsymbol{{\\mu}}^{{\\top}}\\mathbf{{x}} + "
        f"\\lambda_{{\\mathrm{{risk}}}} \\mathbf{{x}}^{{\\top}}\\Sigma \\mathbf{{x}}\\) "
        f"with \\(\\lambda_{{\\mathrm{{risk}}}} = {lambda_risk}\\)."
    )
    lines.append(
        f"- **Budget penalty:** add \\(\\lambda_{{\\mathrm{{budget}}}} (\\sum_i x_i - B)^2\\) "
        f"with \\(\\lambda_{{\\mathrm{{budget}}}} = {lambda_budget}\\), \\(B = {budget_B}\\)."
    )
    lines.append("- **Combined file:** `Q_min_with_budget_penalty.csv`")
    lines.append("")
    lines.append(_matrix_to_markdown_table(Q_min_total, ids, ids))
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Ising form (equivalent energy for every spin configuration)")
    lines.append("")
    lines.append("Spins \\(s_i \\in \\{-1,+1\\}\\), with \\(x_i = (1+s_i)/2\\).")
    lines.append("")
    lines.append("Any symmetric QUBO maps to")
    lines.append("")
    lines.append("$$E_{\\mathrm{Ising}}(\\mathbf{s}) = C_0 + \\sum_i h_i s_i + \\sum_{i<j} J_{ij} s_i s_j$$")
    lines.append("")
    lines.append("with")
    lines.append("")
    lines.append("$$h_i = \\frac{1}{2} \\sum_j Q_{ij}, \\qquad J_{ij} = \\frac{Q_{ij}}{2}\\ (i<j),$$")
    lines.append("")
    lines.append("$$C_0 = \\frac{1}{2}\\sum_i Q_{ii} + \\frac{1}{2}\\sum_{i<j} Q_{ij}.$$")
    lines.append("")
    lines.append("*(Same energy as \\(E_{\\mathrm{QUBO}}\\) after substituting \\(x_i=(1+s_i)/2\\); verified in code.)*")
    lines.append("")
    lines.append("### 2a. Ising parameters for hackathon-style \\(Q\\)")
    lines.append("")
    lines.append(f"- **\\(C_0\\)** = `{c1:.10g}`")
    lines.append("")
    lines.append("**Linear fields \\(h_i\\)** (same order as assets):")
    lines.append("")
    lines.append("| asset | h_i |")
    lines.append("| --- | --- |")
    for a, hi in zip(ids, h1):
        lines.append(f"| {a} | {hi:.10g} |")
    lines.append("")
    lines.append("**Couplings \\(J_{ij}\\)** (symmetric; only \\(i<j\\) listed):")
    lines.append("")
    lines.append("| i | j | J_ij |")
    lines.append("| --- | --- | --- |")
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            lines.append(f"| {ids[i]} | {ids[j]} | {J1[i, j]:.10g} |")
    lines.append("")
    lines.append("### 2b. Ising parameters for minimization QUBO + budget penalty")
    lines.append("")
    lines.append(f"- **\\(C_0\\)** = `{c2:.10g}`")
    lines.append("")
    lines.append("| asset | h_i |")
    lines.append("| --- | --- |")
    for a, hi in zip(ids, h2):
        lines.append(f"| {a} | {hi:.10g} |")
    lines.append("")
    lines.append("| i | j | J_ij |")
    lines.append("| --- | --- | --- |")
    for i in range(n):
        for j in range(i + 1, n):
            lines.append(f"| {ids[i]} | {ids[j]} | {J2[i, j]:.10g} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. How this connects to Phase 4 (Bloqade)")
    lines.append("")
    lines.append(
        "Phase 4 does **not** paste \\(Q\\) into a digital annealer API; it uses a **continuous-time "
        "Rydberg Hamiltonian** (atom positions, global/local detuning, Rabi amplitude). "
        "The QUBO/Ising here is the **discrete portfolio combinatorics encoding**; the analog layer is a **different** "
        "physical realization aimed at similar selection structure (blockade vs correlation, detuning vs returns)."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def minimization_Q_return_risk(
    mu: np.ndarray,
    sigma: np.ndarray,
    lambda_risk: float,
) -> np.ndarray:
    """
    Encode (approximately) minimize  -μ^T x + λ_risk * x^T Σ x  on binary x.

    Using E(x)=x^T Q x with symmetric Q:
      x^T Σ x = sum_i Σ_ii x_i + 2 sum_{i<j} Σ_ij x_i x_j
    Match linear part: Q_ii contribution to x^T Q x is Q_ii x_i, so set Q_ii = -μ_i + λ_risk * Σ_ii.
    Match off-diagonal: 2 Q_ij = 2 λ_risk Σ_ij  =>  Q_ij = λ_risk Σ_ij for i≠j.
    """
    n = len(mu)
    Q = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(Q, -mu + lambda_risk * np.diag(sigma))
    for i in range(n):
        for j in range(i + 1, n):
            v = lambda_risk * sigma[i, j]
            Q[i, j] = v
            Q[j, i] = v
    return Q


def budget_penalty_contribution(
    n: int,
    lambda_budget: float,
    budget_B: float,
) -> np.ndarray:
    """
    λ_budget * (sum_i x_i - B)^2 expanded for binary x_i.

    (sum x - B)^2 = (sum x)^2 - 2B sum x + B^2
    For binary x: = sum_i x_i + 2 sum_{i<j} x_i x_j - 2B sum_i x_i + B^2

    In x^T Q x: Q_ii = λ(1 - 2B), Q_ij = λ for i≠j. Constant λ B^2 omitted.
    """
    Q_add = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(Q_add, lambda_budget * (1.0 - 2.0 * budget_B))
    for i in range(n):
        for j in range(i + 1, n):
            Q_add[i, j] = lambda_budget
            Q_add[j, i] = lambda_budget
    return Q_add


def save_matrix_csv(Q: np.ndarray, ids: list[str], path: Path) -> None:
    df = pd.DataFrame(Q, index=ids, columns=ids)
    df.to_csv(path)


def plot_matrix_heatmap(Q: np.ndarray, ids: list[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    vmax = float(np.max(np.abs(Q))) or 1.0
    im = ax.imshow(Q, cmap="RdBu_r", aspect="equal", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(ids)))
    ax.set_yticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ids, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.78, label="Q value")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_slide_narrative(path: Path) -> None:
    body = """# Slide: From λ penalties to physics

## Headline
**Math penalties (λ) are inefficient. What if we used physics instead?**

## The setup (say this out loud)
- Classical portfolio constraints (budget, concentration, “don’t pile into correlated risk”) get **squashed into a single unconstrained objective** by adding **quadratic penalty terms** weighted by **λ**.
- In the hackathon formulation, a budget like **∑_i w_i = B** becomes **λ(∑_i w_i − B)²** so feasible portfolios sit in **wells** of the energy landscape.

## The frustrating λ tradeoff
- **λ too large:** penalties dominate → the optimizer “forgets” returns and chases feasibility / trivial low-energy states.
- **λ too small:** constraints are weak → **infeasible** portfolios (wrong cardinality, wrong budget) look artificially good.

## Pivot line (your novelty hook)
- **Classical QUBO / penalty methods = hand-tuned λ** to fake constraints inside one objective.
- **Physical devices (Hamiltonian dynamics, hardware graphs, noise)** impose **interactions and locality** that are not the same as a spreadsheet penalty — the question is whether **physics** can implement **structure** more naturally than **λ whack-a-mole**.

## Optional one-liner for the next slide
“We still build **Q** — but the story moves from **tuning λ** to **engineering the right physical Hamiltonian and topology**.”
"""
    path.write_text(body.strip() + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    ids = load_eight_assets()
    mu = load_mu(ids)
    sigma = load_cov_submatrix(ids)

    Q_combo = build_Q_returns_diag_cov_off(mu, sigma)
    save_matrix_csv(Q_combo, ids, OUT_DIR / "Q_returns_diag_cov_off.csv")

    lambda_risk = float(os.environ.get("LAMBDA_RISK", "12.0"))
    lambda_budget = float(os.environ.get("LAMBDA_BUDGET", "5.0"))
    budget_B = float(os.environ.get("BUDGET_B", "4.0"))

    Q_markowitz_min = minimization_Q_return_risk(mu, sigma, lambda_risk=lambda_risk)
    Q_budget_add = budget_penalty_contribution(len(ids), lambda_budget, budget_B)
    Q_total_min = Q_markowitz_min + Q_budget_add

    save_matrix_csv(Q_markowitz_min, ids, OUT_DIR / "Q_min_return_plus_risk.csv")
    save_matrix_csv(Q_budget_add, ids, OUT_DIR / "Q_add_budget_penalty_only.csv")
    save_matrix_csv(Q_total_min, ids, OUT_DIR / "Q_min_with_budget_penalty.csv")

    # Tiny sanity: energy dimensions
    rng = np.random.default_rng(0)
    x_demo = (rng.random(len(ids)) > 0.5).astype(np.float64)

    summary = {
        "assets": ids,
        "step_3_1_Q_structure": {
            "description": "Diagonal = expected return μ_i; off-diagonal = covariance Σ_ij (i≠j).",
            "file": "Q_returns_diag_cov_off.csv",
            "matrix": Q_combo.tolist(),
        },
        "qubo_energy_note": (
            "For binary x, E(x)=x^T Q x = sum_i Q_ii x_i + 2 sum_{i<j} Q_ij x_i x_j. "
            "Signs must match whether the solver minimizes energy and whether you maximize return."
        ),
        "lambda_penalty_narrative": {
            "budget_penalty_readme": "λ(∑_i w_i − B)² as in challenge README — λ must be tuned so solutions are feasible yet non-trivial.",
            "too_high": "Penalties dominate → poor objective / trivial feasible states.",
            "too_low": "Constraints ignored → attractive but invalid portfolios.",
            "physics_hook": "Hardware Hamiltonian + graph structure vs hand-tuned λ to encode constraints.",
        },
        "optional_minimization_encoding": {
            "lambda_risk": lambda_risk,
            "lambda_budget": lambda_budget,
            "budget_B": budget_B,
            "files": {
                "return_plus_risk_min_Q": "Q_min_return_plus_risk.csv",
                "budget_penalty_increment": "Q_add_budget_penalty_only.csv",
                "combined": "Q_min_with_budget_penalty.csv",
            },
            "formula": (
                "Q_ii = -μ_i + λ_risk Σ_ii + λ_budget (1 - 2B); "
                "Q_ij = λ_risk Σ_ij + λ_budget for i≠j (symmetric)."
            ),
        },
        "demo_random_binary_energy": {
            "x": {a: int(v) for a, v in zip(ids, x_demo)},
            "E_Q_combo": qubo_energy_symmetric(x_demo, Q_combo),
            "E_Q_total_min": qubo_energy_symmetric(x_demo, Q_total_min),
        },
    }

    with open(OUT_DIR / "phase3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_slide_narrative(OUT_DIR / "slide_lambda_penalty_narrative.md")

    write_qubo_ising_representation_report(
        OUT_DIR / "QUBO_and_Ising_representation.md",
        ids,
        Q_combo,
        Q_total_min,
        lambda_risk=lambda_risk,
        lambda_budget=lambda_budget,
        budget_B=budget_B,
    )

    plot_matrix_heatmap(
        Q_combo,
        ids,
        OUT_DIR / "Q_returns_diag_cov_off_heatmap.png",
        "Q: μ on diagonal, Σᵢⱼ off-diagonal (i≠j)",
    )
    plot_matrix_heatmap(
        Q_total_min,
        ids,
        OUT_DIR / "Q_min_with_budget_penalty_heatmap.png",
        f"Minimize-Q encoding: return+risk (λᵣ={lambda_risk}) + budget penalty (λᵦ={lambda_budget}, B={budget_B})",
    )

    print("Phase 3 complete:", OUT_DIR.resolve())
    print("Primary Q (μ on diag, Σ off-diag):", OUT_DIR / "Q_returns_diag_cov_off.csv")
    print("QUBO + Ising deliverable:", OUT_DIR / "QUBO_and_Ising_representation.md")
    print("Slide copy:", OUT_DIR / "slide_lambda_penalty_narrative.md")


if __name__ == "__main__":
    main()
