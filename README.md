# Quantum-Enhanced Insurance Portfolio Optimization

**YQuantum 2025 Hackathon** — The Hartford x Capgemini Quantum Lab x QuEra

An end-to-end pipeline that uses neutral-atom (Rydberg) analog quantum computing
to optimize insurance investment portfolios. Instead of tuning penalty parameters
in a QUBO formulation, we encode portfolio constraints directly into the physics
of atom placement and laser pulse design on QuEra's Bloqade platform.

---

## Pipeline Overview

```
INPUT                           CLASSICAL FOUNDATION                    QUANTUM LAYER                         VALIDATION
─────                           ────────────────────                    ─────────────                         ──────────

 50 assets                       Markowitz mean-variance                Atom layout from correlation          Ideal vs noisy
 1200 scenarios          ──>     optimization (SLSQP)           ──>    + adiabatic Rydberg pulse      ──>    bitstring comparison
 6 sectors                       on 8 cluster representatives          on Bloqade (8 qubits)                 + 11-method benchmark

        Phase 1                        Phase 2                              Phase 3-4                            Phase 5
   Clustering & Selection         Classical Baseline                  QUBO + Analog Quantum                Noise & Benchmark
```

**Phase 1 — Smart Data Preparation**
```
50 assets ──> hierarchical clustering on correlation distance ──> 8 clusters
         ──> select highest Sharpe ratio asset per cluster ──> 8 representative assets
```

**Phase 2 — Classical Markowitz Baseline**
```
8 assets ──> maximize  mu^T w - (lambda/2) w^T Sigma w
         ──> subject to sum(w)=1, 0<=w<=1
         ──> result: 6.09% return, 6.14% volatility, Sharpe 0.99
```

**Phase 3 — QUBO / Ising Formulation**
```
Markowitz objective ──> QUBO matrix Q (8x8)
                    ──> Ising parameters (h, J, C0) via x = (1+s)/2
                    ──> documents the lambda penalty problem
```

**Phase 4 — Analog Rydberg on Bloqade**
```
correlation matrix ──> atom positions in 2D (high |rho| = close = blockade)
expected returns   ──> per-atom local detuning (high mu = deeper bias)
                   ──> adiabatic pulse: Omega ramp + Delta sweep
                   ──> 1000-shot measurement ──> portfolio bitstrings
```

**Phase 5 — Noise Analysis**
```
ideal bitstrings ──> apply bit-flip (10%) + uniform mix (4%)
                 ──> compare entropy, top-bitstring survival
                 ──> key finding: quantum signal persists under noise
```

**Benchmark — 11 Methods Compared**
```
EW50, random EW8, top-Sharpe EW8, cluster EW8, Markowitz,
analog return-weighted, analog Sharpe-weighted,
hybrid (quantum select + classical optimize), ensemble
──> evaluated on 1200 scenarios including 120 stress scenarios
```

---

## Core Idea: Physics Replaces Penalty Tuning

Classical QUBO approaches require hand-tuning a penalty parameter lambda to enforce
constraints like budget or diversification. Too high and the optimizer ignores returns;
too low and it produces infeasible portfolios.

Our approach encodes constraints through the physics of Rydberg atoms:

```
CLASSICAL (QUBO)                          ANALOG (Rydberg)
────────────────                          ────────────────
lambda * (sum w_i - B)^2                  Rydberg blockade: atoms within ~7.5 um
  -> hand-tuned penalty                     cannot both be in |1> (selected)
  -> fragile, problem-specific              -> physical law, no tuning needed

Covariance in Q matrix                    Atom distance from |rho_ij|
  -> abstract math encoding                 -> correlated assets placed close
                                            -> blockade prevents co-selection

No return preference                      Per-atom local detuning proportional to mu_i
  -> all assets treated equally              -> higher return = deeper bias toward selection
```

This is not a digital gate circuit. QuEra's hardware is natively analog — we use
it in its native mode, which is a deliberate design choice that avoids the SWAP
overhead required by gate-based approaches on fixed-topology processors.

---

## Key Results

### Best Performer: Hybrid Quantum-Classical

The hybrid approach — quantum Sharpe-weighted bitstring selects assets, then classical
Markowitz re-optimizes weights on that subset — achieves the best risk-adjusted performance:

| Method | Sharpe (all) | CVaR 5% | Sharpe (stress) |
|--------|-------------|---------|-----------------|
| EW50 baseline | 0.071 | -0.0067 | -4.48 |
| Classical Markowitz | 0.107 | -0.0078 | -1.83 |
| Analog return-weighted | 0.114 | -0.0093 | -1.98 |
| **Hybrid Sharpe + Markowitz** | **0.131** | **-0.0018** | **-1.06** |
| Ensemble Sharpe-weighted | 0.117 | -0.0069 | -2.04 |

Under stress scenarios (worst 10% of market conditions), the hybrid method reduces
tail risk by 73% (CVaR) compared to equal-weight baseline.

### Three Detuning Strategies

| Strategy | Top bitstring | Freq | Portfolio | Rationale |
|----------|--------------|------|-----------|-----------|
| Uniform | 00101110 | 10.8% | 4 assets | No return preference |
| Return-weighted | 01101101 | 18.5% | 4 assets, E[R]=6.40% | Bias toward high-return |
| Sharpe-weighted | 00101110 | 16.2% | 4 assets, E[R]=6.09% | Bias toward risk-adjusted |

### Noise Resilience

| Metric | Ideal | Noisy (10% flip + 4% mix) |
|--------|-------|---------------------------|
| Shannon entropy | 3.45 nats | 4.60 nats (+33%) |
| Top bitstring probability | 9.2% | 5.0% |

Distribution flattens under noise, but top bitstrings remain in the upper ranks —
the quantum signal survives, validating the hybrid approach where quantum selects
candidates and classical optimizes weights.

---

## Hardware Connectivity Argument

Insurance portfolios produce **densely connected** problem graphs — every asset pair
has a nonzero covariance. This creates a fundamental mismatch with fixed-topology
quantum processors:

```
Superconducting (IBM-style)              Neutral Atom (QuEra-style)
───────────────────────────              ─────────────────────────
Fixed grid connectivity                  Free 2D atom placement
Dense graph -> SWAP chains               Dense graph -> place atoms at target distances
Each SWAP = more depth + error           No SWAPs needed
Portfolio structure lost in routing      Portfolio structure preserved in geometry
```

Our Phase 4 directly demonstrates this: correlation structure maps to atom distances,
and the Rydberg blockade radius naturally enforces diversification constraints without
additional circuit depth.

---

## Repository Structure

```
.
├── phase1_smart_data_prep.py          # Clustering 50 -> 8 assets
├── phase2_classical_benchmark.py      # Markowitz baseline
├── phase3_qubo_setup.py               # QUBO/Ising formulation
├── phase4_analog_rydberg.py           # Bloqade analog simulation
├── phase5_noise_topology.py           # Noise analysis
├── scenario_portfolio_benchmark.py    # 11-method comparison
│
├── phase1_output/
│   ├── correlation_heatmap_by_cluster.png
│   ├── dendrogram.png
│   ├── eight_winners_sharpe_barh.png
│   └── eight_qubit_assets.csv
│
├── phase2_output/
│   ├── classical_weights_pie.png
│   └── assets_bubble_risk_return_with_portfolio.png
│
├── phase3_output/
│   ├── Q_returns_diag_cov_off_heatmap.png
│   └── QUBO_and_Ising_representation.md
│
├── phase4_output/
│   ├── atom_layout_um.png
│   ├── pulse_waveforms_enhanced.png
│   ├── uniform_vs_return_weighted_bitstrings.png
│   ├── scale_sweep_risk_return.png
│   └── pair_distance_vs_correlation_scatter.png
│
├── phase5_output/
│   ├── ideal_vs_noisy_top_bitstrings.png
│   └── entropy_ideal_vs_noisy.png
│
├── scenario_benchmark_output/
│   ├── metrics_mean_sharpe_cvar5.png
│   ├── stress_scenario_mean_return_by_method.png
│   └── BENCHMARK_REPORT.md
│
├── investment_dataset_scenarios.csv   # 1200 x 50 return scenarios
├── investment_dataset_assets.csv      # Asset metadata
├── investment_dataset_covariance.csv  # 50 x 50 covariance matrix
├── investment_dataset_correlation.csv # 50 x 50 correlation matrix
└── investment_dataset_full.xlsx       # Combined dataset
```

---

## How to Run

**Requirements:** Python 3.10-3.12 (Bloqade does not support 3.13+)

```bash
# Phase 1-3, 5, benchmark (standard Python)
pip install numpy pandas scipy matplotlib seaborn openpyxl
python phase1_smart_data_prep.py
python phase2_classical_benchmark.py
python phase3_qubo_setup.py

# Phase 4-5 (requires bloqade-analog)
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install bloqade-analog numpy pandas scipy matplotlib
python phase4_analog_rydberg.py
python phase5_noise_topology.py

# Benchmark (after Phase 4 completes)
python scenario_portfolio_benchmark.py
```

---

## Technical Details

### Rydberg Hamiltonian

The system evolves under the standard neutral-atom Hamiltonian:

```
H(t) = (Omega(t)/2) sum_i sigma_x^(i) - Delta(t) sum_i n_i + sum_{i<j} (C6/R_ij^6) n_i n_j
```

We do not modify the Hamiltonian. Our contribution is in the **parameter mapping**:

| Parameter | Standard usage | Our mapping |
|-----------|---------------|-------------|
| R_ij (atom distance) | Arbitrary placement | Derived from correlation: high \|rho\| -> close -> blockade |
| Delta_i (local detuning) | Uniform for all atoms | Proportional to mu_i or Sharpe_i per asset |
| Omega(t), Delta(t) (pulse shape) | Application-dependent | Adiabatic sweep: 0.18 + 2.2 + 0.18 us |

### Adiabatic Protocol

```
t = 0.00 us : Omega=0,   Delta=-14  ->  ground state, all assets unselected
t = 0.18 us : Omega=2.8, Delta=-14  ->  laser on, quantum exploration begins
t = 2.38 us : Omega=2.8, Delta=+14  ->  selection becomes favorable
t = 2.56 us : Omega=0,   Delta=+14  ->  laser off, measure final portfolio
```

### Geometry Optimization

Atom positions are optimized to minimize stress between target and actual pairwise distances:

```
minimize sum_{i<j} (||r_i - r_j|| - d_target(|rho_ij|))^2

where d_target:
  |rho| >= 0.30  ->  6.0 um  (within blockade, co-selection suppressed)
  |rho| <= 0.12  ->  16.0 um (beyond blockade, independent selection)
  between        ->  linear interpolation
```

### Noise Model

Phenomenological post-measurement noise applied to ideal Bloqade results:
- **Bit-flip:** 10% per-qubit probability (models readout error / spontaneous emission)
- **Uniform mix:** 4% per-shot probability of complete decoherence

---

## Challenge Requirements Checklist

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Formulate as optimization problem | Done | Markowitz mean-variance (Phase 2) |
| Construct QUBO or Ising representation | Done | 8x8 Q matrix + Ising (h, J, C0) with verification (Phase 3) |
| Run on Bloqade with 8 qubits | Done | Analog mode, 3 detuning strategies, 1000 shots (Phase 4) |
| Scale and add noise | Done | Geometry scale sweep (0.8x-1.5x) + noise model (Phase 4-5) |
| Investigate connectivity and noise | Done | Neutral atom vs superconducting analysis + entropy comparison |
| Quantum-based optimization strategies | Done | Return-weighted, Sharpe-weighted, ensemble, hybrid approaches |
| Hardware assumptions affect outcomes | Done | Scale sweep shows geometry directly controls portfolio selection |
| Demo applicable to stakeholders | Done | 11-method benchmark with stress scenarios, visual outputs |
