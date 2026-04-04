# QUBO and Ising representation (8-asset portfolio)

This document satisfies the requirement **“Constructs a QUBO or Ising representation”**: we give explicit **QUBO** matrices \(Q\) and the equivalent **Ising** Hamiltonian parameters \((h, J, C_0)\) obtained by the change of variables \(x_i = (1+s_i)/2\), \(s_i \in \{-1,+1\}\).

---

## 1. QUBO form

Binary decision variables \(x_i \in \{0,1\}\) (e.g. include asset \(i\) or not).

For **symmetric** \(Q\), the QUBO energy is

$$E_{\mathrm{QUBO}}(\mathbf{x}) = \mathbf{x}^{\top} Q \mathbf{x}
= \sum_i Q_{ii} x_i + 2 \sum_{i<j} Q_{ij} x_i x_j.$$

CSV files with full numeric matrices live in this same folder (`phase3_output/`).

### 1a. Hackathon-style \(Q\) (returns on diagonal, covariances off-diagonal)

- **Construction:** \(Q_{ii} = \mu_i\), \(Q_{ij} = \Sigma_{ij}\) for \(i \neq j\).
- **File:** `Q_returns_diag_cov_off.csv`

| | A027 | A018 | A008 | A022 | A038 | A004 | A048 | A034 |
|---|---|---|---|---|---|---|---|---|
| A027 | 0.036878 | 0.000179625 | 0.00109232 | 0.00096671 | 0.00106527 | 0.00138466 | 5.33496e-05 | 0.000650326 |
| A018 | 0.000179625 | 0.0209197 | 0.000271323 | 0.000407488 | 0.00048829 | 0.000729145 | 9.25151e-05 | 0.000570456 |
| A008 | 0.00109232 | 0.000271323 | 0.0846366 | 0.00481517 | 0.00173439 | 0.00387096 | 0.000260282 | 0.00392239 |
| A022 | 0.00096671 | 0.000407488 | 0.00481517 | 0.0607111 | 0.0029817 | 0.0020029 | 0.000240869 | 0.0018098 |
| A038 | 0.00106527 | 0.00048829 | 0.00173439 | 0.0029817 | 0.0628648 | 0.000996235 | 0.000206099 | 0.00162648 |
| A004 | 0.00138466 | 0.000729145 | 0.00387096 | 0.0020029 | 0.000996235 | 0.0823264 | 0.000145163 | 0.00320953 |
| A048 | 5.33496e-05 | 9.25151e-05 | 0.000260282 | 0.000240869 | 0.000206099 | 0.000145163 | 0.0136317 | 0.0001941 |
| A034 | 0.000650326 | 0.000570456 | 0.00392239 | 0.0018098 | 0.00162648 | 0.00320953 | 0.0001941 | 0.0690703 |

### 1b. Minimization QUBO (return − risk + budget penalty)

- **Risk–return piece:** minimize \(-\boldsymbol{\mu}^{\top}\mathbf{x} + \lambda_{\mathrm{risk}} \mathbf{x}^{\top}\Sigma \mathbf{x}\) with \(\lambda_{\mathrm{risk}} = 12.0\).
- **Budget penalty:** add \(\lambda_{\mathrm{budget}} (\sum_i x_i - B)^2\) with \(\lambda_{\mathrm{budget}} = 5.0\), \(B = 4.0\).
- **Combined file:** `Q_min_with_budget_penalty.csv`

| | A027 | A018 | A008 | A022 | A038 | A004 | A048 | A034 |
|---|---|---|---|---|---|---|---|---|
| A027 | -34.9909 | 5.00216 | 5.01311 | 5.0116 | 5.01278 | 5.01662 | 5.00064 | 5.0078 |
| A018 | 5.00216 | -34.9969 | 5.00326 | 5.00489 | 5.00586 | 5.00875 | 5.00111 | 5.00685 |
| A008 | 5.01311 | 5.00326 | -34.6094 | 5.05778 | 5.02081 | 5.04645 | 5.00312 | 5.04707 |
| A022 | 5.0116 | 5.00489 | 5.05778 | -34.8381 | 5.03578 | 5.02403 | 5.00289 | 5.02172 |
| A038 | 5.01278 | 5.00586 | 5.02081 | 5.03578 | -34.9046 | 5.01195 | 5.00247 | 5.01952 |
| A004 | 5.01662 | 5.00875 | 5.04645 | 5.02403 | 5.01195 | -34.7234 | 5.00174 | 5.03851 |
| A048 | 5.00064 | 5.00111 | 5.00312 | 5.00289 | 5.00247 | 5.00174 | -35.0119 | 5.00233 |
| A034 | 5.0078 | 5.00685 | 5.04707 | 5.02172 | 5.01952 | 5.03851 | 5.00233 | -34.8939 |

---

## 2. Ising form (equivalent energy for every spin configuration)

Spins \(s_i \in \{-1,+1\}\), with \(x_i = (1+s_i)/2\).

Any symmetric QUBO maps to

$$E_{\mathrm{Ising}}(\mathbf{s}) = C_0 + \sum_i h_i s_i + \sum_{i<j} J_{ij} s_i s_j$$

with

$$h_i = \frac{1}{2} \sum_j Q_{ij}, \qquad J_{ij} = \frac{Q_{ij}}{2}\ (i<j),$$

$$C_0 = \frac{1}{2}\sum_i Q_{ii} + \frac{1}{2}\sum_{i<j} Q_{ij}.$$

*(Same energy as \(E_{\mathrm{QUBO}}\) after substituting \(x_i=(1+s_i)/2\); verified in code.)*

### 2a. Ising parameters for hackathon-style \(Q\)

- **\(C_0\)** = `0.2335030894`

**Linear fields \(h_i\)** (same order as assets):

| asset | h_i |
| --- | --- |
| A027 | 0.0211351223 |
| A018 | 0.01182929305 |
| A008 | 0.050301731 |
| A022 | 0.036967855 |
| A038 | 0.0359816415 |
| A004 | 0.0473324815 |
| A048 | 0.00741203735 |
| A034 | 0.040526701 |

**Couplings \(J_{ij}\)** (symmetric; only \(i<j\) listed):

| i | j | J_ij |
| --- | --- | --- |
| A027 | A018 | 8.98125e-05 |
| A027 | A008 | 0.0005461625 |
| A027 | A022 | 0.000483355 |
| A027 | A038 | 0.000532633 |
| A027 | A004 | 0.0006923325 |
| A027 | A048 | 2.66748e-05 |
| A027 | A034 | 0.000325163 |
| A018 | A008 | 0.0001356615 |
| A018 | A022 | 0.000203744 |
| A018 | A038 | 0.000244145 |
| A018 | A004 | 0.0003645725 |
| A018 | A048 | 4.625755e-05 |
| A018 | A034 | 0.000285228 |
| A008 | A022 | 0.0024075835 |
| A008 | A038 | 0.000867197 |
| A008 | A004 | 0.00193548 |
| A008 | A048 | 0.000130141 |
| A008 | A034 | 0.0019611945 |
| A022 | A038 | 0.0014908525 |
| A022 | A004 | 0.00100145 |
| A022 | A048 | 0.0001204345 |
| A022 | A034 | 0.0009048995 |
| A038 | A004 | 0.0004981175 |
| A038 | A048 | 0.0001030495 |
| A038 | A034 | 0.0008132375 |
| A004 | A048 | 7.25815e-05 |
| A004 | A034 | 0.001604763 |
| A048 | A034 | 9.705e-05 |

### 2b. Ising parameters for minimization QUBO + budget penalty

- **\(C_0\)** = `-69.26884615`

| asset | h_i |
| --- | --- |
| A027 | 0.0368813746 |
| A018 | 0.0179620566 |
| A008 | 0.291080107 |
| A022 | 0.16031523 |
| A038 | 0.1022884145 |
| A004 | 0.2123182815 |
| A048 | 0.0011869437 |
| A034 | 0.1249267265 |

| i | j | J_ij |
| --- | --- | --- |
| A027 | A018 | 2.50107775 |
| A027 | A008 | 2.50655395 |
| A027 | A022 | 2.50580026 |
| A027 | A038 | 2.506391596 |
| A027 | A004 | 2.50830799 |
| A027 | A048 | 2.500320098 |
| A027 | A034 | 2.503901956 |
| A018 | A008 | 2.501627938 |
| A018 | A022 | 2.502444928 |
| A018 | A038 | 2.50292974 |
| A018 | A004 | 2.50437487 |
| A018 | A048 | 2.500555091 |
| A018 | A034 | 2.503422736 |
| A008 | A022 | 2.528891002 |
| A008 | A038 | 2.510406364 |
| A008 | A004 | 2.52322576 |
| A008 | A048 | 2.501561692 |
| A008 | A034 | 2.523534334 |
| A022 | A038 | 2.51789023 |
| A022 | A004 | 2.5120174 |
| A022 | A048 | 2.501445214 |
| A022 | A034 | 2.510858794 |
| A038 | A004 | 2.50597741 |
| A038 | A048 | 2.501236594 |
| A038 | A034 | 2.50975885 |
| A004 | A048 | 2.500870978 |
| A004 | A034 | 2.519257156 |
| A048 | A034 | 2.5011646 |

---

## 3. How this connects to Phase 4 (Bloqade)

Phase 4 does **not** paste \(Q\) into a digital annealer API; it uses a **continuous-time Rydberg Hamiltonian** (atom positions, global/local detuning, Rabi amplitude). The QUBO/Ising here is the **discrete portfolio combinatorics encoding**; the analog layer is a **different** physical realization aimed at similar selection structure (blockade vs correlation, detuning vs returns).
