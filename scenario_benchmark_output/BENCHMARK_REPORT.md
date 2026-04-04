# 시나리오 기반 벤치마크: 일반 전략 vs YQuantum 파이프라인

## 데이터

- **입력:** `investment_dataset_scenarios.csv` (1200행 × 50자산 수익률).
- **외부 데이터 없음** — 모든 비교는 이 시뮬레이션 표본 안에서만 정의합니다.

## 스트레스(의사 침체) 정의

- **시장 프록시:** 각 행에서 50자산 **동일가중** 수익률.
- **스트레스 행:** 그 프록시가 **하위 10% 분위** 이하인 시나리오 (약 **120행**).
- 해석: “시장 전체가 크게 깨진 날/경로”에 가깝게 두고, 그 부분집합에서만 평균·CVaR을 다시 봅니다.

## 비교한 방법 (10가지)

### 베이스라인 (1–5)

1. **EW50** — 50자산 동일가중 (가장 단순한 분산 투자).
2. **EW8_random** — 무작위 8자산 동일가중 (시드 고정, `RANDOM8_SEED`).
3. **EW8_top_Sharpe_no_cluster** — 시나리오 기준 Sharpe 상위 8자산 동일가중 (**클러스터 없음**).
4. **EW8_cluster_winners** — Phase 1과 동일한 8종(클러스터별 우승) **동일가중** (최적화 없음).
5. **Markowitz_cluster8** — 같은 8종에 대해 Phase 2 Markowitz 가중 (`classical_optimal_weights.csv`).

### 양자/하이브리드 (6–10)

6. **Analog_return_wtd_EW** — Phase 4 **수익 가중 detuning** 최빈 비트스트링 → 동일가중.
7. **Analog_sharpe_wtd_EW** — Phase 4 **Sharpe 가중 detuning** (μ/σ) 최빈 비트스트링 → 동일가중.
8. **Hybrid_sharpe_select_Markowitz** — Sharpe 가중 비트스트링 **선택** + 선택된 부분집합에 **Markowitz** 최적화.
9. **Hybrid_return_select_Markowitz** — 수익 가중 비트스트링 선택 + Markowitz 최적화.
10. **Analog_sharpe_ensemble** — Sharpe 가중 run의 **전체 비트스트링 분포**를 확률 가중 포트폴리오로 환산.

## 산출 파일

| 파일 | 내용 |
| --- | --- |
| `method_metrics_full_and_stress.csv` | 방법별 전체/스트레스 평균, 표준편차, Sharpe, CVaR5% 등 |
| `per_scenario_portfolio_returns.csv` | 행마다 시장프록시, 스트레스 플래그, 방법별 포트 수익 |
| `deltas_vs_EW50.csv` | EW50 대비 평균 수익 차이(전체·스트레스), CVaR 차이 |
| `benchmark_summary.json` | 위 내용 + 사용 자산 목록·비트스트링 |
| `*.png` | 막대/수평막대 요약 그래프 |

## 숫자 요약 (표)

```
                          method  n_scenarios  mean_all  std_all  sharpe_all   min_all  cvar5pct_all  mean_stress_mkt  std_stress_mkt  sharpe_stress_mkt  cvar5pct_stress_mkt  worst_stress_mkt
               1_EW50_all_assets         1200  0.000238 0.003333    0.071481 -0.009622     -0.006662        -0.005698        0.001271          -4.482778            -0.009162         -0.009622
                    2_EW8_random         1200  0.000390 0.005542    0.070461 -0.021466     -0.011463        -0.008318        0.003663          -2.271000            -0.015821         -0.021466
     3_EW8_top_Sharpe_no_cluster         1200  0.000588 0.005133    0.114589 -0.015185     -0.009734        -0.005180        0.004170          -1.242247            -0.013262         -0.015185
      4_EW8_cluster_winners_only         1200  0.000416 0.003506    0.118769 -0.011153     -0.007024        -0.004736        0.002289          -2.069246            -0.009399         -0.011153
         5_Markowitz_on_cluster8         1200  0.000417 0.003885    0.107218 -0.013182     -0.007780        -0.005107        0.002786          -1.833044            -0.011305         -0.013182
          6_Analog_return_wtd_EW         1200  0.000530 0.004633    0.114412 -0.016357     -0.009320        -0.006210        0.003135          -1.980715            -0.013150         -0.016357
          7_Analog_sharpe_wtd_EW         1200  0.000464 0.004939    0.093909 -0.017888     -0.009799        -0.006228        0.003456          -1.802038            -0.013068         -0.017888
8_Hybrid_sharpe_select_MaxSharpe         1200  0.000126 0.000966    0.130797 -0.002685     -0.001785        -0.000888        0.000840          -1.057340            -0.002460         -0.002685
9_Hybrid_return_select_MaxSharpe         1200  0.000502 0.003971    0.126385 -0.013331     -0.007975        -0.004826        0.002942          -1.640356            -0.010896         -0.013331
          10_Ensemble_sharpe_wtd         1200  0.000404 0.003449    0.117028 -0.010758     -0.006916        -0.004631        0.002268          -2.042187            -0.009189         -0.010758
          11_Ensemble_return_wtd         1200  0.000450 0.003964    0.113604 -0.013527     -0.007949        -0.005387        0.002597          -2.073968            -0.010725         -0.013527
```

## EW50 대비 스트레스 구간 평균 수익 차이

```
                  method_vs_EW50  mean_all_delta  mean_stress_delta  cvar5_all_delta
                    2_EW8_random        0.000152          -0.002620        -0.004801
     3_EW8_top_Sharpe_no_cluster        0.000350           0.000518        -0.003071
      4_EW8_cluster_winners_only        0.000178           0.000962        -0.000362
         5_Markowitz_on_cluster8        0.000178           0.000591        -0.001118
          6_Analog_return_wtd_EW        0.000292          -0.000512        -0.002658
          7_Analog_sharpe_wtd_EW        0.000226          -0.000530        -0.003136
8_Hybrid_sharpe_select_MaxSharpe       -0.000112           0.004810         0.004878
9_Hybrid_return_select_MaxSharpe        0.000264           0.000872        -0.001312
          10_Ensemble_sharpe_wtd        0.000165           0.001067        -0.000253
          11_Ensemble_return_wtd        0.000212           0.000311        -0.001287
```

## 주의 (심사/발표용)

- 시나리오가 **시간 순서**가 아니면 “낙폭(drawdown)”은 정의하지 않았고, **단면 분포** 기준 지표만 사용했습니다.
- **양자(아날로그) 결과**는 에뮬레이터 샘플에서 나온 **비트스트링 선택**을 여기서는 해당 방식(동일가중/Markowitz/앙상블)으로 매핑합니다.
- Bloqade 재실행 시 `phase4_summary.json`이 바뀌면 방법 6–10이 모두 갱신됩니다.

---

## English summary (for judges)

- **Stress set:** scenarios in the bottom **10%** of the equal-weight 50-asset market return (~**120** of **1200** rows).
- **Classical baselines:** EW50, random EW8, top-Sharpe EW8, cluster winners EW8, Markowitz on cluster-8.
- **Quantum-assisted (methods 6–10):**
  - **Return-weighted** (raw μ) + EW on modal bitstring.
  - **Sharpe-weighted** (μ/σ) + EW on modal bitstring — risk-adjusted Hamiltonian.
  - **Hybrid:** quantum Sharpe/return selects assets, classical Markowitz optimizes weights on that subset.
  - **Ensemble:** frequency-weighted average over the full Sharpe-run bitstring distribution.
- **Key CSVs:** `method_metrics_full_and_stress.csv`, `deltas_vs_EW50.csv`, `per_scenario_portfolio_returns.csv`.

**Run:** `python scenario_portfolio_benchmark.py`  (optional: `STRESS_MARKET_QUANTILE=0.1`, `RANDOM8_SEED=42`).
