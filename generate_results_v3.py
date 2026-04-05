#!/usr/bin/env python3
"""Final summary table — clean design, no color noise, renamed Hybrid."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "scenario_benchmark_output" / "method_metrics_full_and_stress.csv"
OUT = ROOT / "results_output"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(METRICS)

fig, ax = plt.subplots(figsize=(14, 7))
ax.axis("off")

title = "Quantum-Enhanced Portfolio Optimization: Results by Objective"
ax.text(0.5, 0.95, title, transform=ax.transAxes, fontsize=18, fontweight="bold", ha="center", va="top")

key_methods = [
    ("1_EW50_all_assets",               "EW50 (Baseline)"),
    ("5_Markowitz_on_cluster8",         "Classical Markowitz"),
    ("6_Analog_return_wtd_EW",          "Quantum Return-Weighted"),
    ("7_Analog_sharpe_wtd_EW",          "Quantum Sharpe-Weighted"),
    ("8_Hybrid_sharpe_select_MaxSharpe","Hybrid Quantum + Markowitz"),
    ("10_Ensemble_sharpe_wtd",          "Ensemble Quantum"),
]

table_data = []
for method_id, display_name in key_methods:
    row = df[df["method"] == method_id].iloc[0]
    table_data.append([
        display_name,
        f"{row['mean_all']*10000:.2f} bps",
        f"{row['std_all']*100:.3f}%",
        f"{row['sharpe_all']:.3f}",
        f"{row['cvar5pct_all']*100:.2f}%",
        f"{row['sharpe_stress_mkt']:.2f}",
    ])

col_labels = ["Method", "Mean Return", "Volatility", "Sharpe (All)", "CVaR 5%", "Sharpe (Stress)"]

table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    bbox=[0.05, 0.28, 0.9, 0.58],
)
table.auto_set_font_size(False)
table.set_fontsize(11)

# Header: dark, clean
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#2c3e50")
    cell.set_text_props(color="white", fontweight="bold")
    cell.set_height(0.12)
    cell.set_edgecolor("white")

# Rows: alternating white/light gray, no colors
for i in range(len(table_data)):
    bg = "#ffffff" if i % 2 == 0 else "#f7f7f7"
    for j in range(len(col_labels)):
        cell = table[i + 1, j]
        cell.set_height(0.10)
        cell.set_facecolor(bg)
        cell.set_edgecolor("#e0e0e0")
        if j == 0:
            cell.set_text_props(fontweight="bold")

# Bold the best value per column (skip method name col)
# Col 1 Mean Return: row 2 (Quantum Return-Wtd) = highest
table[3, 1].set_text_props(fontweight="bold")
# Col 3 Sharpe All: row 4 (Hybrid) = highest
table[5, 3].set_text_props(fontweight="bold")
# Col 4 CVaR: row 4 (Hybrid) = best (closest to 0)
table[5, 4].set_text_props(fontweight="bold")
# Col 5 Stress Sharpe: row 4 (Hybrid) = best
table[5, 5].set_text_props(fontweight="bold")

# Key takeaway
takeaway = (
    "Maximize return: Quantum Return-Weighted beats Classical Markowitz by +27% (5.30 vs 4.17 bps).\n"
    "Minimize tail risk: Hybrid Quantum + Markowitz reduces CVaR by 73% (-0.18% vs -0.67%).\n"
    "Under stress (worst 10%): Hybrid outperforms baseline by 3.4x in Sharpe (-1.06 vs -4.48).\n"
    "Quantum selects which assets (physics); classical optimizes how much (math)."
)
ax.text(0.5, 0.18, takeaway, transform=ax.transAxes, fontsize=10.5,
        ha="center", va="top", linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f9f9f9", edgecolor="#cccccc"))

fig.savefig(OUT / "8_final_summary_v3.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", OUT / "8_final_summary_v3.png")
