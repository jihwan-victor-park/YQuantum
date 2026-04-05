#!/usr/bin/env python3
"""Final summary table — matching the screenshot style exactly."""

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

key_methods = [
    ("1_EW50_all_assets",               "EW50 (Baseline)"),
    ("5_Markowitz_on_cluster8",         "Classical Markowitz"),
    ("6_Analog_return_wtd_EW",          "Quantum Return-Weighted"),
    ("7_Analog_sharpe_wtd_EW",          "Quantum Sharpe-Weighted"),
    ("8_Hybrid_sharpe_select_MaxSharpe","Hybrid Quantum + Markowitz"),
    ("10_Ensemble_sharpe_wtd",          "Ensemble Quantum"),
]

col_labels = ["Method", "Sharpe\n(All)", "CVaR 5%\n(All)", "Sharpe\n(Stress)", "Mean Return\n(All)"]

table_data = []
for method_id, display_name in key_methods:
    row = df[df["method"] == method_id].iloc[0]
    table_data.append([
        display_name,
        f"{row['sharpe_all']:.3f}",
        f"{row['cvar5pct_all']*100:.2f}%",
        f"{row['sharpe_stress_mkt']:.2f}",
        f"{row['mean_all']*10000:.2f} bps",
    ])

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Title
ax.text(0.5, 0.94, "Quantum-Enhanced Portfolio Optimization: Key Findings",
        transform=ax.transAxes, fontsize=17, fontweight="bold", ha="center", va="top",
        fontfamily="sans-serif")

# Table
table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    bbox=[0.02, 0.12, 0.96, 0.74],
    colWidths=[0.32, 0.14, 0.14, 0.14, 0.16],
)
table.auto_set_font_size(False)
table.set_fontsize(12)

n_cols = len(col_labels)
n_rows = len(table_data)

# Header row
for j in range(n_cols):
    cell = table[0, j]
    cell.set_facecolor("#34495e")
    cell.set_text_props(color="white", fontweight="bold", fontsize=11,
                        fontfamily="sans-serif")
    cell.set_edgecolor("#34495e")
    cell.set_height(0.11)

# Data rows — clean alternating
for i in range(n_rows):
    bg = "#ffffff" if i % 2 == 0 else "#f4f6f7"
    for j in range(n_cols):
        cell = table[i + 1, j]
        cell.set_facecolor(bg)
        cell.set_edgecolor("#d5d8dc")
        cell.set_height(0.095)
        cell.set_text_props(fontsize=12, fontfamily="sans-serif")
        if j == 0:
            cell.set_text_props(fontweight="bold", fontsize=11, fontfamily="sans-serif")

# Bold best values
table[3, 4].set_text_props(fontweight="bold", fontsize=12, fontfamily="sans-serif")  # Mean Return: Quantum Return-Wtd
table[5, 1].set_text_props(fontweight="bold", fontsize=12, fontfamily="sans-serif")  # Sharpe All: Hybrid
table[5, 2].set_text_props(fontweight="bold", fontsize=12, fontfamily="sans-serif")  # CVaR: Hybrid
table[5, 3].set_text_props(fontweight="bold", fontsize=12, fontfamily="sans-serif")  # Stress Sharpe: Hybrid

# Takeaway text
takeaway = (
    "Quantum Return-Weighted: +27% higher return than Classical Markowitz (5.30 vs 4.17 bps).\n"
    "Hybrid Quantum + Markowitz: 73% lower tail risk, 3.4x better under stress.\n"
    "Quantum selects which assets (physics); classical optimizes how much (math)."
)
ax.text(0.5, 0.06, takeaway, transform=ax.transAxes, fontsize=10,
        ha="center", va="top", style="italic", linespacing=1.5, fontfamily="sans-serif",
        color="#2c3e50")

fig.savefig(OUT / "8_final_summary_clean.png", dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print("Saved:", OUT / "8_final_summary_clean.png")
