#!/usr/bin/env python3
"""Generate presentation-ready result charts from benchmark data."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "scenario_benchmark_output" / "method_metrics_full_and_stress.csv"
RETURNS = ROOT / "scenario_benchmark_output" / "per_scenario_portfolio_returns.csv"
OUT = ROOT / "results_output"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(METRICS)
ret_df = pd.read_csv(RETURNS)

# Clean method names for display
SHORT_NAMES = {
    "1_EW50_all_assets":              "EW50\n(Baseline)",
    "2_EW8_random":                   "EW8\nRandom",
    "3_EW8_top_Sharpe_no_cluster":    "EW8\nTop Sharpe",
    "4_EW8_cluster_winners_only":     "EW8\nClustered",
    "5_Markowitz_on_cluster8":        "Markowitz\n(Classical)",
    "6_Analog_return_wtd_EW":         "Quantum\nReturn-Wtd",
    "7_Analog_sharpe_wtd_EW":         "Quantum\nSharpe-Wtd",
    "8_Hybrid_sharpe_select_MaxSharpe":"Hybrid\nSharpe+Mark",
    "9_Hybrid_return_select_MaxSharpe":"Hybrid\nReturn+Mark",
    "10_Ensemble_sharpe_wtd":         "Ensemble\nSharpe",
    "11_Ensemble_return_wtd":         "Ensemble\nReturn",
}

# Color coding: gray=classical, blue=quantum, red=hybrid(best)
COLORS = {
    "1_EW50_all_assets":               "#bdc3c7",
    "2_EW8_random":                    "#bdc3c7",
    "3_EW8_top_Sharpe_no_cluster":     "#95a5a6",
    "4_EW8_cluster_winners_only":      "#95a5a6",
    "5_Markowitz_on_cluster8":         "#2c3e50",
    "6_Analog_return_wtd_EW":          "#3498db",
    "7_Analog_sharpe_wtd_EW":          "#2980b9",
    "8_Hybrid_sharpe_select_MaxSharpe":"#e74c3c",
    "9_Hybrid_return_select_MaxSharpe":"#e67e22",
    "10_Ensemble_sharpe_wtd":          "#1abc9c",
    "11_Ensemble_return_wtd":          "#16a085",
}

df["short"] = df["method"].map(SHORT_NAMES)
df["color"] = df["method"].map(COLORS)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 1: Sharpe Ratio Comparison (All Scenarios)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(df))
bars = ax.bar(x, df["sharpe_all"], color=df["color"], edgecolor="white", linewidth=0.8, width=0.75)

# Highlight best
best_idx = df["sharpe_all"].idxmax()
bars[best_idx].set_edgecolor("#c0392b")
bars[best_idx].set_linewidth(2.5)

# Value labels
for i, (val, bar) in enumerate(zip(df["sharpe_all"], bars)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.002,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold" if i == best_idx else "normal")

ax.set_xticks(x)
ax.set_xticklabels(df["short"], fontsize=8.5)
ax.set_ylabel("Sharpe Ratio", fontsize=12)
ax.set_title("Portfolio Sharpe Ratio — All 1200 Scenarios", fontsize=14, fontweight="bold")
ax.axhline(df.loc[0, "sharpe_all"], color="#bdc3c7", ls="--", lw=1, alpha=0.7, label="EW50 Baseline")
ax.grid(True, axis="y", alpha=0.2)

# Legend
legend_patches = [
    mpatches.Patch(color="#bdc3c7", label="Naive baselines"),
    mpatches.Patch(color="#2c3e50", label="Classical Markowitz"),
    mpatches.Patch(color="#3498db", label="Quantum analog"),
    mpatches.Patch(color="#e74c3c", label="Hybrid (quantum + classical)"),
    mpatches.Patch(color="#1abc9c", label="Ensemble"),
]
ax.legend(handles=legend_patches, loc="upper left", fontsize=9, framealpha=0.9)
ax.set_ylim(0, df["sharpe_all"].max() * 1.2)
fig.tight_layout()
fig.savefig(OUT / "1_sharpe_all_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("1. Sharpe ratio comparison saved")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 2: Stress Scenario Performance (Sharpe under worst 10%)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(x, df["sharpe_stress_mkt"], color=df["color"], edgecolor="white", linewidth=0.8, width=0.75)

best_stress_idx = df["sharpe_stress_mkt"].idxmax()
bars[best_stress_idx].set_edgecolor("#c0392b")
bars[best_stress_idx].set_linewidth(2.5)

for i, (val, bar) in enumerate(zip(df["sharpe_stress_mkt"], bars)):
    y_offset = 0.08 if val < 0 else -0.08
    va = "top" if val < 0 else "bottom"
    ax.text(bar.get_x() + bar.get_width()/2, val + y_offset,
            f"{val:.2f}", ha="center", va=va, fontsize=8.5,
            fontweight="bold" if i == best_stress_idx else "normal")

ax.set_xticks(x)
ax.set_xticklabels(df["short"], fontsize=8.5)
ax.set_ylabel("Sharpe Ratio (Stress)", fontsize=12)
ax.set_title("Stress Scenario Performance — Worst 10% Market Conditions (120 scenarios)",
             fontsize=14, fontweight="bold")
ax.axhline(0, color="black", lw=0.5)
ax.grid(True, axis="y", alpha=0.2)
ax.legend(handles=legend_patches, loc="lower left", fontsize=9, framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT / "2_sharpe_stress_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("2. Stress Sharpe comparison saved")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 3: CVaR 5% (Tail Risk) — Lower = Worse
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(14, 6))
# Multiply by 100 for percentage
cvar_pct = df["cvar5pct_all"] * 100
bars = ax.bar(x, cvar_pct, color=df["color"], edgecolor="white", linewidth=0.8, width=0.75)

best_cvar_idx = df["cvar5pct_all"].idxmax()  # closest to 0 = best
bars[best_cvar_idx].set_edgecolor("#c0392b")
bars[best_cvar_idx].set_linewidth(2.5)

for i, (val, bar) in enumerate(zip(cvar_pct, bars)):
    ax.text(bar.get_x() + bar.get_width()/2, val - 0.02,
            f"{val:.2f}%", ha="center", va="top", fontsize=8,
            fontweight="bold" if i == best_cvar_idx else "normal")

ax.set_xticks(x)
ax.set_xticklabels(df["short"], fontsize=8.5)
ax.set_ylabel("CVaR 5% (%)", fontsize=12)
ax.set_title("Tail Risk (CVaR 5%) — Average Loss in Worst 5% of Scenarios\n(closer to 0 = better)",
             fontsize=14, fontweight="bold")
ax.axhline(0, color="black", lw=0.5)
ax.grid(True, axis="y", alpha=0.2)
ax.legend(handles=legend_patches, loc="lower left", fontsize=9, framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT / "3_cvar_tail_risk.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("3. CVaR tail risk saved")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 4: Risk-Return Scatter (All Methods)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(10, 8))

for i, row in df.iterrows():
    vol = row["std_all"] * 100  # to percentage
    ret = row["mean_all"] * 100
    s = 250 if "Hybrid_sharpe" in row["method"] else 150
    marker = "*" if "Hybrid_sharpe" in row["method"] else "o"
    zorder = 10 if "Hybrid_sharpe" in row["method"] else 5
    ax.scatter(vol, ret, s=s, c=row["color"], edgecolors="black", linewidth=1,
               marker=marker, zorder=zorder)
    # Label positioning
    offset_x, offset_y = 0.02, 0.001
    if "EW50" in row["method"]:
        offset_x, offset_y = 0.02, -0.003
    elif "Hybrid_sharpe" in row["method"]:
        offset_x, offset_y = 0.02, 0.002
    name = row["short"].replace("\n", " ")
    fontw = "bold" if "Hybrid_sharpe" in row["method"] else "normal"
    ax.annotate(name, (vol, ret), xytext=(vol + offset_x, ret + offset_y),
                fontsize=7.5, fontweight=fontw)

ax.set_xlabel("Portfolio Volatility (%)", fontsize=12)
ax.set_ylabel("Expected Return (%)", fontsize=12)
ax.set_title("Risk-Return Profile — All 11 Methods", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.25)
ax.legend(handles=legend_patches, loc="upper left", fontsize=9, framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT / "4_risk_return_scatter.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("4. Risk-return scatter saved")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 5: Head-to-Head — Top 5 Methods Radar-style Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
top5_methods = [
    "1_EW50_all_assets",
    "5_Markowitz_on_cluster8",
    "6_Analog_return_wtd_EW",
    "8_Hybrid_sharpe_select_MaxSharpe",
    "10_Ensemble_sharpe_wtd",
]
top5 = df[df["method"].isin(top5_methods)].copy()
top5 = top5.set_index("method").loc[top5_methods].reset_index()

metrics = ["sharpe_all", "cvar5pct_all", "sharpe_stress_mkt", "mean_all"]
metric_labels = ["Sharpe Ratio\n(All)", "CVaR 5%\n(closer to 0 = better)", "Sharpe\n(Stress)", "Mean Return\n(All)"]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Head-to-Head: Key Methods Comparison", fontsize=15, fontweight="bold", y=1.02)

for ax, metric, label in zip(axes, metrics, metric_labels):
    vals = top5[metric].values
    colors_sub = [COLORS[m] for m in top5_methods]
    names = [SHORT_NAMES[m].replace("\n", " ") for m in top5_methods]

    barh = ax.barh(np.arange(len(top5)), vals, color=colors_sub, edgecolor="white", height=0.6)

    # Highlight best
    if "cvar" in metric:
        best = np.argmax(vals)  # closest to 0
    else:
        best = np.argmax(vals)
    barh[best].set_edgecolor("#c0392b")
    barh[best].set_linewidth(2)

    for j, (v, b) in enumerate(zip(vals, barh)):
        fmt = f"{v:.4f}" if abs(v) < 0.01 else f"{v:.3f}"
        x_pos = v + (max(vals) - min(vals)) * 0.03 if v >= 0 else v - (max(vals) - min(vals)) * 0.03
        ha = "left" if v >= 0 else "right"
        ax.text(x_pos, j, fmt, va="center", ha=ha, fontsize=8,
                fontweight="bold" if j == best else "normal")

    ax.set_yticks(np.arange(len(top5)))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel(label, fontsize=9)
    ax.grid(True, axis="x", alpha=0.2)
    ax.axvline(0, color="black", lw=0.5)
    ax.invert_yaxis()

fig.tight_layout()
fig.savefig(OUT / "5_head_to_head.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("5. Head-to-head comparison saved")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 6: Cumulative Return Distribution (Box Plot)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
plot_methods_cols = [
    "1_EW50_all_assets",
    "5_Markowitz_on_cluster8",
    "6_Analog_return_wtd_EW",
    "7_Analog_sharpe_wtd_EW",
    "8_Hybrid_sharpe_select_MaxSharpe",
    "10_Ensemble_sharpe_wtd",
]
plot_labels = [SHORT_NAMES[m].replace("\n", " ") for m in plot_methods_cols]
plot_colors = [COLORS[m] for m in plot_methods_cols]

data_for_box = [ret_df[m].values * 100 for m in plot_methods_cols]  # to percentage

fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot(data_for_box, labels=plot_labels, patch_artist=True,
                widths=0.55, showfliers=True,
                flierprops=dict(marker=".", markersize=3, alpha=0.3),
                medianprops=dict(color="black", linewidth=1.5))

for patch, color in zip(bp["boxes"], plot_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
ax.set_ylabel("Scenario Return (%)", fontsize=12)
ax.set_title("Return Distribution Across 1200 Scenarios — Key Methods",
             fontsize=14, fontweight="bold")
ax.grid(True, axis="y", alpha=0.2)

# Add median labels
for i, d in enumerate(data_for_box):
    med = np.median(d)
    ax.text(i + 1, med + 0.03, f"med={med:.3f}%", ha="center", va="bottom", fontsize=7.5,
            fontweight="bold", color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))

fig.tight_layout()
fig.savefig(OUT / "6_return_distribution_boxplot.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("6. Return distribution boxplot saved")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 7: Stress vs Normal — Grouped Bar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(14, 6))
x_pos = np.arange(len(df))
w = 0.35

bars1 = ax.bar(x_pos - w/2, df["mean_all"] * 10000, w,
               color=[COLORS[m] for m in df["method"]], edgecolor="white",
               label="All scenarios", alpha=0.9)
bars2 = ax.bar(x_pos + w/2, df["mean_stress_mkt"] * 10000, w,
               color=[COLORS[m] for m in df["method"]], edgecolor="black", linewidth=0.8,
               label="Stress scenarios (worst 10%)", alpha=0.5, hatch="//")

ax.set_xticks(x_pos)
ax.set_xticklabels(df["short"], fontsize=8.5)
ax.set_ylabel("Mean Return (bps)", fontsize=12)
ax.set_title("Mean Return: Normal vs Stress Conditions (basis points)",
             fontsize=14, fontweight="bold")
ax.axhline(0, color="black", lw=0.5)
ax.grid(True, axis="y", alpha=0.2)

# Custom legend
normal_patch = mpatches.Patch(facecolor="#3498db", edgecolor="white", label="All 1200 scenarios")
stress_patch = mpatches.Patch(facecolor="#3498db", edgecolor="black", alpha=0.5, hatch="//",
                               label="Stress (worst 10%)")
ax.legend(handles=[normal_patch, stress_patch] + legend_patches,
          loc="lower left", fontsize=8, framealpha=0.9, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "7_normal_vs_stress.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("7. Normal vs stress comparison saved")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 8: Final Summary — The "Money Slide"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(14, 7))
ax.axis("off")

title = "Quantum-Enhanced Portfolio Optimization: Key Findings"
ax.text(0.5, 0.95, title, transform=ax.transAxes, fontsize=18, fontweight="bold",
        ha="center", va="top")

# Table data — key methods only
table_data = []
table_colors = []
key_methods = [
    ("1_EW50_all_assets", "EW50 (Baseline)"),
    ("5_Markowitz_on_cluster8", "Classical Markowitz"),
    ("6_Analog_return_wtd_EW", "Quantum Return-Weighted"),
    ("7_Analog_sharpe_wtd_EW", "Quantum Sharpe-Weighted"),
    ("8_Hybrid_sharpe_select_MaxSharpe", "Hybrid Sharpe + Markowitz"),
    ("10_Ensemble_sharpe_wtd", "Ensemble Sharpe"),
]

for method_id, display_name in key_methods:
    row = df[df["method"] == method_id].iloc[0]
    sharpe_all = f"{row['sharpe_all']:.3f}"
    cvar = f"{row['cvar5pct_all']*100:.2f}%"
    sharpe_stress = f"{row['sharpe_stress_mkt']:.2f}"
    mean_ret = f"{row['mean_all']*10000:.2f} bps"
    table_data.append([display_name, sharpe_all, cvar, sharpe_stress, mean_ret])
    table_colors.append(COLORS[method_id])

col_labels = ["Method", "Sharpe\n(All)", "CVaR 5%\n(All)", "Sharpe\n(Stress)", "Mean Return\n(All)"]

table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    bbox=[0.05, 0.15, 0.9, 0.65],
)
table.auto_set_font_size(False)
table.set_fontsize(11)

# Style header
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#2c3e50")
    cell.set_text_props(color="white", fontweight="bold")
    cell.set_height(0.12)

# Style rows
for i, (_, color) in enumerate(zip(table_data, table_colors)):
    for j in range(len(col_labels)):
        cell = table[i + 1, j]
        cell.set_height(0.10)
        if j == 0:
            cell.set_facecolor(color)
            cell.set_alpha(0.3)
            cell.set_text_props(fontweight="bold")
        # Highlight the best row (Hybrid)
        if "Hybrid" in table_data[i][0]:
            cell.set_facecolor("#fadbd8")
            cell.set_text_props(fontweight="bold")
            if j == 0:
                cell.set_facecolor("#e74c3c")
                cell.set_text_props(color="white", fontweight="bold")

# Key takeaway text
takeaway = (
    "Hybrid Sharpe + Markowitz achieves the highest Sharpe ratio (0.131) and lowest tail risk (CVaR -0.18%).\n"
    "Under stress, it outperforms the baseline by 3.4x in Sharpe ratio (-1.06 vs -4.48).\n"
    "Quantum selects which assets; classical optimizes how much — best of both worlds."
)
ax.text(0.5, 0.08, takeaway, transform=ax.transAxes, fontsize=11,
        ha="center", va="top", style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#eaf2f8", edgecolor="#aed6f1", alpha=0.9))

fig.tight_layout()
fig.savefig(OUT / "8_final_summary_table.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("8. Final summary table saved")

print(f"\nAll charts saved to: {OUT}")
