#!/usr/bin/env python3
"""Generate enhanced pulse waveform chart for presentation."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent / "phase4_output"

# Pulse parameters
dr, ds = 0.18, 2.2
t = np.array([0.0, dr, dr + ds, dr + ds + dr])  # [0, 0.18, 2.38, 2.56]
omega = [0.0, 2.8, 2.8, 0.0]
delta = [-14.0, -14.0, 14.0, 14.0]

# Phase boundaries
t_init_end = dr        # 0.18
t_explore_end = dr + ds  # 2.38
t_total = dr + ds + dr   # 2.56

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
fig.subplots_adjust(hspace=0.25)

# ── 1. Phase shading (both axes) ──
for ax in (ax1, ax2):
    ax.axvspan(0, t_init_end, alpha=0.10, color='#3498db', zorder=0)
    ax.axvspan(t_init_end, t_explore_end, alpha=0.08, color='#2ecc71', zorder=0)
    ax.axvspan(t_explore_end, t_total, alpha=0.10, color='#e74c3c', zorder=0)

# Phase labels on top axis
label_y = 3.55
ax1.text(t_init_end / 2, label_y, 'INIT', ha='center', va='bottom',
         fontsize=9, fontweight='bold', color='#2980b9',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#3498db', alpha=0.15, edgecolor='none'))
ax1.text((t_init_end + t_explore_end) / 2, label_y, 'EXPLORATION', ha='center', va='bottom',
         fontsize=9, fontweight='bold', color='#27ae60',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#2ecc71', alpha=0.15, edgecolor='none'))
ax1.text((t_explore_end + t_total) / 2, label_y, 'FREEZE', ha='center', va='bottom',
         fontsize=9, fontweight='bold', color='#c0392b',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#e74c3c', alpha=0.15, edgecolor='none'))

# ── 2. Rabi amplitude (top) ──
ax1.plot(t, omega, 'o-', color='#2c3e50', lw=2.5, markersize=7, zorder=5)
ax1.fill_between(t, omega, alpha=0.12, color='#3498db')
ax1.set_ylabel('$\\Omega$ (rad/$\\mu$s)', fontsize=12)
ax1.set_title('Exploration Strength  $\\Omega(t)$  — state switching rate', fontsize=13, fontweight='bold')
ax1.set_ylim(-0.3, 4.2)
ax1.grid(True, alpha=0.25)

# Annotations for Rabi
ax1.annotate('Laser OFF\nAll assets unselected',
             xy=(0.03, 0.0), xytext=(0.5, 1.0),
             fontsize=8.5, color='#2c3e50',
             arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.2),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#bdc3c7', alpha=0.9))

ax1.annotate('Peak drive\nQuantum superposition\nof all portfolios',
             xy=(1.3, 2.8), xytext=(1.3, 1.2),
             fontsize=8.5, color='#2c3e50', ha='center',
             arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.2),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#bdc3c7', alpha=0.9))

ax1.annotate('Laser OFF\nPortfolio frozen',
             xy=(2.53, 0.0), xytext=(2.05, 1.0),
             fontsize=8.5, color='#2c3e50',
             arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.2),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#bdc3c7', alpha=0.9))

# ── 3. Detuning (bottom) with color gradient ──
# Create fine-grained line for gradient
t_fine = np.linspace(0, t_total, 500)
delta_fine = np.interp(t_fine, t, delta)

# Plot gradient segments
for i in range(len(t_fine) - 1):
    # Normalize delta to [0, 1] for colormap: negative=blue, positive=red
    norm_val = (delta_fine[i] + 14) / 28  # maps [-14, 14] to [0, 1]
    color = plt.cm.coolwarm(norm_val)
    ax2.plot(t_fine[i:i+2], delta_fine[i:i+2], color=color, lw=3.0, solid_capstyle='round')

# Knot markers
ax2.plot(t, delta, 'o', color='#2c3e50', markersize=7, zorder=5)

# Zero line
ax2.axhline(0, color='gray', ls=':', lw=1, alpha=0.5)

ax2.set_ylabel('$\\Delta$ (rad/$\\mu$s)', fontsize=12)
ax2.set_xlabel('Time ($\\mu$s)', fontsize=12)
ax2.set_title('Selection Bias  $\\Delta(t)$  — favor selecting assets', fontsize=13, fontweight='bold')
ax2.set_ylim(-18, 20)
ax2.grid(True, alpha=0.25)

# Annotations for Detuning
ax2.annotate('$\\Delta < 0$: All assets OFF\n(unselected = lower energy)',
             xy=(0.09, -14), xytext=(0.55, -8),
             fontsize=8.5, color='#2471a3',
             arrowprops=dict(arrowstyle='->', color='#2471a3', lw=1.2),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#d6eaf8', edgecolor='#85c1e9', alpha=0.9))

ax2.annotate('$\\Delta$ crosses 0\nSelection becomes\nfavorable',
             xy=(1.28, 0), xytext=(0.55, 8),
             fontsize=8.5, color='#7d3c98',
             arrowprops=dict(arrowstyle='->', color='#7d3c98', lw=1.2),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f4ecf7', edgecolor='#c39bd3', alpha=0.9))

ax2.annotate('$\\Delta > 0$: Optimal portfolio\nlocked in (selected = lower energy)',
             xy=(2.47, 14), xytext=(1.55, 17),
             fontsize=8.5, color='#c0392b',
             arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8', edgecolor='#f1948a', alpha=0.9))

# Colorbar for detuning gradient
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(-14, 14))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax2, orientation='vertical', shrink=0.8, pad=0.02, aspect=20)
cbar.set_label('Selection pressure', fontsize=9)
cbar.set_ticks([-14, 0, 14])
cbar.set_ticklabels(['Unselect', 'Neutral', 'Select'])

fig.tight_layout()
fig.savefig(OUT / "pulse_waveforms_enhanced.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", OUT / "pulse_waveforms_enhanced.png")
