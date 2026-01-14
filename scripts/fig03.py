"""
Figure 3

This script generates Figure 3 of the manuscript submitted to Nature Climate Change.

Required input:
- hail outputs from HAILCAST (Zenodo archive)

Output:
- Figure saved as PDF/PNG.
"""

import numpy as np
from glob import glob
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch


mpl.rcParams.update({
    # font
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,

    # ticks
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,

    # legend
    "legend.fontsize": 8,

    # line widths
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

# -------------------------
# User settings / paths
# -------------------------
dhail = "25"
path = "<your path here>" #directory Zenodo archive


#BOXPLOT:
#historical
data_dir = path+"/DHAIL_hexbin_data/historical"
files = sorted(glob(os.path.join(data_dir, "hexbin_*.npz"))) #FLH anomalies
dhail_all = []
hzero_all = []
for f in files:
    d = np.load(f)
    dhail_all.append(d["dhail"])
    hzero_all.append(d["hzero"])
dhail_all = np.concatenate(dhail_all)
hzero_all = np.concatenate(hzero_all)

#future
data_dir = path+"/DHAIL_hexbin_data/future/"
files = sorted(glob(os.path.join(data_dir, "hexbin_*.npz"))) #FLH anomalies
dhail_all_future = []
hzero_all_future = []
for f in files:
    d = np.load(f)
    dhail_all_future.append(d["dhail"])
    hzero_all_future.append(d["hzero"])
dhail_all_future = np.concatenate(dhail_all_future)
hzero_all_future = np.concatenate(hzero_all_future)

# === FLH bin ===
bins = np.arange(-2000, 2000 + 500, 500)  # m
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# historical
hail_per_bin_hist = []
digitized_hist = np.digitize(hzero_all, bins)
for i in range(1, len(bins)):
    sel = digitized_hist == i
    hail_per_bin_hist.append(dhail_all[sel])

# future (PGW)
hail_per_bin_fut = []
digitized_fut = np.digitize(hzero_all_future, bins)
for i in range(1, len(bins)):
    sel = digitized_fut == i
    hail_per_bin_fut.append(dhail_all_future[sel])


#PLOT:
fig, axes = plt.subplots(
    1, 2,
    figsize=(7.1, 3.6),
    gridspec_kw={"wspace": 0.15},
    constrained_layout=True
)
axes = axes.ravel()

# Panel 1: boxplot
ax = axes[0]
ax.text(0, 1.07, 'a', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')

positions = bin_centers
offset = 90
width = 140

# boxplot historical
bp_hist = ax.boxplot(
    hail_per_bin_hist,
    positions=positions - offset,
    widths=width,
    patch_artist=True,
    showfliers=False
)
for box in bp_hist["boxes"]:
    box.set(facecolor="#a6cee3", alpha=0.7)

# boxplot future
bp_fut = ax.boxplot(
    hail_per_bin_fut,
    positions=positions + offset,
    widths=width,
    patch_artist=True,
    showfliers=False
)
for box in bp_fut["boxes"]:
    box.set(facecolor="#fdbf6f", alpha=0.7)


ax.set_xlabel("FLH anomaly (m)")
ax.set_ylabel("Hail diameter (mm)")
ax.set_ylim(24, 45)
ax.grid(axis="y", alpha=0.3)

ax.set_xticks(positions, [f"{int(p)}" for p in positions], rotation=30)


legend_elements = [
    Patch(facecolor="#a6cee3", edgecolor="black", alpha=0.7, label="historical"),
    Patch(facecolor="#fdbf6f", edgecolor="black", alpha=0.7, label="PGW"),
]
ax.legend(loc="upper left", handles=legend_elements, frameon=True)


#PDF:
maxvalue="50"

years_past=['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
years_future=['2085', '2086', '2087', '2088', '2089', '2090', '2091', '2092', '2093', '2094', '2095']
months=['06', '07', '08']

#past dhail25
DHAIL_counts_sum = np.zeros(25)
count_month=0
for year in years_past:
    for month in months:
        fname = path+"/DHAIL_PDF/DHAIL25_HAILCAST_PDF_"+year+"_"+month+"_land.npz"
        if os.path.exists(fname):
            DHAIL_counts_sum += np.load(fname)["pdf"]
            count_month+=1

DHAIL_counts_mean = DHAIL_counts_sum/count_month
DHAIL_bins_mean = np.load(path+"/DHAIL_PDF/DHAIL25_HAILCAST_PDF_"+year+"_"+month+"_land.npz")["bin_edges"]

#future dhail25
DHAIL_counts_future_sum = np.zeros(25)
count_month=0
for year in years_future:
    for month in months:
        fname = path+"/DHAIL_PDF/DHAIL25_HAILCAST_PDF_"+year+"_"+month+"_land.npz"
        if os.path.exists(fname):
            DHAIL_counts_future_sum += np.load(fname)["pdf"]
            count_month+=1

DHAIL_counts_future_mean = DHAIL_counts_future_sum/count_month


# Panel 2: PDF
ax = axes[1]
ax.text(0, 1.07, 'b', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')


# ---- sx PDF (log) ----
ax.plot(DHAIL_bins_mean[1:], DHAIL_counts_mean, label="historical", color="tab:blue")
ax.plot(DHAIL_bins_mean[1:], DHAIL_counts_future_mean, label="PGW", color="tab:orange")
ax.set_yscale('log')
ax.set_xlabel('hail diameter (mm)')
ax.set_ylabel('PDF (log)')
ax.set_xlim(25, int(maxvalue))
# ---- dx difference ----
ax2 = ax.twinx()
diff = DHAIL_counts_future_mean - DHAIL_counts_mean
ax2.plot(DHAIL_bins_mean[1:], diff, color="0.2", label="PGW - historical")
ax2.set_ylabel(r'$\Delta$ PDF')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="lower center", frameon=True)

ax2.axhline(0, color="0.4", linewidth=1)

plt.savefig(path + "/figure/fig3.pdf", bbox_inches="tight")
plt.savefig(path + "/figure/fig3.png", dpi=300, bbox_inches="tight")