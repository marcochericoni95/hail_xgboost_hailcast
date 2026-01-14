"""
Figure 1

This script generates Figure 1 of the manuscript submitted to Nature Climate Change.

Required input:
- hail outputs from XGBoost and HAILCAST (Zenodo archive)

Output:
- Figure saved as PDF/PNG.
"""

import numpy as np
from glob import glob
import xarray as xr
import os
import re
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
from scipy.ndimage import uniform_filter
from scipy.optimize import minimize
import matplotlib.axes as maxes

from mpl_toolkits.axes_grid1 import make_axes_locatable


mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})
mpl.rcParams["pdf.compression"] = 9


# -------------------------
# User settings / paths
# -------------------------
dhail = "25"
path = "<your path here>" #directory Zenodo archive


# -------------------------
# Load landmask
# -------------------------
ds_land = xr.open_mfdataset(
    glob(path + "/const_scclim_cosmo_clm.nc"),
    combine="by_coords"
)
landmask = (ds_land.FIS.values > 0)


# -------------------------
# load monthly files and concat on a synthetic time axis
# -------------------------
def load_monthly_concat(file_paths, varname):
    datasets = []
    ds_last = None
    for fp in sorted(file_paths):
        match = re.search(r'(\d{4})_(\d{2})', os.path.basename(fp))
        if match is None:
            continue
        year = int(match.group(1))
        month = int(match.group(2))
        timestamp = pd.Timestamp(f"{year}-{month:02d}-01")
        ds = xr.open_mfdataset(fp)
        ds_last = ds
        data = ds[varname].expand_dims(dim={"time": [timestamp]})
        datasets.append(data)
    if len(datasets) == 0:
        raise FileNotFoundError(f"No files matched for {varname}.")
    return xr.concat(datasets, dim="time"), ds_last


# -------------------------
# Load HAILCAST (historical)
# -------------------------
hailcast_paths = glob(path + f"/HAILCAST/historical/hailcast_{dhail}_daily_*.nc")
hailcast_daily_2d, ds_last = load_monthly_concat(hailcast_paths, "hailcast")
lat = np.array(ds_last["lat"])
lon = np.array(ds_last["lon"])


# -------------------------
# Load XGB (historical)
# -------------------------
xgb_paths = glob(path + "/XGBoost/historical/XGB_global_daily_*.nc")
XGB_global_2d_daily_cpm, _ = load_monthly_concat(xgb_paths, "XGB")
XGB_global_2d_daily_cpm = XGB_global_2d_daily_cpm.astype(float)


# -------------------------
# DIURNAL CYCLE (historical)
# -------------------------
file_hail_past = sorted(glob(path + "/HAILCAST/diurnal_cycle/historical/HAILCAST_diurnal_cycle_*.npy"))
file_xgb_past  = sorted(glob(path + "/XGBoost/diurnal_cycle/historical/XGB_global_diurnal_cycle_*.npy"))

hail_freqs_past = np.stack([np.load(f) for f in file_hail_past])
xgb_freqs_past  = np.stack([np.load(f) for f in file_xgb_past])

hail_freq_past_mean = hail_freqs_past.mean(axis=0)
xgb_freq_past_mean  = xgb_freqs_past.mean(axis=0)

# close the curve (24 -> 0)
hail_freq_past_mean_new = np.empty(25)
hail_freq_past_mean_new[:24] = hail_freq_past_mean
hail_freq_past_mean_new[-1]  = hail_freq_past_mean[0]

xgb_freq_past_mean_new = np.empty(25)
xgb_freq_past_mean_new[:24] = xgb_freq_past_mean
xgb_freq_past_mean_new[-1]  = xgb_freq_past_mean[0]


# -------------------------
# Colormaps
# -------------------------
colors = [
    [255, 255, 255],
    [120, 170, 230],
    [140, 200, 140],
    [255, 255, 100],
    [250, 160, 60],
    [230, 80, 50],
    [90, 50, 140]
]
colors = [[r/255, g/255, b/255] for r, g, b in colors]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap_main", colors)

colors_diff = [
    [84, 48, 5], [140, 81, 10], [191, 129, 45], [223, 194, 125], [246, 232, 195],
    [245, 245, 245], [199, 234, 229], [128, 205, 193], [53, 151, 143], [1, 102, 94], [0, 60, 48]
]
colors_diff = [[r/255, g/255, b/255] for r, g, b in colors_diff]
cmap_diff = mcolors.LinearSegmentedColormap.from_list("custom_cmap_diff", colors_diff)


# -------------------------
# Binning / norms
# -------------------------
bins = 8
bins_bar = 8
maxvalue = 7

diff_bins = 11
diff_bins_bar = 6
diffvalue = 5

levels_main = MaxNLocator(nbins=bins).tick_values(0, maxvalue)
norm_main = BoundaryNorm(levels_main, ncolors=cmap.N, clip=True)

levels_diff = MaxNLocator(nbins=diff_bins).tick_values(-diffvalue, diffvalue)
norm_diff = BoundaryNorm(levels_diff, ncolors=cmap_diff.N, clip=True)


# -------------------------
# Rotated pole estimation
# -------------------------
file_ref = path + "/lffd20130715020000.nz"
ds_ref = xr.open_mfdataset(file_ref, combine="by_coords")
rlat = np.array(ds_ref["rlat"])
rlon = np.array(ds_ref["rlon"])

i_samples = [0, lat.shape[0]//2, lat.shape[0]-1]
j_samples = [0, lon.shape[1]//2, lon.shape[1]-1]

lat_points = lat[np.ix_(i_samples, j_samples)].flatten()
lon_points = lon[np.ix_(i_samples, j_samples)].flatten()

rlat_points = rlat[i_samples]
rlon_points = rlon[j_samples]

def error(pole):
    lat_pole, lon_pole = pole
    proj_tmp = ccrs.RotatedPole(pole_latitude=lat_pole, pole_longitude=lon_pole)
    rlon_calc, rlat_calc = proj_tmp.transform_points(
        ccrs.PlateCarree(), lon_points, lat_points
    )[:, :2].T
    rlat_target = np.repeat(rlat_points, len(rlon_points))
    rlon_target = np.tile(rlon_points, len(rlat_points))
    return np.sum((rlat_calc - rlat_target)**2 + (rlon_calc - rlon_target)**2)

res = minimize(error, x0=[39.25, -162.0])
lat_pole_est, lon_pole_est = res.x

proj_rotated = ccrs.RotatedPole(
    pole_latitude=lat_pole_est,
    pole_longitude=lon_pole_est
)


# -------------------------
# Fields
# -------------------------
hailcast_smooth = uniform_filter(hailcast_daily_2d.sum(axis=0), size=4) * landmask[0]
hailcast_smooth[np.abs(hailcast_smooth) < 0.01] = np.nan

xgb_smooth = uniform_filter(XGB_global_2d_daily_cpm.sum(axis=0), size=4) * landmask[0]
xgb_smooth[np.abs(xgb_smooth) < 0.01] = np.nan

diff_xgb_minus_hailcast = (XGB_global_2d_daily_cpm.sum(axis=0).values -
                           hailcast_daily_2d.sum(axis=0).values)
diff_smooth = uniform_filter(diff_xgb_minus_hailcast, size=4) * landmask[0]
diff_smooth[np.abs(diff_smooth) <= 1] = np.nan


# -------------------------
# Plot helpers
# -------------------------
def style_map_ax(ax):
    ax.add_feature(cfeature.COASTLINE, linewidth=1, alpha=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=1, alpha=0.5)
    gl = ax.gridlines(
        color="black", linestyle="dotted", linewidth=1, alpha=0.5,
        draw_labels=True, x_inline=False, y_inline=False
    )
    gl.right_labels = False
    gl.top_labels = False
    return gl


def add_cbar_below(fig, ax, mappable, *, extend, ticks, label, fraction=0.035, pad="12%"):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size=f"{fraction*100:.0f}%", pad=pad, axes_class=maxes.Axes)
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", extend=extend)
    cb.set_ticks(ticks)
    cb.set_label(label)
    return cb

# -------------------------
# PLOT: 7.1 inch wide, double height, 4 panels (2x2)
# Maps keep identical sizes (same gridspec cells). Each map gets its own cbar below.
# -------------------------
fig = plt.figure(figsize=(7.1, 7.2))
gs = fig.add_gridspec(2, 2, wspace=0.22, hspace=0.33)  # hspace gives room for cbars

ax_a = fig.add_subplot(gs[0, 0], projection=proj_rotated)
ax_b = fig.add_subplot(gs[0, 1], projection=proj_rotated)
ax_c = fig.add_subplot(gs[1, 0], projection=proj_rotated)
ax_d = fig.add_subplot(gs[1, 1])

# ---- Panel a ----
style_map_ax(ax_a)
ax_a.set_title("HAILCAST", y=1.00)
ax_a.text(0.0, 1.02, "a", transform=ax_a.transAxes, fontsize=10, fontweight="bold",
          va="bottom", ha="left")
m_a = ax_a.pcolormesh(
    rlon, rlat, hailcast_smooth,
    cmap=cmap, norm=norm_main,
    transform=proj_rotated, shading="auto",
    rasterized=True
)
add_cbar_below(
    fig, ax_a, m_a,
    extend="max",
    ticks=np.linspace(0, maxvalue, bins_bar),
    label="total hail days",
    fraction=0.035,
    pad="12%"
)

# ---- Panel b ----
style_map_ax(ax_b)
ax_b.set_title("XGBoost", y=1.00)
ax_b.text(0.0, 1.02, "b", transform=ax_b.transAxes, fontsize=10, fontweight="bold",
          va="bottom", ha="left")
m_b = ax_b.pcolormesh(
    rlon, rlat, xgb_smooth,
    cmap=cmap, norm=norm_main,
    transform=proj_rotated, shading="auto",
    rasterized=True
)
add_cbar_below(
    fig, ax_b, m_b,
    extend="max",
    ticks=np.linspace(0, maxvalue, bins_bar),
    label="total hail days",
    fraction=0.035,
    pad="12%"
)

# ---- Panel c ----
style_map_ax(ax_c)
ax_c.set_title("XGBoost - HAILCAST", y=1.00)
ax_c.text(0.0, 1.02, "c", transform=ax_c.transAxes, fontsize=10, fontweight="bold",
          va="bottom", ha="left")
m_c = ax_c.pcolormesh(
    rlon, rlat, diff_smooth,
    cmap=cmap_diff, norm=norm_diff,
    transform=proj_rotated, shading="auto",
    rasterized=True
)
add_cbar_below(
    fig, ax_c, m_c,
    extend="both",
    ticks=np.linspace(-diffvalue, diffvalue, diff_bins_bar),
    label="total hail days",
    fraction=0.035,
    pad="12%"
)

# ---- Panel d (diurnal cycle) ----
ax_d.set_title("diurnal cycle", y=1.00)
ax_d.text(0.0, 1.02, "d", transform=ax_d.transAxes, fontsize=10, fontweight="bold",
          va="bottom", ha="left")

hours = np.arange(0, 25)
ax_d.plot(hours, hail_freq_past_mean_new, label="HAILCAST", color="black", linewidth=1.6)
ax_d.plot(hours, xgb_freq_past_mean_new, "--", label="XGBoost", color="black", linewidth=1.6)
ax_d.set_xlabel("hours")
ax_d.set_ylabel("frequency")
ax_d.set_xlim(0, 24)
ax_d.set_xticks(np.arange(0, 25, 4))
ax_d.grid(alpha=0.35)
ax_d.legend(loc="upper left", frameon=True)

# -------------------------
# Save
# -------------------------
out_pdf = path + "/fig1.pdf"
out_png = path + "/fig1.png"

plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(fig)
