#!/usr/bin/env python3
"""
Step 6 – Diagnostic plots for the Roman Bulge calibration sample
=================================================================
Produces a suite of publication-quality plots aimed at the science
goals of Projects B1 and B2:

  B1: Metallicity distributions along each sightline
  B2: Calibration set completeness (Teff, L, logg, [M/H], mass, age)

Plots produced
--------------
Sky coverage
  01_sky_footprint.png         – l/b map, bulge-centred, with Roman tile outlines
  02_sky_reddening.png         – same, coloured by E(J-Ks)
  03_sky_metallicity.png       – same, coloured by [M/H]
  04_sky_teff.png              – same, coloured by Teff

HR diagram
  05_hrd_lum_teff.png          – log L vs Teff (best values), coloured by [M/H]
  06_hrd_sources.png           – three luminosity estimates overlaid
  07_kiel_diagram.png          – logg vs Teff, coloured by [M/H]

CMDs
  08_cmd_observed.png          – (J-Ks) vs Ks, coloured by E(J-Ks)
  09_cmd_dereddened.png        – (J-Ks)0 vs Ks0, coloured by [M/H]

Metallicity (B1)
  10_metallicity_hist.png      – overall [M/H] distribution
  11_metallicity_by_sightline.png – per-sightline [M/H] violin/strip plot
  12_metallicity_sources.png   – source comparison (ASTRA vs XGBoost vs GSP-Spec)

Parameter coverage (B2)
  13_parameter_completeness.png – horizontal bar chart of N stars per parameter
  14_age_mass_flame.png         – FLAME age vs mass, coloured by [M/H]
  15_lum_comparison.png        – three luminosity estimates vs each other

Reddening
  16_reddening_hist.png        – E(J-Ks) distribution
  17_reddening_vs_b.png        – E(J-Ks) vs Galactic latitude

Input  : roman_master.fits  (output of step 5)
Output : plots/  directory
"""

import pathlib
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import warnings
from astropy.io.fits.verify import VerifyWarning
from astropy.units import UnitsWarning

warnings.filterwarnings("ignore", category=VerifyWarning)
warnings.filterwarnings("ignore", category=UnitsWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_FITS = pathlib.Path("roman_master.fits")
OUTDIR     = pathlib.Path("plots")

# ---------------------------------------------------------------------------
# Roman GBTDS overguide tile centres and geometry (Table 9, arXiv:2505.10574)
# ---------------------------------------------------------------------------
TILE_W_DEG  = (49.4 * u.arcmin).to(u.deg).value
TILE_H_DEG  = (25.3 * u.arcmin).to(u.deg).value

TILE_W_DEG  = (45.0 * u.arcmin).to(u.deg).value
TILE_H_DEG  = (23.0 * u.arcmin).to(u.deg).value

TILE_PA_DEG = 90.6

GBTDS_CENTERS = np.array([
    [-0.417948, -1.200],
    [-0.008974, -1.200],
    [ 0.400000, -1.200],
    [ 0.808974, -1.200],
    [ 1.217948, -1.200],
    [ 0.000000, -0.1250],
], dtype=float)

# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------
C_ALL    = "#adb5bd"
C_PHOT   = "#4895ef"
C_READY  = "#f72585"
C_VIRAC  = "#3a0ca3"
C_2MASS  = "#4cc9f0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(fits_path):
    tbl = Table.read(fits_path)
    scalar = [n for n in tbl.colnames if len(getattr(tbl[n], "shape", ())) <= 1]
    return tbl[scalar].to_pandas()


def col(df, candidates, default=np.nan):
    """Return float Series for first matching column (tries _1 suffix too)."""
    lower = {str(c).lower(): str(c) for c in df.columns}
    for cand in candidates:
        key = str(cand).lower()
        if key in lower:
            return pd.to_numeric(df[lower[key]], errors="coerce")
        if key + "_1" in lower:
            return pd.to_numeric(df[lower[key + "_1"]], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def flag(df, name):
    """Return boolean Series for a has_* flag column."""
    if name not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[name]
    if s.dtype == bool or s.dtype == np.bool_:
        return s.fillna(False)
    return s.fillna(0).astype(bool)


def finite(*arrays):
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(np.asarray(a, dtype=float))
    return mask


def savefig(fig, path, dpi=200):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def tile_outline_lb(l0, b0):
    """Return (l_array, b_array) of the outline of one Roman tile in Galactic coords."""
    center    = SkyCoord(l=l0 * u.deg, b=b0 * u.deg, frame="galactic")
    off_frame = SkyOffsetFrame(origin=center)
    hw, hh    = TILE_W_DEG / 2, TILE_H_DEG / 2
    corners   = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh], [-hw, -hh]])
    theta     = np.deg2rad(TILE_PA_DEG)
    dx = corners[:, 0] * np.cos(theta) - corners[:, 1] * np.sin(theta)
    dy = corners[:, 0] * np.sin(theta) + corners[:, 1] * np.cos(theta)
    outline = SkyCoord(lon=dx * u.deg, lat=dy * u.deg,
                       frame=off_frame).transform_to("galactic")
    return outline.l.deg, outline.b.deg


def bulge_l(l_deg):
    """Remap l to (−180, +180] so the Galactic centre is at 0."""
    return ((np.asarray(l_deg) + 180.0) % 360.0) - 180.0


def sky_ax_setup(ax):
    """Configure a sky-plot axis centred on the Galactic bulge."""
    ax.set_xlabel("Galactic longitude  l  [deg]", fontsize=11)
    ax.set_ylabel("Galactic latitude  b  [deg]", fontsize=11)
    ax.invert_xaxis()          # east to the left, west to the right
    ax.grid(alpha=0.2, lw=0.5)
    # Draw tile outlines
    for l0, b0 in GBTDS_CENTERS:
        tl, tb = tile_outline_lb(l0, b0)
        tl = bulge_l(tl)
        ax.plot(tl, tb, color="k", lw=1.2, zorder=5)


# ---------------------------------------------------------------------------
# 01 – Sky footprint coloured by status
# ---------------------------------------------------------------------------

def plot_sky_footprint(df, outdir):
    l = bulge_l(col(df, ["l", "glon"]).to_numpy())
    b = col(df, ["b", "glat"]).to_numpy()
    cal_phot  = flag(df, "calibration_ready_phot").to_numpy()
    cal_full  = flag(df, "calibration_ready").to_numpy()
    good      = finite(l, b)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(l[good & ~cal_phot], b[good & ~cal_phot],
               s=18, color=C_ALL,   alpha=0.6, label="All selected", zorder=2)
    #ax.scatter(l[good & cal_phot & ~cal_full], b[good & cal_phot & ~cal_full],
    #           s=22, color=C_PHOT,  alpha=0.8, label="Phot-ready", zorder=3)
    ax.scatter(l[good & cal_full], b[good & cal_full],
               s=28, color=C_READY, alpha=0.95, label="Fully calibration-ready", zorder=4)
    sky_ax_setup(ax)
    #ax.set_title("Roman GBTDS overguide footprint – calibration sample", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    savefig(fig, outdir / "01_sky_footprint.png")


# ---------------------------------------------------------------------------
# 02–04 – Sky maps coloured by a quantity
# ---------------------------------------------------------------------------

def plot_sky_colored(df, outdir, val_candidates, fname, title, cbar_label, cmap="viridis"):
    l   = bulge_l(col(df, ["l", "glon"]).to_numpy())
    b   = col(df, ["b", "glat"]).to_numpy()
    z   = col(df, val_candidates).to_numpy()
    good = finite(l, b, z)
    if good.sum() == 0:
        print(f"  skipped {fname} (no data)")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(l[good], b[good], c=z[good], s=24, alpha=0.9,
                    cmap=cmap, zorder=3)
    sky_ax_setup(ax)
    #ax.set_title(title, fontsize=12)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(cbar_label, fontsize=10)
    savefig(fig, outdir / fname)


# ---------------------------------------------------------------------------
# 05 – HR diagram (log L vs Teff), coloured by metallicity
# ---------------------------------------------------------------------------

def plot_hrd(df, outdir):
    teff  = col(df, ["teff_best"]).to_numpy()
    logl  = col(df, ["log10_lum_best_lsun"]).to_numpy()
    metal = col(df, ["metallicity_best"]).to_numpy()
    good  = finite(teff, logl)
    if good.sum() == 0:
        print("  skipped 05_hrd (no data)")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    sc_no_metal = ax.scatter(teff[good & ~np.isfinite(metal)],
                              logl[good & ~np.isfinite(metal)],
                              s=22, color=C_ALL, alpha=0.5, label="No [M/H]", zorder=2)
    good_m = good & np.isfinite(metal)
    if good_m.sum() > 0:
        sc = ax.scatter(teff[good_m], logl[good_m], c=metal[good_m],
                        s=28, alpha=0.9, cmap="RdYlBu_r",
                        vmin=np.nanpercentile(metal[good_m], 5),
                        vmax=np.nanpercentile(metal[good_m], 95),
                        zorder=3)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("[M/H]  [dex]", fontsize=10)
    ax.set_xlabel(r"$T_\mathrm{eff}$  [K]", fontsize=12)
    ax.set_ylabel(r"$\log_{10}(L/L_\odot)$", fontsize=12)
    #ax.set_title("HR diagram – Roman bulge calibration sample", fontsize=12)
    ax.invert_xaxis()
    ax.grid(alpha=0.2)
    handles = [mpatches.Patch(color=C_ALL, label=f"No [M/H]  (N={int((good & ~np.isfinite(metal)).sum())})")]
    if good_m.sum() > 0:
        handles.append(mpatches.Patch(color="grey", label=f"Has [M/H]  (N={int(good_m.sum())})"))
    ax.legend(handles=handles, frameon=False, fontsize=9)
    savefig(fig, outdir / "05_hrd_lum_teff.png")


# ---------------------------------------------------------------------------
# 06 – HR diagram showing all three luminosity sources
# ---------------------------------------------------------------------------

def plot_hrd_lum_sources(df, outdir):
    teff     = col(df, ["teff_best"]).to_numpy()
    lum_fl   = col(df, ["lum_flame_lsun"]).to_numpy()
    lum_rt   = col(df, ["lum_r_teff_lsun"]).to_numpy()
    lum_plx  = col(df, ["lum_parallax_lsun"]).to_numpy()

    any_lum  = np.isfinite(lum_fl) | np.isfinite(lum_rt) | np.isfinite(lum_plx)
    good_t   = np.isfinite(teff) & any_lum
    if good_t.sum() == 0:
        print("  skipped 06_hrd_sources (no data)")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    for lum_arr, label, color, marker in [
        (lum_plx, "Parallax",    "#e63946", "^"),
        (lum_rt,  "R·Teff",      "#457b9d", "s"),
        (lum_fl,  "FLAME",       "#2dc653", "o"),
    ]:
        g = good_t & np.isfinite(lum_arr)
        if g.sum() == 0:
            continue
        logl = np.log10(lum_arr[g])
        ax.scatter(teff[g], logl, s=20, alpha=0.7,
                   color=color, marker=marker, label=f"{label}  (N={g.sum()})", zorder=3)

    ax.set_xlabel(r"$T_\mathrm{eff}$  [K]", fontsize=12)
    ax.set_ylabel(r"$\log_{10}(L/L_\odot)$", fontsize=12)
    #ax.set_title("HR diagram – three independent luminosity estimates", fontsize=12)
    ax.invert_xaxis()
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    savefig(fig, outdir / "06_hrd_sources.png")


# ---------------------------------------------------------------------------
# 07 – Kiel diagram (logg vs Teff), coloured by metallicity
# ---------------------------------------------------------------------------

def plot_kiel(df, outdir):
    teff  = col(df, ["teff_best"]).to_numpy()
    logg  = col(df, ["logg_best"]).to_numpy()
    metal = col(df, ["metallicity_best"]).to_numpy()
    good  = finite(teff, logg)
    if good.sum() == 0:
        print("  skipped 07_kiel (no data)")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    good_m = good & np.isfinite(metal)
    good_nm = good & ~np.isfinite(metal)
    ax.scatter(teff[good_nm], logg[good_nm], s=20, color=C_ALL, alpha=0.5,
               label=f"No [M/H]  (N={good_nm.sum()})", zorder=2)
    if good_m.sum() > 0:
        sc = ax.scatter(teff[good_m], logg[good_m], c=metal[good_m],
                        s=26, alpha=0.9, cmap="RdYlBu_r",
                        vmin=np.nanpercentile(metal[good_m], 5),
                        vmax=np.nanpercentile(metal[good_m], 95),
                        zorder=3)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("[M/H]  [dex]", fontsize=10)
        ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel(r"$T_\mathrm{eff}$  [K]", fontsize=12)
    ax.set_ylabel(r"$\log\,g$", fontsize=12)
    #ax.set_title("Kiel diagram – Roman bulge calibration sample", fontsize=12)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(alpha=0.2)
    savefig(fig, outdir / "07_kiel_diagram.png")


# ---------------------------------------------------------------------------
# 08 – Observed CMD
# ---------------------------------------------------------------------------

def plot_cmd_observed(df, outdir):
    color = col(df, ["j_minus_ks_best"]).to_numpy()
    mag   = col(df, ["ks_mag_best"]).to_numpy()
    ejks  = col(df, ["ext_e_jks"]).to_numpy()
    good  = finite(color, mag)
    if good.sum() == 0:
        print("  skipped 08_cmd_observed (no data)")
        return

    fig, ax = plt.subplots(figsize=(7, 8))
    good_e = good & np.isfinite(ejks)
    good_ne = good & ~np.isfinite(ejks)
    ax.scatter(color[good_ne], mag[good_ne], s=18, color=C_ALL, alpha=0.5,
               label=f"No reddening  (N={good_ne.sum()})", zorder=2)
    if good_e.sum() > 0:
        sc = ax.scatter(color[good_e], mag[good_e], c=ejks[good_e],
                        s=22, alpha=0.85, cmap="hot_r", zorder=3)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("E(J-Ks)", fontsize=10)
    ax.set_xlabel("J – Ks  [mag]", fontsize=12)
    ax.set_ylabel("Ks  [mag]", fontsize=12)
    #ax.set_title("Observed CMD – coloured by reddening", fontsize=12)
    ax.invert_yaxis()
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    savefig(fig, outdir / "08_cmd_observed.png")


# ---------------------------------------------------------------------------
# 09 – Dereddened CMD
# ---------------------------------------------------------------------------

def plot_cmd_dereddened(df, outdir):
    color = col(df, ["j0_minus_ks0"]).to_numpy()
    mag   = col(df, ["ks0"]).to_numpy()
    metal = col(df, ["metallicity_best"]).to_numpy()
    good  = finite(color, mag)
    if good.sum() == 0:
        print("  skipped 09_cmd_dereddened (no data)")
        return

    fig, ax = plt.subplots(figsize=(7, 8))
    good_m = good & np.isfinite(metal)
    good_nm = good & ~np.isfinite(metal)
    ax.scatter(color[good_nm], mag[good_nm], s=18, color=C_ALL, alpha=0.5,
               label=f"No [M/H]  (N={good_nm.sum()})", zorder=2)
    if good_m.sum() > 0:
        sc = ax.scatter(color[good_m], mag[good_m], c=metal[good_m],
                        s=24, alpha=0.9, cmap="RdYlBu_r",
                        vmin=np.nanpercentile(metal[good_m], 5),
                        vmax=np.nanpercentile(metal[good_m], 95),
                        zorder=3)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("[M/H]  [dex]", fontsize=10)
    ax.set_xlabel(r"$(J-K_s)_0$  [mag]", fontsize=12)
    ax.set_ylabel(r"$K_{s,0}$  [mag]", fontsize=12)
    #ax.set_title("Dereddened CMD – coloured by [M/H]", fontsize=12)
    ax.invert_yaxis()
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    savefig(fig, outdir / "09_cmd_dereddened.png")


# ---------------------------------------------------------------------------
# 10 – Overall metallicity distribution
# ---------------------------------------------------------------------------

def plot_metallicity_hist(df, outdir):
    metal_cols = [
        ("metallicity_best", "Best",    C_READY),
        ("mh_astra",         "ASTRA",   "#4895ef"),
        ("feh_bdbs",         "BDBS",    "#f77f00"),
        ("mh_gspspec",       "GSP-Spec","#7b2d8b"),
        ("mh_xgboost",       "XGBoost", "#2dc653"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-2.5, 0.8, 40)
    for col_name, label, color in metal_cols:
        vals = col(df, [col_name]).to_numpy()
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, histtype="step", lw=2,
                color=color, label=f"{label}  (N={len(vals)})", alpha=0.85)
    ax.set_xlabel("[M/H] or [Fe/H]  [dex]", fontsize=12)
    ax.set_ylabel("Number of stars", fontsize=12)
    #ax.set_title("Metallicity distributions – all available sources", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)
    savefig(fig, outdir / "10_metallicity_hist.png")


# ---------------------------------------------------------------------------
# 11 – Metallicity per sightline (strip plot)
# ---------------------------------------------------------------------------

def plot_metallicity_by_sightline(df, outdir):
    sid   = df["sightline_id"].fillna("").astype(str) if "sightline_id" in df.columns \
            else pd.Series("", index=df.index)
    metal = col(df, ["metallicity_best"]).to_numpy()
    good  = (sid != "") & np.isfinite(metal)
    if good.sum() == 0:
        print("  skipped 11_metallicity_by_sightline (no data)")
        return

    # Keep top sightlines by count
    top = sid[good].value_counts().head(15).index.tolist()
    sub_mask = good & sid.isin(top)
    if sub_mask.sum() == 0:
        return

    sids_ordered = sid[sub_mask].value_counts().index.tolist()
    idx_map = {s: i for i, s in enumerate(sids_ordered)}

    fig, ax = plt.subplots(figsize=(11, 6))
    jitter = np.random.default_rng(42).uniform(-0.25, 0.25, sub_mask.sum())
    x_pos  = np.array([idx_map[s] for s in sid[sub_mask]])
    sc = ax.scatter(x_pos + jitter, metal[sub_mask],
                    c=metal[sub_mask], cmap="RdYlBu_r", s=20, alpha=0.7,
                    vmin=-1.5, vmax=0.5, zorder=3)
    # Median per sightline
    for i, s in enumerate(sids_ordered):
        m = sid[sub_mask] == s
        med = np.nanmedian(metal[sub_mask][m])
        ax.hlines(med, i - 0.4, i + 0.4, color="k", lw=2, zorder=4)

    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("[M/H]  [dex]", fontsize=10)

    # Short labels: strip the tile prefix
    labels = [s.split("|")[-2] + "|" + s.split("|")[-1]
              if "|" in s else s[:18] for s in sids_ordered]
    ax.set_xticks(range(len(sids_ordered)))
    ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=7)
    ax.set_ylabel("[M/H]  [dex]", fontsize=12)
    #ax.set_title("Metallicity distribution per sightline  (horizontal bar = median)", fontsize=12)
    ax.grid(axis="y", alpha=0.2)
    savefig(fig, outdir / "11_metallicity_by_sightline.png")


# ---------------------------------------------------------------------------
# 12 – Metallicity source comparison
# ---------------------------------------------------------------------------

def plot_metallicity_sources(df, outdir):
    pairs = [
        ("mh_astra",    "mh_xgboost",  "ASTRA [M/H]",   "XGBoost [M/H]"),
        ("mh_astra",    "mh_gspspec",  "ASTRA [M/H]",   "GSP-Spec [M/H]"),
        ("mh_astra",    "feh_bdbs",    "ASTRA [M/H]",   "BDBS [Fe/H]"),
    ]
    valid_pairs = [(a, b, la, lb) for a, b, la, lb in pairs
                   if col(df, [a]).notna().any() and col(df, [b]).notna().any()]
    if not valid_pairs:
        print("  skipped 12_metallicity_sources (no pairs)")
        return

    n = len(valid_pairs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (c1, c2, l1, l2) in zip(axes, valid_pairs):
        v1 = col(df, [c1]).to_numpy()
        v2 = col(df, [c2]).to_numpy()
        good = finite(v1, v2)
        if good.sum() == 0:
            continue
        lim = [min(v1[good].min(), v2[good].min()) - 0.1,
               max(v1[good].max(), v2[good].max()) + 0.1]
        ax.plot(lim, lim, "k--", lw=1, alpha=0.5, zorder=1)
        ax.scatter(v1[good], v2[good], s=16, alpha=0.6, color=C_PHOT, zorder=2)
        diff = v2[good] - v1[good]
        ax.set_xlabel(l1 + "  [dex]", fontsize=10)
        ax.set_ylabel(l2 + "  [dex]", fontsize=10)
        #ax.set_title(f"N={good.sum()}  |  Δ={np.nanmedian(diff):+.3f}  σ={np.nanstd(diff):.3f}",
        #             fontsize=9)
        ax.grid(alpha=0.2)
    fig.suptitle("Metallicity source comparison", fontsize=12, y=1.01)
    savefig(fig, outdir / "12_metallicity_sources.png")


# ---------------------------------------------------------------------------
# 13 – Parameter completeness bar chart
# ---------------------------------------------------------------------------

def plot_parameter_completeness(df, outdir):
    params = [
        ("Teff (best)",         "teff_best"),
        ("Teff – ASTRA",        "teff_astra"),
        ("Teff – GSP-Spec",     "teff_gspspec"),
        ("Teff – Gaia AP",      "teff_gaia_ap"),
        ("Teff – ZGR",          "teff_zgr"),
        ("Teff – XGBoost",      "teff_xgboost"),
        ("logg (best)",         "logg_best"),
        ("[M/H] (best)",        "metallicity_best"),
        ("[M/H] – ASTRA",       "mh_astra"),
        ("[Fe/H] – BDBS",       "feh_bdbs"),
        ("[M/H] – XGBoost",     "mh_xgboost"),
        ("Radius (best)",       "radius_best_rsun"),
        ("Radius – FLAME",      "radius_flame_rsun"),
        ("Lum (best)",          "luminosity_best_lsun"),
        ("Lum – FLAME",         "lum_flame_lsun"),
        ("Lum – R·Teff",        "lum_r_teff_lsun"),
        ("Lum – parallax",      "lum_parallax_lsun"),
        ("Mass – FLAME",        "mass_flame_msun"),
        ("Age – FLAME",         "age_flame_gyr"),
        ("E(J-Ks)",             "ext_e_jks"),
        ("J0 (dereddened)",     "j0"),
        ("Parallax",            "parallax"),
    ]
    n_total = len(df)
    labels, counts, colors = [], [], []
    for label, colname in params:
        if colname in df.columns:
            n = int(np.isfinite(pd.to_numeric(df[colname], errors="coerce")).sum())
        else:
            n = 0
        labels.append(label)
        counts.append(n)
        frac = n / n_total if n_total > 0 else 0
        colors.append(plt.cm.RdYlGn(frac))

    fig, ax = plt.subplots(figsize=(8, 9))
    y = np.arange(len(labels))
    bars = ax.barh(y, counts, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Number of stars", fontsize=11)
    #ax.set_title(f"Parameter coverage  (total N = {n_total})", fontsize=12)
    ax.axvline(n_total, color="k", lw=1, ls="--", alpha=0.4)
    for bar, n in zip(bars, counts):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                str(n), va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    ax.invert_yaxis()
    savefig(fig, outdir / "13_parameter_completeness.png")


# ---------------------------------------------------------------------------
# 14 – FLAME age vs mass
# ---------------------------------------------------------------------------

def plot_age_mass(df, outdir):
    age   = col(df, ["age_flame_gyr"]).to_numpy()
    mass  = col(df, ["mass_flame_msun"]).to_numpy()
    metal = col(df, ["metallicity_best"]).to_numpy()
    good  = finite(age, mass)
    if good.sum() == 0:
        print("  skipped 14_age_mass (no FLAME data)")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    good_m = good & np.isfinite(metal)
    ax.scatter(mass[good & ~good_m], age[good & ~good_m],
               s=22, color=C_ALL, alpha=0.5, label="No [M/H]", zorder=2)
    if good_m.sum() > 0:
        sc = ax.scatter(mass[good_m], age[good_m], c=metal[good_m],
                        s=26, alpha=0.9, cmap="RdYlBu_r",
                        vmin=-1.5, vmax=0.5, zorder=3)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("[M/H]  [dex]", fontsize=10)
    ax.set_xlabel(r"Mass  $[M_\odot]$", fontsize=12)
    ax.set_ylabel("Age  [Gyr]", fontsize=12)
    #ax.set_title(f"FLAME mass vs age  (N={good.sum()})", fontsize=12)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    savefig(fig, outdir / "14_age_mass_flame.png")


# ---------------------------------------------------------------------------
# 15 – Luminosity comparison (FLAME vs R·Teff vs parallax)
# ---------------------------------------------------------------------------

def plot_lum_comparison(df, outdir):
    lum_fl  = col(df, ["lum_flame_lsun"]).to_numpy()
    lum_rt  = col(df, ["lum_r_teff_lsun"]).to_numpy()
    lum_plx = col(df, ["lum_parallax_lsun"]).to_numpy()

    pairs = [
        (lum_fl,  lum_rt,  r"FLAME  $L/L_\odot$",     r"R·Teff  $L/L_\odot$",   "15a"),
        (lum_fl,  lum_plx, r"FLAME  $L/L_\odot$",     r"Parallax  $L/L_\odot$",  "15b"),
        (lum_rt,  lum_plx, r"R·Teff  $L/L_\odot$",    r"Parallax  $L/L_\odot$",  "15c"),
    ]
    valid = [(a, b, la, lb, tag) for a, b, la, lb, tag in pairs
             if np.isfinite(a).any() and np.isfinite(b).any()]
    if not valid:
        print("  skipped 15_lum_comparison (no data)")
        return

    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (a, b, la, lb, _) in zip(axes, valid):
        good = finite(a, b) & (a > 0) & (b > 0)
        if good.sum() == 0:
            continue
        la_arr = np.log10(a[good])
        lb_arr = np.log10(b[good])
        lim = [min(la_arr.min(), lb_arr.min()) - 0.1,
               max(la_arr.max(), lb_arr.max()) + 0.1]
        ax.plot(lim, lim, "k--", lw=1, alpha=0.4)
        ax.scatter(la_arr, lb_arr, s=16, alpha=0.6, color=C_PHOT, zorder=2)
        diff = lb_arr - la_arr
        ax.set_xlabel(f"log({la})", fontsize=10)
        ax.set_ylabel(f"log({lb})", fontsize=10)
        #ax.set_title(f"N={good.sum()}  Δ={np.nanmedian(diff):+.3f}  σ={np.nanstd(diff):.3f}",
        #             fontsize=9)
        ax.grid(alpha=0.2)
    fig.suptitle("Luminosity estimate comparison", fontsize=12, y=1.01)
    savefig(fig, outdir / "15_lum_comparison.png")


# ---------------------------------------------------------------------------
# 16 – Reddening histogram
# ---------------------------------------------------------------------------

def plot_reddening_hist(df, outdir):
    ejks = col(df, ["ext_e_jks"]).to_numpy()
    ejks = ejks[np.isfinite(ejks)]
    if len(ejks) == 0:
        print("  skipped 16_reddening_hist (no data)")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(ejks, bins=30, color=C_PHOT, edgecolor="white", alpha=0.85)
    ax.axvline(np.median(ejks), color="k", lw=1.5, ls="--",
               label=f"Median = {np.median(ejks):.3f}")
    ax.set_xlabel("E(J – Ks)  [mag]", fontsize=12)
    ax.set_ylabel("Number of stars", fontsize=12)
    #ax.set_title(f"Reddening distribution  (N={len(ejks)})", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=0.2)
    savefig(fig, outdir / "16_reddening_hist.png")


# ---------------------------------------------------------------------------
# 17 – Reddening vs Galactic latitude
# ---------------------------------------------------------------------------

def plot_reddening_vs_b(df, outdir):
    b    = col(df, ["b", "glat"]).to_numpy()
    ejks = col(df, ["ext_e_jks"]).to_numpy()
    good = finite(b, ejks)
    if good.sum() == 0:
        print("  skipped 17_reddening_vs_b (no data)")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(b[good], ejks[good], s=18, alpha=0.7, color=C_PHOT)
    ax.set_xlabel("Galactic latitude  b  [deg]", fontsize=12)
    ax.set_ylabel("E(J – Ks)  [mag]", fontsize=12)
    #ax.set_title("Reddening vs Galactic latitude", fontsize=12)
    ax.grid(alpha=0.2)
    savefig(fig, outdir / "17_reddening_vs_b.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading {INPUT_FITS} ...")
    df = load(INPUT_FITS)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")
    print(f"Writing plots to {OUTDIR}/")

    plot_sky_footprint(df, OUTDIR)
    plot_sky_colored(df, OUTDIR, ["ext_e_jks"],        "02_sky_reddening.png",   "Sky map – E(J-Ks)",   "E(J-Ks)  [mag]",   cmap="hot_r")
    plot_sky_colored(df, OUTDIR, ["metallicity_best"],  "03_sky_metallicity.png", "Sky map – [M/H]",     "[M/H]  [dex]",     cmap="RdYlBu_r")
    plot_sky_colored(df, OUTDIR, ["teff_best"],         "04_sky_teff.png",        "Sky map – Teff",      "Teff  [K]",        cmap="RdYlBu_r")
    plot_hrd(df, OUTDIR)
    plot_hrd_lum_sources(df, OUTDIR)
    plot_kiel(df, OUTDIR)
    plot_cmd_observed(df, OUTDIR)
    plot_cmd_dereddened(df, OUTDIR)
    plot_metallicity_hist(df, OUTDIR)
    plot_metallicity_by_sightline(df, OUTDIR)
    plot_metallicity_sources(df, OUTDIR)
    plot_parameter_completeness(df, OUTDIR)
    plot_age_mass(df, OUTDIR)
    plot_lum_comparison(df, OUTDIR)
    plot_reddening_hist(df, OUTDIR)
    plot_reddening_vs_b(df, OUTDIR)

    print(f"\nDone. Plots in {OUTDIR}/")


if __name__ == "__main__":
    main()
