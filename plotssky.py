#!/usr/bin/env python3
"""
Data coverage sky map — publication quality for proposal
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
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import warnings
from astropy.io.fits.verify import VerifyWarning
from astropy.units import UnitsWarning

warnings.filterwarnings("ignore", category=VerifyWarning)
warnings.filterwarnings("ignore", category=UnitsWarning)

for candidate in ["roman_master_bdbs.fits", "roman_master.fits"]:
    INPUT_FITS = pathlib.Path(candidate)
    if INPUT_FITS.exists():
        break

OUTDIR = pathlib.Path("plots")
OUTDIR.mkdir(exist_ok=True)

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


def bulge_l(l):
    return ((np.asarray(l, dtype=float) + 180.0) % 360.0) - 180.0


def tile_outline(l0, b0):
    center    = SkyCoord(l=l0*u.deg, b=b0*u.deg, frame="galactic")
    off_frame = SkyOffsetFrame(origin=center)
    hw, hh    = TILE_W_DEG / 2, TILE_H_DEG / 2
    corners   = np.array([[-hw,-hh],[hw,-hh],[hw,hh],[-hw,hh],[-hw,-hh]])
    th = np.deg2rad(TILE_PA_DEG)
    dx = corners[:,0]*np.cos(th) - corners[:,1]*np.sin(th)
    dy = corners[:,0]*np.sin(th) + corners[:,1]*np.cos(th)
    pts = SkyCoord(lon=dx*u.deg, lat=dy*u.deg,
                   frame=off_frame).transform_to("galactic")
    return bulge_l(pts.l.deg), pts.b.deg


def load(path):
    tbl = Table.read(path)
    scalar = [n for n in tbl.colnames if len(getattr(tbl[n], "shape", ())) <= 1]
    return tbl[scalar].to_pandas()


def finite(df, *cols):
    result = pd.Series(False, index=df.index)
    for c in cols:
        if c in df.columns:
            result |= np.isfinite(pd.to_numeric(df[c], errors="coerce"))
    return result


print(f"Reading {INPUT_FITS} ...")
df = load(INPUT_FITS)
print(f"  {len(df):,} rows")

l = bulge_l(pd.to_numeric(df.get("l", pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy())
b = pd.to_numeric(df.get("b",  pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy()

# SHAPE
has_astra = finite(df, "mh_astra", "feh_astra")
has_gsp   = (~has_astra) & finite(df, "mh_gspspec", "feh_gspspec", "teff_gspspec", "teff_gaia_ap", "feh_gaia_ap")
has_xgb   = (~has_astra) & (~has_gsp) & finite(df, "mh_xgboost", "teff_xgboost")
has_none  = (~has_astra) & (~has_gsp) & (~has_xgb)

# FILL
has_dered = finite(df, "j0", "h0", "ks0") & finite(df, "ext_e_jks")
has_nir   = finite(df, "j_mag_best", "h_mag_best", "ks_mag_best") & (~has_dered)
has_nonir = ~finite(df, "j_mag_best", "h_mag_best", "ks_mag_best")

# EDGE: BDBS
bdbs_cols = [c for c in df.columns if "bdbs" in c.lower() and
             any(x in c.lower() for x in ["g0","r0","i0","gmag","rmag","imag","mag_g","mag_r","mag_i"])]
has_bdbs = pd.Series(False, index=df.index)
for c in bdbs_cols:
    has_bdbs |= finite(df, c)
if not has_bdbs.any():
    has_bdbs = finite(df, "g0_bdbs", "mag_g_bdbs", "gmag_bdbs")

# SIZE
has_age = finite(df, "age_flame_gyr")

print(f"  ASTRA: {has_astra.sum()}  GSP/GaiaAP: {has_gsp.sum()}  XGBoost: {has_xgb.sum()}  None: {has_none.sum()}")
print(f"  Dereddened: {has_dered.sum()}  NIR only: {has_nir.sum()}  No NIR: {has_nonir.sum()}")
print(f"  BDBS: {has_bdbs.sum()}  FLAME age: {has_age.sum()}")

# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.linewidth":  0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top":       True,
    "ytick.right":     True,
})

fig, ax = plt.subplots(figsize=(10, 5.8))

for l0, b0 in GBTDS_CENTERS:
    tl, tb = tile_outline(l0, b0)
    ax.plot(tl, tb, color="black", lw=1.1, zorder=10, alpha=0.55)

# Encoding tables
FILL   = {"dered": "#2980b9", "nir": "#95a5a6", "none": "white"}
EDGE   = {"bdbs": ("#e67e22", 2.0), "no": ("black", 0.5)}
SIZE   = {True: 80, False: 32}
MARKER = {"astra": "o", "gsp": "^", "xgb": "s", "none": "x"}
ZORDER = {"none": 2, "xgb": 3, "gsp": 4, "astra": 6}

shape_masks = {"none": has_none, "xgb": has_xgb, "gsp": has_gsp, "astra": has_astra}
fill_masks  = {"none": has_nonir, "nir": has_nir, "dered": has_dered}

for sk, smask in shape_masks.items():
    for fk, fmask in fill_masks.items():
        for bk, (ec, elw) in EDGE.items():
            bmask = has_bdbs if bk == "bdbs" else ~has_bdbs
            for age_flag in [True, False]:
                amask = has_age if age_flag else ~has_age
                mask  = (smask & fmask & bmask & amask).to_numpy()
                if not mask.any():
                    continue
                ax.scatter(
                    l[mask], b[mask],
                    marker=MARKER[sk], s=SIZE[age_flag],
                    facecolors=FILL[fk], edgecolors=ec,
                    linewidths=elw, alpha=0.9, zorder=ZORDER[sk],
                )

ax.invert_xaxis()
ax.set_xlabel(r"Galactic longitude $\ell$ [deg]")
ax.set_ylabel(r"Galactic latitude $b$ [deg]")

# ---------------------------------------------------------------------------
# Legend — plain, journal style
# ---------------------------------------------------------------------------
mkw = dict(ls="none", markersize=7, markeredgewidth=0.6)
C   = "#2980b9"   # representative filled colour for legend entries

def header(text):
    return mpatches.Patch(visible=False, label=text)

def sp():
    return mpatches.Patch(visible=False, label=" ")

legend_handles = [
    header("Spectroscopic source (shape)"),
    mlines.Line2D([], [], marker="o", color=C, markeredgecolor="black", label="ASTRA",              **mkw),
    mlines.Line2D([], [], marker="^", color=C, markeredgecolor="black", label="GSP-Spec / Gaia-AP", **mkw),
    mlines.Line2D([], [], marker="s", color=C, markeredgecolor="black", label="XGBoost (photometric)", **mkw),
    mlines.Line2D([], [], marker="x", color="grey", markeredgecolor="grey", markeredgewidth=1.2,
                  label="No stellar parameters", ls="none", markersize=7),
    sp(),
    header("NIR photometry (fill)"),
    mlines.Line2D([], [], marker="o", color=C,        markeredgecolor="black", label=r"Dereddened ($J_0 H_0 K_{s,0}$)", **mkw),
    mlines.Line2D([], [], marker="o", color="#95a5a6", markeredgecolor="black", label="NIR only (no reddening)",          **mkw),
    mlines.Line2D([], [], marker="o", color="white",   markeredgecolor="black", label="No NIR photometry",                **mkw),
    sp(),
    header("BDBS optical (edge)"),
    mlines.Line2D([], [], marker="o", color=C, markeredgecolor="#e67e22", markeredgewidth=2.0,
                  label="BDBS data", ls="none", markersize=7),
    mlines.Line2D([], [], marker="o", color=C, markeredgecolor="black",   markeredgewidth=0.5,
                  label="No BDBS",   ls="none", markersize=7),
    sp(),
    header("FLAME age (size)"),
    mlines.Line2D([], [], marker="o", color=C, markeredgecolor="black", markeredgewidth=0.6,
                  label="Age available", ls="none", markersize=10),
    mlines.Line2D([], [], marker="o", color=C, markeredgecolor="black", markeredgewidth=0.6,
                  label="No age",        ls="none", markersize=5),
]

leg = ax.legend(
    handles=legend_handles,
    loc="upper left",
    fontsize=8,
    frameon=True,
    framealpha=1.0,
    edgecolor="0.6",
    handlelength=1.2,
    labelspacing=0.3,
    borderpad=0.7,
)

for text in leg.get_texts():
    t = text.get_text()
    if t.endswith(("(shape)", "(fill)", "(edge)", "(size)")):
        text.set_fontweight("bold")
        text.set_fontsize(7.5)

ax.text(0.99, 0.03, f"$N = {len(df):,}$",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10)

plt.tight_layout()
for ext in ("pdf", "png"):
    p = OUTDIR / f"sky_data_coverage.{ext}"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"Saved: {p}")
plt.close(fig)