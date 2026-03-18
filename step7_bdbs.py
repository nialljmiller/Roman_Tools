#!/usr/bin/env python3
"""
Step 7 – BDBS derived quantities and updated plots
====================================================
Reads roman_master.fits after you have cross-matched the BDBS CSV onto it
in TOPCAT.  Expects BDBS columns with the prefix bdbs_ (TOPCAT default for
a left-join from a file named bdbs_*.csv or similar).

Quantities derived
------------------
Dereddened BDBS magnitudes
    g0_bdbs, r0_bdbs, i0_bdbs, z0_bdbs, y0_bdbs, u0_bdbs
    (missing-data values of 99.999 are masked to NaN)

Dereddened BDBS colours
    gr0_bdbs, gi0_bdbs, ri0_bdbs, gz0_bdbs, zy0_bdbs

Teff from (g-i)_0  [Casagrande et al. 2010 giant calibration]
    teff_bdbs_opt

Photometric [Fe/H] from RGB locus offset
    feh_empirical_bdbs
    Method: uses the dereddened (g-i)_0 colour at known absolute
    Ks magnitude (from Bailer-Jones r_med_photogeo_pc + ks0) and
    compares to the Zoccali et al. (2003) / Nataf et al. RGB ridge
    line grid parameterised by metallicity.

Outputs
-------
roman_master_bdbs.fits / .csv   – updated master with all new columns
Updated plots in plots/ :
    03_sky_metallicity.png       – now includes feh_empirical_bdbs where available
    10_metallicity_hist.png      – BDBS photometric [Fe/H] added
    11_metallicity_by_sightline.png
    12_metallicity_sources.png   – BDBS phot vs ASTRA/spectroscopic
    18_bdbs_cmd.png              – BDBS optical (g-i)0 vs g0 CMD
    19_bdbs_teff_comparison.png  – teff_bdbs_opt vs teff_astra
    20_bdbs_feh_comparison.png   – feh_empirical_bdbs vs spectroscopic [M/H]
    21_sky_feh_bdbs.png          – sky map coloured by feh_empirical_bdbs

Input  : roman_master.fits  (TOPCAT-merged with BDBS)
Outputs: roman_master_bdbs.fits / .csv  +  plots/
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
import warnings
from astropy.io.fits.verify import VerifyWarning
from astropy.units import UnitsWarning

warnings.filterwarnings("ignore", category=VerifyWarning)
warnings.filterwarnings("ignore", category=UnitsWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_FITS   = pathlib.Path("roman_master.fits")
OUTPUT_FITS  = pathlib.Path("roman_master_bdbs.fits")
OUTPUT_CSV   = pathlib.Path("roman_master_bdbs.csv")
OUTDIR       = pathlib.Path("plots")

MISSING_MAG = 99.0     # BDBS uses 99.999 for missing; mask anything >= this

# ---------------------------------------------------------------------------
# Roman tile geometry (for sky plots)
# ---------------------------------------------------------------------------
TILE_W_DEG  = (45.0 * u.arcmin).to(u.deg).value
TILE_H_DEG  = (23.0 * u.arcmin).to(u.deg).value
#TILE_W_DEG  = (49.4 * u.arcmin).to(u.deg).value
#TILE_H_DEG  = (25.3 * u.arcmin).to(u.deg).value
TILE_PA_DEG = 90.6
GBTDS_CENTERS = np.array([
    [-0.417948, -1.200], [-0.008974, -1.200], [ 0.400000, -1.200],
    [ 0.808974, -1.200], [ 1.217948, -1.200], [ 0.000000, -0.1250],
], dtype=float)

C_ALL   = "#adb5bd"
C_PHOT  = "#4895ef"
C_READY = "#f72585"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_col(columns, candidates):
    lower = {str(c).lower(): str(c) for c in columns}
    for cand in candidates:
        key = str(cand).lower()
        for suffix in ["", "_1", "_bdbs", "_bdbs_1"]:
            if key + suffix in lower:
                return lower[key + suffix]
    return None


def get(df, candidates, default=np.nan):
    c = find_col(df.columns, candidates)
    if c is None:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[c], errors="coerce")


def mask_missing(series, threshold=MISSING_MAG):
    """Replace values >= threshold (BDBS missing-data sentinel) with NaN."""
    s = pd.to_numeric(series, errors="coerce").copy()
    s[s >= threshold] = np.nan
    return s


def as_float(s):
    return pd.to_numeric(s, errors="coerce")


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
    center    = SkyCoord(l=l0 * u.deg, b=b0 * u.deg, frame="galactic")
    off_frame = SkyOffsetFrame(origin=center)
    hw, hh    = TILE_W_DEG / 2, TILE_H_DEG / 2
    corners   = np.array([[-hw,-hh],[hw,-hh],[hw,hh],[-hw,hh],[-hw,-hh]])
    theta     = np.deg2rad(TILE_PA_DEG)
    dx = corners[:,0]*np.cos(theta) - corners[:,1]*np.sin(theta)
    dy = corners[:,0]*np.sin(theta) + corners[:,1]*np.cos(theta)
    outline = SkyCoord(lon=dx*u.deg, lat=dy*u.deg,
                       frame=off_frame).transform_to("galactic")
    return outline.l.deg, outline.b.deg


def bulge_l(l_deg):
    return ((np.asarray(l_deg) + 180.0) % 360.0) - 180.0


def sky_ax_setup(ax):
    ax.set_xlabel("Galactic longitude  l  [deg]", fontsize=11)
    ax.set_ylabel("Galactic latitude  b  [deg]", fontsize=11)
    ax.invert_xaxis()
    ax.grid(alpha=0.2, lw=0.5)
    for l0, b0 in GBTDS_CENTERS:
        tl, tb = tile_outline_lb(l0, b0)
        ax.plot(bulge_l(tl), tb, color="k", lw=1.2, zorder=5)


def load(fits_path):
    tbl = Table.read(fits_path)
    scalar = [n for n in tbl.colnames if len(getattr(tbl[n], "shape", ())) <= 1]
    return tbl[scalar].to_pandas()

# ---------------------------------------------------------------------------
# Empirical photometric [Fe/H] calibration
#
# Method:
#   1. Use the subset of stars that have BOTH a measured (g-i)_0 from BDBS
#      AND a spectroscopic metallicity (metallicity_best from ASTRA/GSP-Spec).
#   2. Fit a linear model:  [Fe/H]_spec = a + b*(g-i)_0 + c*M_Ks
#      via ordinary least squares using numpy.  M_Ks is the absolute Ks
#      magnitude from ks0 + Bailer-Jones photogeometric distance.
#   3. Apply the fitted coefficients to all stars that have (g-i)_0 and M_Ks
#      but lack a spectroscopic [M/H], to produce feh_empirical_bdbs.
#   4. Report the fit coefficients, N_calibrators, and RMS residual so the
#      quality of the calibration is transparent.
#
# This is explicitly an empirical fit to THIS sample, not a published
# calibration.  It is labelled accordingly in all outputs.
# ---------------------------------------------------------------------------

def fit_empirical_feh(gi0, M_Ks, feh_spec):
    """
    Fit  [Fe/H]_spec = a + b*(g-i)_0 + c*M_Ks  by OLS.

    Parameters
    ----------
    gi0, M_Ks, feh_spec : 1-D numpy arrays, all finite, same length.

    Returns
    -------
    coeffs : (a, b, c)
    rms    : float, RMS of residuals in dex
    """
    A = np.column_stack([np.ones(len(gi0)), gi0, M_Ks])
    coeffs, _, _, _ = np.linalg.lstsq(A, feh_spec, rcond=None)
    residuals = feh_spec - (A @ coeffs)
    rms = float(np.sqrt(np.mean(residuals**2)))
    return tuple(coeffs), rms


def apply_empirical_feh(gi0, M_Ks, coeffs):
    """Apply fitted coefficients to predict [Fe/H]."""
    a, b, c = coeffs
    feh = a + b * gi0 + c * M_Ks
    bad = ~np.isfinite(gi0) | ~np.isfinite(M_Ks)
    feh[bad] = np.nan
    return feh


# ---------------------------------------------------------------------------
# Main derivation function
# ---------------------------------------------------------------------------

def derive_bdbs_quantities(df):
    """
    Compute all BDBS-derived quantities and add them to df.
    Returns the augmented DataFrame.
    """
    out = df.copy()

    # ------------------------------------------------------------------
    # Identify BDBS columns.  TOPCAT will have added a suffix; we scan
    # for all plausible names.
    # ------------------------------------------------------------------
    band_map = {
        "u": (["umag_bdbs", "umag"],       ["umag_err_bdbs", "u_err"],  ["u_ext_bdbs", "u_ext"]),
        "g": (["gmag_bdbs", "gmag"],       ["gmag_err_bdbs", "g_err"],  ["g_ext_bdbs", "g_ext"]),
        "r": (["rmag_bdbs", "rmag"],       ["rmag_err_bdbs", "r_err"],  ["r_ext_bdbs", "r_ext"]),
        "i": (["imag_bdbs", "imag"],       ["imag_err_bdbs", "i_err"],  ["i_ext_bdbs", "i_ext"]),
        "z": (["zmag_bdbs", "zmag"],       ["zmag_err_bdbs", "z_err"],  ["z_ext_bdbs", "z_ext"]),
        "y": (["ymag_bdbs", "ymag"],       ["ymag_err_bdbs", "y_err"],  ["y_ext_bdbs", "y_ext"]),
    }

    print("  Dereddening BDBS magnitudes ...")
    n_bdbs = 0
    for band, (mag_cands, err_cands, ext_cands) in band_map.items():
        mag_col = find_col(df.columns, mag_cands)
        err_col = find_col(df.columns, err_cands)
        ext_col = find_col(df.columns, ext_cands)

        if mag_col is None:
            continue

        mag = mask_missing(df[mag_col])
        out[f"mag_{band}_bdbs"] = mag

        if err_col is not None:
            err = mask_missing(df[err_col])
            out[f"e_mag_{band}_bdbs"] = err

        if ext_col is not None:
            ext = as_float(df[ext_col])
            out[f"{band}0_bdbs"] = mag - ext
        else:
            out[f"{band}0_bdbs"] = mag   # no extinction correction available

        if band == "g":
            n_bdbs = int(np.isfinite(mag).sum())

    print(f"    BDBS stars with g magnitude: {n_bdbs}")

    # ------------------------------------------------------------------
    # Dereddened colours
    # ------------------------------------------------------------------
    print("  Computing BDBS dereddened colours ...")
    g0 = as_float(out.get("g0_bdbs", pd.Series(np.nan, index=out.index)))
    r0 = as_float(out.get("r0_bdbs", pd.Series(np.nan, index=out.index)))
    i0 = as_float(out.get("i0_bdbs", pd.Series(np.nan, index=out.index)))
    z0 = as_float(out.get("z0_bdbs", pd.Series(np.nan, index=out.index)))
    y0 = as_float(out.get("y0_bdbs", pd.Series(np.nan, index=out.index)))

    out["gr0_bdbs"] = g0 - r0
    out["gi0_bdbs"] = g0 - i0
    out["ri0_bdbs"] = r0 - i0
    out["gz0_bdbs"] = g0 - z0
    out["zy0_bdbs"] = z0 - y0

    # ------------------------------------------------------------------
    # Empirical photometric [Fe/H] calibrated against this sample's
    # own spectroscopic metallicities
    # ------------------------------------------------------------------
    print("  Fitting empirical [Fe/H] calibration from (g-i)_0 and M_Ks ...")

    # Distance: prefer photogeometric, fall back to geometric
    dist_pc  = as_float(out.get("r_med_photogeo_pc",
                                pd.Series(np.nan, index=out.index))).to_numpy()
    dist_geo = as_float(out.get("r_med_geo_pc",
                                pd.Series(np.nan, index=out.index))).to_numpy()
    dist_pc  = np.where(np.isfinite(dist_pc), dist_pc, dist_geo)

    ks0_arr  = as_float(out.get("ks0", pd.Series(np.nan, index=out.index))).to_numpy()
    gi0_arr  = as_float(out["gi0_bdbs"]).to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        mu   = np.where(dist_pc > 0, 5.0 * np.log10(dist_pc) - 5.0, np.nan)
    M_Ks = ks0_arr - mu

    out["M_Ks_bdbs"]    = M_Ks
    out["dist_used_pc"] = dist_pc

    # Calibration subset: stars with finite (g-i)_0, M_Ks, AND a strictly
    # spectroscopic metallicity.  We use mh_astra or feh_astra directly rather
    # than metallicity_best, which may contain XGBoost (photometric) values.
    # Using a photometric metallicity as ground truth would make the calibration
    # circular.
    feh_spec_strict = as_float(out.get("mh_astra",
                               pd.Series(np.nan, index=out.index))).to_numpy()
    feh_astra_fe    = as_float(out.get("feh_astra",
                                pd.Series(np.nan, index=out.index))).to_numpy()
    # Prefer mh_astra, fall back to feh_astra
    feh_spec_strict = np.where(np.isfinite(feh_spec_strict),
                               feh_spec_strict, feh_astra_fe)

    cal_mask = np.isfinite(gi0_arr) & np.isfinite(M_Ks) & np.isfinite(feh_spec_strict)
    n_cal    = int(cal_mask.sum())

    if n_cal < 3:
        print(f"    Only {n_cal} calibration stars with both BDBS colours and "
              f"spectroscopic [M/H] — skipping empirical [Fe/H] fit.")
        out["feh_empirical_bdbs"]        = np.nan
        out["feh_empirical_bdbs_source"] = ""
    else:
        coeffs, rms = fit_empirical_feh(
            gi0_arr[cal_mask], M_Ks[cal_mask], feh_spec_strict[cal_mask]
        )
        a, b, c = coeffs
        print(f"    Calibration fit:  [Fe/H] = {a:+.3f} {b:+.3f}*(g-i)_0 {c:+.3f}*M_Ks")
        print(f"    N_calibrators = {n_cal},  RMS = {rms:.3f} dex")

        # Apply to ALL stars with the required photometry
        apply_mask = np.isfinite(gi0_arr) & np.isfinite(M_Ks)
        feh_emp    = np.full(len(out), np.nan)
        feh_emp[apply_mask] = apply_empirical_feh(
            gi0_arr[apply_mask], M_Ks[apply_mask], coeffs
        )

        # Label source as empirical so it is never confused with a real calibration
        source = np.full(len(out), "", dtype=object)
        source[apply_mask & ~cal_mask] = "BDBS_EMPIRICAL_FIT"
        source[cal_mask]               = "CALIBRATOR (has spec [M/H])"

        out["feh_empirical_bdbs"]        = feh_emp
        out["feh_empirical_bdbs_source"] = source

        n_applied = int(apply_mask.sum())
        n_new     = int((apply_mask & ~cal_mask).sum())
        print(f"    Applied to {n_applied} stars ({n_new} with no prior spec [M/H])")

    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_sky_feh_bdbs(df, outdir):
    l     = bulge_l(get(df, ["l", "glon"]).to_numpy())
    b     = get(df, ["b", "glat"]).to_numpy()
    feh   = as_float(df.get("feh_empirical_bdbs", pd.Series(np.nan, index=df.index))).to_numpy()
    good  = finite(l, b, feh)
    if good.sum() == 0:
        print("  skipped 21_sky_feh_bdbs (no data)")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    # Background: all stars
    all_good = finite(l, b)
    ax.scatter(l[all_good & ~good], b[all_good & ~good],
               s=14, color=C_ALL, alpha=0.4, label="No BDBS empirical [Fe/H]", zorder=2)
    sc = ax.scatter(l[good], b[good], c=feh[good], s=24, alpha=0.9,
                    cmap="RdYlBu_r", vmin=-1.5, vmax=0.5, zorder=3)
    sky_ax_setup(ax)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("BDBS empirical [Fe/H]  [dex]", fontsize=10)
    #ax.set_title("Sky map – BDBS empirical metallicity (self-calibrated)", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    savefig(fig, outdir / "21_sky_feh_bdbs.png")


def plot_bdbs_cmd(df, outdir):
    gi0 = as_float(df.get("gi0_bdbs", pd.Series(np.nan, index=df.index))).to_numpy()
    g0  = as_float(df.get("g0_bdbs",  pd.Series(np.nan, index=df.index))).to_numpy()
    feh = as_float(df.get("feh_empirical_bdbs", pd.Series(np.nan, index=df.index))).to_numpy()
    good = finite(gi0, g0)
    if good.sum() == 0:
        print("  skipped 18_bdbs_cmd (no BDBS optical data)")
        return

    fig, ax = plt.subplots(figsize=(7, 8))
    good_f  = good & np.isfinite(feh)
    good_nf = good & ~np.isfinite(feh)
    ax.scatter(gi0[good_nf], g0[good_nf], s=18, color=C_ALL, alpha=0.5,
               label=f"No phot [Fe/H]  (N={good_nf.sum()})", zorder=2)
    if good_f.sum() > 0:
        sc = ax.scatter(gi0[good_f], g0[good_f], c=feh[good_f],
                        s=26, alpha=0.9, cmap="RdYlBu_r",
                        vmin=-1.5, vmax=0.5, zorder=3)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("BDBS empirical [Fe/H]  [dex]", fontsize=10)
    ax.set_xlabel(r"$(g - i)_0$  [mag]", fontsize=12)
    ax.set_ylabel(r"$g_0$  [mag]", fontsize=12)
    #ax.set_title(f"BDBS optical CMD  (N={good.sum()})", fontsize=12)
    ax.invert_yaxis()
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    savefig(fig, outdir / "18_bdbs_cmd.png")


def plot_bdbs_feh_comparison(df, outdir):
    feh_phot = as_float(df.get("feh_empirical_bdbs",
                                pd.Series(np.nan, index=df.index))).to_numpy()
    feh_spec = as_float(df.get("metallicity_best",
                                pd.Series(np.nan, index=df.index))).to_numpy()
    good = finite(feh_phot, feh_spec)
    if good.sum() == 0:
        print("  skipped 20_bdbs_feh_comparison (no overlapping data)")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    lim = [-2.6, 0.7]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.4)
    diff = feh_phot[good] - feh_spec[good]
    sc = ax.scatter(feh_spec[good], feh_phot[good],
                    c=diff, cmap="RdBu_r", s=28, alpha=0.8,
                    vmin=-0.5, vmax=0.5, zorder=2)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$\Delta$[Fe/H]  (BDBS empirical − spec)  [dex]", fontsize=10)
    ax.set_xlabel("Spectroscopic [M/H] (best)  [dex]", fontsize=12)
    ax.set_ylabel("BDBS empirical [Fe/H]  [dex]", fontsize=12)
    #ax.set_title(f"[Fe/H] comparison  (N={good.sum()})  "
    #             f"median Δ = {np.median(diff):+.3f}  "
    #             f"σ = {np.std(diff):.3f}", fontsize=10)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.grid(alpha=0.2)
    savefig(fig, outdir / "20_bdbs_feh_comparison.png")


def plot_metallicity_hist_updated(df, outdir):
    """Remake plot 10 with BDBS photometric metallicity included."""
    metal_cols = [
        ("metallicity_best", "Best (spec)",  "#f72585"),
        ("mh_astra",         "ASTRA",        "#4895ef"),
        ("feh_bdbs",         "BDBS spectro", "#f77f00"),
        ("mh_gspspec",       "GSP-Spec",     "#7b2d8b"),
        ("mh_xgboost",       "XGBoost",      "#2dc653"),
        ("feh_empirical_bdbs",    "BDBS empirical fit", "#e63946"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-2.5, 0.8, 40)
    for col_name, label, color in metal_cols:
        if col_name not in df.columns:
            continue
        vals = as_float(df[col_name]).to_numpy()
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


def plot_metallicity_by_sightline_updated(df, outdir):
    """Remake plot 11 using best metallicity (now includes BDBS phot where spec absent)."""
    # Merge: prefer spectroscopic best, fall back to BDBS photometric
    metal_best = as_float(df.get("metallicity_best",
                                  pd.Series(np.nan, index=df.index))).to_numpy()
    feh_phot   = as_float(df.get("feh_empirical_bdbs",
                                  pd.Series(np.nan, index=df.index))).to_numpy()
    metal_merged = np.where(np.isfinite(metal_best), metal_best, feh_phot)

    sid  = df["sightline_id"].fillna("").astype(str) \
           if "sightline_id" in df.columns \
           else pd.Series("", index=df.index)
    good = (sid != "") & np.isfinite(metal_merged)
    if good.sum() == 0:
        print("  skipped 11 update (no data)")
        return

    top = sid[good].value_counts().head(15).index.tolist()
    sub = good & sid.isin(top)
    if sub.sum() == 0:
        return

    ordered = sid[sub].value_counts().index.tolist()
    idx_map = {s: i for i, s in enumerate(ordered)}

    fig, ax = plt.subplots(figsize=(11, 6))
    rng     = np.random.default_rng(42)
    x_pos   = np.array([idx_map[s] for s in sid[sub]])
    jitter  = rng.uniform(-0.25, 0.25, sub.sum())

    # Colour by whether spec or photometric
    is_spec = np.isfinite(metal_best[sub])
    ax.scatter(x_pos[~is_spec] + jitter[~is_spec], metal_merged[sub][~is_spec],
               c=metal_merged[sub][~is_spec], cmap="RdYlBu_r",
               s=18, alpha=0.6, marker="^", vmin=-1.5, vmax=0.5,
               label="BDBS empirical", zorder=2)
    sc = ax.scatter(x_pos[is_spec] + jitter[is_spec], metal_merged[sub][is_spec],
                    c=metal_merged[sub][is_spec], cmap="RdYlBu_r",
                    s=22, alpha=0.8, marker="o", vmin=-1.5, vmax=0.5,
                    label="Spectroscopic", zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("[M/H]  [dex]", fontsize=10)

    for i, s in enumerate(ordered):
        m = (sid[sub] == s)
        med = np.nanmedian(metal_merged[sub][m])
        ax.hlines(med, i - 0.4, i + 0.4, color="k", lw=2, zorder=4)

    labels = [s.split("|")[-2] + "|" + s.split("|")[-1]
              if "|" in s else s[:18] for s in ordered]
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=7)
    ax.set_ylabel("[M/H]  [dex]", fontsize=12)
    #ax.set_title("Metallicity per sightline  (● spec, ▲ BDBS empirical, bar = median)",
    #             fontsize=12)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    savefig(fig, outdir / "11_metallicity_by_sightline.png")


def plot_metallicity_sources_updated(df, outdir):
    """Remake plot 12 including BDBS phot vs spectroscopic."""
    pairs = [
        ("mh_astra",     "mh_xgboost",   "ASTRA [M/H]",        "XGBoost [M/H]"),
        ("mh_astra",     "mh_gspspec",   "ASTRA [M/H]",        "GSP-Spec [M/H]"),
        ("mh_astra",     "feh_empirical_bdbs","ASTRA [M/H]",        "BDBS empirical [Fe/H]"),
        ("metallicity_best","feh_empirical_bdbs","Best spec [M/H]", "BDBS empirical [Fe/H]"),
    ]
    valid = [(a, b, la, lb) for a, b, la, lb in pairs
             if a in df.columns and b in df.columns
             and as_float(df[a]).notna().any()
             and as_float(df[b]).notna().any()]
    if not valid:
        print("  skipped 12 update (no valid pairs)")
        return

    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1:
        axes = [axes]
    for ax, (c1, c2, l1, l2) in zip(axes, valid):
        v1 = as_float(df[c1]).to_numpy()
        v2 = as_float(df[c2]).to_numpy()
        good = finite(v1, v2)
        if good.sum() == 0:
            continue
        lim = [-2.6, 0.7]
        ax.plot(lim, lim, "k--", lw=1, alpha=0.4)
        ax.scatter(v1[good], v2[good], s=18, alpha=0.65, color=C_PHOT, zorder=2)
        diff = v2[good] - v1[good]
        ax.set_xlabel(l1 + "  [dex]", fontsize=9)
        ax.set_ylabel(l2 + "  [dex]", fontsize=9)
        #ax.set_title(f"N={good.sum()}  Δ={np.nanmedian(diff):+.3f}  σ={np.nanstd(diff):.3f}",
        #             fontsize=9)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.grid(alpha=0.2)
    fig.suptitle("Metallicity source comparison", fontsize=12, y=1.01)
    savefig(fig, outdir / "12_metallicity_sources.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading {INPUT_FITS} ...")
    df = load(INPUT_FITS)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    # Check BDBS data is present
    bdbs_present = any("bdbs" in c.lower() or "umag" in c.lower()
                       or "gmag" in c.lower()
                       for c in df.columns)
    if not bdbs_present:
        print("WARNING: no BDBS columns detected.  "
              "Make sure you merged the BDBS CSV in TOPCAT first.")

    print("\nDeriving BDBS quantities ...")
    df = derive_bdbs_quantities(df)

    print(f"\nWriting {OUTPUT_FITS} ...")
    Table.from_pandas(df).write(OUTPUT_FITS, overwrite=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    print(f"\nWriting plots to {OUTDIR}/ ...")
    plot_bdbs_cmd(df, OUTDIR)
    plot_bdbs_feh_comparison(df, OUTDIR)
    plot_sky_feh_bdbs(df, OUTDIR)
    plot_metallicity_hist_updated(df, OUTDIR)
    plot_metallicity_by_sightline_updated(df, OUTDIR)
    plot_metallicity_sources_updated(df, OUTDIR)

    print("\nDone.")


if __name__ == "__main__":
    main()