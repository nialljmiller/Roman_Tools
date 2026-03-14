#!/usr/bin/env python3
"""
Step 5 – Parameters, flags, pruning, and final output
======================================================
Finalises the catalog:

  1. Harvest best Teff, logg, [M/H], radius, luminosity across all sources
     (ASTRA > XGBoost > Gaia GSP-Spec > Gaia AP > BDBS > Bensby)
  2. Add boolean provenance flags (has_virac2, has_reddening, …)
  3. Add calibration_ready flag
  4. Drop SDSS/ASTRA pipeline bookkeeping columns
  5. Reorder columns and write final FITS + CSV + summary

Input  : step4_photometry.fits
Outputs: roman_master.fits / .csv
         roman_calibration_ready.fits / .csv
         roman_summary.txt
"""

import pathlib
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits as astrofits
import warnings
from astropy.io.fits.verify import VerifyWarning
from astropy.units import UnitsWarning

warnings.filterwarnings("ignore", category=VerifyWarning)
warnings.filterwarnings("ignore", category=UnitsWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_FITS       = pathlib.Path("step4_photometry.fits")
MASTER_FITS      = pathlib.Path("roman_master.fits")
MASTER_CSV       = pathlib.Path("roman_master.csv")
READY_FITS       = pathlib.Path("roman_calibration_ready.fits")
READY_CSV        = pathlib.Path("roman_calibration_ready.csv")
SUMMARY_TXT      = pathlib.Path("roman_summary.txt")

T_SUN = 5772.0

# ---------------------------------------------------------------------------
# Bookkeeping columns to drop
# ---------------------------------------------------------------------------
BOOKKEEPING_COLUMNS = [
    "sdss5_target_flags",
    "sdss4_apogee_target1_flags",
    "sdss4_apogee_target2_flags",
    "sdss4_apogee2_target1_flags",
    "sdss4_apogee2_target2_flags",
    "sdss4_apogee2_target3_flags",
    "sdss4_apogee_member_flags",
    "sdss4_apogee_extra_target_flags",
    "lead", "version_id", "task_pk", "source_pk", "v_astra",
    "created", "t_elapsed", "t_overhead", "tag",
    "stellar_parameters_task_pk",
    "al_h_task_pk", "c_12_13_task_pk", "ca_h_task_pk", "ce_h_task_pk",
    "c_1_h_task_pk", "c_h_task_pk", "co_h_task_pk", "cr_h_task_pk",
    "cu_h_task_pk", "fe_h_task_pk", "k_h_task_pk", "mg_h_task_pk",
    "mn_h_task_pk", "na_h_task_pk", "nd_h_task_pk", "ni_h_task_pk",
    "n_h_task_pk", "o_h_task_pk", "p_h_task_pk", "si_h_task_pk",
    "s_h_task_pk", "ti_h_task_pk", "ti_2_h_task_pk", "v_h_task_pk",
]

# Front-of-table column ordering
FRONT_COLUMNS = [
    "gaia_dr3_source_id", "catalogid", "sdss_id",
    "ra_icrs", "dec_icrs", "l", "b",
    "parallax", "parallax_error",
    "teff_best", "teff_source_best",
    "logg_best", "logg_source_best",
    "metallicity_best", "metallicity_source_best",
    "radius_best_rsun", "radius_source_best",
    "luminosity_best_lsun", "log10_luminosity_best_lsun", "luminosity_source_best",
    "gaia_flux_g", "gaia_flux_bp", "gaia_flux_rp",
    "gaia_g0", "gaia_bp0", "gaia_rp0", "gaia_bp_rp0",
    "gaia_lum_parallax_lsun", "gaia_abs_g0",
    "j_mag_best", "h_mag_best", "ks_mag_best",
    "j_mag_source", "h_mag_source", "ks_mag_source", "phot_source_overall",
    "j_minus_h_best", "h_minus_ks_best", "j_minus_ks_best",
    "ext_e_jks", "a_g", "a_j", "a_h", "a_ks",
    "j0", "h0", "ks0",
    "j0_minus_h0", "h0_minus_ks0", "j0_minus_ks0",
    "sightline_id", "sightline_glon", "sightline_glat", "sightline_bin_0p1deg",
    "has_virac2", "has_2mass", "has_reddening",
    "has_dereddened_phot", "has_core_params",
    "calibration_ready_phot", "calibration_ready_params", "calibration_ready",
]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def lower_column_map(columns):
    return {str(c).lower(): str(c) for c in columns}


def find_column(columns, candidates):
    mapping = lower_column_map(columns)
    for candidate in candidates:
        key = str(candidate).lower()
        if key in mapping:
            return mapping[key]
    return None


def as_float(series):
    return pd.to_numeric(series, errors="coerce")

# ---------------------------------------------------------------------------
# Gaia flux and parallax-based luminosity
# ---------------------------------------------------------------------------

# Zero-point flux densities for converting Gaia Vega magnitudes to flux.
# From Gaia DR3 documentation (Riello et al. 2021):
#   G  : 3631 Jy (AB), but Gaia uses Vega; effective zero-point ~3660 Jy
# We use the simpler approach of working directly with the mean flux columns
# (e/s) which Gaia provides, and separately computing luminosity from parallax.

# Bolometric correction to G (very rough; for red giants in the bulge).
# A proper BC_G requires Teff; here we apply a constant median value as a
# first-order estimate.  Replace with a Teff-dependent table if available.
BC_G_GIANTS = 0.0      # set to 0 to report G-band luminosity only; adjust if needed
L_SUN_W     = 3.828e26  # W  (IAU 2015)
F_SUN_G_W_M2_NM = 1.84e-9  # approximate solar flux density in Gaia G band at 1 AU


def compute_gaia_flux_columns(df):
    """
    Extract Gaia raw fluxes and compute dereddened Gaia magnitudes.

    Columns read (from the merged Gaia table or ASTRA cross-match):
      phot_g_mean_flux, phot_bp_mean_flux, phot_rp_mean_flux  (e/s)
      phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag     (Vega mag)
      a_g_val   – Gaia-reported G-band extinction (mag)
      e_bp_min_rp_val – Gaia E(BP-RP)

    Outputs:
      gaia_flux_g / _bp / _rp   – raw mean fluxes (e/s), renamed for clarity
      gaia_g0 / bp0 / rp0       – extinction-corrected Gaia magnitudes
      gaia_bp_rp0               – dereddened BP-RP colour
    """
    print("Computing Gaia flux and dereddened magnitude columns ...")

    flux_g_col  = find_column(df.columns, ["phot_g_mean_flux",  "g_mean_flux"])
    flux_bp_col = find_column(df.columns, ["phot_bp_mean_flux", "bp_mean_flux"])
    flux_rp_col = find_column(df.columns, ["phot_rp_mean_flux", "rp_mean_flux"])

    g_mag_col   = find_column(df.columns, ["phot_g_mean_mag",  "g_mean_mag",  "Gmag"])
    bp_mag_col  = find_column(df.columns, ["phot_bp_mean_mag", "bp_mean_mag", "BPmag"])
    rp_mag_col  = find_column(df.columns, ["phot_rp_mean_mag", "rp_mean_mag", "RPmag"])

    a_g_col     = find_column(df.columns, ["a_g_val",  "ag_val",  "A_G",  "a_g"])
    e_bprp_col  = find_column(df.columns, ["e_bp_min_rp_val", "ebpminrp_val", "e_bprp"])

    # Raw fluxes
    df["gaia_flux_g"]  = as_float(df[flux_g_col])  if flux_g_col  is not None else np.nan
    df["gaia_flux_bp"] = as_float(df[flux_bp_col]) if flux_bp_col is not None else np.nan
    df["gaia_flux_rp"] = as_float(df[flux_rp_col]) if flux_rp_col is not None else np.nan

    # Apparent magnitudes
    g_app  = as_float(df[g_mag_col])  if g_mag_col  is not None else pd.Series(np.nan, index=df.index)
    bp_app = as_float(df[bp_mag_col]) if bp_mag_col is not None else pd.Series(np.nan, index=df.index)
    rp_app = as_float(df[rp_mag_col]) if rp_mag_col is not None else pd.Series(np.nan, index=df.index)

    # Extinction
    a_g    = as_float(df[a_g_col])    if a_g_col    is not None else pd.Series(np.nan, index=df.index)
    e_bprp = as_float(df[e_bprp_col]) if e_bprp_col is not None else pd.Series(np.nan, index=df.index)

    # Dereddened magnitudes
    df["gaia_g0"]      = g_app  - a_g.fillna(0.0)
    df["gaia_bp0"]     = bp_app - (e_bprp.fillna(0.0) * 1.21)   # A_BP ~ 1.21 * E(BP-RP)
    df["gaia_rp0"]     = rp_app - (e_bprp.fillna(0.0) * 0.21)   # A_RP ~ 0.21 * E(BP-RP)
    df["gaia_bp_rp0"]  = df["gaia_bp0"] - df["gaia_rp0"]
    df["a_g"]          = a_g

    # Mask rows where G mag is not available
    no_g = ~np.isfinite(g_app.to_numpy())
    for col in ["gaia_g0", "gaia_bp0", "gaia_rp0", "gaia_bp_rp0"]:
        df.loc[no_g, col] = np.nan

    n_g   = np.isfinite(df["gaia_flux_g"]).sum()
    n_g0  = np.isfinite(df["gaia_g0"]).sum()
    print(f"  Gaia G flux   : {n_g:,} stars")
    print(f"  Gaia G0 (dered): {n_g0:,} stars")
    return df


def compute_gaia_parallax_luminosity(df):
    """
    Estimate luminosity from Gaia parallax and dereddened G magnitude.

    Method:
      1. distance_pc = 1000 / parallax_mas   (naive inversion; good for
         well-measured parallaxes; do not use for plx_snr < 5)
      2. distance_modulus mu = 5 * log10(distance_pc) - 5
      3. Absolute G0 magnitude: M_G0 = gaia_g0 - mu
      4. Luminosity in solar units using solar absolute G magnitude M_G_sun = 5.12
         (Casagrande & VandenBerg 2018, for Gaia G band):
         log10(L/L_sun) = (M_G_sun - M_G0) / 2.5

    Outputs:
      gaia_distance_pc       – naive parallax inversion (pc)
      gaia_abs_g0            – absolute dereddened G magnitude
      gaia_lum_parallax_lsun – G-band luminosity in solar units
    """
    print("Computing Gaia parallax-based luminosity ...")

    M_G_SUN = 5.12   # absolute G magnitude of the Sun (Casagrande & VandenBerg 2018)
    PLX_SNR_MIN = 5.0

    plx_col     = find_column(df.columns, ["parallax",       "Plx",  "plx"])
    plx_err_col = find_column(df.columns, ["parallax_error", "e_Plx","e_plx"])

    if plx_col is None:
        print("  WARNING: parallax column not found; skipping parallax luminosity.")
        df["gaia_distance_pc"]        = np.nan
        df["gaia_abs_g0"]             = np.nan
        df["gaia_lum_parallax_lsun"]  = np.nan
        return df

    # Standardise parallax column name for FRONT_COLUMNS
    df["parallax"] = as_float(df[plx_col])
    if plx_err_col is not None:
        df["parallax_error"] = as_float(df[plx_err_col])
    else:
        df["parallax_error"] = np.nan

    plx     = df["parallax"].to_numpy()
    plx_err = df["parallax_error"].to_numpy()

    # Only use well-measured parallaxes (SNR >= PLX_SNR_MIN, plx > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        plx_snr = np.where(plx_err > 0, plx / plx_err, np.nan)
    good_plx = np.isfinite(plx) & (plx > 0) & (plx_snr >= PLX_SNR_MIN)

    distance_pc = np.full(len(df), np.nan)
    distance_pc[good_plx] = 1000.0 / plx[good_plx]
    df["gaia_distance_pc"] = distance_pc

    g0 = df.get("gaia_g0", pd.Series(np.nan, index=df.index))
    g0_arr = as_float(g0).to_numpy()

    mu      = np.full(len(df), np.nan)
    good_mu = good_plx & np.isfinite(g0_arr)
    mu[good_mu] = 5.0 * np.log10(distance_pc[good_mu]) - 5.0

    abs_g0 = np.full(len(df), np.nan)
    abs_g0[good_mu] = g0_arr[good_mu] - mu[good_mu]
    df["gaia_abs_g0"] = abs_g0

    lum = np.full(len(df), np.nan)
    good_lum = good_mu & np.isfinite(abs_g0)
    lum[good_lum] = 10.0 ** ((M_G_SUN - abs_g0[good_lum]) / 2.5)
    df["gaia_lum_parallax_lsun"] = lum

    n_lum = np.isfinite(lum).sum()
    print(f"  Parallax luminosity available: {n_lum:,} stars  "
          f"(parallax SNR >= {PLX_SNR_MIN})")
    return df



def _pick_best_value(df, candidates, value_col, source_col):
    """
    Walk through a priority-ordered list of candidate columns.
    Write the first finite value found into value_col and record its
    source label in source_col.

    Each candidate is a dict with keys 'col' and 'label'.
    """
    best_val = np.full(len(df), np.nan)
    best_src = np.full(len(df), "", dtype=object)

    for candidate in candidates:
        col   = candidate["col"]
        label = candidate["label"]
        if col not in df.columns:
            continue
        vals = as_float(df[col]).to_numpy()
        fill = np.isfinite(vals) & ~np.isfinite(best_val)
        best_val[fill] = vals[fill]
        best_src[fill] = label

    df[value_col]  = best_val
    df[source_col] = best_src
    return df


def harvest_best_parameters(df):
    print("Harvesting best stellar parameters ...")

    df = compute_gaia_flux_columns(df)
    df = compute_gaia_parallax_luminosity(df)

    teff_candidates = [
        {"col": "teff",          "label": "ASTRA"},
        {"col": "xgb_teff",      "label": "XGB_GAIA"},
        {"col": "Teff-S",        "label": "GAIA_GSPSPEC"},
        {"col": "Teff_x",        "label": "GAIA_AP"},
        {"col": "doppler_teff",  "label": "ASTRA_DOPPLER"},
        {"col": "raw_teff",      "label": "ASTRA_RAW"},
    ]
    df = _pick_best_value(df, teff_candidates, "teff_best", "teff_source_best")

    logg_candidates = [
        {"col": "logg",          "label": "ASTRA"},
        {"col": "xgb_logg",      "label": "XGB_GAIA"},
        {"col": "logg-S",        "label": "GAIA_GSPSPEC"},
        {"col": "logg_x",        "label": "GAIA_AP"},
        {"col": "doppler_logg",  "label": "ASTRA_DOPPLER"},
        {"col": "raw_logg",      "label": "ASTRA_RAW"},
    ]
    df = _pick_best_value(df, logg_candidates, "logg_best", "logg_source_best")

    metal_candidates = [
        {"col": "xgb_mh",        "label": "XGB_GAIA"},
        {"col": "xgb_m_h",       "label": "XGB_GAIA"},
        {"col": "m_h_atm",       "label": "ASTRA"},
        {"col": "fe_h",          "label": "ASTRA"},
        {"col": "[M/H]-S",       "label": "GAIA_GSPSPEC"},
        {"col": "[Fe/H]-S",      "label": "GAIA_GSPSPEC"},
        {"col": "bdbs_[Fe/H]",   "label": "BDBS"},
        {"col": "bdbs_fe_h",     "label": "BDBS"},
        {"col": "bensby_[Fe/H]", "label": "BENSBY"},
        {"col": "bensby_fe_h",   "label": "BENSBY"},
        {"col": "raw_m_h_atm",   "label": "ASTRA_RAW"},
    ]
    df = _pick_best_value(df, metal_candidates, "metallicity_best", "metallicity_source_best")

    radius_candidates = [
        {"col": "radius",    "label": "ASTRA"},
        {"col": "Rad",       "label": "GAIA_AP"},
        {"col": "Rad-Flame", "label": "GAIA_FLAME"},
    ]
    df = _pick_best_value(df, radius_candidates, "radius_best_rsun", "radius_source_best")

    lum_candidates = [
        {"col": "luminosity",              "label": "ASTRA"},
        {"col": "Lum-Flame",               "label": "GAIA_FLAME"},
        {"col": "gaia_lum_parallax_lsun",  "label": "GAIA_PARALLAX"},
    ]
    df = _pick_best_value(df, lum_candidates, "luminosity_best_lsun", "luminosity_source_best")

    # Derive luminosity from R and Teff where direct value is missing
    derived_mask = (
        ~np.isfinite(as_float(df["luminosity_best_lsun"])) &
         np.isfinite(as_float(df["radius_best_rsun"]))      &
         np.isfinite(as_float(df["teff_best"]))
    )
    if derived_mask.sum() > 0:
        r   = as_float(df.loc[derived_mask, "radius_best_rsun"])
        t   = as_float(df.loc[derived_mask, "teff_best"])
        lum = r**2 * (t / T_SUN)**4
        df.loc[derived_mask, "luminosity_best_lsun"]   = lum
        df.loc[derived_mask, "luminosity_source_best"] = "DERIVED_R_TEFF"

    with np.errstate(divide="ignore", invalid="ignore"):
        lum_vals = as_float(df["luminosity_best_lsun"])
        df["log10_luminosity_best_lsun"] = np.where(
            lum_vals > 0, np.log10(lum_vals), np.nan
        )

    for col, label in [
        ("teff_best",          "Teff"),
        ("logg_best",          "logg"),
        ("metallicity_best",   "[M/H]"),
        ("radius_best_rsun",   "radius"),
        ("luminosity_best_lsun", "luminosity"),
    ]:
        n = np.isfinite(as_float(df[col])).sum()
        print(f"  {label:<12}: {n:,}")

    return df

# ---------------------------------------------------------------------------
# Provenance flags
# ---------------------------------------------------------------------------

def add_provenance_flags(df):
    print("Adding provenance and calibration_ready flags ...")

    def col_is_finite(col_name):
        if col_name not in df.columns:
            return pd.Series(False, index=df.index)
        return np.isfinite(as_float(df[col_name]))

    def col_is_nonempty(col_name):
        if col_name not in df.columns:
            return pd.Series(False, index=df.index)
        return df[col_name].fillna("").astype(str).str.strip() != ""

    df["has_virac2"]          = col_is_nonempty("vvv_srcid")   | col_is_finite("virac2_sep_arcsec")
    df["has_2mass"]           = col_is_nonempty("tmass_2MASS")  | col_is_finite("tmass_sep_arcsec")
    df["has_reddening"]       = col_is_finite("ext_e_jks")
    df["has_gaia_flux"]       = col_is_finite("gaia_flux_g")
    df["has_parallax_lum"]    = col_is_finite("gaia_lum_parallax_lsun")
    df["has_best_j"]          = col_is_finite("j_mag_best")
    df["has_best_h"]          = col_is_finite("h_mag_best")
    df["has_best_ks"]         = col_is_finite("ks_mag_best")
    df["has_best_phot"]       = df["has_best_j"] & df["has_best_h"] & df["has_best_ks"]
    df["has_dereddened_phot"] = col_is_finite("j0") & col_is_finite("h0") & col_is_finite("ks0")
    df["has_sightline"]       = col_is_nonempty("sightline_id")
    df["has_teff_best"]       = col_is_finite("teff_best")
    df["has_logg_best"]       = col_is_finite("logg_best")
    df["has_metallicity_best"]= col_is_finite("metallicity_best")
    df["has_radius_best"]     = col_is_finite("radius_best_rsun")
    df["has_luminosity_best"] = col_is_finite("luminosity_best_lsun")
    df["has_core_params"]     = (
        df["has_teff_best"] & df["has_logg_best"] &
        df["has_metallicity_best"] & df["has_luminosity_best"]
    )

    df["calibration_ready_phot"]   = df["has_dereddened_phot"] & df["has_reddening"]
    df["calibration_ready_params"] = df["has_core_params"]
    df["calibration_ready"]        = df["calibration_ready_phot"] & df["calibration_ready_params"]

    print(f"  calibration_ready          : {df['calibration_ready'].sum():,}")
    print(f"  calibration_ready_phot     : {df['calibration_ready_phot'].sum():,}")
    print(f"  has_core_params            : {df['has_core_params'].sum():,}")
    return df

# ---------------------------------------------------------------------------
# Prune bookkeeping
# ---------------------------------------------------------------------------

def prune_bookkeeping(df):
    print("Pruning bookkeeping columns ...")
    to_drop = [c for c in BOOKKEEPING_COLUMNS if c in df.columns]
    df = df.drop(columns=to_drop)
    print(f"  Dropped {len(to_drop)} columns. Remaining: {len(df.columns)}")
    return df

# ---------------------------------------------------------------------------
# Column ordering
# ---------------------------------------------------------------------------

def _phot_source_overall(df):
    if "j_mag_source" not in df.columns:
        return pd.Series("NONE", index=df.index)
    j = df["j_mag_source"].fillna("").astype(str)
    h = df.get("h_mag_source",  pd.Series("", index=df.index)).fillna("").astype(str)
    k = df.get("ks_mag_source", pd.Series("", index=df.index)).fillna("").astype(str)
    result = np.where(
        (j == "VIRAC2") & (h == "VIRAC2") & (k == "VIRAC2"), "VIRAC2",
        np.where(
            (j == "2MASS") & (h == "2MASS") & (k == "2MASS"), "2MASS",
            np.where((j != "") | (h != "") | (k != ""), "MIXED", "NONE")
        )
    )
    return pd.Series(result, index=df.index)


def reorder_columns(df):
    df["phot_source_overall"] = _phot_source_overall(df)
    front = [c for c in FRONT_COLUMNS if c in df.columns]
    rest  = [c for c in df.columns   if c not in front]
    return df[front + rest]

# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

def write_outputs(df):
    ready = df.loc[df["calibration_ready"]].copy()

    Table.from_pandas(df).write(MASTER_FITS, overwrite=True)
    df.to_csv(MASTER_CSV, index=False)
    print(f"  Master    : {MASTER_FITS}  ({len(df):,} rows)")

    Table.from_pandas(ready).write(READY_FITS, overwrite=True)
    ready.to_csv(READY_CSV, index=False)
    print(f"  Cal-ready : {READY_FITS}  ({len(ready):,} rows)")

    _write_summary(df, ready)
    print(f"  Summary   : {SUMMARY_TXT}")


def _write_summary(master, ready):
    ejks = as_float(master.get("ext_e_jks", pd.Series(dtype=float))).to_numpy()

    lines = [
        "Roman Bulge Calibration Pipeline – Final Summary",
        "=" * 50,
        f"Total stars (footprint + cuts)         : {len(master):,}",
        f"Calibration-ready (phot + params)      : {len(ready):,}",
        f"Calibration-ready (phot only)          : {master['calibration_ready_phot'].sum():,}",
        f"Has core params (Teff+logg+[M/H]+L)    : {master['has_core_params'].sum():,}",
        "",
        "Photometry",
        "-" * 20,
        f"VIRAC2 matched                         : {master['has_virac2'].sum():,}",
        f"2MASS matched                          : {master['has_2mass'].sum():,}",
        f"Has E(J-Ks)                            : {master['has_reddening'].sum():,}",
        f"Dereddened J0/H0/Ks0                   : {master['has_dereddened_phot'].sum():,}",
        f"Has Gaia G flux                        : {master['has_gaia_flux'].sum():,}",
        f"Has parallax luminosity                : {master['has_parallax_lum'].sum():,}",
    ]

    if np.isfinite(ejks).any():
        lines += [
            "",
            "Reddening",
            "-" * 20,
            f"E(J-Ks) median                         : {np.nanmedian(ejks):.4f}",
            f"E(J-Ks) 16th pct                       : {np.nanpercentile(ejks, 16):.4f}",
            f"E(J-Ks) 84th pct                       : {np.nanpercentile(ejks, 84):.4f}",
        ]

    lines += [
        "",
        "Parameters",
        "-" * 20,
        f"has_teff_best                          : {master['has_teff_best'].sum():,}",
        f"has_logg_best                          : {master['has_logg_best'].sum():,}",
        f"has_metallicity_best                   : {master['has_metallicity_best'].sum():,}",
        f"has_luminosity_best                    : {master['has_luminosity_best'].sum():,}",
    ]

    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def read_fits_all_columns(fits_path):
    with astrofits.open(fits_path, memmap=True) as hdul:
        tbl = Table(hdul[1].data)
    rows = {}
    for name in tbl.colnames:
        col   = tbl[name]
        shape = getattr(col, "shape", ())
        if len(shape) > 1:
            rows[name] = [str(v.tolist()) for v in col]
        else:
            try:
                rows[name] = col.data.tolist()
            except AttributeError:
                rows[name] = list(col)
    return pd.DataFrame(rows)


def main():
    print(f"Reading {INPUT_FITS} ...")
    df = read_fits_all_columns(INPUT_FITS)
    print(f"  {len(df):,} rows")

    df = harvest_best_parameters(df)
    df = add_provenance_flags(df)
    df = prune_bookkeeping(df)
    df = reorder_columns(df)

    print("\nWriting outputs ...")
    write_outputs(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
