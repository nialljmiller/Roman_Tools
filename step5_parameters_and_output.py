#!/usr/bin/env python3
"""
Step 5 – Parameter harvesting and final output
===============================================
Builds a clean, science-ready catalog from the cross-matched and
dereddened sample.  The philosophy is to keep everything useful
and label it clearly by source, rather than picking a single "best"
value and discarding the rest.

What is kept
------------
Identifiers
    sdss_id, catalogid variants, Gaia DR2/DR3 source_id

Astrometry  (all with errors where available)
    ra, dec, l, b, parallax, proper motions, radial velocity,
    Bailer-Jones geometric and photogeometric distances

Teff  (one column per source, plus errors)
    teff_astra, teff_gspspec, teff_gaia_ap, teff_zgr,
    teff_doppler, teff_xgboost, teff_irfm
    + teff_best (priority pick) and teff_source_best

logg  (one column per source, plus errors)
    logg_astra, logg_gspspec, logg_gaia_ap, logg_zgr,
    logg_doppler, logg_xgboost
    + logg_best and logg_source_best

Metallicity  (one column per source, plus errors)
    feh_astra, mh_astra, feh_doppler, mh_gspspec, feh_gspspec,
    feh_gaia_ap, feh_zgr, mh_xgboost, feh_bdbs, feh_bensby
    + metallicity_best and metallicity_source_best

Radius  (one column per source, plus errors)
    radius_astra_rsun, radius_gaia_ap_rsun, radius_flame_rsun
    + radius_best_rsun and radius_source_best

Luminosity  (three independent estimates + best)
    lum_flame_lsun       – Gaia FLAME direct
    lum_r_teff_lsun      – derived from best radius + best Teff
    lum_parallax_lsun    – from parallax + dereddened G mag
    luminosity_best_lsun – FLAME > R·Teff > parallax
    log10_lum_best_lsun

Mass
    mass_flame_msun  (+ confidence interval)

Age
    age_flame_gyr    (+ confidence interval)

Photometry (pipeline products, then all raw mags at the end)
    j_mag_best/h_mag_best/ks_mag_best, source labels, errors
    j0/h0/ks0, colours
    gaia_g0/bp0/rp0
    ext_e_jks, a_j/a_h/a_ks/a_g
    All raw magnitudes: Gaia G/BP/RP, 2MASS JHK, WISE W1/W2,
    Spitzer 4.5um, VIRAC2 ZYJHKs, BDBS optical + errors

Sightlines
    sightline_id, sightline_glon/glat, sightline_tile

Flags
    has_* provenance booleans, calibration_ready

What is dropped
---------------
SDSS/APOGEE target-selection flags and ASTRA pipeline task-PK columns.

Input  : step3_xmatched.fits   (or step4_photometry.fits if you ran step 4)
Outputs: roman_master.fits / .csv
         roman_calibration_ready.fits / .csv
         roman_summary.txt
"""

import pathlib
import numpy as np
import pandas as pd
from astropy.table import Table
import warnings
from astropy.io.fits.verify import VerifyWarning
from astropy.units import UnitsWarning

warnings.filterwarnings("ignore", category=VerifyWarning)
warnings.filterwarnings("ignore", category=UnitsWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_FITS  = pathlib.Path("step3_xmatched.fits")
MASTER_FITS = pathlib.Path("roman_master.fits")
MASTER_CSV  = pathlib.Path("roman_master.csv")
READY_FITS  = pathlib.Path("roman_calibration_ready.fits")
READY_CSV   = pathlib.Path("roman_calibration_ready.csv")
SUMMARY_TXT = pathlib.Path("roman_summary.txt")

T_SUN       = 5772.0
M_G_SUN     = 5.12      # Casagrande & VandenBerg 2018
PLX_SNR_MIN = 5.0

# ---------------------------------------------------------------------------
# Bookkeeping columns to drop  (with and without _1 TOPCAT suffix)
# ---------------------------------------------------------------------------
BOOKKEEPING_COLUMNS = [
    "lead", "version_id", "task_pk", "source_pk", "v_astra",
    "created", "t_elapsed", "t_overhead", "tag",
    "stellar_parameters_task_pk",
    "al_h_task_pk", "c_12_13_task_pk", "ca_h_task_pk", "ce_h_task_pk",
    "c_1_h_task_pk", "c_h_task_pk", "co_h_task_pk", "cr_h_task_pk",
    "cu_h_task_pk", "fe_h_task_pk", "k_h_task_pk", "mg_h_task_pk",
    "mn_h_task_pk", "na_h_task_pk", "nd_h_task_pk", "ni_h_task_pk",
    "n_h_task_pk", "o_h_task_pk", "p_h_task_pk", "si_h_task_pk",
    "s_h_task_pk", "ti_h_task_pk", "ti_2_h_task_pk", "v_h_task_pk",
    "sdss5_target_flags", "sdss4_apogee_target1_flags",
    "sdss4_apogee_target2_flags", "sdss4_apogee2_target1_flags",
    "sdss4_apogee2_target2_flags", "sdss4_apogee2_target3_flags",
    "sdss4_apogee_member_flags", "sdss4_apogee_extra_target_flags",
    # _1 variants (TOPCAT merge)
    "lead_1", "version_id_1", "task_pk_1", "source_pk_1", "v_astra_1",
    "created_1", "t_elapsed_1", "t_overhead_1", "tag_1",
    "stellar_parameters_task_pk_1",
    "al_h_task_pk_1", "c_12_13_task_pk_1", "ca_h_task_pk_1", "ce_h_task_pk_1",
    "c_1_h_task_pk_1", "c_h_task_pk_1", "co_h_task_pk_1", "cr_h_task_pk_1",
    "cu_h_task_pk_1", "fe_h_task_pk_1", "k_h_task_pk_1", "mg_h_task_pk_1",
    "mn_h_task_pk_1", "na_h_task_pk_1", "nd_h_task_pk_1", "ni_h_task_pk_1",
    "n_h_task_pk_1", "o_h_task_pk_1", "p_h_task_pk_1", "si_h_task_pk_1",
    "s_h_task_pk_1", "ti_h_task_pk_1", "ti_2_h_task_pk_1", "v_h_task_pk_1",
]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def find_column(columns, candidates):
    """
    Case-insensitive column lookup.  Tries each candidate as-is, then
    appends _1 (TOPCAT convention) and _x (pandas merge convention).
    """
    lower_map = {str(c).lower(): str(c) for c in columns}
    for candidate in candidates:
        key = str(candidate).lower()
        if key in lower_map:
            return lower_map[key]
        if key + "_1" in lower_map:
            return lower_map[key + "_1"]
        if key + "_x" in lower_map:
            return lower_map[key + "_x"]
    return None


def as_float(series):
    return pd.to_numeric(series, errors="coerce")


def copy_col(df, out, new_name, candidates):
    """Copy first matching column into out[new_name] as float. No-op if absent."""
    col = find_column(df.columns, candidates)
    if col is not None:
        out[new_name] = as_float(df[col])


def pick_best(out, col_priority, best_col, source_col):
    """Fill best_col with first finite value from col_priority; record source."""
    best_val = np.full(len(out), np.nan)
    best_src = np.full(len(out), "", dtype=object)
    for col in col_priority:
        if col not in out.columns:
            continue
        v    = as_float(out[col]).to_numpy()
        fill = np.isfinite(v) & ~np.isfinite(best_val)
        best_val[fill] = v[fill]
        best_src[fill] = col
    out[best_col]  = best_val
    out[source_col] = best_src

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_input(fits_path):
    tbl = Table.read(fits_path)
    scalar_names = [n for n in tbl.colnames if len(getattr(tbl[n], "shape", ())) <= 1]
    dropped = [n for n in tbl.colnames if n not in scalar_names]
    if dropped:
        print(f"  Dropping {len(dropped)} multi-dim columns")
    return tbl[scalar_names].to_pandas()

# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------

def build_identifiers(df, out):
    for new, candidates in [
        ("sdss_id",            ["sdss_id"]),
        ("sdss4_apogee_id",    ["sdss4_apogee_id"]),
        ("gaia_dr2_source_id", ["gaia_dr2_source_id"]),
        ("gaia_dr3_source_id", ["gaia_dr3_source_id", "source_id"]),
        ("tic_v8_id",          ["tic_v8_id"]),
        ("catalogid",          ["catalogid"]),
        ("catalogid21",        ["catalogid21"]),
        ("catalogid25",        ["catalogid25"]),
        ("catalogid31",        ["catalogid31"]),
        ("row_id",             ["row_id"]),
    ]:
        col = find_column(df.columns, candidates)
        if col is not None:
            out[new] = df[col]

# ---------------------------------------------------------------------------
# Astrometry
# ---------------------------------------------------------------------------

def build_astrometry(df, out):
    for new, candidates in [
        ("ra",                ["ra_1", "ra_icrs", "ra", "RAdeg", "RAJ2000"]),
        ("dec",               ["dec_1", "dec_icrs", "dec", "DEdeg", "DEJ2000"]),
        ("l",                 ["l", "glon"]),
        ("b",                 ["b", "glat"]),
        ("parallax",          ["plx", "parallax", "Plx"]),
        ("e_parallax",        ["e_plx", "parallax_error", "e_Plx"]),
        ("pmra",              ["pmra", "pmRA"]),
        ("e_pmra",            ["e_pmra", "e_pmRA"]),
        ("pmde",              ["pmde", "pmDE"]),
        ("e_pmde",            ["e_pmde", "e_pmDE"]),
        ("v_rad",             ["gaia_v_rad", "v_rad"]),
        ("e_v_rad",           ["gaia_e_v_rad", "e_v_rad"]),
        ("r_med_geo_pc",      ["r_med_geo"]),
        ("r_lo_geo_pc",       ["r_lo_geo"]),
        ("r_hi_geo_pc",       ["r_hi_geo"]),
        ("r_med_photogeo_pc", ["r_med_photogeo"]),
        ("r_lo_photogeo_pc",  ["r_lo_photogeo"]),
        ("r_hi_photogeo_pc",  ["r_hi_photogeo"]),
    ]:
        copy_col(df, out, new, candidates)

# ---------------------------------------------------------------------------
# Teff
# ---------------------------------------------------------------------------

def build_teff_columns(df, out):
    sources = [
        ("teff_astra",   ["teff"],          ["e_teff"]),
        ("teff_doppler", ["doppler_teff"],   ["doppler_e_teff"]),
        ("teff_gspspec", ["Teff-S"],         ["b_Teff-S", "B_Teff-S"]),
        ("teff_gaia_ap", ["Teff_x", "Teff"], ["b_Teff_x", "B_Teff_xa"]),
        ("teff_zgr",     ["zgr_teff"],        ["zgr_e_teff"]),
        ("teff_xgboost", ["teff_xgboost"],    []),
        ("teff_irfm",    ["irfm_teff"],        []),
    ]
    for val_name, val_cands, err_cands in sources:
        col = find_column(df.columns, val_cands)
        if col is not None:
            out[val_name] = as_float(df[col])
            if err_cands:
                err_col = find_column(df.columns, err_cands)
                if err_col is not None:
                    out[f"e_{val_name}"] = as_float(df[err_col])

    pick_best(out,
              ["teff_astra", "teff_gspspec", "teff_gaia_ap",
               "teff_zgr", "teff_doppler", "teff_xgboost", "teff_irfm"],
              "teff_best", "teff_source_best")

# ---------------------------------------------------------------------------
# logg
# ---------------------------------------------------------------------------

def build_logg_columns(df, out):
    sources = [
        ("logg_astra",   ["logg"],          ["e_logg"]),
        ("logg_doppler", ["doppler_logg"],   ["doppler_e_logg"]),
        ("logg_gspspec", ["logg-S"],         ["b_logg-S", "B_logg-S"]),
        ("logg_gaia_ap", ["logg_x", "logg"], ["b_logg_x", "B_logg_xa"]),
        ("logg_zgr",     ["zgr_logg"],        ["zgr_e_logg"]),
        ("logg_xgboost", ["logg_xgboost"],    []),
    ]
    for val_name, val_cands, err_cands in sources:
        if val_name in out.columns:
            continue
        col = find_column(df.columns, val_cands)
        if col is not None:
            out[val_name] = as_float(df[col])
            if err_cands:
                err_col = find_column(df.columns, err_cands)
                if err_col is not None:
                    out[f"e_{val_name}"] = as_float(df[err_col])

    pick_best(out,
              ["logg_astra", "logg_gspspec", "logg_gaia_ap",
               "logg_zgr", "logg_doppler", "logg_xgboost"],
              "logg_best", "logg_source_best")

# ---------------------------------------------------------------------------
# Metallicity
# ---------------------------------------------------------------------------

def build_metallicity_columns(df, out):
    sources = [
        ("feh_astra",   ["fe_h"],                           ["e_fe_h"]),
        ("mh_astra",    ["m_h_atm"],                         ["e_m_h_atm"]),
        ("feh_doppler", ["doppler_fe_h"],                    ["doppler_e_fe_h"]),
        ("mh_gspspec",  ["[M/H]-S"],                          ["b_[M/H]-S"]),
        ("feh_gspspec", ["[Fe/H]-S"],                         ["b_[Fe/H]-S"]),
        ("feh_gaia_ap", ["[Fe/H]_1"],                         ["b_[Fe/H]_x"]),
        ("feh_zgr",     ["zgr_fe_h"],                          ["zgr_e_fe_h"]),
        ("mh_xgboost",  ["mh_xgboost"],                        []),
        ("feh_bdbs",    ["[Fe/H]_2", "feh_bdbs_spec"],          ["e_[Fe/H]", "e_feh_bdbs_spec"]),
        ("feh_bensby",  ["bensby_fe_h", "bensby_[Fe/H]"],      []),
    ]
    for val_name, val_cands, err_cands in sources:
        col = find_column(df.columns, val_cands)
        if col is not None:
            out[val_name] = as_float(df[col])
            if err_cands:
                err_col = find_column(df.columns, err_cands)
                if err_col is not None:
                    out[f"e_{val_name}"] = as_float(df[err_col])

    pick_best(out,
              ["mh_astra", "feh_astra", "mh_gspspec", "feh_gspspec",
               "feh_gaia_ap", "feh_zgr", "feh_doppler",
               "feh_bdbs", "feh_bensby", "mh_xgboost"],
              "metallicity_best", "metallicity_source_best")

# ---------------------------------------------------------------------------
# Radius
# ---------------------------------------------------------------------------

def build_radius_columns(df, out):
    sources = [
        ("radius_astra_rsun",   ["radius"],    []),
        ("radius_gaia_ap_rsun", ["Rad"],       ["b_Rad_x", "B_Rad_xa"]),
        ("radius_flame_rsun",   ["Rad-Flame"], ["b_Rad-Flame_x", "B_Rad-Flame_xa"]),
    ]
    for val_name, val_cands, err_cands in sources:
        col = find_column(df.columns, val_cands)
        if col is not None:
            out[val_name] = as_float(df[col])
            if err_cands:
                err_col = find_column(df.columns, err_cands)
                if err_col is not None:
                    out[f"e_{val_name}"] = as_float(df[err_col])

    pick_best(out,
              ["radius_flame_rsun", "radius_astra_rsun", "radius_gaia_ap_rsun"],
              "radius_best_rsun", "radius_source_best")

# ---------------------------------------------------------------------------
# Luminosity  (three independent estimates)
# ---------------------------------------------------------------------------

def build_luminosity_columns(df, out):
    # 1. FLAME direct
    col = find_column(df.columns, ["Lum-Flame"])
    if col is not None:
        out["lum_flame_lsun"] = as_float(df[col])
        err_col = find_column(df.columns, ["b_Lum-Flame_x", "B_Lum-Flame_xa"])
        if err_col is not None:
            out["e_lum_flame_lsun"] = as_float(df[err_col])

    # 2. Derived from best radius + best Teff
    if "radius_best_rsun" in out.columns and "teff_best" in out.columns:
        r = as_float(out["radius_best_rsun"]).to_numpy()
        t = as_float(out["teff_best"]).to_numpy()
        out["lum_r_teff_lsun"] = np.where(
            (r > 0) & (t > 0), r**2 * (t / T_SUN)**4, np.nan
        )

    # 3. From parallax + dereddened G magnitude
    plx   = as_float(out.get("parallax",   pd.Series(np.nan, index=out.index))).to_numpy()
    e_plx = as_float(out.get("e_parallax", pd.Series(np.nan, index=out.index))).to_numpy()

    # Build gaia_g0 if step 3 didn't produce it
    if "gaia_g0" not in out.columns:
        g_col  = find_column(df.columns, ["g_mag", "phot_g_mean_mag", "Gmag"])
        ag_col = find_column(df.columns, ["AG", "a_g_val", "A0"])
        if g_col is not None:
            g_app = as_float(df[g_col]).to_numpy()
            a_g   = as_float(df[ag_col]).to_numpy() if ag_col is not None else np.zeros(len(df))
            out["gaia_g0"] = g_app - np.where(np.isfinite(a_g), a_g, 0.0)
            out["a_g"]     = a_g

    if "gaia_g0" in out.columns:
        g0   = as_float(out["gaia_g0"]).to_numpy()
        teff = as_float(out.get("teff_best",
                                pd.Series(np.nan, index=out.index))).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            plx_snr = np.where(e_plx > 0, plx / e_plx, np.nan)

        # Require parallax SNR >= 10 (tightened from 5) and cap distance at
        # 20 kpc to exclude clearly unphysical inversions for distant bulge stars.
        # Bulge stars at ~8 kpc have plx ~ 0.12 mas; anything with plx < 0.05 mas
        # (>20 kpc) is unreliable for luminosity purposes.
        good = (
            np.isfinite(plx) & (plx > 0) &
            (plx_snr >= 10.0) &
            (1000.0 / plx <= 20000.0) &
            np.isfinite(g0)
        )

        dist_pc = np.full(len(out), np.nan)
        mu      = np.full(len(out), np.nan)
        abs_g0  = np.full(len(out), np.nan)
        lum_plx = np.full(len(out), np.nan)

        dist_pc[good] = 1000.0 / plx[good]
        mu[good]      = 5.0 * np.log10(dist_pc[good]) - 5.0
        abs_g0[good]  = g0[good] - mu[good]

        # Bolometric correction BC_G(Teff) from Andrae et al. 2018 (Table A1),
        # valid for giants.  BC_G is defined so that
        #   M_bol = M_G + BC_G
        # Anchor points (Teff, BC_G):
        #   3500 K: -0.90,  4000 K: -0.38,  4500 K: -0.13,
        #   5000 K: -0.02,  5500 K: +0.01,  6000 K: +0.02
        # BC_G_sun = -0.08  (Casagrande & VandenBerg 2018, consistent with M_G_sun=5.12)
        # The luminosity ratio uses delta_BC = BC_G_sun - BC_G_star.
        BC_G_SUN     = -0.08
        bc_teff_grid = np.array([3500., 4000., 4500., 5000., 5500., 6000.])
        bc_val_grid  = np.array([-0.90, -0.38, -0.13, -0.02,  0.01,  0.02])

        teff_good = teff[good]
        teff_clamped = np.clip(teff_good, bc_teff_grid[0], bc_teff_grid[-1])
        bc_g_star = np.interp(teff_clamped, bc_teff_grid, bc_val_grid)
        # Where Teff is not available, set BC_G_star = BC_G_sun (no correction)
        no_teff = ~np.isfinite(teff_good)
        bc_g_star[no_teff] = BC_G_SUN

        delta_bc = BC_G_SUN - bc_g_star   # positive for cool giants
        lum_plx[good] = 10.0 ** ((M_G_SUN - abs_g0[good] + delta_bc) / 2.5)

        out["gaia_distance_pc"]  = dist_pc
        out["gaia_abs_g0"]       = abs_g0
        out["lum_parallax_lsun"] = lum_plx

    pick_best(out,
              ["lum_flame_lsun", "lum_r_teff_lsun", "lum_parallax_lsun"],
              "luminosity_best_lsun", "luminosity_source_best")

    lum = as_float(out["luminosity_best_lsun"]).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log10_lum_best_lsun"] = np.where(lum > 0, np.log10(lum), np.nan)

# ---------------------------------------------------------------------------
# Mass
# ---------------------------------------------------------------------------

def build_mass_columns(df, out):
    col = find_column(df.columns, ["Mass-Flame"])
    if col is not None:
        out["mass_flame_msun"] = as_float(df[col])
        err_col = find_column(df.columns, ["b_Mass-Flame_x", "B_Mass-Flame_xa"])
        if err_col is not None:
            out["e_mass_flame_msun"] = as_float(df[err_col])

# ---------------------------------------------------------------------------
# Age
# ---------------------------------------------------------------------------

def build_age_columns(df, out):
    col = find_column(df.columns, ["Age-Flame"])
    if col is not None:
        out["age_flame_gyr"] = as_float(df[col])
        err_col = find_column(df.columns, ["b_Age-Flame_x", "B_Age-Flame_xa"])
        if err_col is not None:
            out["e_age_flame_gyr"] = as_float(df[err_col])

# ---------------------------------------------------------------------------
# Photometry and reddening (pipeline products)
# ---------------------------------------------------------------------------

def build_photometry_columns(df, out):
    # Pipeline-derived products from step 3
    passthrough = [
        "j_mag_best", "h_mag_best", "ks_mag_best",
        "j_mag_source", "h_mag_source", "ks_mag_source",
        "e_j_mag_best", "e_h_mag_best", "e_ks_mag_best",
        "j_minus_h_best", "h_minus_ks_best", "j_minus_ks_best",
        "ext_e_jks", "ext_e_jks_err",
        "a_j", "a_h", "a_ks",
        "j0", "h0", "ks0",
        "j0_minus_h0", "h0_minus_ks0", "j0_minus_ks0",
        "sightline_id", "sightline_glon", "sightline_glat",
        "sightline_tile", "sightline_bin_0p1deg",
        "ext_map_match_sep_arcsec",
    ]
    for col in passthrough:
        if col in df.columns and col not in out.columns:
            out[col] = df[col]

    # Gaia optical dereddened
    if "gaia_g0" not in out.columns:
        g_col  = find_column(df.columns, ["g_mag",  "phot_g_mean_mag"])
        bp_col = find_column(df.columns, ["bp_mag", "phot_bp_mean_mag"])
        rp_col = find_column(df.columns, ["rp_mag", "phot_rp_mean_mag"])
        ag_col = find_column(df.columns, ["AG",     "a_g_val", "A0"])
        eb_col = find_column(df.columns, ["E(BP-RP)", "e_bp_min_rp_val"])
        if g_col is not None:
            g_app  = as_float(df[g_col])
            a_g    = as_float(df[ag_col])   if ag_col  is not None else pd.Series(0.0, index=df.index)
            out["gaia_g0"] = g_app - a_g.fillna(0.0)
            out["a_g"]     = a_g
            # BP/RP dereddening is NOT computed here. Gaia-provided AG is reliable
            # for G0, but the corresponding A_BP and A_RP are strongly
            # colour-/temperature-dependent for cool reddened giants and cannot
            # be approximated with a fixed E(BP-RP) ratio.

    # EBV estimates
    for new, cands in [
        ("ebv",              ["ebv"]),
        ("e_ebv",            ["e_ebv"]),
        ("ebv_sfd",          ["ebv_sfd"]),
        ("ebv_rjce_glimpse", ["ebv_rjce_glimpse"]),
        ("ebv_rjce_allwise", ["ebv_rjce_allwise"]),
    ]:
        if new not in out.columns:
            copy_col(df, out, new, cands)

# ---------------------------------------------------------------------------
# Raw magnitudes (all available, at the end)
# ---------------------------------------------------------------------------

def append_raw_magnitudes(df, out):
    mag_cols = [
        # Gaia
        ("mag_g",            ["g_mag",  "phot_g_mean_mag"]),
        ("mag_bp",           ["bp_mag", "phot_bp_mean_mag"]),
        ("mag_rp",           ["rp_mag", "phot_rp_mean_mag"]),
        # 2MASS from ASTRA cross-match
        ("mag_j_astra",      ["j_mag"]),
        ("e_mag_j_astra",    ["e_j_mag"]),
        ("mag_h_astra",      ["h_mag"]),
        ("e_mag_h_astra",    ["e_h_mag"]),
        ("mag_k_astra",      ["k_mag"]),
        ("e_mag_k_astra",    ["e_k_mag"]),
        # WISE
        ("mag_w1",           ["w1_mag"]),
        ("e_mag_w1",         ["e_w1_mag"]),
        ("mag_w2",           ["w2_mag"]),
        ("e_mag_w2",         ["e_w2_mag"]),
        # Spitzer 4.5um
        ("mag_4_5",          ["mag4_5"]),
        ("e_mag_4_5",        ["d4_5m"]),
        # VIRAC2
        ("mag_z_virac",      ["vvv_Zmag"]),
        ("e_mag_z_virac",    ["vvv_e_Zmag"]),
        ("mag_y_virac",      ["vvv_Ymag"]),
        ("e_mag_y_virac",    ["vvv_e_Ymag"]),
        ("mag_j_virac",      ["vvv_Jmag"]),
        ("e_mag_j_virac",    ["vvv_e_Jmag"]),
        ("mag_h_virac",      ["vvv_Hmag"]),
        ("e_mag_h_virac",    ["vvv_e_Hmag"]),
        ("mag_ks_virac",     ["vvv_Ksmag"]),
        ("e_mag_ks_virac",   ["vvv_e_Ksmag"]),
        # 2MASS from CDS cross-match
        ("mag_j_2mass",      ["tmass_Jmag"]),
        ("e_mag_j_2mass",    ["tmass_e_Jmag"]),
        ("mag_h_2mass",      ["tmass_Hmag"]),
        ("e_mag_h_2mass",    ["tmass_e_Hmag"]),
        ("mag_k_2mass",      ["tmass_Kmag"]),
        ("e_mag_k_2mass",    ["tmass_e_Kmag"]),
        # BDBS optical
        ("mag_u_jkc",        ["u_jkc_mag"]),
        ("mag_b_jkc",        ["b_jkc_mag"]),
        ("mag_v_jkc",        ["v_jkc_mag"]),
        ("mag_r_jkc",        ["r_jkc_mag"]),
        ("mag_i_jkc",        ["i_jkc_mag"]),
        ("mag_u_sdss",       ["u_sdss_mag"]),
        ("mag_g_sdss",       ["g_sdss_mag"]),
        ("mag_r_sdss",       ["r_sdss_mag"]),
        ("mag_i_sdss",       ["i_sdss_mag"]),
        ("mag_z_sdss",       ["z_sdss_mag"]),
        ("mag_y_ps1",        ["y_ps1_mag"]),
        # BDBS spectroscopic metallicity (this is a spectroscopic value, not photometric)
        ("feh_bdbs_spec",    ["[Fe/H]_2"]),
        ("e_feh_bdbs_spec",  ["e_[Fe/H]"]),
    ]
    for new_name, cands in mag_cols:
        if new_name not in out.columns:
            copy_col(df, out, new_name, cands)

# ---------------------------------------------------------------------------
# Provenance flags
# ---------------------------------------------------------------------------

def build_flags(out):
    def finite(col):
        if col not in out.columns:
            return pd.Series(False, index=out.index)
        return np.isfinite(as_float(out[col]))

    def nonempty(col):
        if col not in out.columns:
            return pd.Series(False, index=out.index)
        return out[col].fillna("").astype(str).str.strip() != ""

    out["has_virac2"]          = finite("mag_j_virac")
    out["has_2mass"]           = finite("mag_j_2mass")
    out["has_reddening"]       = finite("ext_e_jks")
    out["has_dereddened_phot"] = finite("j0") & finite("h0") & finite("ks0")
    out["has_sightline"]       = nonempty("sightline_id")
    out["has_teff_best"]       = finite("teff_best")
    out["has_logg_best"]       = finite("logg_best")
    out["has_metallicity_best"]= finite("metallicity_best")
    out["has_radius_best"]     = finite("radius_best_rsun")
    out["has_luminosity_best"] = finite("luminosity_best_lsun")
    out["has_lum_flame"]       = finite("lum_flame_lsun")
    out["has_lum_r_teff"]      = finite("lum_r_teff_lsun")
    out["has_lum_parallax"]    = finite("lum_parallax_lsun")
    out["has_mass_flame"]      = finite("mass_flame_msun")
    out["has_age_flame"]       = finite("age_flame_gyr")
    out["has_core_params"]     = (out["has_teff_best"] & out["has_logg_best"] &
                                  out["has_metallicity_best"] & out["has_luminosity_best"])

    out["calibration_ready_phot"]   = out["has_dereddened_phot"] & out["has_reddening"]
    out["calibration_ready_params"] = out["has_core_params"]
    out["calibration_ready"]        = out["calibration_ready_phot"] & out["calibration_ready_params"]

    print(f"  calibration_ready          : {out['calibration_ready'].sum():,}")
    print(f"  calibration_ready_phot     : {out['calibration_ready_phot'].sum():,}")
    print(f"  has_core_params            : {out['has_core_params'].sum():,}")
    print(f"  has_mass_flame             : {out['has_mass_flame'].sum():,}")
    print(f"  has_age_flame              : {out['has_age_flame'].sum():,}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(out, ready):
    def n(col):
        if col not in out.columns:
            return 0
        return int(np.isfinite(as_float(out[col])).sum())

    ejks = as_float(out.get("ext_e_jks", pd.Series(dtype=float))).to_numpy()

    lines = [
        "Roman Bulge Calibration Pipeline – Final Summary",
        "=" * 50,
        f"Total stars                            : {len(out):,}",
        f"Calibration-ready (phot + params)      : {len(ready):,}",
        f"Calibration-ready (phot only)          : {out['calibration_ready_phot'].sum():,}",
        f"Has core params (Teff+logg+[M/H]+L)    : {out['has_core_params'].sum():,}",
        "",
        "Teff (per source)",
        "-" * 20,
        f"  teff_best                            : {n('teff_best'):,}",
        f"  teff_astra                           : {n('teff_astra'):,}",
        f"  teff_gspspec                         : {n('teff_gspspec'):,}",
        f"  teff_gaia_ap                         : {n('teff_gaia_ap'):,}",
        f"  teff_zgr                             : {n('teff_zgr'):,}",
        f"  teff_xgboost                         : {n('teff_xgboost'):,}",
        f"  teff_irfm                            : {n('teff_irfm'):,}",
        "",
        "logg (per source)",
        "-" * 20,
        f"  logg_best                            : {n('logg_best'):,}",
        f"  logg_astra                           : {n('logg_astra'):,}",
        f"  logg_gspspec                         : {n('logg_gspspec'):,}",
        f"  logg_gaia_ap                         : {n('logg_gaia_ap'):,}",
        f"  logg_zgr                             : {n('logg_zgr'):,}",
        "",
        "Metallicity (per source)",
        "-" * 20,
        f"  metallicity_best                     : {n('metallicity_best'):,}",
        f"  mh_astra                             : {n('mh_astra'):,}",
        f"  feh_astra                            : {n('feh_astra'):,}",
        f"  mh_gspspec                           : {n('mh_gspspec'):,}",
        f"  feh_gaia_ap                          : {n('feh_gaia_ap'):,}",
        f"  feh_zgr                              : {n('feh_zgr'):,}",
        f"  mh_xgboost                           : {n('mh_xgboost'):,}",
        f"  feh_bdbs                             : {n('feh_bdbs'):,}",
        "",
        "Radius (per source)",
        "-" * 20,
        f"  radius_best_rsun                     : {n('radius_best_rsun'):,}",
        f"  radius_flame_rsun                    : {n('radius_flame_rsun'):,}",
        f"  radius_astra_rsun                    : {n('radius_astra_rsun'):,}",
        "",
        "Luminosity (independent estimates)",
        "-" * 20,
        f"  luminosity_best_lsun                 : {n('luminosity_best_lsun'):,}",
        f"  lum_flame_lsun                       : {n('lum_flame_lsun'):,}",
        f"  lum_r_teff_lsun                      : {n('lum_r_teff_lsun'):,}",
        f"  lum_parallax_lsun                    : {n('lum_parallax_lsun'):,}",
        "",
        "Mass and age",
        "-" * 20,
        f"  mass_flame_msun                      : {n('mass_flame_msun'):,}",
        f"  age_flame_gyr                        : {n('age_flame_gyr'):,}",
        "",
        "Photometry",
        "-" * 20,
        f"  VIRAC2 matched                       : {out['has_virac2'].sum():,}",
        f"  2MASS matched                        : {out['has_2mass'].sum():,}",
        f"  Has E(J-Ks)                          : {out['has_reddening'].sum():,}",
        f"  Dereddened J0/H0/Ks0                 : {out['has_dereddened_phot'].sum():,}",
    ]

    if np.isfinite(ejks).any():
        lines += [
            "",
            "Reddening",
            "-" * 20,
            f"  E(J-Ks) median                       : {np.nanmedian(ejks):.4f}",
            f"  E(J-Ks) 16th pct                     : {np.nanpercentile(ejks, 16):.4f}",
            f"  E(J-Ks) 84th pct                     : {np.nanpercentile(ejks, 84):.4f}",
        ]

    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Reading {INPUT_FITS} ...")
    df = load_input(INPUT_FITS)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    # Prune bookkeeping from input before anything else
    to_drop = [c for c in BOOKKEEPING_COLUMNS if c in df.columns]
    df = df.drop(columns=to_drop)
    print(f"  Dropped {len(to_drop)} bookkeeping columns")

    # Build the output DataFrame column-group by column-group
    out = pd.DataFrame(index=df.index)

    print("Building identifiers ...")
    build_identifiers(df, out)

    print("Building astrometry ...")
    build_astrometry(df, out)

    print("Building Teff columns ...")
    build_teff_columns(df, out)

    print("Building logg columns ...")
    build_logg_columns(df, out)

    print("Building metallicity columns ...")
    build_metallicity_columns(df, out)

    print("Building radius columns ...")
    build_radius_columns(df, out)

    print("Building luminosity columns ...")
    build_luminosity_columns(df, out)

    print("Building mass columns ...")
    build_mass_columns(df, out)

    print("Building age columns ...")
    build_age_columns(df, out)

    print("Building photometry columns ...")
    build_photometry_columns(df, out)

    print("Appending raw magnitudes ...")
    append_raw_magnitudes(df, out)

    print("Adding provenance flags ...")
    build_flags(out)

    print("\nHarvest summary:")
    for label, col in [
        ("Teff (best)",     "teff_best"),
        ("logg (best)",     "logg_best"),
        ("[M/H] (best)",    "metallicity_best"),
        ("Radius (best)",   "radius_best_rsun"),
        ("Lum (best)",      "luminosity_best_lsun"),
        ("Lum (FLAME)",     "lum_flame_lsun"),
        ("Lum (R·Teff)",    "lum_r_teff_lsun"),
        ("Lum (parallax)",  "lum_parallax_lsun"),
        ("Mass (FLAME)",    "mass_flame_msun"),
        ("Age (FLAME)",     "age_flame_gyr"),
    ]:
        nval = int(np.isfinite(as_float(out[col])).sum()) if col in out.columns else 0
        print(f"  {label:<20}: {nval:,}")

    ready = out.loc[out["calibration_ready"]].copy()

    print(f"\nWriting outputs ...")
    Table.from_pandas(out).write(MASTER_FITS, overwrite=True)
    out.to_csv(MASTER_CSV, index=False)
    print(f"  Master    : {MASTER_FITS}  ({len(out):,} rows, {len(out.columns)} columns)")

    Table.from_pandas(ready).write(READY_FITS, overwrite=True)
    ready.to_csv(READY_CSV, index=False)
    print(f"  Cal-ready : {READY_FITS}  ({len(ready):,} rows)")

    write_summary(out, ready)
    print(f"  Summary   : {SUMMARY_TXT}")
    print("\nDone.")


if __name__ == "__main__":
    main()