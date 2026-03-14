#!/usr/bin/env python3
"""
Step 4 – Photometry, extinction, dereddened magnitudes, sightlines
===================================================================
Builds the full photometric picture from the cross-matched catalog:

  1. Best J/H/Ks  – prefer VIRAC2, fall back to 2MASS
  2. Extinction   – E(J-Ks) -> A_J, A_H, A_Ks  (Nishiyama bulge law)
  3. Dereddened   – J0, H0, Ks0 and colours
  4. Sightlines   – label each star with its VVV reddening-map cell

Input  : step3_xmatched.fits
Output : step4_photometry.fits
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
INPUT_FITS  = pathlib.Path("step3_xmatched.fits")
OUTPUT_FITS = pathlib.Path("step4_photometry.fits")

# ---------------------------------------------------------------------------
# Extinction law coefficients  (Nishiyama et al. style, bulge)
# ---------------------------------------------------------------------------
A_KS_PER_E_JKS = 0.528
A_H_PER_E_JKS  = 0.857
A_J_PER_E_JKS  = 1.528

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
# Best J/H/Ks photometry
# ---------------------------------------------------------------------------

def build_best_photometry(df):
    print("Building best J/H/Ks photometry (VIRAC2 preferred, 2MASS fallback) ...")

    vvv_j  = find_column(df.columns, ["vvv_Jmag",  "vvv_j",  "vvv_j_mag"])
    vvv_h  = find_column(df.columns, ["vvv_Hmag",  "vvv_h",  "vvv_h_mag"])
    vvv_ks = find_column(df.columns, ["vvv_Ksmag", "vvv_ks", "vvv_ks_mag", "vvv_K"])

    tm_j   = find_column(df.columns, ["tmass_Jmag", "tmass_j_m",  "tmass_j"])
    tm_h   = find_column(df.columns, ["tmass_Hmag", "tmass_h_m",  "tmass_h"])
    tm_ks  = find_column(df.columns, ["tmass_Kmag", "tmass_ks_m", "tmass_ks", "tmass_k_m"])

    def merge_band(vvv_col, tmass_col, band_name):
        best   = np.full(len(df), np.nan)
        source = np.full(len(df), "", dtype=object)

        if vvv_col is not None:
            v    = as_float(df[vvv_col]).to_numpy()
            good = np.isfinite(v)
            best[good]   = v[good]
            source[good] = "VIRAC2"

        if tmass_col is not None:
            t    = as_float(df[tmass_col]).to_numpy()
            good = np.isfinite(t) & ~np.isfinite(best)
            best[good]   = t[good]
            source[good] = "2MASS"

        df[f"{band_name}_mag_best"]   = best
        df[f"{band_name}_mag_source"] = source

    merge_band(vvv_j,  tm_j,  "j")
    merge_band(vvv_h,  tm_h,  "h")
    merge_band(vvv_ks, tm_ks, "ks")

    df["j_minus_h_best"]  = df["j_mag_best"]  - df["h_mag_best"]
    df["h_minus_ks_best"] = df["h_mag_best"]  - df["ks_mag_best"]
    df["j_minus_ks_best"] = df["j_mag_best"]  - df["ks_mag_best"]

    print(f"  J best : {np.isfinite(df['j_mag_best']).sum():,}")
    print(f"  H best : {np.isfinite(df['h_mag_best']).sum():,}")
    print(f"  Ks best: {np.isfinite(df['ks_mag_best']).sum():,}")
    return df

# ---------------------------------------------------------------------------
# Extinction
# ---------------------------------------------------------------------------

def compute_extinction(df):
    print("Computing extinction from E(J-Ks) ...")

    ejks_col = find_column(df.columns, ["ext_EJKs", "ext_e_jks", "ext_ejks",
                                         "ext_E_JKs", "ext_EJKs_1"])
    if ejks_col is None:
        print("  WARNING: E(J-Ks) column not found; A_J/A_H/A_Ks set to NaN.")
        df["ext_e_jks"] = np.nan
        df["a_j"]  = np.nan
        df["a_h"]  = np.nan
        df["a_ks"] = np.nan
        return df

    ejks = as_float(df[ejks_col]).to_numpy()
    df["ext_e_jks"] = ejks
    df["a_j"]       = A_J_PER_E_JKS  * ejks
    df["a_h"]       = A_H_PER_E_JKS  * ejks
    df["a_ks"]      = A_KS_PER_E_JKS * ejks

    finite = np.isfinite(ejks)
    print(f"  E(J-Ks) available: {finite.sum():,}  "
          f"median={np.nanmedian(ejks):.3f}")
    return df

# ---------------------------------------------------------------------------
# Dereddened magnitudes
# ---------------------------------------------------------------------------

def compute_dereddened_mags(df):
    print("Computing dereddened magnitudes ...")

    df["j0"]  = df["j_mag_best"]  - df["a_j"]
    df["h0"]  = df["h_mag_best"]  - df["a_h"]
    df["ks0"] = df["ks_mag_best"] - df["a_ks"]

    df["j0_minus_h0"]  = df["j0"]  - df["h0"]
    df["h0_minus_ks0"] = df["h0"]  - df["ks0"]
    df["j0_minus_ks0"] = df["j0"]  - df["ks0"]

    print(f"  Ks0 available: {np.isfinite(df['ks0']).sum():,}")
    return df

# ---------------------------------------------------------------------------
# Sightline tagging
# ---------------------------------------------------------------------------

def tag_sightlines(df):
    print("Tagging sightlines from VVV reddening-map cell centres ...")

    glon_col = find_column(df.columns, ["ext_GLON", "ext_glon", "ext_l"])
    glat_col = find_column(df.columns, ["ext_GLAT", "ext_glat", "ext_b"])

    if glon_col is not None and glat_col is not None:
        df["sightline_glon"] = as_float(df[glon_col])
        df["sightline_glat"] = as_float(df[glat_col])
    else:
        l_col = find_column(df.columns, ["l", "glon"])
        b_col = find_column(df.columns, ["b", "glat"])
        if l_col and b_col:
            df["sightline_glon"] = (as_float(df[l_col]) * 2).round() / 2
            df["sightline_glat"] = (as_float(df[b_col]) * 2).round() / 2
        else:
            df["sightline_glon"] = np.nan
            df["sightline_glat"] = np.nan

    glon = df["sightline_glon"].to_numpy()
    glat = df["sightline_glat"].to_numpy()

    sightline_ids = []
    bin_ids       = []
    for i in range(len(df)):
        gl = glon[i]
        gb = glat[i]
        if np.isfinite(gl) and np.isfinite(gb):
            sightline_ids.append(f"l{gl:+.3f}_b{gb:+.3f}")
        else:
            sightline_ids.append("")

    l_col = find_column(df.columns, ["l", "glon"])
    b_col = find_column(df.columns, ["b", "glat"])
    if l_col and b_col:
        l_arr = as_float(df[l_col]).to_numpy()
        b_arr = as_float(df[b_col]).to_numpy()
        for i in range(len(df)):
            gl = l_arr[i]
            gb = b_arr[i]
            if np.isfinite(gl) and np.isfinite(gb):
                bin_ids.append(
                    f"l{round(gl / 0.1) * 0.1:+.1f}_b{round(gb / 0.1) * 0.1:+.1f}"
                )
            else:
                bin_ids.append("")
    else:
        bin_ids = [""] * len(df)

    df["sightline_id"]        = sightline_ids
    df["sightline_bin_0p1deg"] = bin_ids

    n_tagged = sum(s != "" for s in sightline_ids)
    print(f"  {n_tagged:,} stars tagged with sightline IDs")
    return df

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

    df = build_best_photometry(df)
    df = compute_extinction(df)
    df = compute_dereddened_mags(df)
    df = tag_sightlines(df)

    Table.from_pandas(df).write(OUTPUT_FITS, overwrite=True)
    print(f"\nWritten: {OUTPUT_FITS}  ({len(df):,} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
