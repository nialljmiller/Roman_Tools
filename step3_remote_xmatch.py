#!/usr/bin/env python3
"""
Step 3 – Remote CDS cross-matches
===================================
Uploads the selected sample to the CDS XMatch service and cross-matches
against three remote VizieR catalogs:

  - VIRAC2         (II/387/virac2)            – VVV/VVVX near-IR photometry
  - 2MASS PSC      (II/246/out)               – near-IR photometry fallback
  - VVV reddening  (J/A+A/644/A140/ejkmap)   – high-res E(J-Ks) map

Results are left-joined back onto the input table.

Input  : step2_selected.fits
Output : step3_xmatched.fits
"""

import pathlib
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import Table
from astropy.io import fits as astrofits
from astroquery.xmatch import XMatch
import warnings
from astropy.io.fits.verify import VerifyWarning
from astropy.units import UnitsWarning

warnings.filterwarnings("ignore", category=VerifyWarning)
warnings.filterwarnings("ignore", category=UnitsWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_FITS  = pathlib.Path("step2_selected.fits")
OUTPUT_FITS = pathlib.Path("step3_xmatched.fits")

# ---------------------------------------------------------------------------
# Remote catalog identifiers
# ---------------------------------------------------------------------------
VIRAC2_TABLE   = "vizier:II/387/virac2"
TMASS_TABLE    = "vizier:II/246/out"
VVV_RED_TABLE  = "vizier:J/A+A/644/A140/ejkmap"

VIRAC2_RADIUS_ARCSEC  = 1.0
TMASS_RADIUS_ARCSEC   = 2.0
VVV_RED_RADIUS_ARCSEC = 30.0   # reddening map cells are coarse

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def as_float(series):
    return pd.to_numeric(series, errors="coerce")

# ---------------------------------------------------------------------------
# XMatch helper
# ---------------------------------------------------------------------------

def _upload_xmatch(df, remote_table, radius_arcsec):
    """
    Upload ra_icrs / dec_icrs to CDS XMatch and return the joined result.
    Returns an empty DataFrame on failure.
    """
    upload = df[["ra_icrs", "dec_icrs"]].copy().reset_index(drop=False)
    upload.rename(columns={"index": "_local_idx",
                            "ra_icrs": "RA",
                            "dec_icrs": "DEC"}, inplace=True)

    result_tbl = XMatch.query(
        cat1=upload,
        cat2=remote_table,
        max_distance=radius_arcsec * u.arcsec,
        colRA1="RA",
        colDec1="DEC",
    )

    if result_tbl is None or len(result_tbl) == 0:
        return pd.DataFrame()

    rows = {}
    for name in result_tbl.colnames:
        col   = result_tbl[name]
        shape = getattr(col, "shape", ())
        if len(shape) > 1:
            rows[name] = [str(v.tolist()) for v in col]
        else:
            try:
                rows[name] = col.data.tolist()
            except AttributeError:
                rows[name] = list(col)
    return pd.DataFrame(rows)


def xmatch_and_join(df, remote_table, radius_arcsec, col_prefix, sep_col_name):
    """
    Run CDS XMatch and left-join the result back onto df.
    Keeps only the closest match per source.
    """
    print(f"  XMatch: {remote_table}  (radius={radius_arcsec}\") ...")
    try:
        result = _upload_xmatch(df, remote_table, radius_arcsec)
    except Exception as exc:
        print(f"    WARNING: failed ({exc}); continuing without this catalog.")
        return df

    if result.empty:
        print("    No matches returned.")
        return df

    rename_map = {c: f"{col_prefix}_{c}" for c in result.columns
                  if c not in ("_local_idx", "angDist")}
    result.rename(columns=rename_map, inplace=True)
    result.rename(columns={"angDist": sep_col_name}, inplace=True)

    result.sort_values(sep_col_name, inplace=True)
    result.drop_duplicates(subset=["_local_idx"], keep="first", inplace=True)
    result.set_index("_local_idx", inplace=True)

    df = df.join(result, how="left")
    matched = df[sep_col_name].notna().sum()
    print(f"    Matched {matched:,} / {len(df):,} rows")
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

    print("\nRunning remote CDS cross-matches ...")
    df = xmatch_and_join(df, VIRAC2_TABLE,  VIRAC2_RADIUS_ARCSEC,  "vvv",  "virac2_sep_arcsec")
    df = xmatch_and_join(df, TMASS_TABLE,   TMASS_RADIUS_ARCSEC,   "tmass","tmass_sep_arcsec")
    df = xmatch_and_join(df, VVV_RED_TABLE, VVV_RED_RADIUS_ARCSEC, "ext",  "ext_map_match_sep_arcsec")

    Table.from_pandas(df).write(OUTPUT_FITS, overwrite=True)
    print(f"\nWritten: {OUTPUT_FITS}  ({len(df):,} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
