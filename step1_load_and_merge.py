#!/usr/bin/env python3
"""
Step 1 – Load ASTRA and merge local catalogs
=============================================
Reads astra.fits and left-joins three local catalogs:
  - gaiaXGBoost.fits  (Gaia DR3 XGBoost stellar parameters, joined by source_id)
  - bdbs.fits         (BDBS photometry, positional 1 arcsec)
  - bensby.fits       (Bensby high-res spectroscopy, positional 2 arcsec)

Memory strategy
---------------
- astra.fits (2.2 GB): opened with memmap=True so data is paged from disk
  on demand rather than copied wholesale into RAM.  All columns are kept.
  Multi-dimensional columns (bit-flag arrays etc.) are stored as string
  representations in the DataFrame so nothing is silently lost.

- gaiaXGBoost.fits (8 GB): the source_id column is read first (tiny).
  A row-index mask is built from the ASTRA source_ids.  Only the matching
  rows are then loaded from disk.  The full 8 GB is never in RAM.

- bdbs.fits (307 MB) and bensby.fits (830 KB): small enough to load normally.

Input  : astra.fits
Output : step1_merged.fits
"""

import pathlib
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.io import fits as astrofits
from astropy.io.fits.verify import VerifyWarning
from astropy.units import UnitsWarning
import warnings

warnings.filterwarnings("ignore", category=VerifyWarning)
warnings.filterwarnings("ignore", category=UnitsWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ASTRA_FITS   = pathlib.Path("astra.fits")
XGBOOST_FITS = pathlib.Path("gaiaXGBoost.fits")
BDBS_FITS    = pathlib.Path("bdbs.fits")
BENSBY_FITS  = pathlib.Path("bensby.fits")
OUTPUT_FITS  = pathlib.Path("step1_merged.fits")

BDBS_RADIUS_ARCSEC   = 1.0
BENSBY_RADIUS_ARCSEC = 2.0

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


def table_to_dataframe_all_columns(tbl):
    """
    Convert an astropy Table to a pandas DataFrame keeping every column.

    Multi-dimensional columns cannot be stored as numeric arrays in pandas,
    so they are converted to their string representation.  This preserves
    them in the output FITS file (as string columns) without crashing pandas.
    """
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

# ---------------------------------------------------------------------------
# Load ASTRA  (memory-mapped: 2.2 GB file, all columns kept)
# ---------------------------------------------------------------------------

def load_astra(fits_path):
    """
    Load ASTRA (HDU 2) using astropy memory-mapping.

    memmap=True means astropy maps the file data section into virtual memory
    pages.  Data is read from disk only when a column is accessed, so the
    full 2.2 GB is never copied into RAM at once.
    """
    print(f"Loading ASTRA from {fits_path} (HDU 2, memmap) ...")
    with astrofits.open(fits_path, memmap=True) as hdul:
        tbl = Table(hdul[2].data)
    df = table_to_dataframe_all_columns(tbl)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")
    return df

# ---------------------------------------------------------------------------
# Merge Gaia XGBoost  (row-filtered: the 8 GB file is never fully loaded)
# ---------------------------------------------------------------------------

def merge_gaia_xgboost(df, xgboost_path):
    """
    Left-join the Gaia XGBoost parameter table onto ASTRA by Gaia DR3 source_id.

    Only the rows whose source_id matches an ASTRA source are loaded from
    the 8 GB file.  All columns from those matched rows are kept.
    """
    print(f"Merging Gaia XGBoost from {xgboost_path} ...")

    astra_id_col = find_column(df.columns, ["gaia_dr3_source_id", "source_id", "gaia_source_id"])
    if astra_id_col is None:
        print("  WARNING: Gaia source_id not found in ASTRA; skipping XGBoost merge.")
        return df

    astra_ids    = pd.to_numeric(df[astra_id_col], errors="coerce").dropna().astype(np.int64)
    astra_id_set = set(astra_ids.tolist())
    print(f"  ASTRA has {len(astra_id_set):,} unique Gaia source_ids to match")

    with astrofits.open(xgboost_path, memmap=True) as hdul:
        hdu_data  = hdul[1].data
        col_names = hdu_data.names

        xgb_id_col = None
        for candidate in ["source_id", "gaia_dr3_source_id", "SOURCE_ID"]:
            if candidate in col_names:
                xgb_id_col = candidate
                break
        if xgb_id_col is None:
            print("  WARNING: source_id column not found in XGBoost file; skipping merge.")
            return df

        print(f"  Reading XGBoost source_id column from "
              f"{xgboost_path.stat().st_size / 1e9:.1f} GB file ...")
        xgb_ids = pd.to_numeric(
            pd.Series(np.array(hdu_data[xgb_id_col])), errors="coerce"
        ).to_numpy()

        match_mask = np.isin(xgb_ids, np.fromiter(astra_id_set, dtype=np.float64))
        n_match    = int(match_mask.sum())
        print(f"  {n_match:,} XGBoost rows match ASTRA; loading only those rows ...")

        if n_match == 0:
            print("  No matches; skipping XGBoost merge.")
            return df

        rows = {}
        for col_name in col_names:
            col_data = np.array(hdu_data[col_name])[match_mask]
            if col_data.ndim > 1:
                rows[col_name] = [str(v.tolist()) for v in col_data]
            else:
                rows[col_name] = col_data.tolist()

    xgb_matched = pd.DataFrame(rows)
    xgb_matched["_xgb_key"] = pd.to_numeric(xgb_matched[xgb_id_col], errors="coerce")
    xgb_matched = xgb_matched.rename(
        columns={c: f"xgb_{c}" for c in xgb_matched.columns if c != "_xgb_key"}
    )

    df["_xgb_key"] = pd.to_numeric(df[astra_id_col], errors="coerce")
    merged = df.merge(xgb_matched, on="_xgb_key", how="left")
    merged.drop(columns=["_xgb_key"], inplace=True)

    xgb_cols = [c for c in merged.columns if c.startswith("xgb_")]
    matched  = merged[xgb_cols].notna().any(axis=1).sum() if xgb_cols else 0
    print(f"  Matched {matched:,} / {len(merged):,} ASTRA rows to XGBoost")
    return merged

# ---------------------------------------------------------------------------
# Positional merge helper  (used for BDBS and Bensby)
# ---------------------------------------------------------------------------

def positional_merge(df, catalog_path, radius_arcsec, prefix, hdu=1):
    print(f"Positional merge to {catalog_path.name} (radius={radius_arcsec}\") ...")
    with astrofits.open(catalog_path, memmap=True) as hdul:
        tbl = Table(hdul[hdu].data)
    cat = table_to_dataframe_all_columns(tbl)
    print(f"  Catalog rows: {len(cat):,}")

    ra_col  = find_column(df.columns,  ["ra_icrs", "ra", "RA", "raj2000"])
    dec_col = find_column(df.columns,  ["dec_icrs", "dec", "DEC", "dej2000"])
    cat_ra  = find_column(cat.columns, ["ra", "RA", "raj2000", "ra_deg"])
    cat_dec = find_column(cat.columns, ["dec", "DEC", "dej2000", "dec_deg"])

    if any(c is None for c in [ra_col, dec_col, cat_ra, cat_dec]):
        print(f"  WARNING: could not resolve coordinates; skipping {prefix} merge.")
        return df

    src     = SkyCoord(ra=as_float(df[ra_col]).to_numpy()   * u.deg,
                       dec=as_float(df[dec_col]).to_numpy()  * u.deg)
    cat_crd = SkyCoord(ra=as_float(cat[cat_ra]).to_numpy()  * u.deg,
                       dec=as_float(cat[cat_dec]).to_numpy() * u.deg)

    idx, sep, _ = src.match_to_catalog_sky(cat_crd)
    within = sep.arcsec <= radius_arcsec

    cat_cols    = [c for c in cat.columns if c not in (cat_ra, cat_dec)]
    cat_matched = cat.iloc[idx][cat_cols].copy()
    cat_matched.columns = [f"{prefix}_{c}" for c in cat_matched.columns]
    cat_matched.index   = df.index

    for col in cat_matched.columns:
        cat_matched.loc[~within, col] = np.nan

    df = pd.concat([df, cat_matched], axis=1)
    df[f"{prefix}_match_sep_arcsec"] = sep.arcsec
    df.loc[~within, f"{prefix}_match_sep_arcsec"] = np.nan

    print(f"  Matched {within.sum():,} / {len(df):,} rows")
    return df

# ---------------------------------------------------------------------------
# Ensure ICRS coordinates  (needed before positional merges)
# ---------------------------------------------------------------------------

def ensure_icrs_coordinates(df):
    out = df.copy()
    ra_col  = find_column(out.columns, ["ra", "ra_deg", "raj2000", "ra_icrs", "RAdeg"])
    dec_col = find_column(out.columns, ["dec", "dec_deg", "dej2000", "dec_icrs", "DEdeg"])
    if ra_col is not None and dec_col is not None:
        out["ra_icrs"]  = as_float(out[ra_col])
        out["dec_icrs"] = as_float(out[dec_col])
        return out
    l_col = find_column(out.columns, ["l", "glon", "gal_l"])
    b_col = find_column(out.columns, ["b", "glat", "gal_b"])
    if l_col is None or b_col is None:
        raise KeyError("Cannot find RA/Dec or Galactic l/b columns.")
    coords = SkyCoord(l=as_float(out[l_col]).to_numpy() * u.deg,
                      b=as_float(out[b_col]).to_numpy() * u.deg, frame="galactic")
    out["ra_icrs"]  = coords.icrs.ra.deg
    out["dec_icrs"] = coords.icrs.dec.deg
    return out

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_astra(ASTRA_FITS)
    df = ensure_icrs_coordinates(df)
    df = merge_gaia_xgboost(df, XGBOOST_FITS)
    df = positional_merge(df, BDBS_FITS,   BDBS_RADIUS_ARCSEC,   prefix="bdbs")
    df = positional_merge(df, BENSBY_FITS, BENSBY_RADIUS_ARCSEC, prefix="bensby")

    Table.from_pandas(df).write(OUTPUT_FITS, overwrite=True)
    print(f"\nWritten: {OUTPUT_FITS}  ({len(df):,} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
    