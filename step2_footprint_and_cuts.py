#!/usr/bin/env python3
"""
Step 2 – Roman footprint and stellar cuts
==========================================
Applies the GBTDS overguide tile geometry and the Weiss-Zinn
asteroseismology detectability cuts.

Cuts applied:
  - Inside at least one GBTDS overguide tile (6 rotated rectangles)
  - Teff <= 5500 K  (Weiss et al. 2025 §3.3)
  - logg finite
  - H <= 17 (Roman F146 brightness proxy)

Input  : step1_merged.fits
Output : step2_selected.fits
"""

import pathlib
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame
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
INPUT_FITS  = pathlib.Path("merged.fits")
OUTPUT_FITS = pathlib.Path("step2_selected.fits")

# ---------------------------------------------------------------------------
# Roman GBTDS configuration
# ---------------------------------------------------------------------------
TILE_W_DEG  = (49.4 * u.arcmin).to(u.deg).value
TILE_H_DEG  = (25.3 * u.arcmin).to(u.deg).value
TILE_PA_DEG = 90.6

GBTDS_CENTERS_OVERGUIDE = np.array([
    [-0.417948, -1.200],
    [-0.008974, -1.200],
    [ 0.400000, -1.200],
    [ 0.808974, -1.200],
    [ 1.217948, -1.200],
    [ 0.000000, -0.1250],
], dtype=float)

# Weiss-Zinn cuts
TEFF_MAX  = 5500.0   # K
H_MAG_MAX = 17.0     # mag

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


def wrap_galactic_longitude(l_deg):
    return ((l_deg + 180.0) % 360.0) - 180.0

# ---------------------------------------------------------------------------
# Footprint
# ---------------------------------------------------------------------------

def _points_in_rotated_rectangle(l_deg, b_deg, l0_deg, b0_deg,
                                  width_deg, height_deg, pa_deg):
    center    = SkyCoord(l=l0_deg * u.deg, b=b0_deg * u.deg, frame="galactic")
    off_frame = SkyOffsetFrame(origin=center)
    pts = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg,
                   frame="galactic").transform_to(off_frame)
    dx = pts.lon.to(u.deg).value
    dy = pts.lat.to(u.deg).value
    theta = np.deg2rad(pa_deg)
    x_rot =  dx * np.cos(theta) + dy * np.sin(theta)
    y_rot = -dx * np.sin(theta) + dy * np.cos(theta)
    return (np.abs(x_rot) <= width_deg / 2.0) & (np.abs(y_rot) <= height_deg / 2.0)


def apply_roman_footprint(df):
    print(f"Applying Roman GBTDS overguide footprint ({len(GBTDS_CENTERS_OVERGUIDE)} tiles) ...")

    l_col = find_column(df.columns, ["l", "glon", "gal_l"])
    b_col = find_column(df.columns, ["b", "glat", "gal_b"])
    if l_col is None or b_col is None:
        raise KeyError("Need Galactic l and b columns for footprint selection.")

    l = wrap_galactic_longitude(as_float(df[l_col]).to_numpy())
    b = as_float(df[b_col]).to_numpy()

    centers = GBTDS_CENTERS_OVERGUIDE.copy()
    centers[:, 0] = wrap_galactic_longitude(centers[:, 0])

    in_footprint = np.zeros(len(df), dtype=bool)
    for l0, b0 in centers:
        in_footprint |= _points_in_rotated_rectangle(
            l, b, l0, b0, TILE_W_DEG, TILE_H_DEG, TILE_PA_DEG
        )

    result = df.loc[in_footprint].copy().reset_index(drop=True)
    print(f"  {in_footprint.sum():,} / {len(df):,} stars inside footprint")
    return result

# ---------------------------------------------------------------------------
# Stellar cuts
# ---------------------------------------------------------------------------

def apply_stellar_cuts(df):
    print(f"Applying stellar cuts (Teff <= {TEFF_MAX} K, H <= {H_MAG_MAX}) ...")

    teff_col = find_column(df.columns, ["teff", "teff_atm", "TEFF"])
    logg_col = find_column(df.columns, ["logg", "logg_atm", "LOGG"])
    hmag_col = find_column(df.columns, ["h_mag", "h", "H", "Hmag"])

    if teff_col is None or logg_col is None or hmag_col is None:
        raise KeyError(
            f"Need teff/logg/h_mag columns. "
            f"Found: teff={teff_col}, logg={logg_col}, h={hmag_col}"
        )

    teff = as_float(df[teff_col])
    logg = as_float(df[logg_col])
    hmag = as_float(df[hmag_col])

    teff_ok = np.isfinite(teff) & (teff <= TEFF_MAX)
    logg_ok = np.isfinite(logg)
    hmag_ok = np.isfinite(hmag) & (hmag <= H_MAG_MAX)
    mask    = teff_ok & logg_ok & hmag_ok

    print(f"  Teff OK  : {teff_ok.sum():,}")
    print(f"  logg OK  : {logg_ok.sum():,}")
    print(f"  H-mag OK : {hmag_ok.sum():,}")
    print(f"  All cuts : {mask.sum():,} / {len(df):,} kept")

    return df.loc[mask].copy().reset_index(drop=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def read_fits_all_columns(fits_path):
    """
    Read a FITS table using memmap so the full file is never copied into RAM.
    Multi-dimensional columns are stored as strings rather than crashing pandas.
    """
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

    df = apply_roman_footprint(df)
    df = apply_stellar_cuts(df)

    Table.from_pandas(df).write(OUTPUT_FITS, overwrite=True)
    print(f"\nWritten: {OUTPUT_FITS}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
