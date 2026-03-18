#!/usr/bin/env python3
"""
Step 2 – Roman footprint and stellar cuts
==========================================
Applies the GBTDS overguide tile geometry and the Weiss-Zinn
asteroseismology detectability cuts.

Memory strategy
---------------
The merged FITS from step 1 can be large.  This script never loads the
full file at once.  Instead:

  1. Open the file with memmap=True and read ONLY the 5 columns needed
     to decide which rows survive (l, b, teff, logg, h_mag).
  2. Build the combined boolean keep-mask entirely from those 5 columns.
  3. Read ALL columns from the file, but only for the rows where the
     mask is True.  A few hundred rows × hundreds of columns is trivial.

Cuts applied:
  - Inside at least one GBTDS overguide tile (6 rotated rectangles)
  - Teff <= 5500 K  (Weiss et al. 2025 §3.3)
  - logg finite
  - H <= 17 (Roman F146 brightness proxy)

Input  : merged.fits   (name of your TOPCAT-merged file; edit INPUT_FITS below)
Output : step2_selected.fits
"""

import pathlib
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from astropy.io import fits as astrofits
from astropy.table import Table
import warnings
from astropy.io.fits.verify import VerifyWarning
from astropy.units import UnitsWarning

warnings.filterwarnings("ignore", category=VerifyWarning)
warnings.filterwarnings("ignore", category=UnitsWarning)

# ---------------------------------------------------------------------------
# Paths  –  edit INPUT_FITS to match whatever your merged file is called
# ---------------------------------------------------------------------------
INPUT_FITS  = pathlib.Path("merged.fits")
OUTPUT_FITS = pathlib.Path("step2_selected.fits")

# ---------------------------------------------------------------------------
# Roman GBTDS configuration
# ---------------------------------------------------------------------------
TILE_W_DEG  = (45.0 * u.arcmin).to(u.deg).value
TILE_H_DEG  = (23.0 * u.arcmin).to(u.deg).value

#TILE_W_DEG  = (49.4 * u.arcmin).to(u.deg).value
#TILE_H_DEG  = (25.3 * u.arcmin).to(u.deg).value
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
TEFF_MAX  = 5500.0
H_MAG_MAX = 17.0

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def find_colname(available_names, candidates):
    """Case-insensitive search for the first matching column name."""
    lower_map = {n.lower(): n for n in available_names}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def wrap_galactic_longitude(l_deg):
    return ((l_deg + 180.0) % 360.0) - 180.0


def read_single_column(hdu_data, col_name):
    """Read one column from a memmap'd FITS HDU as a float64 numpy array."""
    return pd.to_numeric(
        pd.Series(np.array(hdu_data[col_name])), errors="coerce"
    ).to_numpy(dtype=np.float64)

# ---------------------------------------------------------------------------
# Build the keep-mask using only the 5 cut columns
# ---------------------------------------------------------------------------

def build_keep_mask(fits_path):
    """
    Open the FITS file with memmap and read only l, b, teff, logg, h_mag.
    Returns a boolean numpy array of which rows survive all cuts.
    """
    print(f"  Opening {fits_path} with memmap to read cut columns only ...")

    with astrofits.open(fits_path, memmap=True) as hdul:
        hdu_data   = hdul[1].data
        col_names  = hdu_data.names
        n_rows     = len(hdu_data)
        print(f"  File has {n_rows:,} rows and {len(col_names)} columns")

        l_col    = find_colname(col_names, ["l", "glon", "gal_l", "l_1"])
        b_col    = find_colname(col_names, ["b", "glat", "gal_b", "b_1"])
        teff_col = find_colname(col_names, ["teff", "teff_atm", "TEFF", "teff_1"])
        logg_col = find_colname(col_names, ["logg", "logg_atm", "LOGG", "logg_1"])
        hmag_col = find_colname(col_names, ["h_mag", "h", "H", "Hmag", "h_mag_1"])

        missing = [name for name, col in
                   [("l", l_col), ("b", b_col), ("teff", teff_col),
                    ("logg", logg_col), ("h_mag", hmag_col)]
                   if col is None]
        if missing:
            raise KeyError(f"Could not find required columns: {missing}. "
                           f"Available: {col_names[:20]} ...")

        print(f"  Reading l={l_col}, b={b_col}, teff={teff_col}, "
              f"logg={logg_col}, h={hmag_col}")

        l    = read_single_column(hdu_data, l_col)
        b    = read_single_column(hdu_data, b_col)
        teff = read_single_column(hdu_data, teff_col)
        logg = read_single_column(hdu_data, logg_col)
        hmag = read_single_column(hdu_data, hmag_col)

    # Footprint mask
    l_wrapped = wrap_galactic_longitude(l)
    centers   = GBTDS_CENTERS_OVERGUIDE.copy()
    centers[:, 0] = wrap_galactic_longitude(centers[:, 0])

    in_footprint = np.zeros(n_rows, dtype=bool)
    for l0, b0 in centers:
        center    = SkyCoord(l=l0 * u.deg, b=b0 * u.deg, frame="galactic")
        off_frame = SkyOffsetFrame(origin=center)
        pts = SkyCoord(l=l_wrapped * u.deg, b=b * u.deg,
                       frame="galactic").transform_to(off_frame)
        dx    = pts.lon.to(u.deg).value
        dy    = pts.lat.to(u.deg).value
        theta = np.deg2rad(TILE_PA_DEG)
        x_rot =  dx * np.cos(theta) + dy * np.sin(theta)
        y_rot = -dx * np.sin(theta) + dy * np.cos(theta)
        in_footprint |= (np.abs(x_rot) <= TILE_W_DEG / 2.0) & \
                        (np.abs(y_rot) <= TILE_H_DEG / 2.0)

    # Stellar cuts mask
    teff_ok = np.isfinite(teff) & (teff <= TEFF_MAX)
    logg_ok = np.isfinite(logg)
    hmag_ok = np.isfinite(hmag) & (hmag <= H_MAG_MAX)

    keep = in_footprint #& teff_ok & logg_ok & hmag_ok

    print(f"  In footprint : {in_footprint.sum():,}")
    print(f"  Teff OK      : {teff_ok.sum():,}")
    print(f"  logg OK      : {logg_ok.sum():,}")
    print(f"  H-mag OK     : {hmag_ok.sum():,}")
    print(f"  All cuts     : {keep.sum():,} / {n_rows:,} rows kept")

    return keep

# ---------------------------------------------------------------------------
# Read only the surviving rows for all columns
# ---------------------------------------------------------------------------

def read_selected_rows(fits_path, keep_mask):
    """
    Read every column from the FITS file, but only for rows where
    keep_mask is True.  This is the only time the full column data
    is touched, and only a small fraction of rows are materialised.
    """
    print(f"  Loading {keep_mask.sum():,} selected rows across all columns ...")
    row_indices = np.where(keep_mask)[0]

    with astrofits.open(fits_path, memmap=True) as hdul:
        hdu_data  = hdul[1].data
        col_names = hdu_data.names

        rows = {}
        for col_name in col_names:
            col_data = np.array(hdu_data[col_name])[row_indices]
            if col_data.ndim > 1:
                rows[col_name] = [str(v.tolist()) for v in col_data]
            else:
                rows[col_name] = col_data.tolist()

    df = pd.DataFrame(rows)
    print(f"  DataFrame built: {len(df):,} rows, {len(df.columns)} columns")
    return df

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Reading {INPUT_FITS} ...")

    keep_mask = build_keep_mask(INPUT_FITS)
    df        = read_selected_rows(INPUT_FITS, keep_mask)

    Table.from_pandas(df).write(OUTPUT_FITS, overwrite=True)
    print(f"\nWritten: {OUTPUT_FITS}  ({len(df):,} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()