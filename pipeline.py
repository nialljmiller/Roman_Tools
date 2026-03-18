#!/usr/bin/env python3
"""
Roman Bulge Calibration Pipeline
=================================
Builds a clean, parameter-rich calibration catalog of ASTRA spectroscopic
targets observable by the Roman Space Telescope GBTDS.

Steps
-----
 1. load_astra                – Read astra.fits (HDU 2) into a scalar DataFrame.
 2. merge_gaia_xgboost        – Merge gaiaXGBoost.fits by Gaia DR3 source_id.
 3. merge_bdbs                – Positional cross-match to bdbs.fits (1 arcsec).
 4. merge_bensby              – Positional cross-match to bensby.fits (2 arcsec).
 5. apply_roman_footprint     – Keep stars inside the GBTDS overguide tile geometry.
 6. apply_stellar_cuts        – Weiss-Zinn: Teff ≤ 5500 K, finite logg, H ≤ 17.
 7. ensure_icrs_coordinates   – Convert Galactic l/b → ICRS RA/Dec where missing.
 8. xmatch_virac2             – CDS XMatch to VIRAC2 near-IR photometry.
 9. xmatch_2mass              – CDS XMatch to 2MASS PSC.
10. xmatch_vvv_reddening      – CDS XMatch to VVV high-res reddening map.
11. build_best_photometry     – Prefer VIRAC2, fall back to 2MASS for J/H/Ks.
12. compute_extinction        – E(J-Ks) → A_J, A_H, A_Ks via bulge extinction law.
13. compute_dereddened_mags   – Produce J0, H0, Ks0 and dereddened colours.
14. tag_sightlines            – Label each star with its VVV reddening-map cell.
15. harvest_best_parameters   – Pick best Teff, logg, [M/H], radius, luminosity.
16. add_provenance_flags      – has_virac2, has_reddening, calibration_ready, …
17. prune_bookkeeping         – Drop SDSS/ASTRA task-PK housekeeping columns.
18. write_outputs             – FITS + CSV + plain-text summary.

Inputs (all expected in the working directory)
------
  astra.fits          – ASTRA MWM spectroscopic catalog (HDU 2)
  gaiaXGBoost.fits    – Gaia DR3 XGBoost stellar parameter table
  bdbs.fits           – BDBS photometric catalog
  bensby.fits         – Bensby high-resolution spectroscopic sample
  gaia.fits           – Gaia DR3 source table (used for Gaia AP columns)

Outputs
-------
  roman_pipeline_master.fits / .csv
  roman_pipeline_selected.fits / .csv
  roman_pipeline_summary.txt
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from astropy.table import Table
from astropy.io.fits.verify import VerifyWarning
from astropy.units import UnitsWarning
from astroquery.xmatch import XMatch

warnings.filterwarnings("ignore", category=VerifyWarning)
warnings.filterwarnings("ignore", category=UnitsWarning)


# =============================================================================
# Configuration – edit these paths and constants as needed
# =============================================================================

ASTRA_FITS        = pathlib.Path("astra.fits")
XGBOOST_FITS      = pathlib.Path("gaiaXGBoost.fits")
BDBS_FITS         = pathlib.Path("bdbs.fits")
BENSBY_FITS       = pathlib.Path("bensby.fits")
GAIA_FITS         = pathlib.Path("gaia.fits")

OUT_PREFIX        = pathlib.Path("roman_pipeline")

# Roman GBTDS tile geometry (Table 9, https://arxiv.org/pdf/2505.10574)
TILE_W_DEG        = (49.4 * u.arcmin).to(u.deg).value   # tile width
TILE_H_DEG        = (25.3 * u.arcmin).to(u.deg).value   # tile height
TILE_PA_DEG       = 90.6                                  # position angle

GBTDS_CENTERS_OVERGUIDE = np.array([
    [-0.417948, -1.200],
    [-0.008974, -1.200],
    [ 0.400000, -1.200],
    [ 0.808974, -1.200],
    [ 1.217948, -1.200],
    [ 0.000000, -0.1250],
], dtype=float)

# Weiss-Zinn seismology detectability cuts
TEFF_MAX          = 5500.0   # K  (Weiss et al. 2025 §3.3)
H_MAG_MAX         = 17.0     # mag proxy for Roman F146 brightness limit

# Bulge extinction law  (Nishiyama et al. style)
A_KS_PER_E_JKS   = 0.528
A_H_PER_E_JKS    = 0.857
A_J_PER_E_JKS    = 1.528

# CDS XMatch remote catalog identifiers
VIRAC2_TABLE      = "vizier:II/387/virac2"
TMASS_TABLE       = "vizier:II/246/out"
VVV_RED_TABLE     = "vizier:J/A+A/644/A140/ejkmap"

# Cross-match radii
VIRAC2_RADIUS_ARCSEC  = 1.0
TMASS_RADIUS_ARCSEC   = 2.0
VVV_RED_RADIUS_ARCSEC = 30.0   # reddening map has coarse cells
BDBS_RADIUS_ARCSEC    = 1.0
BENSBY_RADIUS_ARCSEC  = 2.0


# =============================================================================
# Utility functions
# =============================================================================

def lower_column_map(columns):
    """Return a dict mapping lowercase column name → original column name."""
    return {str(c).lower(): str(c) for c in columns}


def find_column(columns, candidates):
    """Return the first column from candidates that exists (case-insensitive)."""
    mapping = lower_column_map(columns)
    for candidate in candidates:
        key = str(candidate).lower()
        if key in mapping:
            return mapping[key]
    return None


def as_float(series):
    """Coerce a pandas Series to float, turning bad values into NaN."""
    return pd.to_numeric(series, errors="coerce")


def wrap_galactic_longitude(l_deg):
    """Remap Galactic longitude to (−180, +180] for consistent bulge comparisons."""
    return ((l_deg + 180.0) % 360.0) - 180.0


def fits_to_scalar_dataframe(fits_path, hdu=1):
    """
    Read a FITS table and return a pandas DataFrame containing only
    scalar (1-D) columns.  Multi-dimensional columns (e.g. ASTRA bitflag
    arrays) are silently dropped because pandas cannot represent them.
    """
    tbl = Table.read(fits_path, hdu=hdu)
    scalar_cols = [
        name for name in tbl.colnames
        if len(getattr(tbl[name], "shape", ())) <= 1
    ]
    dropped = [n for n in tbl.colnames if n not in scalar_cols]
    if dropped:
        print(f"  [fits_to_scalar_dataframe] dropped {len(dropped)} multi-dim columns: "
              f"{dropped[:6]}{'…' if len(dropped) > 6 else ''}")
    return tbl[scalar_cols].to_pandas()


# =============================================================================
# Step 1 – Load ASTRA
# =============================================================================

def load_astra(fits_path):
    """
    Read astra.fits HDU 2 into a scalar pandas DataFrame.

    ASTRA stores its main source table in HDU 2.  Multi-dimensional columns
    (bit-flag arrays etc.) are dropped.
    """
    print(f"\n[Step 1] Loading ASTRA from {fits_path} …")
    df = fits_to_scalar_dataframe(fits_path, hdu=2)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} scalar columns.")
    return df


# =============================================================================
# Step 2 – Merge Gaia XGBoost
# =============================================================================

def merge_gaia_xgboost(astra_df, xgboost_path):
    """
    Left-join the Gaia XGBoost stellar parameter table onto the ASTRA catalog
    using the Gaia DR3 source_id as the key.

    Columns added (prefixed xgb_) include Teff, logg, [M/H] from the
    XGBoost model trained on Gaia photometry + astrometry.
    """
    print(f"\n[Step 2] Merging Gaia XGBoost from {xgboost_path} …")

    xgb = fits_to_scalar_dataframe(xgboost_path, hdu=1)
    print(f"  XGBoost rows: {len(xgb):,}")

    # Identify source_id column in each table
    astra_id_col = find_column(astra_df.columns,
                               ["gaia_dr3_source_id", "source_id", "gaia_source_id"])
    xgb_id_col   = find_column(xgb.columns,
                               ["source_id", "gaia_dr3_source_id", "gaia_source_id"])

    if astra_id_col is None or xgb_id_col is None:
        print("  WARNING: could not locate Gaia source_id columns; skipping XGBoost merge.")
        return astra_df

    # Normalise types before merging
    astra_df["_xgb_key"] = pd.to_numeric(astra_df[astra_id_col], errors="coerce")
    xgb["_xgb_key"]      = pd.to_numeric(xgb[xgb_id_col],        errors="coerce")

    # Rename XGBoost columns to avoid clashes, preserving the originals
    xgb_renamed = xgb.rename(columns={c: f"xgb_{c}" for c in xgb.columns
                                       if c != "_xgb_key"})

    merged = astra_df.merge(xgb_renamed, on="_xgb_key", how="left")
    merged.drop(columns=["_xgb_key"], inplace=True)

    matched = merged[[c for c in merged.columns if c.startswith("xgb_")]]\
                    .notna().any(axis=1).sum()
    print(f"  Matched {matched:,} / {len(merged):,} ASTRA rows to XGBoost.")
    return merged


# =============================================================================
# Step 3 – Merge BDBS
# =============================================================================

def merge_bdbs(df, bdbs_path, radius_arcsec):
    """
    Positional cross-match of the ASTRA working table to the BDBS photometric
    catalog.  Uses astropy SkyCoord nearest-neighbour matching; rows outside
    radius_arcsec are not merged.

    BDBS provides optical+NIR photometry and photometric metallicities for
    stars in the Galactic bulge, which complement the ASTRA spectroscopic data.
    """
    print(f"\n[Step 3] Merging BDBS (radius={radius_arcsec}\") from {bdbs_path} …")

    bdbs = fits_to_scalar_dataframe(bdbs_path, hdu=1)
    print(f"  BDBS rows: {len(bdbs):,}")

    # Resolve coordinates in the working table
    ra_col  = find_column(df.columns, ["ra_icrs", "ra", "RA", "raj2000"])
    dec_col = find_column(df.columns, ["dec_icrs", "dec", "DEC", "dej2000"])

    if ra_col is None or dec_col is None:
        print("  WARNING: RA/Dec not yet available; skipping BDBS merge (run after Step 7).")
        return df

    # Resolve coordinates in BDBS
    bdbs_ra_col  = find_column(bdbs.columns, ["ra", "RA", "raj2000", "ra_deg"])
    bdbs_dec_col = find_column(bdbs.columns, ["dec", "DEC", "dej2000", "dec_deg"])

    if bdbs_ra_col is None or bdbs_dec_col is None:
        print("  WARNING: could not find RA/Dec in BDBS; skipping merge.")
        return df

    src  = SkyCoord(ra=as_float(df[ra_col]).to_numpy()   * u.deg,
                    dec=as_float(df[dec_col]).to_numpy()  * u.deg)
    bdbs_coords = SkyCoord(ra=as_float(bdbs[bdbs_ra_col]).to_numpy()  * u.deg,
                           dec=as_float(bdbs[bdbs_dec_col]).to_numpy() * u.deg)

    idx, sep, _ = src.match_to_catalog_sky(bdbs_coords)
    within = sep.arcsec <= radius_arcsec

    bdbs_cols = [c for c in bdbs.columns
                 if c not in (bdbs_ra_col, bdbs_dec_col)]
    bdbs_matched = bdbs.iloc[idx][bdbs_cols].copy()
    bdbs_matched.columns = [f"bdbs_{c}" for c in bdbs_matched.columns]
    bdbs_matched.index   = df.index

    # Zero out rows that did not match
    for col in bdbs_matched.columns:
        bdbs_matched.loc[~within, col] = np.nan

    df = pd.concat([df, bdbs_matched], axis=1)
    df["bdbs_match_sep_arcsec"] = sep.arcsec
    df.loc[~within, "bdbs_match_sep_arcsec"] = np.nan

    print(f"  Matched {within.sum():,} / {len(df):,} rows to BDBS.")
    return df


# =============================================================================
# Step 4 – Merge Bensby
# =============================================================================

def merge_bensby(df, bensby_path, radius_arcsec):
    """
    Positional cross-match to the Bensby high-resolution spectroscopic sample.
    Bensby et al. provide precise Teff, logg, [Fe/H], and ages for a sample of
    dwarf/subgiant bulge stars, useful as a calibration anchor.
    """
    print(f"\n[Step 4] Merging Bensby (radius={radius_arcsec}\") from {bensby_path} …")

    bensby = fits_to_scalar_dataframe(bensby_path, hdu=1)
    print(f"  Bensby rows: {len(bensby):,}")

    ra_col  = find_column(df.columns, ["ra_icrs", "ra", "RA"])
    dec_col = find_column(df.columns, ["dec_icrs", "dec", "DEC"])

    if ra_col is None or dec_col is None:
        print("  WARNING: RA/Dec not available; skipping Bensby merge.")
        return df

    ben_ra_col  = find_column(bensby.columns, ["ra", "RA", "raj2000"])
    ben_dec_col = find_column(bensby.columns, ["dec", "DEC", "dej2000"])

    if ben_ra_col is None or ben_dec_col is None:
        print("  WARNING: could not find RA/Dec in Bensby; skipping merge.")
        return df

    src     = SkyCoord(ra=as_float(df[ra_col]).to_numpy()         * u.deg,
                       dec=as_float(df[dec_col]).to_numpy()        * u.deg)
    ben_crd = SkyCoord(ra=as_float(bensby[ben_ra_col]).to_numpy() * u.deg,
                       dec=as_float(bensby[ben_dec_col]).to_numpy()* u.deg)

    idx, sep, _ = src.match_to_catalog_sky(ben_crd)
    within = sep.arcsec <= radius_arcsec

    ben_cols = [c for c in bensby.columns
                if c not in (ben_ra_col, ben_dec_col)]
    ben_matched = bensby.iloc[idx][ben_cols].copy()
    ben_matched.columns = [f"bensby_{c}" for c in ben_matched.columns]
    ben_matched.index   = df.index

    for col in ben_matched.columns:
        ben_matched.loc[~within, col] = np.nan

    df = pd.concat([df, ben_matched], axis=1)
    df["bensby_match_sep_arcsec"] = sep.arcsec
    df.loc[~within, "bensby_match_sep_arcsec"] = np.nan

    print(f"  Matched {within.sum():,} / {len(df):,} rows to Bensby.")
    return df


# =============================================================================
# Step 5 – Apply Roman Footprint
# =============================================================================

def _points_in_rotated_rectangle(l_deg, b_deg, l0_deg, b0_deg,
                                  width_deg, height_deg, pa_deg):
    """
    Return a boolean mask of points inside one rotated rectangular Roman tile.

    Uses a SkyOffsetFrame centred on (l0, b0) to project points into the
    tile frame, then rotates by pa_deg before applying half-width/height cuts.
    """
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


def apply_roman_footprint(df, centers, width_deg, height_deg, pa_deg):
    """
    Retain only stars that fall inside at least one GBTDS overguide tile.

    The tile layout is defined by the array of (l, b) centre coordinates
    from Table 9 of the GBTDS definition paper.
    """
    print(f"\n[Step 5] Applying Roman GBTDS overguide footprint "
          f"({len(centers)} tiles) …")

    l_col = find_column(df.columns, ["l", "glon", "gal_l"])
    b_col = find_column(df.columns, ["b", "glat", "gal_b"])

    if l_col is None or b_col is None:
        raise KeyError("Need Galactic l and b columns for footprint selection.")

    l = wrap_galactic_longitude(as_float(df[l_col]).to_numpy())
    b = as_float(df[b_col]).to_numpy()

    tile_centers = centers.copy()
    tile_centers[:, 0] = wrap_galactic_longitude(tile_centers[:, 0])

    in_footprint = np.zeros(len(df), dtype=bool)
    for l0, b0 in tile_centers:
        in_footprint |= _points_in_rotated_rectangle(
            l, b, l0, b0, width_deg, height_deg, pa_deg
        )

    result = df.loc[in_footprint].copy().reset_index(drop=True)
    print(f"  {in_footprint.sum():,} / {len(df):,} stars inside footprint.")
    return result


# =============================================================================
# Step 6 – Apply Stellar Cuts
# =============================================================================

def apply_stellar_cuts(df, teff_max, h_mag_max):
    """
    Apply the Weiss-Zinn asteroseismology detectability cuts.

    Keeps stars with:
      - Teff ≤ teff_max  (cool enough for solar-like oscillations)
      - finite logg      (spectroscopic solution exists)
      - H ≤ h_mag_max    (bright enough for Roman detection)
    """
    print(f"\n[Step 6] Applying stellar cuts "
          f"(Teff ≤ {teff_max} K, H ≤ {h_mag_max}) …")

    teff_col  = find_column(df.columns, ["teff", "teff_atm", "TEFF"])
    logg_col  = find_column(df.columns, ["logg", "logg_atm", "LOGG"])
    hmag_col  = find_column(df.columns, ["h_mag", "h", "H", "Hmag"])

    if teff_col is None or logg_col is None or hmag_col is None:
        raise KeyError(f"Need teff/logg/h_mag columns. "
                       f"Found: teff={teff_col}, logg={logg_col}, h={hmag_col}")

    teff = as_float(df[teff_col])
    logg = as_float(df[logg_col])
    hmag = as_float(df[hmag_col])

    teff_ok  = np.isfinite(teff) & (teff  <= teff_max)
    logg_ok  = np.isfinite(logg)
    hmag_ok  = np.isfinite(hmag) & (hmag  <= h_mag_max)

    mask = teff_ok & logg_ok & hmag_ok

    print(f"  Teff OK   : {teff_ok.sum():,}")
    print(f"  logg OK   : {logg_ok.sum():,}")
    print(f"  H-mag OK  : {hmag_ok.sum():,}")
    print(f"  All cuts  : {mask.sum():,} / {len(df):,} stars kept.")

    return df.loc[mask].copy().reset_index(drop=True)


# =============================================================================
# Step 7 – Ensure ICRS Coordinates
# =============================================================================

def ensure_icrs_coordinates(df):
    """
    Guarantee that every row has ra_icrs and dec_icrs (decimal degrees, J2000).

    Priority order:
      1. Use existing RA/Dec columns if present.
      2. Convert from Galactic l/b using astropy.
    """
    print("\n[Step 7] Ensuring ICRS coordinates …")
    out = df.copy()

    ra_col  = find_column(out.columns, ["ra", "ra_deg", "raj2000", "ra_icrs", "RAdeg"])
    dec_col = find_column(out.columns, ["dec", "dec_deg", "dej2000", "dec_icrs", "DEdeg"])

    if ra_col is not None and dec_col is not None:
        out["ra_icrs"]  = as_float(out[ra_col])
        out["dec_icrs"] = as_float(out[dec_col])
        print(f"  Copied from existing columns: {ra_col}, {dec_col}")
        return out

    l_col = find_column(out.columns, ["l", "glon", "gal_l"])
    b_col = find_column(out.columns, ["b", "glat", "gal_b"])

    if l_col is None or b_col is None:
        raise KeyError("Cannot find RA/Dec or Galactic l/b to determine coordinates.")

    l_vals = as_float(out[l_col]).to_numpy()
    b_vals = as_float(out[b_col]).to_numpy()
    coords = SkyCoord(l=l_vals * u.deg, b=b_vals * u.deg, frame="galactic")

    out["ra_icrs"]  = coords.icrs.ra.deg
    out["dec_icrs"] = coords.icrs.dec.deg
    print(f"  Converted from Galactic l/b.  "
          f"RA range: [{out['ra_icrs'].min():.2f}, {out['ra_icrs'].max():.2f}]")
    return out


# =============================================================================
# Steps 8–10 – CDS Remote Cross-Matches
# =============================================================================

def _upload_xmatch(local_df, remote_table, radius_arcsec):
    """
    Submit a local table to the CDS XMatch service against a VizieR catalog.

    local_df must contain ra_icrs and dec_icrs columns.
    Returns the joined result as a pandas DataFrame, or an empty DataFrame
    on failure.
    """
    upload = local_df[["ra_icrs", "dec_icrs"]].copy().reset_index(drop=False)
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

    return result_tbl.to_pandas()


def xmatch_virac2(df, radius_arcsec):
    """
    Cross-match to VIRAC2 (VVV/VVVX near-IR survey, Vizier II/387/virac2).

    VIRAC2 provides precise J, H, Ks photometry in the Galactic bulge with
    better depth and crowding handling than 2MASS.  Results are left-joined
    back onto df, with column prefix 'vvv_'.
    """
    print(f"\n[Step 8] XMatch to VIRAC2 (radius={radius_arcsec}\") …")
    try:
        result = _upload_xmatch(df, VIRAC2_TABLE, radius_arcsec)
    except Exception as exc:
        print(f"  WARNING: VIRAC2 XMatch failed ({exc}); continuing without it.")
        return df

    if result.empty:
        print("  No VIRAC2 matches returned.")
        return df

    # Rename to avoid column name clashes and track source
    rename_map = {c: f"vvv_{c}" for c in result.columns
                  if c not in ("_local_idx", "angDist")}
    result.rename(columns=rename_map, inplace=True)
    result.rename(columns={"angDist": "virac2_sep_arcsec"}, inplace=True)

    # Keep only the closest match per source
    result.sort_values("virac2_sep_arcsec", inplace=True)
    result.drop_duplicates(subset=["_local_idx"], keep="first", inplace=True)
    result.set_index("_local_idx", inplace=True)

    df = df.join(result, how="left")
    matched = df["virac2_sep_arcsec"].notna().sum()
    print(f"  Matched {matched:,} / {len(df):,} rows to VIRAC2.")
    return df


def xmatch_2mass(df, radius_arcsec):
    """
    Cross-match to the 2MASS Point Source Catalog (Vizier II/246/out).

    Used as a fallback for stars without VIRAC2 photometry, and to fill
    any missing J/H/Ks values.  Results prefixed 'tmass_'.
    """
    print(f"\n[Step 9] XMatch to 2MASS PSC (radius={radius_arcsec}\") …")
    try:
        result = _upload_xmatch(df, TMASS_TABLE, radius_arcsec)
    except Exception as exc:
        print(f"  WARNING: 2MASS XMatch failed ({exc}); continuing without it.")
        return df

    if result.empty:
        print("  No 2MASS matches returned.")
        return df

    rename_map = {c: f"tmass_{c}" for c in result.columns
                  if c not in ("_local_idx", "angDist")}
    result.rename(columns=rename_map, inplace=True)
    result.rename(columns={"angDist": "tmass_sep_arcsec"}, inplace=True)

    result.sort_values("tmass_sep_arcsec", inplace=True)
    result.drop_duplicates(subset=["_local_idx"], keep="first", inplace=True)
    result.set_index("_local_idx", inplace=True)

    df = df.join(result, how="left")
    matched = df["tmass_sep_arcsec"].notna().sum()
    print(f"  Matched {matched:,} / {len(df):,} rows to 2MASS.")
    return df


def xmatch_vvv_reddening(df, radius_arcsec):
    """
    Cross-match to the VVV high-resolution E(J-Ks) reddening map
    (Vizier J/A+A/644/A140/ejkmap, Surot et al. 2020).

    The reddening map has angular resolution of ~1–2 arcmin; the large
    matching radius (30 arcsec default) locates the nearest map cell centre.
    Results prefixed 'ext_'.
    """
    print(f"\n[Step 10] XMatch to VVV reddening map (radius={radius_arcsec}\") …")
    try:
        result = _upload_xmatch(df, VVV_RED_TABLE, radius_arcsec)
    except Exception as exc:
        print(f"  WARNING: VVV reddening XMatch failed ({exc}); continuing.")
        return df

    if result.empty:
        print("  No VVV reddening matches returned.")
        return df

    rename_map = {c: f"ext_{c}" for c in result.columns
                  if c not in ("_local_idx", "angDist")}
    result.rename(columns=rename_map, inplace=True)
    result.rename(columns={"angDist": "ext_map_match_sep_arcsec"}, inplace=True)

    result.sort_values("ext_map_match_sep_arcsec", inplace=True)
    result.drop_duplicates(subset=["_local_idx"], keep="first", inplace=True)
    result.set_index("_local_idx", inplace=True)

    df = df.join(result, how="left")
    matched = df["ext_map_match_sep_arcsec"].notna().sum()
    print(f"  Matched {matched:,} / {len(df):,} rows to VVV reddening map.")
    return df


# =============================================================================
# Step 11 – Build Best J/H/Ks Photometry
# =============================================================================

def build_best_photometry(df):
    """
    Construct best-available J, H, Ks columns by preferring VIRAC2 over 2MASS.

    For each band:
      - If a finite VIRAC2 value exists, use it and label the source 'VIRAC2'.
      - Else if a finite 2MASS value exists, use it and label the source '2MASS'.
      - Else leave as NaN.

    Also computes observed colours: (J-H), (H-Ks), (J-Ks).
    """
    print("\n[Step 11] Building best J/H/Ks photometry …")

    # VIRAC2 band column guesses
    vvv_j  = find_column(df.columns, ["vvv_Jmag",  "vvv_j",  "vvv_j_mag"])
    vvv_h  = find_column(df.columns, ["vvv_Hmag",  "vvv_h",  "vvv_h_mag"])
    vvv_ks = find_column(df.columns, ["vvv_Ksmag", "vvv_ks", "vvv_ks_mag", "vvv_K"])

    # 2MASS band column guesses
    tm_j   = find_column(df.columns, ["tmass_Jmag", "tmass_j_m",  "tmass_j"])
    tm_h   = find_column(df.columns, ["tmass_Hmag", "tmass_h_m",  "tmass_h"])
    tm_ks  = find_column(df.columns, ["tmass_Kmag", "tmass_ks_m", "tmass_ks", "tmass_k_m"])

    def merge_band(vvv_col, tmass_col, band_name):
        j_best   = np.full(len(df), np.nan)
        j_source = np.full(len(df), "", dtype=object)

        if vvv_col is not None:
            v = as_float(df[vvv_col]).to_numpy()
            good = np.isfinite(v)
            j_best[good]   = v[good]
            j_source[good] = "VIRAC2"

        if tmass_col is not None:
            t = as_float(df[tmass_col]).to_numpy()
            good = np.isfinite(t) & ~np.isfinite(j_best)
            j_best[good]   = t[good]
            j_source[good] = "2MASS"

        df[f"{band_name}_mag_best"]    = j_best
        df[f"{band_name}_mag_source"]  = j_source

    merge_band(vvv_j,  tm_j,  "j")
    merge_band(vvv_h,  tm_h,  "h")
    merge_band(vvv_ks, tm_ks, "ks")

    df["j_minus_h_best"]  = df["j_mag_best"]  - df["h_mag_best"]
    df["h_minus_ks_best"] = df["h_mag_best"]  - df["ks_mag_best"]
    df["j_minus_ks_best"] = df["j_mag_best"]  - df["ks_mag_best"]

    n_j  = np.isfinite(df["j_mag_best"]).sum()
    n_h  = np.isfinite(df["h_mag_best"]).sum()
    n_ks = np.isfinite(df["ks_mag_best"]).sum()
    print(f"  J best: {n_j:,}  H best: {n_h:,}  Ks best: {n_ks:,}")
    return df


# =============================================================================
# Step 12 – Compute Extinction
# =============================================================================

def compute_extinction(df, aks_per_ejks, ah_per_ejks, aj_per_ejks):
    """
    Convert E(J-Ks) reddening values from the VVV map into band extinctions.

    Uses a Nishiyama-style bulge extinction law:
      A_Ks = aks_per_ejks * E(J-Ks)
      A_H  = ah_per_ejks  * E(J-Ks)
      A_J  = aj_per_ejks  * E(J-Ks)

    The E(J-Ks) column is expected to come from the VVV reddening cross-match
    as 'ext_EJKs' or similar.
    """
    print("\n[Step 12] Computing extinction …")

    ejks_col = find_column(df.columns, ["ext_EJKs", "ext_e_jks", "ext_ejks",
                                         "ext_E_JKs", "ext_EJKs_1"])
    if ejks_col is None:
        print("  WARNING: E(J-Ks) column not found; setting A_J/A_H/A_Ks to NaN.")
        df["ext_e_jks"] = np.nan
        df["a_j"]  = np.nan
        df["a_h"]  = np.nan
        df["a_ks"] = np.nan
        return df

    ejks = as_float(df[ejks_col]).to_numpy()
    df["ext_e_jks"] = ejks
    df["a_j"]  = aj_per_ejks  * ejks
    df["a_h"]  = ah_per_ejks  * ejks
    df["a_ks"] = aks_per_ejks * ejks

    finite = np.isfinite(ejks)
    print(f"  E(J-Ks) available for {finite.sum():,} stars.  "
          f"Median E(J-Ks) = {np.nanmedian(ejks):.3f}")
    return df


# =============================================================================
# Step 13 – Compute Dereddened Magnitudes
# =============================================================================

def compute_dereddened_mags(df):
    """
    Subtract band extinctions from best observed magnitudes.

    Produces: J0, H0, Ks0 (dereddened magnitudes) and
              (J-H)0, (H-Ks)0, (J-Ks)0 (dereddened colours).
    """
    print("\n[Step 13] Computing dereddened magnitudes …")

    df["j0"]  = df["j_mag_best"]  - df["a_j"]
    df["h0"]  = df["h_mag_best"]  - df["a_h"]
    df["ks0"] = df["ks_mag_best"] - df["a_ks"]

    df["j0_minus_h0"]  = df["j0"]  - df["h0"]
    df["h0_minus_ks0"] = df["h0"]  - df["ks0"]
    df["j0_minus_ks0"] = df["j0"]  - df["ks0"]

    n_dered = np.isfinite(df["ks0"]).sum()
    print(f"  Dereddened Ks0 available for {n_dered:,} stars.")
    return df


# =============================================================================
# Step 14 – Tag Sightlines
# =============================================================================

def tag_sightlines(df):
    """
    Attach a sightline label to each star based on its VVV reddening-map cell.

    The VVV reddening map cells have associated Galactic coordinates.  Each
    star is labelled with:
      sightline_glon / sightline_glat – map cell centre (deg)
      sightline_id                    – human-readable 'lXXX_bXXX' string
      sightline_bin_0p1deg            – 0.1-deg binned position string
    """
    print("\n[Step 14] Tagging sightlines …")

    glon_col = find_column(df.columns, ["ext_GLON", "ext_glon", "ext_l"])
    glat_col = find_column(df.columns, ["ext_GLAT", "ext_glat", "ext_b"])

    if glon_col is not None and glat_col is not None:
        df["sightline_glon"] = as_float(df[glon_col])
        df["sightline_glat"] = as_float(df[glat_col])
    else:
        # Fall back to the star's own position rounded to 0.5 deg
        l_col = find_column(df.columns, ["l", "glon"])
        b_col = find_column(df.columns, ["b", "glat"])
        if l_col and b_col:
            df["sightline_glon"] = (as_float(df[l_col]) * 2).round() / 2
            df["sightline_glat"] = (as_float(df[b_col]) * 2).round() / 2
        else:
            df["sightline_glon"] = np.nan
            df["sightline_glat"] = np.nan

    def make_sightline_id(row):
        gl = row["sightline_glon"]
        gb = row["sightline_glat"]
        if not (np.isfinite(gl) and np.isfinite(gb)):
            return ""
        return f"l{gl:+.3f}_b{gb:+.3f}"

    df["sightline_id"] = df.apply(make_sightline_id, axis=1)

    def make_bin_id(row):
        l_col2 = find_column(df.columns, ["l", "glon"])
        b_col2 = find_column(df.columns, ["b", "glat"])
        if l_col2 is None or b_col2 is None:
            return ""
        gl = row.get(l_col2, np.nan)
        gb = row.get(b_col2, np.nan)
        if not (np.isfinite(gl) and np.isfinite(gb)):
            return ""
        return f"l{round(gl / 0.1) * 0.1:+.1f}_b{round(gb / 0.1) * 0.1:+.1f}"

    df["sightline_bin_0p1deg"] = df.apply(make_bin_id, axis=1)

    n_tagged = (df["sightline_id"] != "").sum()
    print(f"  {n_tagged:,} stars tagged with sightline IDs.")
    return df


# =============================================================================
# Step 15 – Harvest Best Parameters
# =============================================================================

def _pick_best_value(df, candidates, value_col, source_col):
    """
    Scan a priority-ordered list of candidate columns and write the first
    finite value found into value_col, recording the source label in source_col.

    Each entry in candidates is a dict with keys:
      'col'   – column name in df (may be absent)
      'label' – provenance string to record in source_col
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
    """
    Build best-available Teff, logg, metallicity, radius, and luminosity
    columns by scanning all available sources in priority order.

    Priority for Teff:
      ASTRA (m_h_atm) > Gaia XGBoost > Gaia GSP-Spec > Gaia AP > ASTRA raw

    Priority for metallicity:
      Gaia XGBoost [M/H] > ASTRA [M/H] > ASTRA [Fe/H] > BDBS [Fe/H] >
      Bensby [Fe/H] > Gaia GSP-Spec

    Luminosity is estimated from ASTRA radius + Teff when a direct value
    is not present.
    """
    print("\n[Step 15] Harvesting best stellar parameters …")
    T_SUN = 5772.0

    # -- Teff
    teff_candidates = [
        {"col": "teff",          "label": "ASTRA"},
        {"col": "xgb_teff",      "label": "XGB_GAIA"},
        {"col": "Teff-S",        "label": "GAIA_GSPSPEC"},
        {"col": "Teff_x",        "label": "GAIA_AP"},
        {"col": "doppler_teff",  "label": "ASTRA_DOPPLER"},
        {"col": "raw_teff",      "label": "ASTRA_RAW"},
    ]
    df = _pick_best_value(df, teff_candidates, "teff_best", "teff_source_best")

    # -- logg
    logg_candidates = [
        {"col": "logg",          "label": "ASTRA"},
        {"col": "xgb_logg",      "label": "XGB_GAIA"},
        {"col": "logg-S",        "label": "GAIA_GSPSPEC"},
        {"col": "logg_x",        "label": "GAIA_AP"},
        {"col": "doppler_logg",  "label": "ASTRA_DOPPLER"},
        {"col": "raw_logg",      "label": "ASTRA_RAW"},
    ]
    df = _pick_best_value(df, logg_candidates, "logg_best", "logg_source_best")

    # -- Metallicity ([M/H] or [Fe/H])
    metal_candidates = [
        {"col": "xgb_mh",          "label": "XGB_GAIA"},
        {"col": "xgb_m_h",         "label": "XGB_GAIA"},
        {"col": "m_h_atm",         "label": "ASTRA"},
        {"col": "fe_h",            "label": "ASTRA"},
        {"col": "[M/H]-S",         "label": "GAIA_GSPSPEC"},
        {"col": "[Fe/H]-S",        "label": "GAIA_GSPSPEC"},
        {"col": "bdbs_[Fe/H]",     "label": "BDBS"},
        {"col": "bdbs_fe_h",       "label": "BDBS"},
        {"col": "bensby_[Fe/H]",   "label": "BENSBY"},
        {"col": "bensby_fe_h",     "label": "BENSBY"},
        {"col": "raw_m_h_atm",     "label": "ASTRA_RAW"},
    ]
    df = _pick_best_value(df, metal_candidates, "metallicity_best", "metallicity_source_best")

    # -- Radius
    radius_candidates = [
        {"col": "radius",    "label": "ASTRA"},
        {"col": "Rad",       "label": "GAIA_AP"},
        {"col": "Rad-Flame", "label": "GAIA_FLAME"},
    ]
    df = _pick_best_value(df, radius_candidates, "radius_best_rsun", "radius_source_best")

    # -- Luminosity: prefer direct value, then derive from R and Teff
    lum_candidates = [
        {"col": "luminosity",      "label": "ASTRA"},
        {"col": "Lum-Flame",       "label": "GAIA_FLAME"},
    ]
    df = _pick_best_value(df, lum_candidates, "luminosity_best_lsun", "luminosity_source_best")

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

    for col, label in [("teff_best",     "Teff"),
                        ("logg_best",     "logg"),
                        ("metallicity_best", "[M/H]"),
                        ("radius_best_rsun", "radius"),
                        ("luminosity_best_lsun", "luminosity")]:
        n = np.isfinite(as_float(df[col])).sum()
        print(f"  {label:<12}: {n:,} stars")

    return df


# =============================================================================
# Step 16 – Add Provenance Flags
# =============================================================================

def add_provenance_flags(df):
    """
    Add boolean 'has_*' columns and composite 'calibration_ready' flag.

    calibration_ready requires:
      - dereddened J0, H0, Ks0
      - Teff_best, logg_best, metallicity_best, luminosity_best
    """
    print("\n[Step 16] Adding provenance and quality flags …")

    def col_is_finite(col_name):
        if col_name not in df.columns:
            return pd.Series(False, index=df.index)
        return np.isfinite(as_float(df[col_name]))

    def col_is_nonempty(col_name):
        if col_name not in df.columns:
            return pd.Series(False, index=df.index)
        return df[col_name].fillna("").astype(str).str.strip() != ""

    df["has_virac2"]        = col_is_nonempty("vvv_srcid") | \
                              col_is_finite("virac2_sep_arcsec")
    df["has_2mass"]         = col_is_nonempty("tmass_2MASS") | \
                              col_is_finite("tmass_sep_arcsec")
    df["has_reddening"]     = col_is_finite("ext_e_jks")
    df["has_best_j"]        = col_is_finite("j_mag_best")
    df["has_best_h"]        = col_is_finite("h_mag_best")
    df["has_best_ks"]       = col_is_finite("ks_mag_best")
    df["has_best_phot"]     = df["has_best_j"] & df["has_best_h"] & df["has_best_ks"]
    df["has_dereddened_phot"] = col_is_finite("j0") & col_is_finite("h0") & col_is_finite("ks0")
    df["has_sightline"]     = col_is_nonempty("sightline_id")
    df["has_teff_best"]     = col_is_finite("teff_best")
    df["has_logg_best"]     = col_is_finite("logg_best")
    df["has_metallicity_best"]  = col_is_finite("metallicity_best")
    df["has_radius_best"]   = col_is_finite("radius_best_rsun")
    df["has_luminosity_best"]   = col_is_finite("luminosity_best_lsun")
    df["has_core_params"]   = (df["has_teff_best"] & df["has_logg_best"] &
                               df["has_metallicity_best"] & df["has_luminosity_best"])

    df["calibration_ready_phot"]   = df["has_dereddened_phot"] & df["has_reddening"]
    df["calibration_ready_params"] = df["has_core_params"]
    df["calibration_ready"]        = df["calibration_ready_phot"] & df["calibration_ready_params"]

    print(f"  calibration_ready (phot + params): {df['calibration_ready'].sum():,}")
    print(f"  calibration_ready (phot only)    : {df['calibration_ready_phot'].sum():,}")
    print(f"  has_core_params                  : {df['has_core_params'].sum():,}")
    return df


# =============================================================================
# Step 17 – Prune Bookkeeping Columns
# =============================================================================

BOOKKEEPING_COLUMNS = [
    # SDSS/APOGEE target-selection flags
    "sdss5_target_flags",
    "sdss4_apogee_target1_flags",
    "sdss4_apogee_target2_flags",
    "sdss4_apogee2_target1_flags",
    "sdss4_apogee2_target2_flags",
    "sdss4_apogee2_target3_flags",
    "sdss4_apogee_member_flags",
    "sdss4_apogee_extra_target_flags",
    # ASTRA pipeline task housekeeping
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


def prune_bookkeeping(df):
    """
    Drop SDSS/APOGEE target-selection flags and ASTRA task-PK columns.

    These columns are pipeline housekeeping and carry no astrophysical
    information.  All science columns are preserved.
    """
    print("\n[Step 17] Pruning bookkeeping columns …")
    to_drop = [c for c in BOOKKEEPING_COLUMNS if c in df.columns]
    df = df.drop(columns=to_drop)
    print(f"  Dropped {len(to_drop)} bookkeeping columns. "
          f"Remaining columns: {len(df.columns)}")
    return df


# =============================================================================
# Step 18 – Write Outputs
# =============================================================================

FRONT_COLUMNS = [
    "gaia_dr3_source_id", "catalogid", "sdss_id",
    "ra_icrs", "dec_icrs", "l", "b",
    "teff_best", "teff_source_best",
    "logg_best", "logg_source_best",
    "metallicity_best", "metallicity_source_best",
    "radius_best_rsun", "radius_source_best",
    "luminosity_best_lsun", "log10_luminosity_best_lsun", "luminosity_source_best",
    "j_mag_best", "h_mag_best", "ks_mag_best",
    "j_mag_source", "h_mag_source", "ks_mag_source", "phot_source_overall",
    "j_minus_h_best", "h_minus_ks_best", "j_minus_ks_best",
    "ext_e_jks", "a_j", "a_h", "a_ks",
    "j0", "h0", "ks0",
    "j0_minus_h0", "h0_minus_ks0", "j0_minus_ks0",
    "sightline_id", "sightline_glon", "sightline_glat", "sightline_bin_0p1deg",
    "has_virac2", "has_2mass", "has_reddening",
    "has_dereddened_phot", "has_core_params",
    "calibration_ready_phot", "calibration_ready_params", "calibration_ready",
]


def _reorder_columns(df):
    front  = [c for c in FRONT_COLUMNS if c in df.columns]
    rest   = [c for c in df.columns   if c not in front]
    return df[front + rest]


def _phot_source_overall(df):
    """Label photometry source: VIRAC2, 2MASS, MIXED, or NONE."""
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


def write_outputs(df, out_prefix):
    """
    Write the final pipeline products:
      - master catalog (all stars passing Steps 1–6)
      - calibration-ready subset
      - plain-text summary
    """
    print(f"\n[Step 18] Writing outputs with prefix '{out_prefix}' …")

    df["phot_source_overall"] = _phot_source_overall(df)
    df = _reorder_columns(df)

    ready = df.loc[df["calibration_ready"]].copy()

    master_fits = pathlib.Path(f"{out_prefix}_master.fits")
    master_csv  = pathlib.Path(f"{out_prefix}_master.csv")
    ready_fits  = pathlib.Path(f"{out_prefix}_selected.fits")
    ready_csv   = pathlib.Path(f"{out_prefix}_selected.csv")
    summary_txt = pathlib.Path(f"{out_prefix}_summary.txt")

    Table.from_pandas(df).write(master_fits, overwrite=True)
    df.to_csv(master_csv, index=False)
    print(f"  Master catalog: {master_fits}  ({len(df):,} rows)")

    Table.from_pandas(ready).write(ready_fits, overwrite=True)
    ready.to_csv(ready_csv, index=False)
    print(f"  Selected (calibration-ready): {ready_fits}  ({len(ready):,} rows)")

    _write_summary(df, ready, summary_txt)
    print(f"  Summary: {summary_txt}")


def _write_summary(master, ready, path):
    n = len(master)
    lines = [
        "Roman Bulge Calibration Pipeline – Summary",
        "=" * 43,
        f"Stars passing footprint + stellar cuts  : {n:,}",
        f"Calibration-ready (full)               : {len(ready):,}",
        f"Calibration-ready photometry only      : {master['calibration_ready_phot'].sum():,}",
        f"Has core parameters (Teff+logg+Z+L)    : {master['has_core_params'].sum():,}",
        "",
        "Photometry coverage",
        "-" * 20,
        f"VIRAC2 matched                         : {master['has_virac2'].sum():,}",
        f"2MASS matched                          : {master['has_2mass'].sum():,}",
        f"Has reddening E(J-Ks)                  : {master['has_reddening'].sum():,}",
        f"Dereddened J0/H0/Ks0 available         : {master['has_dereddened_phot'].sum():,}",
    ]

    if "ext_e_jks" in master.columns:
        ejks = as_float(master["ext_e_jks"]).to_numpy()
        if np.isfinite(ejks).any():
            lines += [
                "",
                "Reddening statistics",
                "-" * 20,
                f"E(J-Ks) median                         : {np.nanmedian(ejks):.4f}",
                f"E(J-Ks) 16th percentile                : {np.nanpercentile(ejks, 16):.4f}",
                f"E(J-Ks) 84th percentile                : {np.nanpercentile(ejks, 84):.4f}",
            ]

    lines += [
        "",
        "Parameter coverage",
        "-" * 20,
        f"has_teff_best                          : {master['has_teff_best'].sum():,}",
        f"has_logg_best                          : {master['has_logg_best'].sum():,}",
        f"has_metallicity_best                   : {master['has_metallicity_best'].sum():,}",
        f"has_luminosity_best                    : {master['has_luminosity_best'].sum():,}",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Roman Bulge Calibration Pipeline")
    print("=" * 60)

    # Step 1
    df = load_astra(ASTRA_FITS)

    # Step 2
    df = merge_gaia_xgboost(df, XGBOOST_FITS)

    # Steps 3–4: positional merges need coordinates first.
    # ASTRA contains RA/Dec or l/b so we set them up before merging.
    df = ensure_icrs_coordinates(df)   # Step 7 called early for local merges

    # Step 3
    df = merge_bdbs(df, BDBS_FITS, BDBS_RADIUS_ARCSEC)

    # Step 4
    df = merge_bensby(df, BENSBY_FITS, BENSBY_RADIUS_ARCSEC)

    # Step 5
    df = apply_roman_footprint(
        df,
        GBTDS_CENTERS_OVERGUIDE,
        TILE_W_DEG,
        TILE_H_DEG,
        TILE_PA_DEG,
    )

    # Step 6
    df = apply_stellar_cuts(df, TEFF_MAX, H_MAG_MAX)

    # Step 7 already done above; coordinates are already present.

    # Steps 8–10: remote CDS cross-matches
    df = xmatch_virac2(df, VIRAC2_RADIUS_ARCSEC)
    df = xmatch_2mass(df, TMASS_RADIUS_ARCSEC)
    df = xmatch_vvv_reddening(df, VVV_RED_RADIUS_ARCSEC)

    # Step 11
    df = build_best_photometry(df)

    # Step 12
    df = compute_extinction(df, A_KS_PER_E_JKS, A_H_PER_E_JKS, A_J_PER_E_JKS)

    # Step 13
    df = compute_dereddened_mags(df)

    # Step 14
    df = tag_sightlines(df)

    # Step 15
    df = harvest_best_parameters(df)

    # Step 16
    df = add_provenance_flags(df)

    # Step 17
    df = prune_bookkeeping(df)

    # Step 18
    write_outputs(df, OUT_PREFIX)

    print("\nPipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
