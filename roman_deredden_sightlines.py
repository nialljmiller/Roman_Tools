#!/usr/bin/env python3
"""
Build a dereddened, sightline-tagged sample from the ASTRA Roman-footprint sample.

This script is designed to run *after* roman_selection.py has already produced the
selected ASTRA sample, for example:

    astra_overguide_roman_wz_selected.fits

What this script does:
    1. Reads the selected ASTRA sample.
    2. Ensures every row has ICRS coordinates.
    3. Cross-matches the sample against:
         - VIRAC2 (VVV/VVVX near-IR photometry): II/387/virac2
         - 2MASS PSC: II/246/out
         - VVV high-resolution reddening map: J/A+A/644/A140/ejkmap
    4. Builds a single output table with:
         - best-available J/H/Ks photometry (prefer VIRAC2, fallback to 2MASS)
         - reddening values E(J-Ks)
         - simple bulge-law extinction estimates A_J, A_H, A_Ks
         - dereddened magnitudes and colors
         - sightline tags based on the matched reddening-map cell
    5. Writes FITS + CSV + a plain-text summary.

Notes:
    - This uses astroquery.xmatch, which uploads the local source list to CDS XMatch.
    - The local uploaded table must be in ICRS/J2000 coordinates. If your ASTRA
      table only has Galactic coordinates (l, b), the script converts them.
    - The extinction coefficients are configurable. The defaults assume a bulge-style
      law using E(J-Ks) -> A_Ks and derive A_J and A_H from that.

Recommended install:
    python -m pip install astropy astroquery pandas numpy
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.xmatch import XMatch
from astroquery.vizier import Vizier


# -----------------------------------------------------------------------------
# Remote catalog identifiers
# -----------------------------------------------------------------------------
VIRAC2_TABLE = "vizier:II/387/virac2"
TMASS_TABLE = "vizier:II/246/out"
VVV_REDDENING_TABLE = "vizier:J/A+A/644/A140/ejkmap"


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def _lower_name_map(columns: Iterable[str]) -> dict[str, str]:
    """Map lowercase column names to their original names."""
    mapping: dict[str, str] = {}
    for name in columns:
        mapping[str(name).lower()] = str(name)
    return mapping



def find_first_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    """Return the first matching column name, case-insensitively."""
    mapping = _lower_name_map(columns)
    for candidate in candidates:
        key = str(candidate).lower()
        if key in mapping:
            return mapping[key]
    return None



def require_column(columns: Iterable[str], candidates: Iterable[str], label: str) -> str:
    """Find a required column or raise a helpful error."""
    found = find_first_column(columns, candidates)
    if found is None:
        raise KeyError(
            f"Could not find a column for {label}. Tried: {list(candidates)}"
        )
    return found



def as_float_array(series: pd.Series) -> np.ndarray:
    """Convert a pandas Series to a float numpy array, coercing invalid values to NaN."""
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)



def choose_first_finite(primary: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Element-wise choose primary where finite, otherwise fallback."""
    out = np.array(primary, dtype=float, copy=True)
    use_fallback = ~np.isfinite(out) & np.isfinite(fallback)
    out[use_fallback] = fallback[use_fallback]
    return out



def choose_source_label(primary: np.ndarray, fallback: np.ndarray, primary_label: str, fallback_label: str) -> np.ndarray:
    """Create an array of source labels corresponding to chosen finite values."""
    labels = np.full(primary.shape, "", dtype=object)
    primary_ok = np.isfinite(primary)
    fallback_ok = ~primary_ok & np.isfinite(fallback)
    labels[primary_ok] = primary_label
    labels[fallback_ok] = fallback_label
    return labels



def round_if_finite(value: float, ndigits: int) -> str:
    """Round a float to a string if finite, otherwise return 'nan'."""
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.{ndigits}f}"



def table_to_scalar_pandas(tbl: Table) -> tuple[pd.DataFrame, list[str]]:
    """
    Convert an astropy Table to pandas after dropping multidimensional columns.

    FITS products like ASTRA can contain vector-valued columns (for example
    bitflag arrays). pandas cannot represent these cleanly via astropy's
    to_pandas bridge, and they are not needed for the cross-match workflow.

    Returns
    -------
    dataframe, dropped_columns
    """
    scalar_names: list[str] = []
    dropped: list[str] = []
    for name in tbl.colnames:
        shape = getattr(tbl[name], "shape", ())
        if len(shape) <= 1:
            scalar_names.append(name)
        else:
            dropped.append(name)
    return tbl[scalar_names].to_pandas(), dropped


# -----------------------------------------------------------------------------
# Coordinate handling
# -----------------------------------------------------------------------------

def ensure_icrs_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataframe contains ICRS coordinates in decimal degrees.

    Priority:
        1. Existing RA/Dec columns if present.
        2. Convert from Galactic l/b if present.
    """
    out = df.copy()

    ra_col = find_first_column(
        out.columns,
        ["ra", "ra_deg", "radeg", "raj2000", "ra_icrs", "RA", "RAdeg"],
    )
    dec_col = find_first_column(
        out.columns,
        ["dec", "dec_deg", "dedeg", "dej2000", "dec_icrs", "DEC", "DEdeg"],
    )

    if ra_col is not None and dec_col is not None:
        out["ra_icrs"] = pd.to_numeric(out[ra_col], errors="coerce")
        out["dec_icrs"] = pd.to_numeric(out[dec_col], errors="coerce")
        return out

    l_col = find_first_column(out.columns, ["l", "glon", "lon", "gal_l"])
    b_col = find_first_column(out.columns, ["b", "glat", "lat", "gal_b"])

    if l_col is None or b_col is None:
        raise KeyError(
            "Could not determine source coordinates. Need either RA/Dec or Galactic l/b columns."
        )

    l_vals = pd.to_numeric(out[l_col], errors="coerce").to_numpy(dtype=float)
    b_vals = pd.to_numeric(out[b_col], errors="coerce").to_numpy(dtype=float)

    coords = SkyCoord(l=l_vals * u.deg, b=b_vals * u.deg, frame="galactic")
    out["ra_icrs"] = coords.icrs.ra.deg
    out["dec_icrs"] = coords.icrs.dec.deg
    return out


# -----------------------------------------------------------------------------
# XMatch helpers
# -----------------------------------------------------------------------------

def xmatch_local_to_remote(
    local_df: pd.DataFrame,
    remote_table: str,
    radius_arcsec: float,
    output_path: pathlib.Path,
) -> pd.DataFrame:
    """
    Cross-match a local table with a remote CDS/VizieR table.

    The local table must already contain:
        - row_id
        - ra_icrs
        - dec_icrs

    Returns a pandas DataFrame with *all* matches within the search radius.
    """
    local_table = Table.from_pandas(local_df[["row_id", "ra_icrs", "dec_icrs"]].copy())

    result = XMatch.query(
        cat1=local_table,
        cat2=remote_table,
        max_distance=radius_arcsec * u.arcsec,
        colRA1="ra_icrs",
        colDec1="dec_icrs",
    )

    result.write(output_path, overwrite=True)
    return result.to_pandas()



def query_reddening_map_vizier(
    local_df: pd.DataFrame,
    radius_arcsec: float,
    output_path: pathlib.Path,
    progress_every: int = 50,
) -> pd.DataFrame:
    """
    Query the VVV reddening map via VizieR region searches.

    CDS XMatch does not reliably expose every VizieR table for uploaded-table
    cross-matching. The Surot et al. (2020) reddening map can still be queried
    through VizieR by position, which is fine here because the selected sample is
    small.
    """
    viz = Vizier(columns=["*", "+_r"], row_limit=1)
    rows: list[dict[str, object]] = []

    n_total = len(local_df)
    for i, row in enumerate(local_df[["row_id", "ra_icrs", "dec_icrs"]].itertuples(index=False), start=1):
        row_id, ra_deg, dec_deg = row
        if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
            continue

        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        try:
            result = viz.query_region(
                coord,
                radius=radius_arcsec * u.arcsec,
                catalog=VVV_REDDENING_TABLE.replace("vizier:", ""),
            )
        except Exception as exc:
            print(f"  Warning: reddening query failed for row_id={row_id}: {exc}")
            continue

        if len(result) == 0:
            if progress_every > 0 and (i % progress_every == 0 or i == n_total):
                print(f"  Reddening queries completed: {i}/{n_total}")
            continue

        tbl = result[0]
        if len(tbl) == 0:
            if progress_every > 0 and (i % progress_every == 0 or i == n_total):
                print(f"  Reddening queries completed: {i}/{n_total}")
            continue

        match = tbl[0]
        record = {name: match[name] for name in tbl.colnames}
        record["row_id"] = int(row_id)
        rows.append(record)

        if progress_every > 0 and (i % progress_every == 0 or i == n_total):
            print(f"  Reddening queries completed: {i}/{n_total}")

    if rows:
        out_df = pd.DataFrame(rows)
        Table.from_pandas(out_df).write(output_path, overwrite=True)
        return out_df

    empty_df = pd.DataFrame({"row_id": pd.Series(dtype="int64")})
    Table.from_pandas(empty_df).write(output_path, overwrite=True)
    return empty_df



def best_match_per_source(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the nearest match per source and store the multiplicity.

    Handles either the modern `angDist` column or the older `_r` distance column.
    """
    if matches.empty:
        return matches.copy()

    dist_col = None
    if "angDist" in matches.columns:
        dist_col = "angDist"
    elif "_r" in matches.columns:
        dist_col = "_r"
    else:
        raise KeyError(
            "Cross-match result did not include a recognized distance column (angDist or _r)."
        )

    work = matches.copy()
    work[dist_col] = pd.to_numeric(work[dist_col], errors="coerce")
    work = work.sort_values(["row_id", dist_col], ascending=[True, True])

    counts = work.groupby("row_id").size().rename("match_multiplicity")
    best = work.drop_duplicates(subset=["row_id"], keep="first").copy()
    best = best.merge(counts, on="row_id", how="left")
    return best



def prefix_columns(df: pd.DataFrame, prefix: str, protected: Optional[set[str]] = None) -> pd.DataFrame:
    """Prefix all columns except a protected set."""
    if protected is None:
        protected = set()
    rename_map: dict[str, str] = {}
    for col in df.columns:
        if col in protected:
            continue
        rename_map[col] = f"{prefix}{col}"
    return df.rename(columns=rename_map)


# -----------------------------------------------------------------------------
# Catalog-specific extraction and derived quantities
# -----------------------------------------------------------------------------

def build_best_photometry_columns(master: pd.DataFrame) -> pd.DataFrame:
    """Prefer VIRAC2 J/H/Ks, then fall back to 2MASS."""
    out = master.copy()

    vvv_j = pd.to_numeric(out.get("vvv_Jmag"), errors="coerce") if "vvv_Jmag" in out else pd.Series(np.nan, index=out.index)
    vvv_h = pd.to_numeric(out.get("vvv_Hmag"), errors="coerce") if "vvv_Hmag" in out else pd.Series(np.nan, index=out.index)
    vvv_k = pd.to_numeric(out.get("vvv_Ksmag"), errors="coerce") if "vvv_Ksmag" in out else pd.Series(np.nan, index=out.index)

    tmass_j = pd.to_numeric(out.get("tmass_Jmag"), errors="coerce") if "tmass_Jmag" in out else pd.Series(np.nan, index=out.index)
    tmass_h = pd.to_numeric(out.get("tmass_Hmag"), errors="coerce") if "tmass_Hmag" in out else pd.Series(np.nan, index=out.index)
    tmass_k = pd.to_numeric(out.get("tmass_Kmag"), errors="coerce") if "tmass_Kmag" in out else pd.Series(np.nan, index=out.index)

    out["j_mag_best"] = choose_first_finite(vvv_j.to_numpy(dtype=float), tmass_j.to_numpy(dtype=float))
    out["h_mag_best"] = choose_first_finite(vvv_h.to_numpy(dtype=float), tmass_h.to_numpy(dtype=float))
    out["ks_mag_best"] = choose_first_finite(vvv_k.to_numpy(dtype=float), tmass_k.to_numpy(dtype=float))

    out["j_mag_source"] = choose_source_label(vvv_j.to_numpy(dtype=float), tmass_j.to_numpy(dtype=float), "VIRAC2", "2MASS")
    out["h_mag_source"] = choose_source_label(vvv_h.to_numpy(dtype=float), tmass_h.to_numpy(dtype=float), "VIRAC2", "2MASS")
    out["ks_mag_source"] = choose_source_label(vvv_k.to_numpy(dtype=float), tmass_k.to_numpy(dtype=float), "VIRAC2", "2MASS")

    # Best-available uncertainties when present.
    vvv_ej = pd.to_numeric(out.get("vvv_e_Jmag"), errors="coerce") if "vvv_e_Jmag" in out else pd.Series(np.nan, index=out.index)
    vvv_eh = pd.to_numeric(out.get("vvv_e_Hmag"), errors="coerce") if "vvv_e_Hmag" in out else pd.Series(np.nan, index=out.index)
    vvv_ek = pd.to_numeric(out.get("vvv_e_Ksmag"), errors="coerce") if "vvv_e_Ksmag" in out else pd.Series(np.nan, index=out.index)

    tmass_ej = pd.to_numeric(out.get("tmass_e_Jmag"), errors="coerce") if "tmass_e_Jmag" in out else pd.Series(np.nan, index=out.index)
    tmass_eh = pd.to_numeric(out.get("tmass_e_Hmag"), errors="coerce") if "tmass_e_Hmag" in out else pd.Series(np.nan, index=out.index)
    tmass_ek = pd.to_numeric(out.get("tmass_e_Kmag"), errors="coerce") if "tmass_e_Kmag" in out else pd.Series(np.nan, index=out.index)

    out["e_j_mag_best"] = choose_first_finite(vvv_ej.to_numpy(dtype=float), tmass_ej.to_numpy(dtype=float))
    out["e_h_mag_best"] = choose_first_finite(vvv_eh.to_numpy(dtype=float), tmass_eh.to_numpy(dtype=float))
    out["e_ks_mag_best"] = choose_first_finite(vvv_ek.to_numpy(dtype=float), tmass_ek.to_numpy(dtype=float))

    out["j_minus_h_best"] = out["j_mag_best"] - out["h_mag_best"]
    out["h_minus_ks_best"] = out["h_mag_best"] - out["ks_mag_best"]
    out["j_minus_ks_best"] = out["j_mag_best"] - out["ks_mag_best"]

    return out



def add_extinction_columns(
    master: pd.DataFrame,
    aks_per_ejks: float,
    ah_per_ejks: float,
    aj_per_ejks: float,
) -> pd.DataFrame:
    """Add extinction and dereddened quantities."""
    out = master.copy()

    ejks_col = None
    if "ext_E(J-Ks)" in out.columns:
        ejks_col = "ext_E(J-Ks)"
    elif "ext_E(J-Ks)_" in out.columns:
        ejks_col = "ext_E(J-Ks)_"
    elif "ext_EJKs" in out.columns:
        ejks_col = "ext_EJKs"

    if ejks_col is None:
        out["ext_e_jks"] = np.nan
    else:
        out["ext_e_jks"] = pd.to_numeric(out[ejks_col], errors="coerce")

    err_col = None
    for candidate in ["ext_e_", "ext_e_E(J-Ks)", "ext_e_EJKs"]:
        if candidate in out.columns:
            err_col = candidate
            break
    if err_col is not None:
        out["ext_e_jks_err"] = pd.to_numeric(out[err_col], errors="coerce")
    else:
        out["ext_e_jks_err"] = np.nan

    out["a_ks"] = aks_per_ejks * out["ext_e_jks"]
    out["a_h"] = ah_per_ejks * out["ext_e_jks"]
    out["a_j"] = aj_per_ejks * out["ext_e_jks"]

    out["j0"] = out["j_mag_best"] - out["a_j"]
    out["h0"] = out["h_mag_best"] - out["a_h"]
    out["ks0"] = out["ks_mag_best"] - out["a_ks"]

    out["j0_minus_h0"] = out["j0"] - out["h0"]
    out["h0_minus_ks0"] = out["h0"] - out["ks0"]
    out["j0_minus_ks0"] = out["j0"] - out["ks0"]

    return out



def add_sightline_tags(master: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable sightline tags from the reddening-map match."""
    out = master.copy()

    glon_col = find_first_column(out.columns, ["ext_GLON", "ext_glon"])
    glat_col = find_first_column(out.columns, ["ext_GLAT", "ext_glat"])
    tile_col = find_first_column(out.columns, ["ext_Tile", "ext_tile"])
    dist_col = find_first_column(out.columns, ["ext_angDist", "ext__r"])

    if glon_col is not None:
        out["sightline_glon"] = pd.to_numeric(out[glon_col], errors="coerce")
    else:
        out["sightline_glon"] = np.nan

    if glat_col is not None:
        out["sightline_glat"] = pd.to_numeric(out[glat_col], errors="coerce")
    else:
        out["sightline_glat"] = np.nan

    if tile_col is not None:
        out["sightline_tile"] = out[tile_col].astype(str)
        out.loc[out[tile_col].isna(), "sightline_tile"] = ""
    else:
        out["sightline_tile"] = ""

    if dist_col is not None:
        out["ext_map_match_sep_arcsec"] = pd.to_numeric(out[dist_col], errors="coerce")
    else:
        out["ext_map_match_sep_arcsec"] = np.nan

    sightline_ids = []
    coarse_ids = []
    for tile, glon, glat in zip(
        out["sightline_tile"].to_numpy(dtype=object),
        out["sightline_glon"].to_numpy(dtype=float),
        out["sightline_glat"].to_numpy(dtype=float),
    ):
        tile_str = "" if tile is None else str(tile)
        if np.isfinite(glon) and np.isfinite(glat):
            sightline_ids.append(
                f"{tile_str}|l={glon:+08.4f}|b={glat:+08.4f}"
            )
            coarse_ids.append(
                f"l={glon:+05.1f}|b={glat:+05.1f}"
            )
        else:
            sightline_ids.append("")
            coarse_ids.append("")

    out["sightline_id"] = np.array(sightline_ids, dtype=object)
    out["sightline_bin_0p1deg"] = np.array(coarse_ids, dtype=object)
    return out


# -----------------------------------------------------------------------------
# Summary output
# -----------------------------------------------------------------------------

def write_summary(master: pd.DataFrame, path: pathlib.Path) -> None:
    """Write a plain-text summary of match completeness."""
    n_total = len(master)

    def count_nonempty(column: str) -> int:
        if column not in master.columns:
            return 0
        series = master[column]
        if pd.api.types.is_numeric_dtype(series):
            return int(np.isfinite(pd.to_numeric(series, errors="coerce")).sum())
        text = series.fillna("").astype(str).str.strip()
        return int((text != "").sum())

    def count_finite(column: str) -> int:
        if column not in master.columns:
            return 0
        return int(np.isfinite(pd.to_numeric(master[column], errors="coerce")).sum())

    lines = []
    lines.append("Roman deredden / sightline-tagging summary")
    lines.append("=" * 44)
    lines.append(f"Total selected ASTRA stars                 : {n_total}")
    lines.append(f"Matched VIRAC2 rows                        : {count_nonempty('vvv_srcid')}")
    lines.append(f"Matched 2MASS rows                         : {count_nonempty('tmass_2MASS') + count_nonempty('tmass__2MASS')}")
    lines.append(f"Matched reddening-map rows                 : {count_finite('ext_e_jks')}")
    lines.append(f"Best J available                           : {count_finite('j_mag_best')}")
    lines.append(f"Best H available                           : {count_finite('h_mag_best')}")
    lines.append(f"Best Ks available                          : {count_finite('ks_mag_best')}")
    lines.append(f"Dereddened J0 available                    : {count_finite('j0')}")
    lines.append(f"Dereddened H0 available                    : {count_finite('h0')}")
    lines.append(f"Dereddened Ks0 available                   : {count_finite('ks0')}")
    lines.append(f"Unique sightline IDs                       : {count_nonempty('sightline_id')}")

    if "ext_e_jks" in master.columns:
        ejks = pd.to_numeric(master["ext_e_jks"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(ejks).any():
            lines.append("")
            lines.append("Reddening statistics")
            lines.append("-" * 20)
            lines.append(f"E(J-Ks) median                            : {np.nanmedian(ejks):.4f}")
            lines.append(f"E(J-Ks) 16th percentile                   : {np.nanpercentile(ejks, 16):.4f}")
            lines.append(f"E(J-Ks) 84th percentile                   : {np.nanpercentile(ejks, 84):.4f}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Main program
# -----------------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-match the selected ASTRA Roman sample to VIRAC2, 2MASS, and the VVV reddening map."
    )

    parser.add_argument(
        "input_fits",
        nargs="?",
        default="astra_overguide_roman_wz_selected.fits",
        help="Input FITS table produced by roman_selection.py",
    )
    parser.add_argument(
        "--out-prefix",
        default="astra_roman_dereddened",
        help="Prefix for output products",
    )
    parser.add_argument(
        "--virac-radius-arcsec",
        type=float,
        default=1.0,
        help="Cross-match radius for VIRAC2 photometry",
    )
    parser.add_argument(
        "--tmass-radius-arcsec",
        type=float,
        default=1.0,
        help="Cross-match radius for 2MASS photometry",
    )
    parser.add_argument(
        "--reddening-radius-arcsec",
        type=float,
        default=120.0,
        help="Cross-match radius for the VVV reddening map cell centers",
    )
    parser.add_argument(
        "--aks-per-ejks",
        type=float,
        default=0.528,
        help="A_Ks / E(J-Ks) coefficient",
    )
    parser.add_argument(
        "--ah-per-ejks",
        type=float,
        default=0.857,
        help="A_H / E(J-Ks) coefficient",
    )
    parser.add_argument(
        "--aj-per-ejks",
        type=float,
        default=1.528,
        help="A_J / E(J-Ks) coefficient",
    )
    parser.add_argument(
        "--skip-virac2",
        action="store_true",
        help="Do not query VIRAC2",
    )
    parser.add_argument(
        "--skip-2mass",
        action="store_true",
        help="Do not query 2MASS",
    )
    parser.add_argument(
        "--skip-reddening",
        action="store_true",
        help="Do not query the VVV reddening map",
    )

    return parser



def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    input_path = pathlib.Path(args.input_fits).expanduser().resolve()
    out_prefix = pathlib.Path(args.out_prefix).expanduser().resolve()
    out_dir = out_prefix.parent
    stem = out_prefix.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading input table: {input_path}")
    base_table = Table.read(input_path)
    base_df, dropped_columns = table_to_scalar_pandas(base_table)
    if dropped_columns:
        print(
            "Dropping multidimensional columns that cannot be converted to pandas: "
            + ", ".join(dropped_columns)
        )
    base_df = ensure_icrs_coordinates(base_df)
    base_df = base_df.reset_index(drop=True)
    base_df["row_id"] = np.arange(len(base_df), dtype=np.int64)

    print(f"Input rows: {len(base_df)}")

    master = base_df.copy()

    if not args.skip_virac2:
        print(f"Cross-matching to VIRAC2 with radius = {args.virac_radius_arcsec:.3f} arcsec")
        virac_raw_path = out_dir / f"{stem}_virac2_raw_matches.fits"
        virac_raw = xmatch_local_to_remote(
            local_df=base_df,
            remote_table=VIRAC2_TABLE,
            radius_arcsec=args.virac_radius_arcsec,
            output_path=virac_raw_path,
        )
        virac_best = best_match_per_source(virac_raw)
        virac_best = prefix_columns(virac_best, "vvv_", protected={"row_id"})
        master = master.merge(virac_best, on="row_id", how="left")
        print(f"  VIRAC2 matched sources: {virac_best['row_id'].nunique()}")

    if not args.skip_2mass:
        print(f"Cross-matching to 2MASS with radius = {args.tmass_radius_arcsec:.3f} arcsec")
        tmass_raw_path = out_dir / f"{stem}_2mass_raw_matches.fits"
        tmass_raw = xmatch_local_to_remote(
            local_df=base_df,
            remote_table=TMASS_TABLE,
            radius_arcsec=args.tmass_radius_arcsec,
            output_path=tmass_raw_path,
        )
        tmass_best = best_match_per_source(tmass_raw)
        tmass_best = prefix_columns(tmass_best, "tmass_", protected={"row_id"})
        master = master.merge(tmass_best, on="row_id", how="left")
        print(f"  2MASS matched sources: {tmass_best['row_id'].nunique()}")

    if not args.skip_reddening:
        print(f"Cross-matching to VVV reddening map with radius = {args.reddening_radius_arcsec:.3f} arcsec")
        reddening_raw_path = out_dir / f"{stem}_reddening_raw_matches.fits"
        try:
            reddening_raw = xmatch_local_to_remote(
                local_df=base_df,
                remote_table=VVV_REDDENING_TABLE,
                radius_arcsec=args.reddening_radius_arcsec,
                output_path=reddening_raw_path,
            )
            reddening_best = best_match_per_source(reddening_raw)
            print("  Reddening retrieval method: CDS XMatch")
        except Exception as exc:
            print(f"  CDS XMatch reddening lookup failed: {exc}")
            print("  Falling back to direct VizieR region queries for the reddening map")
            reddening_best = query_reddening_map_vizier(
                local_df=base_df,
                radius_arcsec=args.reddening_radius_arcsec,
                output_path=reddening_raw_path,
            )
            print("  Reddening retrieval method: VizieR query_region")
        reddening_best = prefix_columns(reddening_best, "ext_", protected={"row_id"})
        master = master.merge(reddening_best, on="row_id", how="left")
        print(f"  Reddening-map matched sources: {reddening_best['row_id'].nunique()}")

    master = build_best_photometry_columns(master)
    master = add_extinction_columns(
        master,
        aks_per_ejks=args.aks_per_ejks,
        ah_per_ejks=args.ah_per_ejks,
        aj_per_ejks=args.aj_per_ejks,
    )
    master = add_sightline_tags(master)

    # Reorder a useful subset toward the front.
    front_columns = [
        "row_id",
        "ra_icrs",
        "dec_icrs",
        "j_mag_best",
        "h_mag_best",
        "ks_mag_best",
        "j_mag_source",
        "h_mag_source",
        "ks_mag_source",
        "ext_e_jks",
        "ext_e_jks_err",
        "a_j",
        "a_h",
        "a_ks",
        "j0",
        "h0",
        "ks0",
        "j_minus_ks_best",
        "j0_minus_ks0",
        "sightline_tile",
        "sightline_glon",
        "sightline_glat",
        "sightline_id",
        "sightline_bin_0p1deg",
        "ext_map_match_sep_arcsec",
    ]
    existing_front = [col for col in front_columns if col in master.columns]
    remaining = [col for col in master.columns if col not in existing_front]
    master = master[existing_front + remaining]

    master_table = Table.from_pandas(master)

    out_fits = out_dir / f"{stem}.fits"
    out_csv = out_dir / f"{stem}.csv"
    out_summary = out_dir / f"{stem}_summary.txt"

    print(f"Writing FITS: {out_fits}")
    master_table.write(out_fits, overwrite=True)

    print(f"Writing CSV: {out_csv}")
    master.to_csv(out_csv, index=False)

    print(f"Writing summary: {out_summary}")
    write_summary(master, out_summary)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
