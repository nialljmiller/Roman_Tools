#!/usr/bin/env python3
"""
Step 3 – Remote cross-matches, photometry, extinction, sightlines
==================================================================
Cross-matches the selected sample against VIRAC2, 2MASS, and the VVV
reddening map, then builds best photometry, extinction, dereddened
magnitudes, and sightline tags.

This is a direct port of roman_deredden_sightlines.py into the pipeline,
with fixed input/output paths and no argparse.

Input  : step2_selected.fits
Output : step3_xmatched.fits  +  step3_summary.txt
"""

import pathlib
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.xmatch import XMatch
from astroquery.vizier import Vizier
import warnings
from astropy.io.fits.verify import VerifyWarning
from astropy.units import UnitsWarning

warnings.filterwarnings("ignore", category=VerifyWarning)
warnings.filterwarnings("ignore", category=UnitsWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_FITS   = pathlib.Path("step2_selected.fits")
OUTPUT_FITS  = pathlib.Path("step3_xmatched.fits")
SUMMARY_TXT  = pathlib.Path("step3_summary.txt")

# ---------------------------------------------------------------------------
# Remote catalog identifiers
# ---------------------------------------------------------------------------
VIRAC2_TABLE        = "vizier:II/387/virac2"
TMASS_TABLE         = "vizier:II/246/out"
VVV_REDDENING_TABLE = "vizier:J/A+A/644/A140/ejkmap"

VIRAC2_RADIUS_ARCSEC    = 1.0
TMASS_RADIUS_ARCSEC     = 1.0
REDDENING_RADIUS_ARCSEC = 120.0

# ---------------------------------------------------------------------------
# Extinction law coefficients
# ---------------------------------------------------------------------------
AKS_PER_EJKS = 0.528
AH_PER_EJKS  = 0.857
AJ_PER_EJKS  = 1.528

# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def lower_name_map(columns):
    return {str(n).lower(): str(n) for n in columns}


def find_first_column(columns, candidates):
    mapping = lower_name_map(columns)
    for candidate in candidates:
        key = str(candidate).lower()
        if key in mapping:
            return mapping[key]
    return None


def choose_first_finite(primary, fallback):
    out = np.array(primary, dtype=float, copy=True)
    use_fallback = ~np.isfinite(out) & np.isfinite(fallback)
    out[use_fallback] = fallback[use_fallback]
    return out


def choose_source_label(primary, fallback, primary_label, fallback_label):
    labels = np.full(primary.shape, "", dtype=object)
    primary_ok  = np.isfinite(primary)
    fallback_ok = ~primary_ok & np.isfinite(fallback)
    labels[primary_ok]  = primary_label
    labels[fallback_ok] = fallback_label
    return labels

# ---------------------------------------------------------------------------
# Load input
# ---------------------------------------------------------------------------

def load_input(fits_path):
    """Load the FITS table, dropping multi-dim columns that pandas cannot hold."""
    tbl = Table.read(fits_path)
    scalar_names = [n for n in tbl.colnames if len(getattr(tbl[n], "shape", ())) <= 1]
    dropped = [n for n in tbl.colnames if n not in scalar_names]
    if dropped:
        print(f"  Dropping {len(dropped)} multi-dim columns: {dropped[:6]}{'…' if len(dropped)>6 else ''}")
    df = tbl[scalar_names].to_pandas()
    return df

# ---------------------------------------------------------------------------
# Coordinate handling
# ---------------------------------------------------------------------------

def ensure_icrs_coordinates(df):
    out = df.copy()
    ra_col  = find_first_column(out.columns, ["ra", "ra_deg", "radeg", "raj2000", "ra_icrs", "RA", "ra_1"])
    dec_col = find_first_column(out.columns, ["dec", "dec_deg", "dedeg", "dej2000", "dec_icrs", "DEC", "dec_1"])
    if ra_col is not None and dec_col is not None:
        out["ra_icrs"]  = pd.to_numeric(out[ra_col],  errors="coerce")
        out["dec_icrs"] = pd.to_numeric(out[dec_col], errors="coerce")
        return out
    l_col = find_first_column(out.columns, ["l", "glon", "gal_l"])
    b_col = find_first_column(out.columns, ["b", "glat", "gal_b"])
    if l_col is None or b_col is None:
        raise KeyError("Cannot find RA/Dec or Galactic l/b columns.")
    l_vals = pd.to_numeric(out[l_col], errors="coerce").to_numpy(dtype=float)
    b_vals = pd.to_numeric(out[b_col], errors="coerce").to_numpy(dtype=float)
    coords = SkyCoord(l=l_vals * u.deg, b=b_vals * u.deg, frame="galactic")
    out["ra_icrs"]  = coords.icrs.ra.deg
    out["dec_icrs"] = coords.icrs.dec.deg
    return out

# ---------------------------------------------------------------------------
# XMatch helpers
# ---------------------------------------------------------------------------

def xmatch_local_to_remote(local_df, remote_table, radius_arcsec):
    """Upload row_id + ra_icrs + dec_icrs to CDS XMatch and return all matches."""
    local_table = Table.from_pandas(
        local_df[["row_id", "ra_icrs", "dec_icrs"]].copy()
    )
    result = XMatch.query(
        cat1=local_table,
        cat2=remote_table,
        max_distance=radius_arcsec * u.arcsec,
        colRA1="ra_icrs",
        colDec1="dec_icrs",
    )
    return result.to_pandas()


def query_reddening_map_vizier(local_df, radius_arcsec, progress_every=50):
    """
    Fall back to per-star VizieR region queries for the reddening map.
    CDS XMatch does not always expose this table for uploaded-table matching.
    """
    viz    = Vizier(columns=["*", "+_r"], row_limit=1)
    rows   = []
    n_total = len(local_df)
    for i, row in enumerate(
        local_df[["row_id", "ra_icrs", "dec_icrs"]].itertuples(index=False), start=1
    ):
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
            pass
        else:
            tbl = result[0]
            if len(tbl) > 0:
                match  = tbl[0]
                record = {name: match[name] for name in tbl.colnames}
                record["row_id"] = int(row_id)
                rows.append(record)
        if progress_every > 0 and (i % progress_every == 0 or i == n_total):
            print(f"  Reddening queries completed: {i}/{n_total}")
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame({"row_id": pd.Series(dtype="int64")})


def best_match_per_source(matches):
    """Keep the nearest match per source row_id."""
    if matches.empty:
        return matches.copy()
    dist_col = "angDist" if "angDist" in matches.columns else "_r"
    if dist_col not in matches.columns:
        raise KeyError("Cross-match result has no angDist or _r column.")
    work = matches.copy()
    work[dist_col] = pd.to_numeric(work[dist_col], errors="coerce")
    work = work.sort_values(["row_id", dist_col])
    counts = work.groupby("row_id").size().rename("match_multiplicity")
    best   = work.drop_duplicates(subset=["row_id"], keep="first").copy()
    return best.merge(counts, on="row_id", how="left")


def prefix_columns(df, prefix, protected=None):
    if protected is None:
        protected = set()
    return df.rename(columns={c: f"{prefix}{c}" for c in df.columns if c not in protected})

# ---------------------------------------------------------------------------
# Photometry, extinction, sightlines
# ---------------------------------------------------------------------------

def build_best_photometry_columns(master):
    out = master.copy()

    def get(col):
        return pd.to_numeric(out[col], errors="coerce") if col in out.columns \
               else pd.Series(np.nan, index=out.index)

    vvv_j, vvv_h, vvv_k   = get("vvv_Jmag"),   get("vvv_Hmag"),   get("vvv_Ksmag")
    tm_j,  tm_h,  tm_k    = get("tmass_Jmag"), get("tmass_Hmag"), get("tmass_Kmag")
    vvv_ej, vvv_eh, vvv_ek = get("vvv_e_Jmag"), get("vvv_e_Hmag"), get("vvv_e_Ksmag")
    tm_ej,  tm_eh,  tm_ek  = get("tmass_e_Jmag"), get("tmass_e_Hmag"), get("tmass_e_Kmag")

    out["j_mag_best"]  = choose_first_finite(vvv_j.to_numpy(float), tm_j.to_numpy(float))
    out["h_mag_best"]  = choose_first_finite(vvv_h.to_numpy(float), tm_h.to_numpy(float))
    out["ks_mag_best"] = choose_first_finite(vvv_k.to_numpy(float), tm_k.to_numpy(float))

    out["j_mag_source"]  = choose_source_label(vvv_j.to_numpy(float), tm_j.to_numpy(float), "VIRAC2", "2MASS")
    out["h_mag_source"]  = choose_source_label(vvv_h.to_numpy(float), tm_h.to_numpy(float), "VIRAC2", "2MASS")
    out["ks_mag_source"] = choose_source_label(vvv_k.to_numpy(float), tm_k.to_numpy(float), "VIRAC2", "2MASS")

    out["e_j_mag_best"]  = choose_first_finite(vvv_ej.to_numpy(float), tm_ej.to_numpy(float))
    out["e_h_mag_best"]  = choose_first_finite(vvv_eh.to_numpy(float), tm_eh.to_numpy(float))
    out["e_ks_mag_best"] = choose_first_finite(vvv_ek.to_numpy(float), tm_ek.to_numpy(float))

    out["j_minus_h_best"]  = out["j_mag_best"] - out["h_mag_best"]
    out["h_minus_ks_best"] = out["h_mag_best"] - out["ks_mag_best"]
    out["j_minus_ks_best"] = out["j_mag_best"] - out["ks_mag_best"]
    return out


def add_extinction_columns(master):
    out = master.copy()
    ejks_col = None
    for candidate in ["ext_E(J-Ks)", "ext_E(J-Ks)_", "ext_EJKs"]:
        if candidate in out.columns:
            ejks_col = candidate
            break
    out["ext_e_jks"] = pd.to_numeric(out[ejks_col], errors="coerce") \
                       if ejks_col else np.nan
    err_col = next((c for c in ["ext_e_", "ext_e_E(J-Ks)", "ext_e_EJKs"]
                    if c in out.columns), None)
    out["ext_e_jks_err"] = pd.to_numeric(out[err_col], errors="coerce") \
                           if err_col else np.nan

    out["a_ks"] = AKS_PER_EJKS * out["ext_e_jks"]
    out["a_h"]  = AH_PER_EJKS  * out["ext_e_jks"]
    out["a_j"]  = AJ_PER_EJKS  * out["ext_e_jks"]
    out["j0"]   = out["j_mag_best"]  - out["a_j"]
    out["h0"]   = out["h_mag_best"]  - out["a_h"]
    out["ks0"]  = out["ks_mag_best"] - out["a_ks"]
    out["j0_minus_h0"]  = out["j0"] - out["h0"]
    out["h0_minus_ks0"] = out["h0"] - out["ks0"]
    out["j0_minus_ks0"] = out["j0"] - out["ks0"]
    return out


def add_sightline_tags(master):
    out = master.copy()
    glon_col = find_first_column(out.columns, ["ext_GLON", "ext_glon"])
    glat_col = find_first_column(out.columns, ["ext_GLAT", "ext_glat"])
    tile_col = find_first_column(out.columns, ["ext_Tile", "ext_tile"])
    dist_col = find_first_column(out.columns, ["ext_angDist", "ext__r"])

    out["sightline_glon"] = pd.to_numeric(out[glon_col], errors="coerce") if glon_col else np.nan
    out["sightline_glat"] = pd.to_numeric(out[glat_col], errors="coerce") if glat_col else np.nan
    out["sightline_tile"] = out[tile_col].astype(str) if tile_col else ""
    out["ext_map_match_sep_arcsec"] = pd.to_numeric(out[dist_col], errors="coerce") if dist_col else np.nan

    sightline_ids = []
    coarse_ids    = []
    for tile, glon, glat in zip(
        out["sightline_tile"].to_numpy(dtype=object),
        out["sightline_glon"].to_numpy(dtype=float),
        out["sightline_glat"].to_numpy(dtype=float),
    ):
        tile_str = "" if tile is None else str(tile)
        if np.isfinite(glon) and np.isfinite(glat):
            sightline_ids.append(f"{tile_str}|l={glon:+08.4f}|b={glat:+08.4f}")
            coarse_ids.append(f"l={glon:+05.1f}|b={glat:+05.1f}")
        else:
            sightline_ids.append("")
            coarse_ids.append("")

    out["sightline_id"]        = np.array(sightline_ids, dtype=object)
    out["sightline_bin_0p1deg"] = np.array(coarse_ids,   dtype=object)
    return out

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(master, path):
    def count_finite(col):
        if col not in master.columns:
            return 0
        return int(np.isfinite(pd.to_numeric(master[col], errors="coerce")).sum())
    def count_nonempty(col):
        if col not in master.columns:
            return 0
        s = master[col]
        if pd.api.types.is_numeric_dtype(s):
            return int(np.isfinite(pd.to_numeric(s, errors="coerce")).sum())
        return int((s.fillna("").astype(str).str.strip() != "").sum())

    ejks = pd.to_numeric(master.get("ext_e_jks", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    lines = [
        "Step 3 – remote cross-match summary",
        "=" * 37,
        f"Total stars                                : {len(master)}",
        f"Matched VIRAC2 rows                        : {count_nonempty('vvv_srcid')}",
        f"Matched 2MASS rows                         : {count_nonempty('tmass_2MASS') + count_nonempty('tmass__2MASS')}",
        f"Matched reddening-map rows                 : {count_finite('ext_e_jks')}",
        f"Best J available                           : {count_finite('j_mag_best')}",
        f"Best H available                           : {count_finite('h_mag_best')}",
        f"Best Ks available                          : {count_finite('ks_mag_best')}",
        f"Dereddened J0                              : {count_finite('j0')}",
        f"Dereddened H0                              : {count_finite('h0')}",
        f"Dereddened Ks0                             : {count_finite('ks0')}",
        f"Unique sightline IDs                       : {count_nonempty('sightline_id')}",
    ]
    if np.isfinite(ejks).any():
        lines += [
            "",
            "Reddening statistics",
            "-" * 20,
            f"E(J-Ks) median       : {np.nanmedian(ejks):.4f}",
            f"E(J-Ks) 16th pct     : {np.nanpercentile(ejks, 16):.4f}",
            f"E(J-Ks) 84th pct     : {np.nanpercentile(ejks, 84):.4f}",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Reading {INPUT_FITS} ...")
    df = load_input(INPUT_FITS)
    df = ensure_icrs_coordinates(df)
    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    print(f"  {len(df):,} rows")

    # VIRAC2
    print(f"Cross-matching to VIRAC2 (radius={VIRAC2_RADIUS_ARCSEC}\") ...")
    try:
        virac_raw  = xmatch_local_to_remote(df, VIRAC2_TABLE, VIRAC2_RADIUS_ARCSEC)
        virac_best = best_match_per_source(virac_raw)
        virac_best = prefix_columns(virac_best, "vvv_", protected={"row_id"})
        df = df.merge(virac_best, on="row_id", how="left")
        print(f"  VIRAC2 matched: {virac_best['row_id'].nunique():,}")
    except Exception as exc:
        print(f"  WARNING: VIRAC2 failed ({exc}); continuing.")

    # 2MASS
    print(f"Cross-matching to 2MASS (radius={TMASS_RADIUS_ARCSEC}\") ...")
    try:
        tmass_raw  = xmatch_local_to_remote(df, TMASS_TABLE, TMASS_RADIUS_ARCSEC)
        tmass_best = best_match_per_source(tmass_raw)
        tmass_best = prefix_columns(tmass_best, "tmass_", protected={"row_id"})
        df = df.merge(tmass_best, on="row_id", how="left")
        print(f"  2MASS matched: {tmass_best['row_id'].nunique():,}")
    except Exception as exc:
        print(f"  WARNING: 2MASS failed ({exc}); continuing.")

    # VVV reddening map
    print(f"Cross-matching to VVV reddening map (radius={REDDENING_RADIUS_ARCSEC}\") ...")
    try:
        red_raw  = xmatch_local_to_remote(df, VVV_REDDENING_TABLE, REDDENING_RADIUS_ARCSEC)
        red_best = best_match_per_source(red_raw)
        print("  Reddening retrieval: CDS XMatch")
    except Exception as exc:
        print(f"  CDS XMatch failed ({exc})")
        print("  Falling back to VizieR region queries ...")
        red_best = query_reddening_map_vizier(df, REDDENING_RADIUS_ARCSEC)
        print("  Reddening retrieval: VizieR query_region")
    red_best = prefix_columns(red_best, "ext_", protected={"row_id"})
    df = df.merge(red_best, on="row_id", how="left")
    print(f"  Reddening matched: {red_best['row_id'].nunique():,}")

    df = build_best_photometry_columns(df)
    df = add_extinction_columns(df)
    df = add_sightline_tags(df)

    Table.from_pandas(df).write(OUTPUT_FITS, overwrite=True)
    write_summary(df, SUMMARY_TXT)
    print(f"\nWritten: {OUTPUT_FITS}  ({len(df):,} rows, {len(df.columns)} columns)")
    print(f"Summary: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()