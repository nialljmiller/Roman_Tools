#!/usr/bin/env python3
"""
Harvest best-available stellar parameters for the Roman calibration sample.

This script is meant to run after roman_deredden_sightlines.py and complements
roman_calibration_catalog.py. It keeps every row, optionally merges an external
Gaia XGBoost cross-match table, derives additional dereddened photometry where
possible, and builds best-available columns for:

    - effective temperature (teff_best)
    - surface gravity (logg_best)
    - metallicity proxy in dex (metallicity_best)
    - radius (radius_best)
    - luminosity in solar units (luminosity_best_lsun)

The selection logic is intentionally wide-net and non-destructive:
    - no rows are removed
    - source-specific values are preserved
    - new "best" columns include provenance labels
    - simple completeness flags are added

Example
-------
python roman_parameter_catalog.py astra_overguide_roman_dereddened.fits \
    --xgboost gaia_xgboost_matches.csv \
    --out-prefix astra_overguide_roman_parameters
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from astropy.table import Table

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T_SUN = 5772.0


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def _lower_name_map(columns: Iterable[str]) -> dict[str, str]:
    return {str(name).lower(): str(name) for name in columns}



def find_first_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    mapping = _lower_name_map(columns)
    for candidate in candidates:
        key = str(candidate).lower()
        if key in mapping:
            return mapping[key]
    return None



def first_present_value(df: pd.DataFrame, candidates: Iterable[str], default=np.nan) -> pd.Series:
    col = find_first_column(df.columns, candidates)
    if col is None:
        return pd.Series(default, index=df.index)
    return df[col]



def as_numeric(series: pd.Series | np.ndarray | list | tuple) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")



def clean_text_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()



def table_to_scalar_pandas(tbl: Table) -> tuple[pd.DataFrame, list[str]]:
    scalar_names: list[str] = []
    dropped: list[str] = []
    for name in tbl.colnames:
        shape = getattr(tbl[name], "shape", ())
        if len(shape) <= 1:
            scalar_names.append(name)
        else:
            dropped.append(name)
    return tbl[scalar_names].to_pandas(), dropped



def read_table_like(path: pathlib.Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".fits", ".fit", ".fits.gz"}:
        tbl = Table.read(path)
        df, _ = table_to_scalar_pandas(tbl)
        return df
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".parquet"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")



def make_join_key(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype("Int64")
    out = vals.astype(str)
    out = out.replace("<NA>", "")
    return out



def choose_text(primary: pd.Series, fallback: pd.Series, empty: str = "") -> pd.Series:
    p = clean_text_series(primary)
    f = clean_text_series(fallback)
    out = p.copy()
    use_fallback = (out == "") & (f != "")
    out.loc[use_fallback] = f.loc[use_fallback]
    out.loc[out == ""] = empty
    return out



def count_finite(series: pd.Series) -> int:
    return int(np.isfinite(as_numeric(series)).sum())



def count_true(series: pd.Series | np.ndarray) -> int:
    return int(np.asarray(series, dtype=bool).sum())



def add_if_absent(df: pd.DataFrame, name: str, values) -> pd.DataFrame:
    if name not in df.columns:
        df[name] = values
    return df


# -----------------------------------------------------------------------------
# Optional merge: Gaia XGBoost table
# -----------------------------------------------------------------------------

def merge_xgboost_if_requested(master: pd.DataFrame, xgboost_path: Optional[pathlib.Path]) -> pd.DataFrame:
    out = master.copy()

    already_has = any(col in out.columns for col in ["teff_xgboost", "logg_xgboost", "mh_xgboost"])
    if xgboost_path is None:
        return out

    xg_df = read_table_like(xgboost_path)
    if len(xg_df) == 0:
        return out

    base_id_col = find_first_column(out.columns, ["gaia_dr3_source_id", "source_id"])
    xgb_id_col = find_first_column(xg_df.columns, ["source_id", "gaia_dr3_source_id", "SOURCE_ID"])
    if base_id_col is None:
        raise KeyError("Could not find Gaia DR3 source ID in the base catalog for XGBoost merge.")
    if xgb_id_col is None:
        raise KeyError("Could not find source_id in the XGBoost table.")

    if already_has:
        print("XGBoost-like columns already exist in the input table; merging anyway, with incoming collisions renamed using xgb_.")

    base = out.copy()
    xgb = xg_df.copy()
    base["_gaia_join_key"] = make_join_key(base[base_id_col])
    xgb["_gaia_join_key"] = make_join_key(xgb[xgb_id_col])
    xgb = xgb.loc[xgb["_gaia_join_key"] != ""].copy()
    xgb = xgb.drop_duplicates(subset=["_gaia_join_key"], keep="first")

    rename_map: dict[str, str] = {}
    for col in xgb.columns:
        if col == "_gaia_join_key":
            continue
        if col in base.columns:
            rename_map[col] = f"xgb_{col}"
    if rename_map:
        xgb = xgb.rename(columns=rename_map)

    merged = base.merge(xgb, on="_gaia_join_key", how="left")
    return merged.drop(columns=["_gaia_join_key"])


# -----------------------------------------------------------------------------
# Dereddening helpers
# -----------------------------------------------------------------------------

def ensure_jhk_best(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "j_mag_best" not in out.columns:
        out["j_mag_best"] = as_numeric(first_present_value(out, ["j_mag", "Jmag", "tmass_Jmag", "vvv_Jmag"]))
    if "h_mag_best" not in out.columns:
        out["h_mag_best"] = as_numeric(first_present_value(out, ["h_mag", "Hmag", "tmass_Hmag", "vvv_Hmag"]))
    if "ks_mag_best" not in out.columns:
        out["ks_mag_best"] = as_numeric(first_present_value(out, ["k_mag", "ks_mag", "Ksmag", "tmass_Kmag", "vvv_Ksmag"]))

    return out



def add_dereddened_photometry(
    df: pd.DataFrame,
    aks_per_ejks: float,
    ah_per_ejks: float,
    aj_per_ejks: float,
) -> pd.DataFrame:
    out = ensure_jhk_best(df)

    # NIR dereddening from E(J-Ks) if needed.
    ejks = as_numeric(first_present_value(out, ["ext_e_jks", "E_JKs", "ejks"]))
    out = add_if_absent(out, "ext_e_jks", ejks)
    out = add_if_absent(out, "a_ks", aks_per_ejks * out["ext_e_jks"])
    out = add_if_absent(out, "a_h", ah_per_ejks * out["ext_e_jks"])
    out = add_if_absent(out, "a_j", aj_per_ejks * out["ext_e_jks"])
    out = add_if_absent(out, "j0", as_numeric(out["j_mag_best"]) - as_numeric(out["a_j"]))
    out = add_if_absent(out, "h0", as_numeric(out["h_mag_best"]) - as_numeric(out["a_h"]))
    out = add_if_absent(out, "ks0", as_numeric(out["ks_mag_best"]) - as_numeric(out["a_ks"]))
    out = add_if_absent(out, "j0_minus_h0", as_numeric(out["j0"]) - as_numeric(out["h0"]))
    out = add_if_absent(out, "h0_minus_ks0", as_numeric(out["h0"]) - as_numeric(out["ks0"]))
    out = add_if_absent(out, "j0_minus_ks0", as_numeric(out["j0"]) - as_numeric(out["ks0"]))

    # Gaia broad-band dereddening using Gaia-provided extinction estimates.
    g_mag = as_numeric(first_present_value(out, ["g_mag"]))
    bp_mag = as_numeric(first_present_value(out, ["bp_mag"]))
    rp_mag = as_numeric(first_present_value(out, ["rp_mag"]))
    ag = as_numeric(first_present_value(out, ["AG_1", "AG-HS", "AG"]))
    abp = as_numeric(first_present_value(out, ["ABP"]))
    arp = as_numeric(first_present_value(out, ["ARP"]))

    out["gaia_ag_best"] = ag
    out["gaia_abp_best"] = abp
    out["gaia_arp_best"] = arp
    out["g0_gaia"] = g_mag - ag
    out["bp0_gaia"] = bp_mag - abp
    out["rp0_gaia"] = rp_mag - arp
    out["bp_rp0_gaia"] = out["bp0_gaia"] - out["rp0_gaia"]

    # BDBS dereddening using survey extinction columns already present in the table.
    umag = as_numeric(first_present_value(out, ["umag"]))
    gmag2 = as_numeric(first_present_value(out, ["gmag_2", "gmag2"]))
    imag = as_numeric(first_present_value(out, ["imag"]))
    au = as_numeric(first_present_value(out, ["Au"]))
    ag2 = as_numeric(first_present_value(out, ["Ag_2", "Ag2"]))
    ai = as_numeric(first_present_value(out, ["Ai"]))

    out["u0_bdbs"] = umag - au
    out["g0_bdbs"] = gmag2 - ag2
    out["i0_bdbs"] = imag - ai
    out["u_g0_bdbs"] = out["u0_bdbs"] - out["g0_bdbs"]
    out["g_i0_bdbs"] = out["g0_bdbs"] - out["i0_bdbs"]

    out["has_nir_dereddening"] = np.isfinite(as_numeric(out["j0"]))
    out["has_gaia_dereddening"] = np.isfinite(as_numeric(out["g0_gaia"]))
    out["has_bdbs_dereddening"] = np.isfinite(as_numeric(out["g0_bdbs"]))
    return out


# -----------------------------------------------------------------------------
# Best-available parameter harvesting
# -----------------------------------------------------------------------------

def _coerce_err(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return as_numeric(df[col])



def pick_best_from_candidates(
    df: pd.DataFrame,
    candidates: Sequence[dict[str, Optional[str]]],
    value_name: str,
    err_name: Optional[str] = None,
    source_name: Optional[str] = None,
    kind_name: Optional[str] = None,
) -> pd.DataFrame:
    out = df.copy()
    values = np.full(len(out), np.nan, dtype=float)
    errors = np.full(len(out), np.nan, dtype=float)
    sources = np.full(len(out), "", dtype=object)
    kinds = np.full(len(out), "", dtype=object)

    filled = np.zeros(len(out), dtype=bool)

    for cand in candidates:
        col = cand.get("col")
        if col is None or col not in out.columns:
            continue
        data = as_numeric(out[col]).to_numpy(dtype=float)
        good = np.isfinite(data) & (~filled)
        if not np.any(good):
            continue

        values[good] = data[good]

        err_col = cand.get("err")
        if err_col is not None and err_col in out.columns:
            err_data = as_numeric(out[err_col]).to_numpy(dtype=float)
            errors[good] = err_data[good]

        label = str(cand.get("label", col))
        sources[good] = label

        kind = str(cand.get("kind", ""))
        if kind:
            kinds[good] = kind

        filled[good] = True

    out[value_name] = values
    if err_name is not None:
        out[err_name] = errors
    if source_name is not None:
        out[source_name] = pd.Series(sources, index=out.index, dtype=object)
    if kind_name is not None:
        out[kind_name] = pd.Series(kinds, index=out.index, dtype=object)
    return out



def compute_luminosity_from_radius_teff(radius_rsun: pd.Series, teff_k: pd.Series) -> pd.Series:
    r = as_numeric(radius_rsun)
    t = as_numeric(teff_k)
    lum = (r ** 2.0) * (t / T_SUN) ** 4.0
    lum = lum.where((r > 0) & (t > 0))
    return lum



def add_best_parameters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    teff_candidates = [
        {"col": "teff", "err": "e_teff", "label": "ASTRA"},
        {"col": "teff_xgboost", "err": None, "label": "GAIA_XGBOOST"},
        {"col": "zgr_teff", "err": "zgr_e_teff", "label": "ZGR"},
        {"col": "Teff_x", "err": None, "label": "GAIA_AP"},
        {"col": "Teff-S", "err": None, "label": "GAIA_GSPSPEC"},
        {"col": "Teff-HS", "err": "e_Teff-HS", "label": "GAIA_HOTSTAR"},
        {"col": "Teff-UCD", "err": "e_Teff-UCD", "label": "GAIA_UCD"},
        {"col": "irfm_teff", "err": None, "label": "IRFM"},
        {"col": "doppler_teff", "err": "doppler_e_teff", "label": "DOPPLER"},
        {"col": "raw_teff", "err": "raw_e_teff", "label": "ASTRA_RAW"},
    ]
    out = pick_best_from_candidates(
        out,
        teff_candidates,
        value_name="teff_best",
        err_name="e_teff_best",
        source_name="teff_source_best",
    )

    logg_candidates = [
        {"col": "logg", "err": "e_logg", "label": "ASTRA"},
        {"col": "logg_xgboost", "err": None, "label": "GAIA_XGBOOST"},
        {"col": "zgr_logg", "err": "zgr_e_logg", "label": "ZGR"},
        {"col": "logg_x", "err": None, "label": "GAIA_AP"},
        {"col": "logg-S", "err": None, "label": "GAIA_GSPSPEC"},
        {"col": "logg-HS", "err": "e_logg-HS", "label": "GAIA_HOTSTAR"},
        {"col": "doppler_logg", "err": "doppler_e_logg", "label": "DOPPLER"},
        {"col": "raw_logg", "err": "raw_e_logg", "label": "ASTRA_RAW"},
    ]
    out = pick_best_from_candidates(
        out,
        logg_candidates,
        value_name="logg_best",
        err_name="e_logg_best",
        source_name="logg_source_best",
    )

    metallicity_candidates = [
        {"col": "mh_xgboost", "err": None, "label": "GAIA_XGBOOST", "kind": "[M/H]"},
        {"col": "m_h_atm", "err": "e_m_h_atm", "label": "ASTRA", "kind": "[M/H]"},
        {"col": "fe_h", "err": "e_fe_h", "label": "ASTRA", "kind": "[Fe/H]"},
        {"col": "[M/H]-S", "err": None, "label": "GAIA_GSPSPEC", "kind": "[M/H]"},
        {"col": "[Fe/H]-S", "err": None, "label": "GAIA_GSPSPEC", "kind": "[Fe/H]"},
        {"col": "[Fe/H]_1", "err": None, "label": "GAIA_AP", "kind": "[Fe/H]"},
        {"col": "zgr_fe_h", "err": "zgr_e_fe_h", "label": "ZGR", "kind": "[Fe/H]"},
        {"col": "[Fe/H]_2", "err": "e_[Fe/H]", "label": "BDBS", "kind": "[Fe/H]"},
        {"col": "raw_m_h_atm", "err": "raw_e_m_h_atm", "label": "ASTRA_RAW", "kind": "[M/H]"},
        {"col": "raw_fe_h", "err": "raw_e_fe_h", "label": "ASTRA_RAW", "kind": "[Fe/H]"},
    ]
    out = pick_best_from_candidates(
        out,
        metallicity_candidates,
        value_name="metallicity_best",
        err_name="e_metallicity_best",
        source_name="metallicity_source_best",
        kind_name="metallicity_kind_best",
    )

    radius_candidates = [
        {"col": "radius", "err": None, "label": "ASTRA"},
        {"col": "Rad", "err": None, "label": "GAIA_AP"},
        {"col": "Rad-Flame", "err": None, "label": "GAIA_FLAME"},
    ]
    out = pick_best_from_candidates(
        out,
        radius_candidates,
        value_name="radius_best_rsun",
        err_name="e_radius_best_rsun",
        source_name="radius_source_best",
    )

    # Source-consistent luminosity estimates.
    out["luminosity_flame_lsun"] = as_numeric(first_present_value(out, ["Lum-Flame"]))
    out.loc[as_numeric(out["luminosity_flame_lsun"]) <= 0, "luminosity_flame_lsun"] = np.nan

    out["luminosity_from_astra_lsun"] = compute_luminosity_from_radius_teff(
        first_present_value(out, ["radius"]),
        first_present_value(out, ["teff"]),
    )
    out["luminosity_from_gaia_ap_lsun"] = compute_luminosity_from_radius_teff(
        first_present_value(out, ["Rad"]),
        first_present_value(out, ["Teff_x"]),
    )
    out["luminosity_from_xgboost_lsun"] = compute_luminosity_from_radius_teff(
        first_present_value(out, ["Rad"]),
        first_present_value(out, ["teff_xgboost"]),
    )
    out["luminosity_from_best_rt_lsun"] = compute_luminosity_from_radius_teff(
        out["radius_best_rsun"],
        out["teff_best"],
    )

    lum_candidates = [
        {"col": "luminosity_flame_lsun", "err": None, "label": "GAIA_FLAME"},
        {"col": "luminosity_from_astra_lsun", "err": None, "label": "ASTRA_RADIUS+ASTRA_TEFF"},
        {"col": "luminosity_from_gaia_ap_lsun", "err": None, "label": "GAIA_RAD+GAIA_TEFF"},
        {"col": "luminosity_from_xgboost_lsun", "err": None, "label": "GAIA_RAD+XGBOOST_TEFF"},
        {"col": "luminosity_from_best_rt_lsun", "err": None, "label": "BEST_RADIUS+BEST_TEFF"},
    ]
    out = pick_best_from_candidates(
        out,
        lum_candidates,
        value_name="luminosity_best_lsun",
        err_name="e_luminosity_best_lsun",
        source_name="luminosity_source_best",
    )

    logl = np.full(len(out), np.nan, dtype=float)
    good = np.isfinite(as_numeric(out["luminosity_best_lsun"])) & (as_numeric(out["luminosity_best_lsun"]) > 0)
    logl[good.to_numpy()] = np.log10(as_numeric(out.loc[good, "luminosity_best_lsun"]))
    out["log10_luminosity_best_lsun"] = logl

    # Convenience aliases for the user's language.
    out["z_best"] = out["metallicity_best"]
    out["e_z_best"] = out["e_metallicity_best"]
    out["z_source_best"] = out["metallicity_source_best"]
    out["z_kind_best"] = out["metallicity_kind_best"]

    out["has_teff_best"] = np.isfinite(as_numeric(out["teff_best"]))
    out["has_logg_best"] = np.isfinite(as_numeric(out["logg_best"]))
    out["has_metallicity_best"] = np.isfinite(as_numeric(out["metallicity_best"]))
    out["has_radius_best"] = np.isfinite(as_numeric(out["radius_best_rsun"]))
    out["has_luminosity_best"] = np.isfinite(as_numeric(out["luminosity_best_lsun"]))
    out["has_core_params"] = (
        out["has_teff_best"]
        & out["has_logg_best"]
        & out["has_metallicity_best"]
        & out["has_luminosity_best"]
    )
    return out


# -----------------------------------------------------------------------------
# Output ordering and stats
# -----------------------------------------------------------------------------

def add_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["source_id_best"] = choose_text(
        first_present_value(out, ["sdss_id"], default=""),
        first_present_value(out, ["gaia_dr3_source_id"], default=""),
        empty="",
    )
    out["gaia_id_best"] = choose_text(
        first_present_value(out, ["gaia_dr3_source_id"], default=""),
        first_present_value(out, ["gaia_dr2_source_id"], default=""),
        empty="",
    )
    return out



def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "source_id_best",
        "sdss_id",
        "sdss4_apogee_id",
        "gaia_id_best",
        "gaia_dr3_source_id",
        "gaia_dr2_source_id",
        "ra_icrs",
        "dec_icrs",
        "l",
        "b",
        "teff_best",
        "e_teff_best",
        "teff_source_best",
        "logg_best",
        "e_logg_best",
        "logg_source_best",
        "metallicity_best",
        "e_metallicity_best",
        "metallicity_source_best",
        "metallicity_kind_best",
        "z_best",
        "e_z_best",
        "z_source_best",
        "z_kind_best",
        "radius_best_rsun",
        "e_radius_best_rsun",
        "radius_source_best",
        "luminosity_best_lsun",
        "log10_luminosity_best_lsun",
        "luminosity_source_best",
        "j_mag_best",
        "h_mag_best",
        "ks_mag_best",
        "ext_e_jks",
        "a_j",
        "a_h",
        "a_ks",
        "j0",
        "h0",
        "ks0",
        "j0_minus_ks0",
        "g0_gaia",
        "bp0_gaia",
        "rp0_gaia",
        "bp_rp0_gaia",
        "u0_bdbs",
        "g0_bdbs",
        "i0_bdbs",
        "g_i0_bdbs",
        "has_teff_best",
        "has_logg_best",
        "has_metallicity_best",
        "has_radius_best",
        "has_luminosity_best",
        "has_core_params",
    ]
    front = [c for c in preferred if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]



def build_source_stats(df: pd.DataFrame, value_col: str, source_col: str) -> pd.DataFrame:
    good = np.isfinite(as_numeric(df[value_col]))
    if source_col not in df.columns:
        return pd.DataFrame(columns=[source_col, "count", "fraction_of_finite"])
    counts = clean_text_series(df.loc[good, source_col]).replace("", "NONE").value_counts(dropna=False)
    out = counts.rename_axis(source_col).reset_index(name="count")
    denom = max(int(good.sum()), 1)
    out["fraction_of_finite"] = out["count"] / denom
    return out



def write_summary(df: pd.DataFrame, path: pathlib.Path) -> None:
    lines: list[str] = []
    lines.append("Roman parameter-harvest summary")
    lines.append("=" * 30)
    lines.append(f"Total rows                               : {len(df)}")
    lines.append(f"Rows with best Teff                      : {count_true(df['has_teff_best'])}")
    lines.append(f"Rows with best logg                      : {count_true(df['has_logg_best'])}")
    lines.append(f"Rows with best metallicity               : {count_true(df['has_metallicity_best'])}")
    lines.append(f"Rows with best radius                    : {count_true(df['has_radius_best'])}")
    lines.append(f"Rows with best luminosity                : {count_true(df['has_luminosity_best'])}")
    lines.append(f"Rows with Teff+logg+Z+L                  : {count_true(df['has_core_params'])}")
    lines.append("")
    for val_col, src_col, title in [
        ("teff_best", "teff_source_best", "Teff source mix"),
        ("logg_best", "logg_source_best", "logg source mix"),
        ("metallicity_best", "metallicity_source_best", "Metallicity source mix"),
        ("luminosity_best_lsun", "luminosity_source_best", "Luminosity source mix"),
    ]:
        stats = build_source_stats(df, val_col, src_col)
        lines.append(title)
        lines.append("-" * len(title))
        if len(stats) == 0:
            lines.append("None")
        else:
            for _, row in stats.iterrows():
                lines.append(f"{row[src_col]:<24} : {int(row['count']):>4d} ({float(row['fraction_of_finite']):.3f})")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")



def write_table(df: pd.DataFrame, fits_path: pathlib.Path, csv_path: pathlib.Path) -> None:
    table = Table.from_pandas(df)
    table.write(fits_path, overwrite=True)
    df.to_csv(csv_path, index=False)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_teff_metallicity_logg(df: pd.DataFrame, output_path: pathlib.Path) -> None:
    teff = as_numeric(df["teff_best"])
    metal = as_numeric(df["metallicity_best"])
    logg = as_numeric(df["logg_best"])
    good = np.isfinite(teff) & np.isfinite(metal) & np.isfinite(logg)
    if int(good.sum()) == 0:
        return

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    sc = ax.scatter(teff[good], metal[good], c=logg[good], s=18, alpha=0.85)
    ax.set_xlabel(r"$T_{\mathrm{eff}}$ [K]")
    ax.set_ylabel("Best metallicity [dex]")
    ax.set_title("Roman calibration sample: Teff vs metallicity, colored by logg")
    ax.invert_xaxis()
    ax.grid(alpha=0.25)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$\log g$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)



def plot_hrd(df: pd.DataFrame, output_path: pathlib.Path) -> None:
    teff = as_numeric(df["teff_best"])
    logl = as_numeric(df["log10_luminosity_best_lsun"])
    metal = as_numeric(df["metallicity_best"])
    good = np.isfinite(teff) & np.isfinite(logl)
    if int(good.sum()) == 0:
        return

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    if np.isfinite(metal[good]).any():
        sc = ax.scatter(teff[good], logl[good], c=metal[good], s=18, alpha=0.85)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("Best metallicity [dex]")
    else:
        ax.scatter(teff[good], logl[good], s=18, alpha=0.85)
    ax.set_xlabel(r"$T_{\mathrm{eff}}$ [K]")
    ax.set_ylabel(r"$\log_{10}(L/L_\odot)$")
    ax.set_title("Roman calibration sample: harvested HR diagram")
    ax.invert_xaxis()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Harvest best-available stellar parameters from the Roman dereddened catalog."
    )
    parser.add_argument(
        "input_table",
        nargs="?",
        default="astra_overguide_roman_dereddened.fits",
        help="Input FITS/CSV table, typically the output from roman_deredden_sightlines.py",
    )
    parser.add_argument(
        "--xgboost",
        default=None,
        help="Optional FITS/CSV/Parquet Gaia XGBoost cross-match table to merge by Gaia DR3 source_id",
    )
    parser.add_argument(
        "--out-prefix",
        default="astra_overguide_roman_parameters",
        help="Prefix for output products",
    )
    parser.add_argument(
        "--aks-per-ejks",
        type=float,
        default=0.528,
        help="A_Ks / E(J-Ks) coefficient used only if a_ks is absent",
    )
    parser.add_argument(
        "--ah-per-ejks",
        type=float,
        default=0.857,
        help="A_H / E(J-Ks) coefficient used only if a_h is absent",
    )
    parser.add_argument(
        "--aj-per-ejks",
        type=float,
        default=1.528,
        help="A_J / E(J-Ks) coefficient used only if a_j is absent",
    )
    return parser



def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    input_path = pathlib.Path(args.input_table).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input table not found: {input_path}")

    xgb_path = None
    if args.xgboost is not None:
        xgb_path = pathlib.Path(args.xgboost).expanduser().resolve()
        if not xgb_path.exists():
            raise FileNotFoundError(f"XGBoost table not found: {xgb_path}")

    out_prefix = pathlib.Path(args.out_prefix)
    if out_prefix.is_absolute():
        stem = out_prefix.name
        out_dir = out_prefix.parent
    else:
        stem = out_prefix.name
        out_dir = pathlib.Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading input table: {input_path}")
    base = read_table_like(input_path)
    print(f"Input rows: {len(base)}")

    if xgb_path is not None:
        print(f"Merging optional XGBoost table: {xgb_path}")
        base = merge_xgboost_if_requested(base, xgb_path)

    work = add_identifier_columns(base)
    work = add_dereddened_photometry(
        work,
        aks_per_ejks=args.aks_per_ejks,
        ah_per_ejks=args.ah_per_ejks,
        aj_per_ejks=args.aj_per_ejks,
    )
    work = add_best_parameters(work)
    work = reorder_columns(work)

    full_fits = out_dir / f"{stem}.fits"
    full_csv = out_dir / f"{stem}.csv"
    core_fits = out_dir / f"{stem}_core_params.fits"
    core_csv = out_dir / f"{stem}_core_params.csv"
    summary_txt = out_dir / f"{stem}_summary.txt"
    teff_metal_png = out_dir / f"{stem}_teff_metallicity_logg.png"
    hrd_png = out_dir / f"{stem}_hrd.png"
    teff_src_csv = out_dir / f"{stem}_teff_source_stats.csv"
    logg_src_csv = out_dir / f"{stem}_logg_source_stats.csv"
    metal_src_csv = out_dir / f"{stem}_metallicity_source_stats.csv"
    lum_src_csv = out_dir / f"{stem}_luminosity_source_stats.csv"

    core = work.loc[work["has_core_params"]].copy().reset_index(drop=True)

    print(f"Writing full parameter catalog FITS: {full_fits}")
    write_table(work, full_fits, full_csv)

    print(f"Writing core-parameter subset FITS: {core_fits}")
    write_table(core, core_fits, core_csv)

    print(f"Writing summary text: {summary_txt}")
    write_summary(work, summary_txt)

    build_source_stats(work, "teff_best", "teff_source_best").to_csv(teff_src_csv, index=False)
    build_source_stats(work, "logg_best", "logg_source_best").to_csv(logg_src_csv, index=False)
    build_source_stats(work, "metallicity_best", "metallicity_source_best").to_csv(metal_src_csv, index=False)
    build_source_stats(work, "luminosity_best_lsun", "luminosity_source_best").to_csv(lum_src_csv, index=False)

    print(f"Writing money plot: {teff_metal_png}")
    plot_teff_metallicity_logg(work, teff_metal_png)

    print(f"Writing HR diagram: {hrd_png}")
    plot_hrd(work, hrd_png)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
