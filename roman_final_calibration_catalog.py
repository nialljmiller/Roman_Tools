#!/usr/bin/env python3
"""
Build the final Roman calibration catalog and diagnostics in one pass.

This combines the key functionality of:
  - roman_calibration_catalog.py
  - roman_calibration_diagnostics.py
  - roman_parameter_catalog.py

What it does
------------
1. Reads the input FITS/CSV table.
2. Optionally merges a Gaia XGBoost cross-match table by Gaia DR3 source_id.
3. Preserves all rows and all scalar columns.
4. Rebuilds/ensures dereddened photometry.
5. Harvests best-available Teff, logg, metallicity/Z proxy, radius, and luminosity.
6. Builds a final master catalog plus ready subsets.
7. Writes statistics tables and a broad diagnostics plot suite.

Notes
-----
- `z_best` here is a metallicity proxy in dex, not an absolute mass fraction Z.
  It aliases `metallicity_best` so the user can work naturally with a broad-net
  "best available Z-like quantity" while preserving provenance via `z_kind_best`.
- No rows are dropped unless you explicitly subset the output files later.
"""

from __future__ import annotations

import argparse
import pathlib
import warnings
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from astropy.table import Table
from astropy.io.fits.verify import VerifyWarning
from astropy.units import UnitsWarning

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=VerifyWarning)
warnings.filterwarnings("ignore", category=UnitsWarning)

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

def VVVlwrap(l_deg):
    return ((l_deg + 180.0) % 360.0) - 180.0


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()



def choose_text(primary: pd.Series, fallback: pd.Series, empty: str = "") -> pd.Series:
    p = clean_text(primary)
    f = clean_text(fallback)
    out = p.copy()
    use_fallback = (out == "") & (f != "")
    out.loc[use_fallback] = f.loc[use_fallback]
    out.loc[out == ""] = empty
    return out



def count_true(mask: pd.Series | np.ndarray) -> int:
    return int(np.asarray(mask, dtype=bool).sum())



def finite_mask(*arrays: pd.Series | np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        vals = np.asarray(arr, dtype=float)
        mask &= np.isfinite(vals)
    return mask



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



def read_table_like(path: pathlib.Path) -> tuple[pd.DataFrame, list[str]]:
    suffix = path.suffix.lower()
    if suffix in {".fits", ".fit", ".fits.gz"}:
        tbl = Table.read(path)
        return table_to_scalar_pandas(tbl)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path), []
    if suffix in {".parquet"}:
        return pd.read_parquet(path), []
    raise ValueError(f"Unsupported table format: {path}")



def write_table(df: pd.DataFrame, fits_path: pathlib.Path, csv_path: pathlib.Path) -> None:
    table = Table.from_pandas(df)
    table.write(fits_path, overwrite=True)
    df.to_csv(csv_path, index=False)



def get_series(df: pd.DataFrame, candidates: Iterable[str], default=np.nan) -> pd.Series:
    return first_present_value(df, candidates, default=default)



def ensure_bool(df: pd.DataFrame, candidates: Iterable[str], fallback: bool = False) -> pd.Series:
    col = find_first_column(df.columns, candidates)
    if col is None:
        return pd.Series(fallback, index=df.index, dtype=bool)
    series = df[col]
    if series.dtype == bool:
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(bool)
    s = clean_text(series).str.lower()
    return s.isin(["true", "1", "yes", "y"])



def savefig(fig: plt.Figure, path: pathlib.Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Optional merge: Gaia XGBoost
# -----------------------------------------------------------------------------

def make_join_key(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype("Int64")
    out = vals.astype(str)
    return out.replace("<NA>", "")



def merge_xgboost_if_requested(master: pd.DataFrame, xgboost_path: Optional[pathlib.Path]) -> pd.DataFrame:
    out = master.copy()
    if xgboost_path is None:
        return out

    xgb_df, _ = read_table_like(xgboost_path)
    if len(xgb_df) == 0:
        return out

    base_id_col = find_first_column(out.columns, ["gaia_dr3_source_id", "source_id"])
    xgb_id_col = find_first_column(xgb_df.columns, ["source_id", "gaia_dr3_source_id", "SOURCE_ID"])
    if base_id_col is None:
        raise KeyError("Could not find Gaia DR3 source ID in the base catalog for XGBoost merge.")
    if xgb_id_col is None:
        raise KeyError("Could not find source_id in the XGBoost table.")

    base = out.copy()
    xgb = xgb_df.copy()
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
# Dereddening and status flags
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



def ensure_jhk_best(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "j_mag_best" not in out.columns:
        out["j_mag_best"] = as_numeric(first_present_value(out, ["j_mag", "Jmag", "tmass_Jmag", "vvv_Jmag"]))
    if "h_mag_best" not in out.columns:
        out["h_mag_best"] = as_numeric(first_present_value(out, ["h_mag", "Hmag", "tmass_Hmag", "vvv_Hmag"]))
    if "ks_mag_best" not in out.columns:
        out["ks_mag_best"] = as_numeric(first_present_value(out, ["k_mag", "ks_mag", "Ksmag", "tmass_Kmag", "vvv_Ksmag"]))

    if "j_mag_source" not in out.columns:
        out["j_mag_source"] = pd.Series(np.where(np.isfinite(as_numeric(first_present_value(out, ["j_mag"]))), "2MASS", ""), index=out.index)
    if "h_mag_source" not in out.columns:
        out["h_mag_source"] = pd.Series(np.where(np.isfinite(as_numeric(first_present_value(out, ["h_mag"]))), "2MASS", ""), index=out.index)
    if "ks_mag_source" not in out.columns:
        out["ks_mag_source"] = pd.Series(np.where(np.isfinite(as_numeric(first_present_value(out, ["k_mag", "ks_mag"]))), "2MASS", ""), index=out.index)

    return out



def add_dereddened_photometry(df: pd.DataFrame, aks_per_ejks: float, ah_per_ejks: float, aj_per_ejks: float) -> pd.DataFrame:
    out = ensure_jhk_best(df)

    ejks = as_numeric(first_present_value(out, ["ext_e_jks", "E_JKs", "ejks"]))
    out["ext_e_jks"] = ejks
    if "ext_e_jks_err" not in out.columns:
        out["ext_e_jks_err"] = as_numeric(first_present_value(out, ["ext_e_jks_err", "ejks_err"]))

    if "a_ks" not in out.columns:
        out["a_ks"] = aks_per_ejks * out["ext_e_jks"]
    if "a_h" not in out.columns:
        out["a_h"] = ah_per_ejks * out["ext_e_jks"]
    if "a_j" not in out.columns:
        out["a_j"] = aj_per_ejks * out["ext_e_jks"]

    if "j0" not in out.columns:
        out["j0"] = as_numeric(out["j_mag_best"]) - as_numeric(out["a_j"])
    if "h0" not in out.columns:
        out["h0"] = as_numeric(out["h_mag_best"]) - as_numeric(out["a_h"])
    if "ks0" not in out.columns:
        out["ks0"] = as_numeric(out["ks_mag_best"]) - as_numeric(out["a_ks"])

    out["j_minus_h_best"] = as_numeric(out["j_mag_best"]) - as_numeric(out["h_mag_best"])
    out["h_minus_ks_best"] = as_numeric(out["h_mag_best"]) - as_numeric(out["ks_mag_best"])
    out["j_minus_ks_best"] = as_numeric(out["j_mag_best"]) - as_numeric(out["ks_mag_best"])
    out["j0_minus_h0"] = as_numeric(out["j0"]) - as_numeric(out["h0"])
    out["h0_minus_ks0"] = as_numeric(out["h0"]) - as_numeric(out["ks0"])
    out["j0_minus_ks0"] = as_numeric(out["j0"]) - as_numeric(out["ks0"])

    # Gaia dereddening
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

    # BDBS dereddening
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

    out["has_nir_dereddening"] = np.isfinite(as_numeric(out["j0"])) & np.isfinite(as_numeric(out["h0"])) & np.isfinite(as_numeric(out["ks0"]))
    out["has_gaia_dereddening"] = np.isfinite(as_numeric(out["g0_gaia"]))
    out["has_bdbs_dereddening"] = np.isfinite(as_numeric(out["g0_bdbs"]))
    return out



def add_status_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    virac_src = first_present_value(out, ["vvv_srcid", "vvv_SrcID", "vvv_sourceid", "vvv_source_id"], default="")
    tmass_src = first_present_value(out, ["tmass_2MASS", "tmass__2MASS", "tmass_2mass", "tmass_id"], default="")

    out["has_virac2"] = clean_text(virac_src) != ""
    out["has_2mass"] = clean_text(tmass_src) != ""

    out["has_best_j"] = np.isfinite(as_numeric(first_present_value(out, ["j_mag_best"])))
    out["has_best_h"] = np.isfinite(as_numeric(first_present_value(out, ["h_mag_best"])))
    out["has_best_ks"] = np.isfinite(as_numeric(first_present_value(out, ["ks_mag_best"])))
    out["has_best_phot"] = out["has_best_j"] & out["has_best_h"] & out["has_best_ks"]

    out["has_reddening"] = np.isfinite(as_numeric(first_present_value(out, ["ext_e_jks"])))
    out["has_j0"] = np.isfinite(as_numeric(first_present_value(out, ["j0"])))
    out["has_h0"] = np.isfinite(as_numeric(first_present_value(out, ["h0"])))
    out["has_ks0"] = np.isfinite(as_numeric(first_present_value(out, ["ks0"])))
    out["has_dereddened_phot"] = out["has_j0"] & out["has_h0"] & out["has_ks0"]

    j_src = clean_text(first_present_value(out, ["j_mag_source"], default=""))
    h_src = clean_text(first_present_value(out, ["h_mag_source"], default=""))
    k_src = clean_text(first_present_value(out, ["ks_mag_source"], default=""))

    overall = np.full(len(out), "", dtype=object)
    virac_all = (j_src == "VIRAC2") & (h_src == "VIRAC2") & (k_src == "VIRAC2")
    tmass_all = (j_src == "2MASS") & (h_src == "2MASS") & (k_src == "2MASS")
    mixed = (~virac_all) & (~tmass_all) & ((j_src != "") | (h_src != "") | (k_src != ""))
    overall[virac_all.to_numpy()] = "VIRAC2"
    overall[tmass_all.to_numpy()] = "2MASS"
    overall[mixed.to_numpy()] = "MIXED"
    out["phot_source_overall"] = pd.Series(overall, index=out.index, dtype=object)

    sightline = clean_text(first_present_value(out, ["sightline_id"], default=""))
    out["has_sightline"] = sightline != ""

    # Two readiness levels: old photometric readiness and final parameter readiness.
    out["calibration_ready_phot"] = out["has_best_phot"] & out["has_reddening"] & out["has_sightline"]
    return out


# -----------------------------------------------------------------------------
# Best-available parameter harvesting
# -----------------------------------------------------------------------------

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
    return lum.where((r > 0) & (t > 0))



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
    out = pick_best_from_candidates(out, teff_candidates, "teff_best", "e_teff_best", "teff_source_best")

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
    out = pick_best_from_candidates(out, logg_candidates, "logg_best", "e_logg_best", "logg_source_best")

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
        "metallicity_best",
        "e_metallicity_best",
        "metallicity_source_best",
        "metallicity_kind_best",
    )

    radius_candidates = [
        {"col": "radius", "err": None, "label": "ASTRA"},
        {"col": "Rad", "err": None, "label": "GAIA_AP"},
        {"col": "Rad-Flame", "err": None, "label": "GAIA_FLAME"},
    ]
    out = pick_best_from_candidates(out, radius_candidates, "radius_best_rsun", "e_radius_best_rsun", "radius_source_best")

    out["luminosity_flame_lsun"] = as_numeric(first_present_value(out, ["Lum-Flame"]))
    out.loc[as_numeric(out["luminosity_flame_lsun"]) <= 0, "luminosity_flame_lsun"] = np.nan
    out["luminosity_from_astra_lsun"] = compute_luminosity_from_radius_teff(first_present_value(out, ["radius"]), first_present_value(out, ["teff"]))
    out["luminosity_from_gaia_ap_lsun"] = compute_luminosity_from_radius_teff(first_present_value(out, ["Rad"]), first_present_value(out, ["Teff_x"]))
    out["luminosity_from_xgboost_lsun"] = compute_luminosity_from_radius_teff(first_present_value(out, ["Rad"]), first_present_value(out, ["teff_xgboost"]))
    out["luminosity_from_best_rt_lsun"] = compute_luminosity_from_radius_teff(out["radius_best_rsun"], out["teff_best"])

    lum_candidates = [
        {"col": "luminosity_flame_lsun", "err": None, "label": "GAIA_FLAME"},
        {"col": "luminosity_from_astra_lsun", "err": None, "label": "ASTRA_RADIUS+ASTRA_TEFF"},
        {"col": "luminosity_from_gaia_ap_lsun", "err": None, "label": "GAIA_RAD+GAIA_TEFF"},
        {"col": "luminosity_from_xgboost_lsun", "err": None, "label": "GAIA_RAD+XGBOOST_TEFF"},
        {"col": "luminosity_from_best_rt_lsun", "err": None, "label": "BEST_RADIUS+BEST_TEFF"},
    ]
    out = pick_best_from_candidates(out, lum_candidates, "luminosity_best_lsun", "e_luminosity_best_lsun", "luminosity_source_best")

    logl = np.full(len(out), np.nan, dtype=float)
    good = np.isfinite(as_numeric(out["luminosity_best_lsun"])) & (as_numeric(out["luminosity_best_lsun"]) > 0)
    logl[np.asarray(good, dtype=bool)] = np.log10(as_numeric(out.loc[good, "luminosity_best_lsun"]))
    out["log10_luminosity_best_lsun"] = logl

    out["z_best"] = out["metallicity_best"]
    out["e_z_best"] = out["e_metallicity_best"]
    out["z_source_best"] = out["metallicity_source_best"]
    out["z_kind_best"] = out["metallicity_kind_best"]

    out["has_teff_best"] = np.isfinite(as_numeric(out["teff_best"]))
    out["has_logg_best"] = np.isfinite(as_numeric(out["logg_best"]))
    out["has_metallicity_best"] = np.isfinite(as_numeric(out["metallicity_best"]))
    out["has_radius_best"] = np.isfinite(as_numeric(out["radius_best_rsun"]))
    out["has_luminosity_best"] = np.isfinite(as_numeric(out["luminosity_best_lsun"]))
    out["has_core_params"] = out["has_teff_best"] & out["has_logg_best"] & out["has_metallicity_best"] & out["has_luminosity_best"]

    out["calibration_ready_params"] = ensure_bool(out, ["calibration_ready_phot"]) & out["has_core_params"]
    out["calibration_ready"] = out["calibration_ready_params"]
    return out


# -----------------------------------------------------------------------------
# Catalog ordering and summary tables
# -----------------------------------------------------------------------------

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "source_id_best", "sdss_id", "sdss4_apogee_id", "gaia_id_best", "gaia_dr3_source_id", "gaia_dr2_source_id", "tic_v8_id",
        "catalogid", "catalogid21", "catalogid25", "catalogid31", "ra", "dec", "ra_icrs", "dec_icrs", "l", "b",
        "teff_best", "e_teff_best", "teff_source_best",
        "logg_best", "e_logg_best", "logg_source_best",
        "metallicity_best", "e_metallicity_best", "metallicity_source_best", "metallicity_kind_best",
        "z_best", "e_z_best", "z_source_best", "z_kind_best",
        "radius_best_rsun", "e_radius_best_rsun", "radius_source_best",
        "luminosity_best_lsun", "log10_luminosity_best_lsun", "luminosity_source_best",
        "j_mag_best", "h_mag_best", "ks_mag_best", "j_mag_source", "h_mag_source", "ks_mag_source", "phot_source_overall",
        "j_minus_h_best", "h_minus_ks_best", "j_minus_ks_best",
        "ext_e_jks", "ext_e_jks_err", "a_j", "a_h", "a_ks", "j0", "h0", "ks0", "j0_minus_h0", "h0_minus_ks0", "j0_minus_ks0",
        "g0_gaia", "bp0_gaia", "rp0_gaia", "bp_rp0_gaia", "u0_bdbs", "g0_bdbs", "i0_bdbs", "g_i0_bdbs",
        "sightline_tile", "sightline_glon", "sightline_glat", "sightline_id", "sightline_bin_0p1deg", "ext_map_match_sep_arcsec",
        "has_virac2", "has_2mass", "has_best_j", "has_best_h", "has_best_ks", "has_best_phot",
        "has_reddening", "has_dereddened_phot", "has_sightline", "has_teff_best", "has_logg_best", "has_metallicity_best", "has_radius_best", "has_luminosity_best",
        "calibration_ready_phot", "has_core_params", "calibration_ready_params", "calibration_ready",
    ]
    front = [c for c in preferred if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]



def build_overlap_table(master: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("total", len(master)),
        ("has_virac2", count_true(master["has_virac2"])),
        ("has_2mass", count_true(master["has_2mass"])),
        ("has_reddening", count_true(master["has_reddening"])),
        ("has_dereddened_phot", count_true(master["has_dereddened_phot"])),
        ("calibration_ready_phot", count_true(master["calibration_ready_phot"])),
        ("has_teff_best", count_true(master["has_teff_best"])),
        ("has_logg_best", count_true(master["has_logg_best"])),
        ("has_metallicity_best", count_true(master["has_metallicity_best"])),
        ("has_luminosity_best", count_true(master["has_luminosity_best"])),
        ("has_core_params", count_true(master["has_core_params"])),
        ("calibration_ready", count_true(master["calibration_ready"])),
    ]
    out = pd.DataFrame(rows, columns=["metric", "count"])
    out["fraction_of_total"] = out["count"] / max(len(master), 1)
    return out



def build_phot_source_table(master: pd.DataFrame) -> pd.DataFrame:
    counts = clean_text(master["phot_source_overall"]).replace("", "NONE").value_counts(dropna=False)
    out = counts.rename_axis("phot_source_overall").reset_index(name="count")
    out["fraction_of_total"] = out["count"] / max(len(master), 1)
    return out



def build_source_stats(df: pd.DataFrame, value_col: str, source_col: str) -> pd.DataFrame:
    good = np.isfinite(as_numeric(df[value_col]))
    if source_col not in df.columns:
        return pd.DataFrame(columns=[source_col, "count", "fraction_of_finite"])
    counts = clean_text(df.loc[good, source_col]).replace("", "NONE").value_counts(dropna=False)
    out = counts.rename_axis(source_col).reset_index(name="count")
    out["fraction_of_finite"] = out["count"] / max(int(good.sum()), 1)
    return out



def build_sightline_table(master: pd.DataFrame) -> pd.DataFrame:
    work = master.loc[master["has_sightline"]].copy()
    if len(work) == 0:
        return pd.DataFrame(columns=[
            "sightline_id", "sightline_tile", "sightline_glon", "sightline_glat",
            "n_total", "n_ready_phot", "n_ready_final", "n_has_core_params",
            "median_e_jks", "median_j0_minus_ks0", "median_ks0"
        ])

    rows = []
    for sid, g in work.groupby("sightline_id", dropna=False):
        rows.append({
            "sightline_id": sid,
            "sightline_tile": clean_text(first_present_value(g, ["sightline_tile"], default="")).iloc[0],
            "sightline_glon": as_numeric(first_present_value(g, ["sightline_glon"], default=np.nan)).iloc[0],
            "sightline_glat": as_numeric(first_present_value(g, ["sightline_glat"], default=np.nan)).iloc[0],
            "n_total": len(g),
            "n_ready_phot": count_true(g["calibration_ready_phot"]),
            "n_ready_final": count_true(g["calibration_ready"]),
            "n_has_core_params": count_true(g["has_core_params"]),
            "median_e_jks": float(np.nanmedian(as_numeric(first_present_value(g, ["ext_e_jks"])))),
            "median_j0_minus_ks0": float(np.nanmedian(as_numeric(first_present_value(g, ["j0_minus_ks0"])))),
            "median_ks0": float(np.nanmedian(as_numeric(first_present_value(g, ["ks0"])))),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["n_total", "n_ready_final", "n_has_core_params"], ascending=[False, False, False]).reset_index(drop=True)



def write_summary_text(master: pd.DataFrame, ready: pd.DataFrame, overlap: pd.DataFrame, phot_sources: pd.DataFrame, sightlines: pd.DataFrame, path: pathlib.Path) -> None:
    def metric(name: str) -> int:
        row = overlap.loc[overlap["metric"] == name, "count"]
        return int(row.iloc[0]) if len(row) else 0

    lines: list[str] = []
    lines.append("Roman final calibration-catalog summary")
    lines.append("=" * 39)
    lines.append(f"Total rows                               : {len(master)}")
    lines.append(f"Rows with reddening                      : {metric('has_reddening')}")
    lines.append(f"Rows ready at phot stage                 : {metric('calibration_ready_phot')}")
    lines.append(f"Rows with best Teff                      : {metric('has_teff_best')}")
    lines.append(f"Rows with best logg                      : {metric('has_logg_best')}")
    lines.append(f"Rows with best metallicity               : {metric('has_metallicity_best')}")
    lines.append(f"Rows with best luminosity                : {metric('has_luminosity_best')}")
    lines.append(f"Rows with Teff+logg+Z+L                  : {metric('has_core_params')}")
    lines.append(f"Final calibration-ready rows             : {len(ready)}")
    lines.append(f"Unique sightlines                        : {len(sightlines)}")
    if len(master) > 0:
        lines.append(f"Final calibration-ready fraction         : {len(ready)/len(master):.4f}")

    lines.append("")
    lines.append("Photometry source mix")
    lines.append("-" * 20)
    for _, row in phot_sources.iterrows():
        lines.append(f"{row['phot_source_overall']:<10} : {int(row['count']):>4d} ({float(row['fraction_of_total']):.3f})")

    for val_col, src_col, title in [
        ("teff_best", "teff_source_best", "Teff source mix"),
        ("logg_best", "logg_source_best", "logg source mix"),
        ("metallicity_best", "metallicity_source_best", "Metallicity source mix"),
        ("luminosity_best_lsun", "luminosity_source_best", "Luminosity source mix"),
    ]:
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))
        stats = build_source_stats(master, val_col, src_col)
        if len(stats) == 0:
            lines.append("None")
        else:
            for _, row in stats.iterrows():
                lines.append(f"{row[src_col]:<24} : {int(row['count']):>4d} ({float(row['fraction_of_finite']):.3f})")

    if len(sightlines) > 0:
        lines.append("")
        lines.append("Top sightlines by row count")
        lines.append("-" * 26)
        for _, row in sightlines.head(10).iterrows():
            lines.append(f"{str(row['sightline_id'])[:55]:<55}  n={int(row['n_total']):>3d}  final={int(row['n_ready_final']):>3d}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Diagnostics plotting
# -----------------------------------------------------------------------------

def plot_glon_glat_by_status(df: pd.DataFrame, outdir: pathlib.Path) -> Optional[str]:
    l = as_numeric(get_series(df, ["l", "sightline_glon"]))
    l = VVVlwrap(l)
    b = as_numeric(get_series(df, ["b", "sightline_glat"]))
    has_red = ensure_bool(df, ["has_reddening"])
    cal_ready = ensure_bool(df, ["calibration_ready"])
    good = finite_mask(l, b)
    if good.sum() == 0:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(l[good & ~has_red], b[good & ~has_red], s=20, alpha=0.65, label="No reddening")
    ax.scatter(l[good & has_red & ~cal_ready], b[good & has_red & ~cal_ready], s=24, alpha=0.8, label="Has reddening / not final-ready")
    ax.scatter(l[good & cal_ready], b[good & cal_ready], s=30, alpha=0.9, label="Final calibration-ready")
    ax.set_xlabel("Galactic longitude l [deg]")
    ax.set_ylabel("Galactic latitude b [deg]")
    ax.set_title("Roman sample sky coverage by final status")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    path = outdir / "sky_status.png"
    savefig(fig, path)
    return path.name



def plot_glon_glat_colored(df: pd.DataFrame, outdir: pathlib.Path, zcol_candidates: list[str], filename: str, title: str, cbar_label: str) -> Optional[str]:
    l = as_numeric(get_series(df, ["l", "sightline_glon"]))
    l = VVVlwrap(l)
    b = as_numeric(get_series(df, ["b", "sightline_glat"]))
    z = as_numeric(get_series(df, zcol_candidates))
    good = finite_mask(l, b, z)
    if good.sum() == 0:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sc = ax.scatter(l[good], b[good], c=z[good], s=28, alpha=0.9)
    ax.set_xlabel("Galactic longitude l [deg]")
    ax.set_ylabel("Galactic latitude b [deg]")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(cbar_label)
    path = outdir / filename
    savefig(fig, path)
    return path.name



def plot_cmd(df: pd.DataFrame, outdir: pathlib.Path, color_candidates: list[str], mag_candidates: list[str], filename: str, title: str, xlabel: str, ylabel: str, color_by: Optional[list[str]] = None, cbar_label: str = "") -> Optional[str]:
    color = as_numeric(get_series(df, color_candidates))
    mag = as_numeric(get_series(df, mag_candidates))
    good = finite_mask(color, mag)
    if good.sum() == 0:
        return None

    fig, ax = plt.subplots(figsize=(7.7, 6.7))
    if color_by is None:
        ax.scatter(color[good], mag[good], s=24, alpha=0.8)
    else:
        z = as_numeric(get_series(df, color_by))
        good &= np.isfinite(np.asarray(z, dtype=float))
        if good.sum() == 0:
            return None
        sc = ax.scatter(color[good], mag[good], c=z[good], s=24, alpha=0.85)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(cbar_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    path = outdir / filename
    savefig(fig, path)
    return path.name



def plot_teff_logg(df: pd.DataFrame, outdir: pathlib.Path) -> Optional[str]:
    teff = as_numeric(get_series(df, ["teff_best", "teff"]))
    logg = as_numeric(get_series(df, ["logg_best", "logg"]))
    metal = as_numeric(get_series(df, ["metallicity_best", "z_best"]))
    good = finite_mask(teff, logg)
    if good.sum() == 0:
        return None

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    if np.isfinite(np.asarray(metal, dtype=float)).sum() > 0:
        good2 = good & np.isfinite(np.asarray(metal, dtype=float))
        sc = ax.scatter(teff[good2], logg[good2], c=metal[good2], s=26, alpha=0.85)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Best metallicity proxy [dex]")
    else:
        ax.scatter(teff[good], logg[good], s=26, alpha=0.85)
    ax.set_xlabel("Teff [K]")
    ax.set_ylabel("log g")
    ax.set_title("Teff-log g plane")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    path = outdir / "teff_logg.png"
    savefig(fig, path)
    return path.name



def plot_teff_vs_colors(df: pd.DataFrame, outdir: pathlib.Path) -> list[str]:
    produced: list[str] = []
    configs = [
        (["j_minus_ks_best", "j0_minus_ks0"], "teff_vs_j_minus_ks.png", "J-Ks"),
        (["j_minus_h_best", "j0_minus_h0"], "teff_vs_j_minus_h.png", "J-H"),
        (["h_minus_ks_best", "h0_minus_ks0"], "teff_vs_h_minus_ks.png", "H-Ks"),
    ]
    teff = as_numeric(get_series(df, ["teff_best", "teff"]))
    for candidates, fname, label in configs:
        color = as_numeric(get_series(df, candidates))
        good = finite_mask(teff, color)
        if good.sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(7.6, 6.2))
        ax.scatter(color[good], teff[good], s=24, alpha=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel("Teff [K]")
        ax.set_title(f"Teff versus {label}")
        ax.invert_yaxis()
        ax.grid(alpha=0.25)
        path = outdir / fname
        savefig(fig, path)
        produced.append(path.name)
    return produced



def plot_histogram(df: pd.DataFrame, outdir: pathlib.Path, candidates: list[str], filename: str, title: str, xlabel: str, bins: int = 30) -> Optional[str]:
    x = as_numeric(get_series(df, candidates))
    x = x[np.isfinite(np.asarray(x, dtype=float))]
    if len(x) == 0:
        return None
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    ax.hist(x, bins=bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of stars")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    path = outdir / filename
    savefig(fig, path)
    return path.name



def plot_hist_by_group(df: pd.DataFrame, outdir: pathlib.Path, value_candidates: list[str], group_candidates: list[str], filename: str, title: str, xlabel: str) -> Optional[str]:
    x = as_numeric(get_series(df, value_candidates))
    g = clean_text(get_series(df, group_candidates, default="NONE")).replace("", "NONE")
    good = np.isfinite(np.asarray(x, dtype=float))
    if good.sum() == 0:
        return None

    levels = [lvl for lvl in ["VIRAC2", "2MASS", "MIXED", "NONE"] if lvl in set(g[good])]
    if not levels:
        return None

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    for lvl in levels:
        m = good & (g == lvl)
        if m.sum() == 0:
            continue
        ax.hist(x[m], bins=25, histtype="step", linewidth=2, label=lvl)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of stars")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    path = outdir / filename
    savefig(fig, path)
    return path.name



def plot_match_separation(df: pd.DataFrame, outdir: pathlib.Path) -> list[str]:
    produced: list[str] = []
    configs = [
        (["vvv_match_sep_arcsec", "virac2_match_sep_arcsec"], "virac2_match_sep_hist.png", "VIRAC2 match separation", "Separation [arcsec]"),
        (["tmass_match_sep_arcsec", "2mass_match_sep_arcsec", "Separation"], "tmass_match_sep_hist.png", "2MASS/BDBS match separation", "Separation [arcsec]"),
        (["ext_map_match_sep_arcsec"], "reddening_match_sep_hist.png", "Reddening-map match separation", "Separation [arcsec]"),
    ]
    for candidates, fname, title, xlabel in configs:
        made = plot_histogram(df, outdir, candidates, fname, title, xlabel, bins=25)
        if made is not None:
            produced.append(made)
    return produced



def plot_sightline_counts(df: pd.DataFrame, outdir: pathlib.Path, top_n: int = 30) -> Optional[str]:
    sid = clean_text(get_series(df, ["sightline_id"], default=""))
    if (sid != "").sum() == 0:
        return None
    counts = sid[sid != ""].value_counts().head(top_n)
    labels = [s[:20] for s in counts.index.astype(str)]
    x = np.arange(len(counts))
    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    ax.bar(x, counts.to_numpy(dtype=float))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("Number of stars")
    ax.set_title(f"Top {len(counts)} sightlines by star count")
    ax.grid(axis="y", alpha=0.25)
    path = outdir / "sightline_counts_top30.png"
    savefig(fig, path)
    return path.name



def plot_sightline_reddening(df: pd.DataFrame, outdir: pathlib.Path, top_n: int = 20) -> Optional[str]:
    sid = clean_text(get_series(df, ["sightline_id"], default=""))
    ejks = as_numeric(get_series(df, ["ext_e_jks"]))
    good = (sid != "") & np.isfinite(np.asarray(ejks, dtype=float))
    if good.sum() == 0:
        return None
    top_ids = sid[good].value_counts().head(top_n).index
    labels = [s[:18] for s in top_ids.astype(str)]
    groups = [as_numeric(ejks[good & sid.isin([sid_val])]) for sid_val in top_ids]
    if not groups:
        return None
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.boxplot(groups, tick_labels=labels, showfliers=False)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("E(J-Ks)")
    ax.set_title("Reddening distribution in top sightlines")
    ax.grid(axis="y", alpha=0.25)
    path = outdir / "sightline_reddening_boxplot.png"
    savefig(fig, path)
    return path.name



def plot_color_color(df: pd.DataFrame, outdir: pathlib.Path, dered: bool = False) -> Optional[str]:
    if dered:
        x = as_numeric(get_series(df, ["j0_minus_h0"]))
        y = as_numeric(get_series(df, ["h0_minus_ks0"]))
        fname = "color_color_dereddened.png"
        title = "Dereddened color-color diagram"
        xlabel = "(J-H)0"
        ylabel = "(H-Ks)0"
    else:
        x = as_numeric(get_series(df, ["j_minus_h_best"]))
        y = as_numeric(get_series(df, ["h_minus_ks_best"]))
        fname = "color_color_observed.png"
        title = "Observed color-color diagram"
        xlabel = "J-H"
        ylabel = "H-Ks"
    good = finite_mask(x, y)
    if good.sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(7.1, 6.4))
    ax.scatter(x[good], y[good], s=24, alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    path = outdir / fname
    savefig(fig, path)
    return path.name



def plot_reddening_vs_position(df: pd.DataFrame, outdir: pathlib.Path) -> list[str]:
    produced: list[str] = []
    ejks = as_numeric(get_series(df, ["ext_e_jks"]))
    for coord_candidates, fname, xlabel in [(["l", "sightline_glon"], "ejks_vs_l.png", "Galactic longitude l [deg]"), (["b", "sightline_glat"], "ejks_vs_b.png", "Galactic latitude b [deg]")]:
        coord = as_numeric(get_series(df, coord_candidates))
        good = finite_mask(coord, ejks)
        if good.sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(7.6, 5.8))
        ax.scatter(coord[good], ejks[good], s=24, alpha=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("E(J-Ks)")
        ax.set_title(f"Reddening versus {xlabel.split()[1]}")
        ax.grid(alpha=0.25)
        path = outdir / fname
        savefig(fig, path)
        produced.append(path.name)
    return produced



def plot_dereddening_shift(df: pd.DataFrame, outdir: pathlib.Path) -> list[str]:
    produced: list[str] = []
    configs = [
        (["j_mag_best"], ["j0"], "delta_j_dereddening_hist.png", "J - J0"),
        (["h_mag_best"], ["h0"], "delta_h_dereddening_hist.png", "H - H0"),
        (["ks_mag_best"], ["ks0"], "delta_ks_dereddening_hist.png", "Ks - Ks0"),
    ]
    for obs_c, dered_c, fname, xlabel in configs:
        obs = as_numeric(get_series(df, obs_c))
        dered = as_numeric(get_series(df, dered_c))
        delta = obs - dered
        good = np.isfinite(np.asarray(delta, dtype=float))
        if good.sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(7.4, 5.6))
        ax.hist(delta[good], bins=25)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of stars")
        ax.set_title(f"Dereddening shift: {xlabel}")
        ax.grid(alpha=0.25)
        path = outdir / fname
        savefig(fig, path)
        produced.append(path.name)
    return produced



def plot_h_vs_teff_by_ready(df: pd.DataFrame, outdir: pathlib.Path) -> Optional[str]:
    h = as_numeric(get_series(df, ["h_mag_best", "h_mag"]))
    teff = as_numeric(get_series(df, ["teff_best", "teff"]))
    ready = ensure_bool(df, ["calibration_ready"])
    good = finite_mask(h, teff)
    if good.sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    ax.scatter(teff[good & ~ready], h[good & ~ready], s=22, alpha=0.7, label="Not ready")
    ax.scatter(teff[good & ready], h[good & ready], s=28, alpha=0.85, label="Final-ready")
    ax.set_xlabel("Teff [K]")
    ax.set_ylabel("H")
    ax.set_title("H versus Teff")
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    path = outdir / "h_vs_teff_ready.png"
    savefig(fig, path)
    return path.name



def plot_teff_metallicity_logg(df: pd.DataFrame, outdir: pathlib.Path) -> Optional[str]:
    teff = as_numeric(get_series(df, ["teff_best", "teff"]))
    metal = as_numeric(get_series(df, ["metallicity_best", "z_best"]))
    logg = as_numeric(get_series(df, ["logg_best", "logg"]))
    good = finite_mask(teff, metal, logg)
    if good.sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    sc = ax.scatter(teff[good], metal[good], c=logg[good], s=18, alpha=0.85)
    ax.set_xlabel("Teff [K]")
    ax.set_ylabel("Best metallicity proxy [dex]")
    ax.set_title("Teff vs metallicity, colored by logg")
    ax.invert_xaxis()
    ax.grid(alpha=0.25)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("log g")
    path = outdir / "teff_metallicity_logg.png"
    savefig(fig, path)
    return path.name



def plot_hrd(df: pd.DataFrame, outdir: pathlib.Path) -> Optional[str]:
    teff = as_numeric(get_series(df, ["teff_best", "teff"]))
    logl = as_numeric(get_series(df, ["log10_luminosity_best_lsun"]))
    metal = as_numeric(get_series(df, ["metallicity_best", "z_best"]))
    good = finite_mask(teff, logl)
    if good.sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    if np.isfinite(np.asarray(metal, dtype=float)).any():
        good2 = good & np.isfinite(np.asarray(metal, dtype=float))
        sc = ax.scatter(teff[good2], logl[good2], c=metal[good2], s=18, alpha=0.85)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("Best metallicity proxy [dex]")
    else:
        ax.scatter(teff[good], logl[good], s=18, alpha=0.85)
    ax.set_xlabel("Teff [K]")
    ax.set_ylabel(r"$\log_{10}(L/L_\odot)$")
    ax.set_title("Harvested HR diagram")
    ax.invert_xaxis()
    ax.grid(alpha=0.25)
    path = outdir / "hrd.png"
    savefig(fig, path)
    return path.name



def write_diagnostics(df: pd.DataFrame, outdir: pathlib.Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    produced: list[str] = []

    for item in [
        plot_glon_glat_by_status(df, outdir),
        plot_glon_glat_colored(df, outdir, ["ext_e_jks"], "sky_reddening.png", "Sky map colored by reddening", "E(J-Ks)"),
        plot_glon_glat_colored(df, outdir, ["teff_best", "teff"], "sky_teff.png", "Sky map colored by Teff", "Teff [K]"),
        plot_glon_glat_colored(df, outdir, ["logg_best", "logg"], "sky_logg.png", "Sky map colored by log g", "log g"),
        plot_glon_glat_colored(df, outdir, ["metallicity_best", "z_best"], "sky_metallicity.png", "Sky map colored by metallicity", "Best metallicity proxy [dex]"),
        plot_cmd(df, outdir, ["j_minus_ks_best"], ["ks_mag_best", "h_mag"], "cmd_observed_jks_ks.png", "Observed CMD", "J-Ks", "Ks", color_by=["ext_e_jks"], cbar_label="E(J-Ks)"),
        plot_cmd(df, outdir, ["j0_minus_ks0"], ["ks0"], "cmd_dereddened_jks0_ks0.png", "Dereddened CMD", "(J-Ks)0", "Ks0", color_by=["teff_best", "teff"], cbar_label="Teff [K]"),
        plot_cmd(df, outdir, ["j_minus_h_best"], ["h_mag_best", "h_mag"], "cmd_observed_jh_h.png", "Observed J-H versus H", "J-H", "H", color_by=["ext_e_jks"], cbar_label="E(J-Ks)"),
        plot_cmd(df, outdir, ["j0_minus_h0"], ["h0"], "cmd_dereddened_jh0_h0.png", "Dereddened J-H versus H0", "(J-H)0", "H0", color_by=["teff_best", "teff"], cbar_label="Teff [K]"),
        plot_teff_logg(df, outdir),
        plot_histogram(df, outdir, ["teff_best", "teff"], "hist_teff.png", "Teff distribution", "Teff [K]"),
        plot_histogram(df, outdir, ["logg_best", "logg"], "hist_logg.png", "log g distribution", "log g"),
        plot_histogram(df, outdir, ["metallicity_best", "z_best"], "hist_metallicity.png", "Metallicity distribution", "Best metallicity proxy [dex]"),
        plot_histogram(df, outdir, ["luminosity_best_lsun"], "hist_luminosity.png", "Luminosity distribution", "L/Lsun"),
        plot_histogram(df, outdir, ["h_mag_best", "h_mag"], "hist_hmag.png", "H distribution", "H"),
        plot_histogram(df, outdir, ["ext_e_jks"], "hist_ejks.png", "Reddening distribution", "E(J-Ks)"),
        plot_histogram(df, outdir, ["j0_minus_ks0"], "hist_j0_minus_ks0.png", "Dereddened color distribution", "(J-Ks)0"),
        plot_hist_by_group(df, outdir, ["ext_e_jks"], ["phot_source_overall"], "hist_ejks_by_photsource.png", "Reddening by photometry source", "E(J-Ks)"),
        plot_hist_by_group(df, outdir, ["teff_best", "teff"], ["phot_source_overall"], "hist_teff_by_photsource.png", "Teff by photometry source", "Teff [K]"),
        plot_sightline_counts(df, outdir),
        plot_sightline_reddening(df, outdir),
        plot_color_color(df, outdir, dered=False),
        plot_color_color(df, outdir, dered=True),
        plot_h_vs_teff_by_ready(df, outdir),
        plot_teff_metallicity_logg(df, outdir),
        plot_hrd(df, outdir),
    ]:
        if item is not None:
            produced.append(item)

    produced.extend(plot_teff_vs_colors(df, outdir))
    produced.extend(plot_match_separation(df, outdir))
    produced.extend(plot_reddening_vs_position(df, outdir))
    produced.extend(plot_dereddening_shift(df, outdir))

    manifest = outdir / "manifest.txt"
    lines = [
        "Roman final calibration diagnostics manifest",
        "=" * 43,
        f"Rows       : {len(df)}",
        "",
        "Plots written:",
    ]
    for name in sorted(set(produced)):
        lines.append(f"- {name}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Build final catalog
# -----------------------------------------------------------------------------

def build_final_catalog(base: pd.DataFrame, aks_per_ejks: float, ah_per_ejks: float, aj_per_ejks: float) -> pd.DataFrame:
    out = add_identifier_columns(base)
    out = add_dereddened_photometry(out, aks_per_ejks=aks_per_ejks, ah_per_ejks=ah_per_ejks, aj_per_ejks=aj_per_ejks)
    out = add_status_flags(out)
    out = add_best_parameters(out)
    out = reorder_columns(out)
    return out


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build the final Roman calibration catalog and diagnostics in one pass.")
    p.add_argument("input_table", nargs="?", default="astra_overguide_roman_dereddened.fits")
    p.add_argument("--xgboost", default=None, help="Optional FITS/CSV/Parquet Gaia XGBoost table to merge by Gaia DR3 source_id")
    p.add_argument("--out-prefix", default="astra_overguide_roman_final_calibration", help="Prefix for output products")
    p.add_argument("--plots-dir", default=None, help="Optional directory for diagnostic plots; default is <out-prefix>_plots")
    p.add_argument("--aks-per-ejks", type=float, default=0.528)
    p.add_argument("--ah-per-ejks", type=float, default=0.857)
    p.add_argument("--aj-per-ejks", type=float, default=1.528)
    return p



def main() -> int:
    args = build_parser().parse_args()

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

    plots_dir = pathlib.Path(args.plots_dir).expanduser().resolve() if args.plots_dir else (out_dir / f"{stem}_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading input table: {input_path}")
    base_df, dropped = read_table_like(input_path)
    if dropped:
        print("Dropping multidimensional columns that cannot be converted to pandas: " + ", ".join(dropped))
    print(f"Input rows: {len(base_df)}")

    if xgb_path is not None:
        print(f"Merging optional XGBoost table: {xgb_path}")
        base_df = merge_xgboost_if_requested(base_df, xgb_path)

    master = build_final_catalog(base_df, aks_per_ejks=args.aks_per_ejks, ah_per_ejks=args.ah_per_ejks, aj_per_ejks=args.aj_per_ejks)
    ready = master.loc[master["calibration_ready"]].copy().reset_index(drop=True)
    phot_ready = master.loc[master["calibration_ready_phot"]].copy().reset_index(drop=True)

    overlap = build_overlap_table(master)
    phot_sources = build_phot_source_table(master)
    sightlines = build_sightline_table(master)
    teff_sources = build_source_stats(master, "teff_best", "teff_source_best")
    logg_sources = build_source_stats(master, "logg_best", "logg_source_best")
    metal_sources = build_source_stats(master, "metallicity_best", "metallicity_source_best")
    lum_sources = build_source_stats(master, "luminosity_best_lsun", "luminosity_source_best")

    master_fits = out_dir / f"{stem}_master.fits"
    master_csv = out_dir / f"{stem}_master.csv"
    ready_fits = out_dir / f"{stem}_calibration_ready.fits"
    ready_csv = out_dir / f"{stem}_calibration_ready.csv"
    phot_ready_fits = out_dir / f"{stem}_phot_ready.fits"
    phot_ready_csv = out_dir / f"{stem}_phot_ready.csv"
    overlap_csv = out_dir / f"{stem}_overlap_stats.csv"
    phot_csv = out_dir / f"{stem}_phot_source_stats.csv"
    sight_csv = out_dir / f"{stem}_sightline_counts.csv"
    teff_src_csv = out_dir / f"{stem}_teff_source_stats.csv"
    logg_src_csv = out_dir / f"{stem}_logg_source_stats.csv"
    metal_src_csv = out_dir / f"{stem}_metallicity_source_stats.csv"
    lum_src_csv = out_dir / f"{stem}_luminosity_source_stats.csv"
    summary_txt = out_dir / f"{stem}_summary.txt"

    print(f"Writing master catalog FITS: {master_fits}")
    write_table(master, master_fits, master_csv)

    print(f"Writing final calibration-ready subset FITS: {ready_fits}")
    write_table(ready, ready_fits, ready_csv)

    print(f"Writing phot-ready subset FITS: {phot_ready_fits}")
    write_table(phot_ready, phot_ready_fits, phot_ready_csv)

    print(f"Writing overlap statistics CSV: {overlap_csv}")
    overlap.to_csv(overlap_csv, index=False)
    print(f"Writing photometry-source statistics CSV: {phot_csv}")
    phot_sources.to_csv(phot_csv, index=False)
    print(f"Writing sightline counts CSV: {sight_csv}")
    sightlines.to_csv(sight_csv, index=False)
    teff_sources.to_csv(teff_src_csv, index=False)
    logg_sources.to_csv(logg_src_csv, index=False)
    metal_sources.to_csv(metal_src_csv, index=False)
    lum_sources.to_csv(lum_src_csv, index=False)

    print(f"Writing summary text: {summary_txt}")
    write_summary_text(master, ready, overlap, phot_sources, sightlines, summary_txt)

    print(f"Writing diagnostics to: {plots_dir}")
    write_diagnostics(master, plots_dir)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
