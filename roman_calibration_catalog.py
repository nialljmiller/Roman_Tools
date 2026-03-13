#!/usr/bin/env python3
"""
Build overlap statistics, a cleaned master calibration catalog, and quick-look
plots from the dereddened/sightline-tagged Roman ASTRA sample.

Designed to run after roman_deredden_sightlines.py, for example:

    python roman_calibration_catalog.py astra_overguide_roman_dereddened.fits \
        --out-prefix astra_overguide_roman_calibration

What this script does:
    1. Reads the dereddened FITS table.
    2. Drops multidimensional columns that do not round-trip cleanly to pandas.
    3. Adds provenance / quality flags:
         - has_virac2
         - has_2mass
         - has_reddening
         - has_dereddened_phot
         - phot_source_overall
         - calibration_ready
    4. Writes:
         - a full master catalog (FITS + CSV)
         - a calibration-ready subset (FITS + CSV)
         - overlap statistics CSV + summary text
         - per-sightline counts CSV
         - quick-look plots (PNG)

The script is intentionally conservative: it does not try to infer metallicity.
It only organizes the current photometry/reddening/sightline state cleanly so
later APOGEE/BDBS information can be merged in without changing the schema.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from astropy.table import Table

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _lower_name_map(columns: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in columns:
        mapping[str(name).lower()] = str(name)
    return mapping



def find_first_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    mapping = _lower_name_map(columns)
    for candidate in candidates:
        key = str(candidate).lower()
        if key in mapping:
            return mapping[key]
    return None



def as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")



def count_true(mask: pd.Series | np.ndarray) -> int:
    arr = np.asarray(mask, dtype=bool)
    return int(arr.sum())



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



def choose_text(primary: pd.Series, fallback: pd.Series, empty: str = "") -> pd.Series:
    p = clean_text_series(primary)
    f = clean_text_series(fallback)
    out = p.copy()
    use_fallback = (out == "") & (f != "")
    out.loc[use_fallback] = f.loc[use_fallback]
    out.loc[out == ""] = empty
    return out



def first_present_value(df: pd.DataFrame, candidates: Iterable[str], default=np.nan) -> pd.Series:
    col = find_first_column(df.columns, candidates)
    if col is None:
        return pd.Series(default, index=df.index)
    return df[col]


# -----------------------------------------------------------------------------
# Catalog augmentation
# -----------------------------------------------------------------------------

def add_status_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    virac_src = first_present_value(out, ["vvv_srcid", "vvv_SrcID", "vvv_sourceid", "vvv_source_id"], default="")
    tmass_src = first_present_value(out, ["tmass_2MASS", "tmass__2MASS", "tmass_2mass", "tmass_id"], default="")

    out["has_virac2"] = clean_text_series(virac_src) != ""
    out["has_2mass"] = clean_text_series(tmass_src) != ""

    out["has_best_j"] = np.isfinite(as_numeric(first_present_value(out, ["j_mag_best"])))
    out["has_best_h"] = np.isfinite(as_numeric(first_present_value(out, ["h_mag_best"])))
    out["has_best_ks"] = np.isfinite(as_numeric(first_present_value(out, ["ks_mag_best"])))
    out["has_best_phot"] = out["has_best_j"] & out["has_best_h"] & out["has_best_ks"]

    out["has_reddening"] = np.isfinite(as_numeric(first_present_value(out, ["ext_e_jks", "E_JKs", "ejks"])))
    out["has_j0"] = np.isfinite(as_numeric(first_present_value(out, ["j0"])))
    out["has_h0"] = np.isfinite(as_numeric(first_present_value(out, ["h0"])))
    out["has_ks0"] = np.isfinite(as_numeric(first_present_value(out, ["ks0"])))
    out["has_dereddened_phot"] = out["has_j0"] & out["has_h0"] & out["has_ks0"]

    j_src = clean_text_series(first_present_value(out, ["j_mag_source"], default=""))
    h_src = clean_text_series(first_present_value(out, ["h_mag_source"], default=""))
    k_src = clean_text_series(first_present_value(out, ["ks_mag_source"], default=""))

    overall = np.full(len(out), "", dtype=object)
    virac_all = (j_src == "VIRAC2") & (h_src == "VIRAC2") & (k_src == "VIRAC2")
    tmass_all = (j_src == "2MASS") & (h_src == "2MASS") & (k_src == "2MASS")
    mixed = (~virac_all) & (~tmass_all) & ((j_src != "") | (h_src != "") | (k_src != ""))
    overall[virac_all.to_numpy()] = "VIRAC2"
    overall[tmass_all.to_numpy()] = "2MASS"
    overall[mixed.to_numpy()] = "MIXED"
    out["phot_source_overall"] = pd.Series(overall, index=out.index, dtype=object)

    sightline = clean_text_series(first_present_value(out, ["sightline_id"], default=""))
    out["has_sightline"] = sightline != ""

    # Conservative current-stage definition. Metallicities can be merged in later.
    out["calibration_ready"] = out["has_best_phot"] & out["has_reddening"] & out["has_sightline"]

    return out



def add_preferred_identifiers(df: pd.DataFrame) -> pd.DataFrame:
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



def build_master_catalog(df: pd.DataFrame) -> pd.DataFrame:
    out = add_status_flags(df)
    out = add_preferred_identifiers(out)

    # Some consistent ordering for useful columns.
    preferred = [
        "source_id_best",
        "sdss_id",
        "sdss4_apogee_id",
        "gaia_id_best",
        "gaia_dr3_source_id",
        "gaia_dr2_source_id",
        "tic_v8_id",
        "catalogid",
        "catalogid21",
        "catalogid25",
        "catalogid31",
        "ra_icrs",
        "dec_icrs",
        "l",
        "b",
        "teff",
        "logg",
        "h_mag",
        "j_mag_best",
        "h_mag_best",
        "ks_mag_best",
        "j_mag_source",
        "h_mag_source",
        "ks_mag_source",
        "phot_source_overall",
        "e_j_mag_best",
        "e_h_mag_best",
        "e_ks_mag_best",
        "j_minus_h_best",
        "h_minus_ks_best",
        "j_minus_ks_best",
        "ext_e_jks",
        "ext_e_jks_err",
        "a_j",
        "a_h",
        "a_ks",
        "j0",
        "h0",
        "ks0",
        "j0_minus_h0",
        "h0_minus_ks0",
        "j0_minus_ks0",
        "sightline_tile",
        "sightline_glon",
        "sightline_glat",
        "sightline_id",
        "sightline_bin_0p1deg",
        "ext_map_match_sep_arcsec",
        "has_virac2",
        "has_2mass",
        "has_best_j",
        "has_best_h",
        "has_best_ks",
        "has_best_phot",
        "has_reddening",
        "has_dereddened_phot",
        "has_sightline",
        "calibration_ready",
    ]

    existing_preferred = [col for col in preferred if col in out.columns]
    trailing = [col for col in out.columns if col not in existing_preferred]
    out = out[existing_preferred + trailing]
    return out


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def build_overlap_table(master: pd.DataFrame) -> pd.DataFrame:
    has_virac2 = master["has_virac2"]
    has_2mass = master["has_2mass"]
    has_reddening = master["has_reddening"]
    has_dered = master["has_dereddened_phot"]
    cal_ready = master["calibration_ready"]

    rows = [
        ("total", len(master)),
        ("has_virac2", count_true(has_virac2)),
        ("has_2mass", count_true(has_2mass)),
        ("has_reddening", count_true(has_reddening)),
        ("has_dereddened_phot", count_true(has_dered)),
        ("calibration_ready", count_true(cal_ready)),
        ("virac2_and_2mass", count_true(has_virac2 & has_2mass)),
        ("virac2_and_reddening", count_true(has_virac2 & has_reddening)),
        ("2mass_and_reddening", count_true(has_2mass & has_reddening)),
        ("virac2_and_2mass_and_reddening", count_true(has_virac2 & has_2mass & has_reddening)),
        ("virac2_without_reddening", count_true(has_virac2 & ~has_reddening)),
        ("2mass_without_reddening", count_true(has_2mass & ~has_reddening)),
        ("reddening_without_virac2", count_true(has_reddening & ~has_virac2)),
        ("reddening_without_2mass", count_true(has_reddening & ~has_2mass)),
    ]

    frac = []
    n_total = max(len(master), 1)
    for _, count in rows:
        frac.append(float(count) / float(n_total))

    out = pd.DataFrame(rows, columns=["metric", "count"])
    out["fraction_of_total"] = frac
    return out



def build_phot_source_table(master: pd.DataFrame) -> pd.DataFrame:
    counts = clean_text_series(master["phot_source_overall"]).replace("", "NONE").value_counts(dropna=False)
    out = counts.rename_axis("phot_source_overall").reset_index(name="count")
    out["fraction_of_total"] = out["count"] / max(len(master), 1)
    return out



def build_sightline_table(master: pd.DataFrame) -> pd.DataFrame:
    work = master.loc[master["has_sightline"]].copy()
    if len(work) == 0:
        return pd.DataFrame(
            columns=[
                "sightline_id",
                "sightline_tile",
                "sightline_glon",
                "sightline_glat",
                "n_total",
                "n_calibration_ready",
                "n_has_virac2",
                "n_has_2mass",
                "n_has_reddening",
                "median_e_jks",
                "median_j0_minus_ks0",
                "median_ks0",
            ]
        )

    grouped = work.groupby("sightline_id", dropna=False)
    rows = []
    for sightline_id, g in grouped:
        tile = clean_text_series(g["sightline_tile"]).iloc[0] if "sightline_tile" in g else ""
        glon = as_numeric(g["sightline_glon"]).iloc[0] if "sightline_glon" in g else np.nan
        glat = as_numeric(g["sightline_glat"]).iloc[0] if "sightline_glat" in g else np.nan
        rows.append(
            {
                "sightline_id": sightline_id,
                "sightline_tile": tile,
                "sightline_glon": glon,
                "sightline_glat": glat,
                "n_total": len(g),
                "n_calibration_ready": count_true(g["calibration_ready"]),
                "n_has_virac2": count_true(g["has_virac2"]),
                "n_has_2mass": count_true(g["has_2mass"]),
                "n_has_reddening": count_true(g["has_reddening"]),
                "median_e_jks": float(np.nanmedian(as_numeric(g["ext_e_jks"]))) if "ext_e_jks" in g else np.nan,
                "median_j0_minus_ks0": float(np.nanmedian(as_numeric(g["j0_minus_ks0"]))) if "j0_minus_ks0" in g else np.nan,
                "median_ks0": float(np.nanmedian(as_numeric(g["ks0"]))) if "ks0" in g else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["n_total", "n_calibration_ready", "median_e_jks"], ascending=[False, False, True]).reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Output writing
# -----------------------------------------------------------------------------

def write_summary_text(
    master: pd.DataFrame,
    ready: pd.DataFrame,
    overlap: pd.DataFrame,
    phot_sources: pd.DataFrame,
    sightlines: pd.DataFrame,
    path: pathlib.Path,
) -> None:
    def metric(name: str) -> int:
        row = overlap.loc[overlap["metric"] == name, "count"]
        return int(row.iloc[0]) if len(row) else 0

    lines: list[str] = []
    lines.append("Roman calibration-catalog summary")
    lines.append("=" * 34)
    lines.append(f"Total rows in dereddened input            : {len(master)}")
    lines.append(f"Rows with VIRAC2                          : {metric('has_virac2')}")
    lines.append(f"Rows with 2MASS                           : {metric('has_2mass')}")
    lines.append(f"Rows with reddening                       : {metric('has_reddening')}")
    lines.append(f"Rows with dereddened J0/H0/Ks0            : {metric('has_dereddened_phot')}")
    lines.append(f"Calibration-ready rows                    : {len(ready)}")
    lines.append(f"Unique sightlines                         : {len(sightlines)}")

    if len(master) > 0:
        lines.append(f"Calibration-ready fraction                : {len(ready)/len(master):.4f}")

    if "ext_e_jks" in master.columns:
        ejks = as_numeric(master["ext_e_jks"]).to_numpy(dtype=float)
        if np.isfinite(ejks).any():
            lines.append("")
            lines.append("Reddening statistics")
            lines.append("-" * 20)
            lines.append(f"E(J-Ks) median                           : {np.nanmedian(ejks):.4f}")
            lines.append(f"E(J-Ks) 16th percentile                  : {np.nanpercentile(ejks, 16):.4f}")
            lines.append(f"E(J-Ks) 84th percentile                  : {np.nanpercentile(ejks, 84):.4f}")

    if len(phot_sources) > 0:
        lines.append("")
        lines.append("Photometry source mix")
        lines.append("-" * 20)
        for _, row in phot_sources.iterrows():
            lines.append(
                f"{row['phot_source_overall']:<10} : {int(row['count']):>4d} ({float(row['fraction_of_total']):.3f})"
            )

    if len(sightlines) > 0:
        lines.append("")
        lines.append("Top sightlines by row count")
        lines.append("-" * 26)
        top = sightlines.head(10)
        for _, row in top.iterrows():
            lines.append(
                f"{str(row['sightline_id'])[:60]:<60}  n={int(row['n_total']):>3d}  ready={int(row['n_calibration_ready']):>3d}"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def write_table(df: pd.DataFrame, fits_path: pathlib.Path, csv_path: pathlib.Path) -> None:
    table = Table.from_pandas(df)
    table.write(fits_path, overwrite=True)
    df.to_csv(csv_path, index=False)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_sky_coverage(master: pd.DataFrame, output_path: pathlib.Path) -> None:
    l_col = find_first_column(master.columns, ["l"])
    b_col = find_first_column(master.columns, ["b"])
    if l_col is None or b_col is None:
        return


    glon = as_numeric(master[l_col]).to_numpy(dtype=float)
    glat = as_numeric(master[b_col]).to_numpy(dtype=float)
    has_red = master["has_reddening"].to_numpy(dtype=bool)
    
    glon = ((glon + 180.0) % 360.0) - 180.0

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.scatter(glon[~has_red], glat[~has_red], s=18, alpha=0.7, label="No reddening match")
    ax.scatter(glon[has_red], glat[has_red], s=24, alpha=0.85, label="Has reddening match")
    ax.set_xlabel("Galactic longitude l [deg]")
    ax.set_ylabel("Galactic latitude b [deg]")
    ax.set_title("Roman ASTRA sample: reddening coverage")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)



def plot_cmd(master: pd.DataFrame, output_path: pathlib.Path) -> None:
    color = as_numeric(first_present_value(master, ["j0_minus_ks0"]))
    mag = as_numeric(first_present_value(master, ["ks0"]))
    good = np.isfinite(color) & np.isfinite(mag)
    if good.sum() == 0:
        return

    virac = master["has_virac2"].to_numpy(dtype=bool)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(color[good & ~virac], mag[good & ~virac], s=22, alpha=0.75, label="Dereddened, no VIRAC2")
    ax.scatter(color[good & virac], mag[good & virac], s=26, alpha=0.85, label="Dereddened + VIRAC2")
    ax.set_xlabel(r"$(J-K_s)_0$")
    ax.set_ylabel(r"$K_{s,0}$")
    ax.set_title("Dereddened CMD for Roman calibration sample")
    ax.invert_yaxis()
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)



def plot_top_sightlines(sightlines: pd.DataFrame, output_path: pathlib.Path, top_n: int = 20) -> None:
    if len(sightlines) == 0:
        return

    top = sightlines.head(top_n).copy()
    labels = []
    for _, row in top.iterrows():
        tile = str(row.get("sightline_tile", "")).strip()
        if tile and tile.lower() != "nan":
            labels.append(tile)
        else:
            labels.append(str(row["sightline_id"])[:18])

    x = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.bar(x, top["n_total"].to_numpy(dtype=float), label="Total")
    ax.bar(x, top["n_calibration_ready"].to_numpy(dtype=float), label="Calibration-ready")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("Number of stars")
    ax.set_title(f"Top {len(top)} sightlines in Roman sample")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the Roman master calibration catalog, overlap stats, and diagnostic plots."
    )
    parser.add_argument(
        "input_fits",
        nargs="?",
        default="astra_overguide_roman_dereddened.fits",
        help="Input FITS table from roman_deredden_sightlines.py",
    )
    parser.add_argument(
        "--out-prefix",
        default="astra_overguide_roman_calibration",
        help="Prefix for output products",
    )
    parser.add_argument(
        "--top-sightlines",
        type=int,
        default=20,
        help="Number of sightlines to include in the bar-chart plot",
    )
    return parser



def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    input_path = pathlib.Path(args.input_fits).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input FITS file not found: {input_path}")

    out_prefix = pathlib.Path(args.out_prefix)
    if out_prefix.is_absolute():
        stem = out_prefix.name
        out_dir = out_prefix.parent
    else:
        stem = out_prefix.name
        out_dir = pathlib.Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading input table: {input_path}")
    input_table = Table.read(input_path)
    base_df, dropped = table_to_scalar_pandas(input_table)
    if dropped:
        print(
            "Dropping multidimensional columns that cannot be converted to pandas: "
            + ", ".join(dropped)
        )
    print(f"Input rows: {len(base_df)}")

    master = build_master_catalog(base_df)
    ready = master.loc[master["calibration_ready"]].copy().reset_index(drop=True)
    overlap = build_overlap_table(master)
    phot_sources = build_phot_source_table(master)
    sightlines = build_sightline_table(master)

    master_fits = out_dir / f"{stem}_master.fits"
    master_csv = out_dir / f"{stem}_master.csv"
    ready_fits = out_dir / f"{stem}_calibration_ready.fits"
    ready_csv = out_dir / f"{stem}_calibration_ready.csv"
    overlap_csv = out_dir / f"{stem}_overlap_stats.csv"
    phot_csv = out_dir / f"{stem}_phot_source_stats.csv"
    sight_csv = out_dir / f"{stem}_sightline_counts.csv"
    summary_txt = out_dir / f"{stem}_summary.txt"
    sky_png = out_dir / f"{stem}_sky_coverage.png"
    cmd_png = out_dir / f"{stem}_cmd_dereddened.png"
    sight_png = out_dir / f"{stem}_top_sightlines.png"

    print(f"Writing master catalog FITS: {master_fits}")
    write_table(master, master_fits, master_csv)

    print(f"Writing calibration-ready subset FITS: {ready_fits}")
    write_table(ready, ready_fits, ready_csv)

    print(f"Writing overlap statistics CSV: {overlap_csv}")
    overlap.to_csv(overlap_csv, index=False)

    print(f"Writing photometry-source statistics CSV: {phot_csv}")
    phot_sources.to_csv(phot_csv, index=False)

    print(f"Writing sightline counts CSV: {sight_csv}")
    sightlines.to_csv(sight_csv, index=False)

    print(f"Writing summary text: {summary_txt}")
    write_summary_text(master, ready, overlap, phot_sources, sightlines, summary_txt)

    print(f"Writing sky coverage plot: {sky_png}")
    plot_sky_coverage(master, sky_png)

    print(f"Writing dereddened CMD plot: {cmd_png}")
    plot_cmd(master, cmd_png)

    print(f"Writing top-sightlines plot: {sight_png}")
    plot_top_sightlines(sightlines, sight_png, top_n=args.top_sightlines)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
