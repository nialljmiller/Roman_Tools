#!/usr/bin/env python3
"""
Generate a broad suite of diagnostic plots for the Roman calibration sample.

Designed to run after roman_calibration_catalog.py, for example:

    python roman_calibration_diagnostics.py \
        astra_overguide_roman_calibration_master.fits \
        --outdir roman_calibration_plots

Outputs a batch of PNG figures plus a manifest text file listing what was made.
The script is intentionally defensive: if some columns are missing, it skips only
those plots that require them.
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
    return {str(c).lower(): str(c) for c in columns}


def find_first_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    mapping = _lower_name_map(columns)
    for candidate in candidates:
        key = str(candidate).lower()
        if key in mapping:
            return mapping[key]
    return None


def as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_text(series: pd.Series) -> pd.Series:
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


def get_series(df: pd.DataFrame, candidates: Iterable[str], default=np.nan) -> pd.Series:
    col = find_first_column(df.columns, candidates)
    if col is None:
        return pd.Series(default, index=df.index)
    return df[col]


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


def finite_mask(*arrays: pd.Series | np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        vals = np.asarray(arr, dtype=float)
        mask &= np.isfinite(vals)
    return mask


# -----------------------------------------------------------------------------
# Plot functions
# -----------------------------------------------------------------------------

def plot_glon_glat_by_status(df: pd.DataFrame, outdir: pathlib.Path) -> Optional[str]:
    l = as_numeric(get_series(df, ["l", "glon"]))
    b = as_numeric(get_series(df, ["b", "glat"]))
    has_red = ensure_bool(df, ["has_reddening"])
    has_virac2 = ensure_bool(df, ["has_virac2"])
    cal_ready = ensure_bool(df, ["calibration_ready"])
    good = finite_mask(l, b)
    if good.sum() == 0:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(l[good & ~has_red], b[good & ~has_red], s=20, alpha=0.65, label="No reddening")
    ax.scatter(l[good & has_red & ~cal_ready], b[good & has_red & ~cal_ready], s=24, alpha=0.8, label="Has reddening")
    ax.scatter(l[good & cal_ready], b[good & cal_ready], s=30, alpha=0.9, label="Calibration-ready")
    ax.set_xlabel("Galactic longitude l [deg]")
    ax.set_ylabel("Galactic latitude b [deg]")
    ax.set_title("Roman sample sky coverage by status")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    path = outdir / "sky_status.png"
    savefig(fig, path)
    return path.name


def plot_glon_glat_colored(df: pd.DataFrame, outdir: pathlib.Path, zcol_candidates: list[str], filename: str, title: str, cbar_label: str) -> Optional[str]:
    l = as_numeric(get_series(df, ["l", "glon"]))
    b = as_numeric(get_series(df, ["b", "glat"]))
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
    teff = as_numeric(get_series(df, ["teff"]))
    logg = as_numeric(get_series(df, ["logg"]))
    red = as_numeric(get_series(df, ["ext_e_jks"]))
    good = finite_mask(teff, logg)
    if good.sum() == 0:
        return None

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    if np.isfinite(np.asarray(red, dtype=float)).sum() > 0:
        good2 = good & np.isfinite(np.asarray(red, dtype=float))
        sc = ax.scatter(teff[good2], logg[good2], c=red[good2], s=26, alpha=0.85)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("E(J-Ks)")
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
    teff = as_numeric(get_series(df, ["teff"]))
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
        (["tmass_match_sep_arcsec", "2mass_match_sep_arcsec"], "tmass_match_sep_hist.png", "2MASS match separation", "Separation [arcsec]"),
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
    subset = df.loc[good & sid.isin(top_ids)].copy()
    subset["sightline_id_short"] = clean_text(subset[find_first_column(subset.columns, ["sightline_id"])])
    subset["sightline_id_short"] = subset["sightline_id_short"].str.slice(0, 18)

    groups = [as_numeric(subset.loc[subset["sightline_id_short"] == s, get_series(subset, ["ext_e_jks"]).name]) for s in subset["sightline_id_short"].dropna().unique()]
    labels = list(subset["sightline_id_short"].dropna().unique())
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
    for coord_candidates, fname, xlabel in [(["l", "glon"], "ejks_vs_l.png", "Galactic longitude l [deg]"), (["b", "glat"], "ejks_vs_b.png", "Galactic latitude b [deg]")]:
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
    h = as_numeric(get_series(df, ["h_mag", "h_mag_best"]))
    teff = as_numeric(get_series(df, ["teff"]))
    ready = ensure_bool(df, ["calibration_ready"])
    good = finite_mask(h, teff)
    if good.sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    ax.scatter(teff[good & ~ready], h[good & ~ready], s=22, alpha=0.7, label="Not ready")
    ax.scatter(teff[good & ready], h[good & ready], s=28, alpha=0.85, label="Calibration-ready")
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


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate a broad diagnostics plot suite for the Roman calibration sample.")
    p.add_argument("input_fits", nargs="?", default="astra_overguide_roman_calibration_master.fits")
    p.add_argument("--outdir", default="roman_calibration_plots")
    return p


def main() -> int:
    args = build_parser().parse_args()
    input_path = pathlib.Path(args.input_fits).expanduser().resolve()
    outdir = pathlib.Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Reading input table: {input_path}")
    table = Table.read(input_path)
    df, dropped = table_to_scalar_pandas(table)
    if dropped:
        print("Dropped multidimensional columns: " + ", ".join(dropped))
    print(f"Input rows: {len(df)}")

    produced: list[str] = []

    for item in [
        plot_glon_glat_by_status(df, outdir),
        plot_glon_glat_colored(df, outdir, ["ext_e_jks"], "sky_reddening.png", "Sky map colored by reddening", "E(J-Ks)"),
        plot_glon_glat_colored(df, outdir, ["teff"], "sky_teff.png", "Sky map colored by Teff", "Teff [K]"),
        plot_glon_glat_colored(df, outdir, ["logg"], "sky_logg.png", "Sky map colored by log g", "log g"),
        plot_cmd(df, outdir, ["j_minus_ks_best"], ["ks_mag_best", "h_mag"], "cmd_observed_jks_ks.png", "Observed CMD", "J-Ks", "Ks", color_by=["ext_e_jks"], cbar_label="E(J-Ks)"),
        plot_cmd(df, outdir, ["j0_minus_ks0"], ["ks0"], "cmd_dereddened_jks0_ks0.png", "Dereddened CMD", "(J-Ks)0", "Ks0", color_by=["teff"], cbar_label="Teff [K]"),
        plot_cmd(df, outdir, ["j_minus_h_best"], ["h_mag_best", "h_mag"], "cmd_observed_jh_h.png", "Observed J-H versus H", "J-H", "H", color_by=["ext_e_jks"], cbar_label="E(J-Ks)"),
        plot_cmd(df, outdir, ["j0_minus_h0"], ["h0"], "cmd_dereddened_jh0_h0.png", "Dereddened J-H versus H0", "(J-H)0", "H0", color_by=["teff"], cbar_label="Teff [K]"),
        plot_teff_logg(df, outdir),
        plot_histogram(df, outdir, ["teff"], "hist_teff.png", "Teff distribution", "Teff [K]"),
        plot_histogram(df, outdir, ["logg"], "hist_logg.png", "log g distribution", "log g"),
        plot_histogram(df, outdir, ["h_mag", "h_mag_best"], "hist_hmag.png", "H distribution", "H"),
        plot_histogram(df, outdir, ["ext_e_jks"], "hist_ejks.png", "Reddening distribution", "E(J-Ks)"),
        plot_histogram(df, outdir, ["j0_minus_ks0"], "hist_j0_minus_ks0.png", "Dereddened color distribution", "(J-Ks)0"),
        plot_hist_by_group(df, outdir, ["ext_e_jks"], ["phot_source_overall"], "hist_ejks_by_photsource.png", "Reddening by photometry source", "E(J-Ks)"),
        plot_hist_by_group(df, outdir, ["teff"], ["phot_source_overall"], "hist_teff_by_photsource.png", "Teff by photometry source", "Teff [K]"),
        plot_sightline_counts(df, outdir),
        plot_sightline_reddening(df, outdir),
        plot_color_color(df, outdir, dered=False),
        plot_color_color(df, outdir, dered=True),
        plot_h_vs_teff_by_ready(df, outdir),
    ]:
        if item is None:
            continue
        produced.append(item)

    produced.extend(plot_teff_vs_colors(df, outdir))
    produced.extend(plot_match_separation(df, outdir))
    produced.extend(plot_reddening_vs_position(df, outdir))
    produced.extend(plot_dereddening_shift(df, outdir))

    manifest = outdir / "manifest.txt"
    lines = [
        "Roman calibration diagnostics plot manifest",
        "=" * 42,
        f"Input file : {input_path}",
        f"Rows       : {len(df)}",
        "",
        "Plots written:",
    ]
    for name in sorted(produced):
        lines.append(f"- {name}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(produced)} plots to: {outdir}")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
