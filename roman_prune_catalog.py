#!/usr/bin/env python3
"""
Conservatively prune obviously non-science bookkeeping columns from a Roman
calibration FITS catalog, while preserving all astrophysical content.

This script ONLY drops columns that were explicitly identified as safe to
remove:
  - SDSS/APOGEE target-selection flags
  - ASTRA task / execution bookkeeping columns

It does NOT drop metallicities, abundances, reddening, Gaia AP params,
BDBS columns, match metadata, or raw/calibrated stellar parameters.

Example
-------
python roman_prune_catalog.py \
    astra_overguide_roman_calibration_master.fits \
    --output astra_overguide_roman_calibration_master_pruned.fits
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Iterable

from astropy.table import Table


# -----------------------------------------------------------------------------
# Columns that are safe to remove for this workflow
# -----------------------------------------------------------------------------

DROP_COLUMNS: list[str] = [
    # SDSS / APOGEE target-selection baggage
    "sdss5_target_flags",
    "sdss4_apogee_target1_flags",
    "sdss4_apogee_target2_flags",
    "sdss4_apogee2_target1_flags",
    "sdss4_apogee2_target2_flags",
    "sdss4_apogee2_target3_flags",
    "sdss4_apogee_member_flags",
    "sdss4_apogee_extra_target_flags",
    # ASTRA task / pipeline bookkeeping
    "lead",
    "version_id",
    "task_pk",
    "source_pk",
    "v_astra",
    "created",
    "t_elapsed",
    "t_overhead",
    "tag",
    "stellar_parameters_task_pk",
    "al_h_task_pk",
    "c_12_13_task_pk",
    "ca_h_task_pk",
    "ce_h_task_pk",
    "c_1_h_task_pk",
    "c_h_task_pk",
    "co_h_task_pk",
    "cr_h_task_pk",
    "cu_h_task_pk",
    "fe_h_task_pk",
    "k_h_task_pk",
    "mg_h_task_pk",
    "mn_h_task_pk",
    "na_h_task_pk",
    "nd_h_task_pk",
    "ni_h_task_pk",
    "n_h_task_pk",
    "o_h_task_pk",
    "p_h_task_pk",
    "si_h_task_pk",
    "s_h_task_pk",
    "ti_h_task_pk",
    "ti_2_h_task_pk",
    "v_h_task_pk",
]


# A modest front-of-table ordering to make the output easier to inspect.
# Any columns not listed here are preserved and appended in their original order.
PREFERRED_FRONT: list[str] = [
    "sdss_id",
    "sdss4_apogee_id",
    "gaia_dr3_source_id",
    "gaia_dr2_source_id",
    "tic_v8_id",
    "catalogid",
    "catalogid21",
    "catalogid25",
    "catalogid31",
    "ra",
    "dec",
    "l",
    "b",
    "plx",
    "e_plx",
    "pmra",
    "e_pmra",
    "pmde",
    "e_pmde",
    "g_mag",
    "bp_mag",
    "rp_mag",
    "j_mag",
    "e_j_mag",
    "h_mag",
    "e_h_mag",
    "k_mag",
    "e_k_mag",
    "teff",
    "e_teff",
    "logg",
    "e_logg",
    "fe_h",
    "e_fe_h",
    "m_h_atm",
    "e_m_h_atm",
    "mass",
    "radius",
    "ebv",
    "e_ebv",
    "ebv_zhang_2023",
    "e_ebv_zhang_2023",
    "ebv_sfd",
    "e_ebv_sfd",
    "ebv_rjce_glimpse",
    "e_ebv_rjce_glimpse",
    "ebv_rjce_allwise",
    "e_ebv_rjce_allwise",
    "ebv_bayestar_2019",
    "e_ebv_bayestar_2019",
    "ebv_edenhofer_2023",
    "e_ebv_edenhofer_2023",
    "zgr_teff",
    "zgr_e_teff",
    "zgr_logg",
    "zgr_e_logg",
    "zgr_fe_h",
    "zgr_e_fe_h",
    "Source_x",
    "RAdeg",
    "DEdeg",
    "Teff_x",
    "logg_x",
    "[Fe/H]_1",
    "Dist",
    "A0",
    "AG_1",
    "E(BP-RP)",
    "Rad",
    "Rad-Flame",
    "Mass-Flame",
    "Age-Flame",
    "NOAId",
    "NOADist",
    "NOADistRank",
    "Flags-NOA",
    "angDist",
    "RAJ2000",
    "DEJ2000",
    "umag",
    "e_umag",
    "gmag_2",
    "e_gmag",
    "imag",
    "e_imag",
    "Au",
    "Ag_2",
    "Ai",
    "D",
    "e_D",
    "[Fe/H]_2",
    "e_[Fe/H]",
    "Separation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drop only the definitely-unneeded bookkeeping columns from a "
            "Roman calibration FITS catalog and write a cleaned FITS file."
        )
    )
    parser.add_argument("input_fits", help="Input FITS table")
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Output FITS path. Default: <input_stem>_pruned.fits in the same "
            "directory"
        ),
    )
    parser.add_argument(
        "--summary",
        help=(
            "Optional text summary path. Default: <output_stem>_summary.txt in "
            "the same directory"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be dropped without writing output",
    )
    return parser.parse_args()



def default_output_path(input_path: pathlib.Path) -> pathlib.Path:
    return input_path.with_name(f"{input_path.stem}_pruned.fits")



def default_summary_path(output_path: pathlib.Path) -> pathlib.Path:
    return output_path.with_name(f"{output_path.stem}_summary.txt")



def reorder_columns(colnames: Iterable[str], preferred_front: Iterable[str]) -> list[str]:
    present_preferred = [name for name in preferred_front if name in colnames]
    remaining = [name for name in colnames if name not in set(present_preferred)]
    return present_preferred + remaining



def build_summary(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    original_ncols: int,
    final_ncols: int,
    removed: list[str],
    missing_requested: list[str],
) -> str:
    lines: list[str] = []
    lines.append("Roman catalog pruning summary")
    lines.append("============================")
    lines.append(f"Input : {input_path}")
    lines.append(f"Output: {output_path}")
    lines.append("")
    lines.append(f"Original columns: {original_ncols}")
    lines.append(f"Final columns   : {final_ncols}")
    lines.append(f"Removed columns : {len(removed)}")
    lines.append("")

    lines.append("Removed column names")
    lines.append("--------------------")
    if removed:
        lines.extend(removed)
    else:
        lines.append("<none>")
    lines.append("")

    lines.append("Requested drop columns not present in input")
    lines.append("--------------------------------------------")
    if missing_requested:
        lines.extend(missing_requested)
    else:
        lines.append("<none>")
    lines.append("")

    return "\n".join(lines)



def main() -> None:
    args = parse_args()

    input_path = pathlib.Path(args.input_fits).expanduser().resolve()
    output_path = pathlib.Path(args.output).expanduser().resolve() if args.output else default_output_path(input_path)
    summary_path = pathlib.Path(args.summary).expanduser().resolve() if args.summary else default_summary_path(output_path)

    print(f"Reading input table: {input_path}")
    table = Table.read(input_path)
    original_colnames = list(table.colnames)
    original_ncols = len(original_colnames)
    print(f"Input rows   : {len(table)}")
    print(f"Input columns: {original_ncols}")

    removed = [name for name in DROP_COLUMNS if name in table.colnames]
    missing_requested = [name for name in DROP_COLUMNS if name not in table.colnames]

    print(f"Columns marked for removal and present: {len(removed)}")
    if removed:
        for name in removed:
            print(f"  DROP {name}")

    if args.dry_run:
        print("Dry run requested; no files written.")
        return

    if removed:
        table.remove_columns(removed)

    new_order = reorder_columns(table.colnames, PREFERRED_FRONT)
    table = table[new_order]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing cleaned FITS: {output_path}")
    table.write(output_path, overwrite=True)

    summary_text = build_summary(
        input_path=input_path,
        output_path=output_path,
        original_ncols=original_ncols,
        final_ncols=len(table.colnames),
        removed=removed,
        missing_requested=missing_requested,
    )
    print(f"Writing summary text: {summary_path}")
    summary_path.write_text(summary_text, encoding="utf-8")

    print("Done.")
    print(f"Final columns: {len(table.colnames)}")


if __name__ == "__main__":
    main()
