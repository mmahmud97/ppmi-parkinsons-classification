"""
Subset and consistency checks between MODEL 1 DATASET and
ALL DATA FILES REPOSITORY.

Uses the mapping produced by data_inventory.build_model1_to_all_mapping.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ppmi_config import ALL_DATA_ROOT, MODEL1_ROOT, ensure_data_roots_exist


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV with reasonable defaults and tolerant encoding."""
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin1")


def choose_key_columns(cols: Sequence[str]) -> List[str]:
    """Heuristic selection of key columns for row-level checks."""
    candidates = [
        ["PATNO", "EVENT_ID"],
        ["PATNO", "REC_ID"],
        ["PATNO"],
        ["REC_ID"],
    ]
    colset = set(cols)
    for cand in candidates:
        if all(c in colset for c in cand):
            return cand
    return []  # fall back to row-count-only checks


def compare_files(
    rel_model1: str,
    rel_all: str,
    out_dir: Path,
) -> Dict[str, object]:
    """
    Compare a MODEL 1 file to its ALL DATA counterpart.

    Returns high-level metrics and writes detailed CSVs where useful.
    """
    model1_path = MODEL1_ROOT / rel_model1
    all_path = ALL_DATA_ROOT / rel_all

    df_m = load_csv(model1_path)
    df_a = load_csv(all_path)

    cols_m = set(df_m.columns)
    cols_a = set(df_a.columns)

    shared_cols = sorted(cols_m & cols_a)
    only_in_m = sorted(cols_m - cols_a)
    only_in_a = sorted(cols_a - cols_m)

    key_cols = choose_key_columns(shared_cols)

    result: Dict[str, object] = {
        "rel_path_model1": rel_model1,
        "rel_path_all": rel_all,
        "n_rows_model1": len(df_m),
        "n_rows_all": len(df_a),
        "n_cols_model1": len(df_m.columns),
        "n_cols_all": len(df_a.columns),
        "n_shared_cols": len(shared_cols),
        "n_only_in_model1": len(only_in_m),
        "n_only_in_all": len(only_in_a),
        "key_columns": ",".join(key_cols),
        "row_subset_check_performed": bool(key_cols),
        "all_model1_keys_in_all": None,
        "sample_value_mismatches": np.nan,
    }

    # Save column comparison detail
    col_report = out_dir / "column_reports"
    col_report.mkdir(parents=True, exist_ok=True)
    col_df = pd.DataFrame(
        {
            "column": sorted(cols_m | cols_a),
            "in_model1": [c in cols_m for c in sorted(cols_m | cols_a)],
            "in_all": [c in cols_a for c in sorted(cols_m | cols_a)],
        }
    )
    col_df.to_csv(
        col_report
        .joinpath(rel_model1.replace("/", "__") + "_columns.csv"),
        index=False,
    )

    if not key_cols:
        return result

    # Row-level subset check
    df_m_keyed = df_m.set_index(key_cols)
    df_a_keyed = df_a.set_index(key_cols)

    keys_m = set(df_m_keyed.index)
    keys_a = set(df_a_keyed.index)

    missing_in_all = keys_m - keys_a
    result["all_model1_keys_in_all"] = len(missing_in_all) == 0
    result["n_missing_model1_keys_in_all"] = len(missing_in_all)

    # Optionally save missing keys detail
    if missing_in_all:
        miss_df = (
            df_m_keyed.loc[list(missing_in_all)]
            .reset_index()[key_cols]
        )
        miss_dir = out_dir / "missing_keys"
        miss_dir.mkdir(parents=True, exist_ok=True)
        miss_df.to_csv(
            miss_dir
            .joinpath(rel_model1.replace("/", "__") + "_missing_keys.csv"),
            index=False,
        )

    # Value equality checks on a sample of overlapping rows and non-key columns
    non_key_cols = [c for c in shared_cols if c not in key_cols]
    if keys_m & keys_a and non_key_cols:
        shared_keys = list(keys_m & keys_a)
        sample_size = min(100, len(shared_keys))
        sample_keys = shared_keys[:sample_size]

        sub_m = df_m_keyed.loc[sample_keys, non_key_cols]
        sub_a = df_a_keyed.loc[sample_keys, non_key_cols]

        # Align column order
        sub_a = sub_a[sub_m.columns]

        mismatches = (sub_m != sub_a) & ~(sub_m.isna() & sub_a.isna())
        n_mismatches = mismatches.values.sum()
        result["sample_value_mismatches"] = int(n_mismatches)

        if n_mismatches:
            mismatch_dir = out_dir / "value_mismatches"
            mismatch_dir.mkdir(parents=True, exist_ok=True)
            diff = pd.DataFrame(
                {
                    "key": list(sub_m.index),
                    "n_mismatched_cols": mismatches.sum(axis=1).values,
                }
            )
            diff.to_csv(
                mismatch_dir.joinpath(
                    rel_model1.replace("/", "__") + "_value_mismatches.csv"
                ),
                index=False,
            )

    return result


def run_subset_checks(
    mapping_csv: Path,
    out_dir: Path,
) -> Path:
    """Run subset checks for all mapped MODEL 1 files."""
    ensure_data_roots_exist()
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = pd.read_csv(mapping_csv)
    # For now focus on CSV files to avoid Excel/binary parsing issues.
    if "ext" in mapping.columns:
        mapping = mapping[mapping["ext"] == ".csv"].copy()
    # If there are multiple ALL files per MODEL1 file, keep them all for now.
    results: List[Dict[str, object]] = []

    for _, row in mapping.iterrows():
        rel_m = row["rel_path_model1"]
        rel_a = row["rel_path_all"]
        if not isinstance(rel_m, str) or not isinstance(rel_a, str):
            continue
        print(f"Comparing {rel_m}  <--  {rel_a}")
        res = compare_files(rel_m, rel_a, out_dir=out_dir)
        results.append(res)

    summary = pd.DataFrame(results)
    summary_path = out_dir / "subset_checks_summary.csv"
    summary.to_csv(summary_path, index=False)
    return summary_path


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run subset checks between MODEL 1 DATASET and ALL DATA FILES REPOSITORY."
    )
    parser.add_argument(
        "--mapping-csv",
        type=str,
        required=True,
        help="Path to model1_to_all_mapping.csv produced by data_inventory.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports/subset_checks",
        help="Directory to write subset check reports.",
    )
    args = parser.parse_args(argv)

    mapping_csv = Path(args.mapping_csv)
    out_dir = Path(args.out_dir)

    summary_path = run_subset_checks(mapping_csv, out_dir)
    print(f"Wrote subset check summary to {summary_path}")


if __name__ == "__main__":
    main()


