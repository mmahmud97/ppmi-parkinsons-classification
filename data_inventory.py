"""
Utilities to inventory files in the PPMI folders and map MODEL 1 DATASET
files back to their sources in ALL DATA FILES REPOSITORY.

This focuses on tabular analysis-ready files (CSV / XLSX) but records others
for completeness.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from ppmi_config import ALL_DATA_ROOT, MODEL1_ROOT, ensure_data_roots_exist


DATE_SUFFIX_PATTERN = re.compile(
    r"(_\d{1,2}[A-Za-z]{3}\d{4}|_\d{8})$"
)  # e.g., _13Oct2024 or _20240401


@dataclasses.dataclass
class FileRecord:
    root: str  # "all" or "model1"
    rel_path: str
    domain: str
    filename: str
    stem: str
    ext: str
    base_key: str


def iter_files(root: Path, root_label: str) -> Iterable[FileRecord]:
    """Yield FileRecord entries for all files under a root directory."""
    for path in root.rglob("*"):
        if not path.is_file():
            continue

        rel_path = str(path.relative_to(root))
        filename = path.name
        stem = path.stem
        ext = path.suffix.lower()

        # Domain = first directory component if present
        parts = rel_path.split("/")
        domain = parts[0] if len(parts) > 1 else ""

        base_key = DATE_SUFFIX_PATTERN.sub("", stem)

        yield FileRecord(
            root=root_label,
            rel_path=rel_path,
            domain=domain,
            filename=filename,
            stem=stem,
            ext=ext,
            base_key=base_key,
        )


def build_inventory() -> pd.DataFrame:
    """Return a DataFrame describing all files in both roots."""
    ensure_data_roots_exist()

    records: List[FileRecord] = []
    records.extend(iter_files(ALL_DATA_ROOT, "all"))
    records.extend(iter_files(MODEL1_ROOT, "model1"))

    df = pd.DataFrame([dataclasses.asdict(r) for r in records])
    return df.sort_values(["root", "domain", "rel_path"]).reset_index(drop=True)


def summarize_by_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Simple per-domain summary of file counts by root."""
    return (
        df.groupby(["root", "domain", "ext"])["rel_path"]
        .count()
        .rename("n_files")
        .reset_index()
        .sort_values(["root", "domain", "ext"])
    )


def build_model1_to_all_mapping(
    df: pd.DataFrame,
    prefer_same_filename: bool = True,
) -> pd.DataFrame:
    """
    Heuristic mapping from MODEL 1 files to their likely ALL DATA sources.

    Strategy:
      - Match on extension (csv/xlsx) and base_key.
      - Prefer matches where the filename is identical (if any).
      - Otherwise keep all matches sharing the same base_key and extension.
    """
    model1 = df[df["root"] == "model1"].copy()
    all_df = df[df["root"] == "all"].copy()

    # Only keep tabular types for mapping
    tabular_exts = {".csv", ".xlsx"}
    model1 = model1[model1["ext"].isin(tabular_exts)]
    all_df = all_df[all_df["ext"].isin(tabular_exts)]

    merged = model1.merge(
        all_df,
        on=["base_key", "ext"],
        how="left",
        suffixes=("_model1", "_all"),
    )

    if merged.empty:
        return merged

    # Rank candidates per model1 file
    merged["filename_match"] = (
        merged["filename_model1"] == merged["filename_all"]
    )

    def choose_best(group: pd.DataFrame) -> pd.DataFrame:
        if prefer_same_filename and group["filename_match"].any():
            return group[group["filename_match"]]
        return group

    best = (
        merged.groupby("rel_path_model1", group_keys=False)
        .apply(choose_best)
        .reset_index(drop=True)
    )

    return best


def save_inventory_and_mapping(
    out_dir: Path,
    inventory_csv: str = "inventory_all_vs_model1.csv",
    domain_summary_csv: str = "inventory_domain_summary.csv",
    mapping_csv: str = "model1_to_all_mapping.csv",
) -> Dict[str, Path]:
    """Compute inventory + mapping and save to CSV files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    inv = build_inventory()
    mapping = build_model1_to_all_mapping(inv)
    dom_summary = summarize_by_domain(inv)

    inventory_path = out_dir / inventory_csv
    domain_summary_path = out_dir / domain_summary_csv
    mapping_path = out_dir / mapping_csv

    inv.to_csv(inventory_path, index=False)
    dom_summary.to_csv(domain_summary_path, index=False)
    mapping.to_csv(mapping_path, index=False)

    return {
        "inventory": inventory_path,
        "domain_summary": domain_summary_path,
        "mapping": mapping_path,
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Inventory PPMI data folders and map MODEL 1 files to their sources."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports/inventory",
        help="Directory to write inventory and mapping CSV files.",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)

    paths = save_inventory_and_mapping(out_dir)
    print("Wrote:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()


