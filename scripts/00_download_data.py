"""Document how to obtain the (access-gated) TB Portals data and wire it into this repo.

The NIH TB Portals data is governed by a Data Use Agreement and cannot be redistributed. This
script does not download anything; it prints the required manual steps and verifies that the
expected files are reachable from :data:`ctb.config.RAW_DIR`.

Usage
-----
    uv run python scripts/00_download_data.py
"""

from __future__ import annotations

import sys

from ctb.config import RAW_DIR, TB_PORTALS_FILES

INSTRUCTIONS = f"""
TB Portals data acquisition (manual, access-gated)
==================================================
1. Request access at https://tbportals.niaid.nih.gov/ and execute the Data Use Agreement (DUA).
2. Download the "Published Clinical Data" CSV bundle (August 2023 release used here).
3. Place (or symlink) the CSV files so they are reachable at:
       {RAW_DIR}
   e.g.:
       mkdir -p data/raw
       ln -s /path/to/TB_Portals_Published_Clinical_Data_August_2023 \\
             data/raw/tb_portals_aug2023
   Or set the environment variable CTB_RAW_DIR to the folder holding the CSVs.

Compliance reminder: raw row-level data must NEVER be committed or shared. data/ is git-ignored.
"""


def main() -> int:
    """Print acquisition instructions and verify file reachability.

    Returns
    -------
    int
        Process exit code (0 if all files found, 1 otherwise).
    """
    print(INSTRUCTIONS)
    print(f"Checking for expected files under: {RAW_DIR}\n")
    missing = []
    for name, fname in TB_PORTALS_FILES.items():
        path = RAW_DIR / fname
        status = "OK" if path.exists() else "MISSING"
        if not path.exists():
            missing.append(name)
        print(f"  [{status:7s}] {name:16s} -> {fname}")
    if missing:
        print(f"\n{len(missing)} file(s) missing. Follow the steps above before running the ETL.")
        return 1
    print("\nAll expected TB Portals files are reachable. You can run scripts/01_build_dataset.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
