"""Download METR-LA files from public mirrors (requires network)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


URLS = {
    "distances_la.csv": "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/distances_la_2012.csv",
    "graph_sensor_locations.csv": "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_locations.csv",
}


def download_file(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["curl.exe", "-L", "-o", str(dest), url, "--max-time", "120"],
            check=True,
            capture_output=True,
        )
        return dest.exists() and dest.stat().st_size > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--source_adj", type=Path, default=None, help="Copy adj_mx.pkl from existing project")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for name, url in URLS.items():
        dest = args.data_dir / name
        if download_file(url, dest):
            print(f"Downloaded {name}")
            ok += 1
        else:
            print(f"Failed {name}; run generate_synthetic_metr_la.py for fallback distances")

    if args.source_adj and args.source_adj.exists():
        shutil.copy(args.source_adj, args.data_dir / "adj_mx.pkl")
        print("Copied adj_mx.pkl")

    h5 = args.data_dir / "metr-la.h5"
    if not h5.exists():
        print("metr-la.h5 not downloaded automatically. Place file manually or run:")
        print("  python scripts/generate_synthetic_metr_la.py --data_dir data")
        sys.exit(1 if ok == 0 else 0)


if __name__ == "__main__":
    main()
