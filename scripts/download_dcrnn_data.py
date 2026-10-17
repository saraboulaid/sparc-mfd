#!/usr/bin/env python3
"""
Download / prepare real DCRNN datasets for SPARC-OTS FULL runs.

Official sources:
  - Google Drive (metr-la.h5 + pems-bay.h5):
    https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX
  - Zenodo PEMS-BAY mirror (pems-bay.h5 + adj_mx_bay.pkl):
    https://zenodo.org/records/4263971
  - DCRNN sensor locations (optional):
    https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparc_mfd.data_loader import inspect_dataset

DCRNN_GDRIVE = "https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX"
ZENODO_PEMS = "https://zenodo.org/records/4263971"

ZENODO_PEMS_FILES = {
    "pems-bay.h5": f"{ZENODO_PEMS}/files/pems-bay.h5?download=1",
    "adj_mx_bay.pkl": f"{ZENODO_PEMS}/files/adj_mx_bay.pkl?download=1",
}

DCRNN_RAW = {
    "distances_la.csv": "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/distances_la_2012.csv",
    "distances_bay.csv": "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/distances_bay_2017.csv",
    "graph_sensor_locations.csv": "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_locations.csv",
    "graph_sensor_locations_bay.csv": "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_locations_bay.csv",
    "graph_sensor_ids.txt": "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_ids.txt",
}


def backup_synthetic_distances(data_dir: Path, dist_name: str) -> None:
    path = data_dir / dist_name
    if not path.exists():
        return
    from sparc_mfd.data_loader import _inspect_distances_csv, DATASET_CONFIG

    name = "metr-la" if "la" in dist_name else "pems-bay"
    meta = _inspect_distances_csv(path, DATASET_CONFIG[name]["n_sensors"])
    if not meta["synthetic"]:
        print(f"  keep distances: {dist_name} ({meta['rows']} rows, official)")
        return
    backup = path.with_name(path.stem + "_synthetic" + path.suffix)
    if backup.exists():
        path.unlink()
        print(f"  removed duplicate synthetic distances: {dist_name}")
        return
    shutil.move(str(path), str(backup))
    print(f"  renamed synthetic distances -> {backup.name}")


def download_file(url: str, dest: Path, timeout: int = 3600) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists): {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    try:
        subprocess.run(
            ["curl.exe", "-L", "-o", str(dest), url, "--max-time", str(timeout)],
            check=True,
            capture_output=True,
        )
        ok = dest.exists() and dest.stat().st_size > 0
        if ok:
            print(f"  downloaded: {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return ok
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  failed: {dest.name} ({exc})")
        return False


def _is_real_h5(data_dir: Path, name: str, filename: str) -> bool:
    path = data_dir / filename
    if not path.exists():
        return False
    report = inspect_dataset(data_dir, name)
    return not report.synthetic and report.ok_for_full()


def backup_synthetic_h5(data_dir: Path, name: str, h5_name: str) -> None:
    path = data_dir / h5_name
    if not path.exists():
        return
    if _is_real_h5(data_dir, name, h5_name):
        print(f"  keep real file: {h5_name}")
        return
    backup = data_dir / h5_name.replace(".h5", "_synthetic.h5")
    if backup.exists():
        print(f"  synthetic backup already exists: {backup.name}")
        path.unlink()
        print(f"  removed duplicate synthetic: {h5_name}")
        return
    shutil.move(str(path), str(backup))
    print(f"  renamed synthetic -> {backup.name}")


def remove_speed_cache(data_dir: Path) -> None:
    npy = data_dir / "speed.npy"
    if npy.exists():
        npy.unlink()
        print("  removed stale speed.npy cache")


def prepare_metr_la(data_dir: Path, skip_download: bool, fetch_optional: bool) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    print("\n[METR-LA]")
    backup_synthetic_h5(data_dir, "metr-la", "metr-la.h5")
    remove_speed_cache(data_dir)
    backup_synthetic_distances(data_dir, "distances_la.csv")
    if not skip_download and fetch_optional:
        download_file(DCRNN_RAW["distances_la.csv"], data_dir / "distances_la.csv")
        download_file(DCRNN_RAW["graph_sensor_locations.csv"], data_dir / "graph_sensor_locations.csv")
        download_file(DCRNN_RAW["graph_sensor_ids.txt"], data_dir / "graph_sensor_ids.txt")
    target = data_dir / "metr-la.h5"
    if not target.exists():
        print(
            "  Download metr-la.h5 manually from the official DCRNN Google Drive:\n"
            f"  {DCRNN_GDRIVE}\n"
            f"  Then copy to: {target}"
        )


def prepare_pems_bay(data_dir: Path, skip_download: bool, fetch_optional: bool) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    print("\n[PEMS-BAY]")
    backup_synthetic_h5(data_dir, "pems-bay", "pems-bay.h5")
    remove_speed_cache(data_dir)
    backup_synthetic_distances(data_dir, "distances_bay.csv")

    if not skip_download:
        download_file(DCRNN_RAW["distances_bay.csv"], data_dir / "distances_bay.csv")
        for zenodo_name, url in ZENODO_PEMS_FILES.items():
            download_file(url, data_dir / zenodo_name)
        bay_adj = data_dir / "adj_mx_bay.pkl"
        target_adj = data_dir / "adj_mx.pkl"
        if bay_adj.exists():
            if target_adj.exists():
                backup_adj = data_dir / "adj_mx_synthetic.pkl"
                if not backup_adj.exists():
                    shutil.copy(target_adj, backup_adj)
                    print("  backed up existing adj_mx.pkl -> adj_mx_synthetic.pkl")
            shutil.copy(bay_adj, target_adj)
            print("  installed real adj_mx.pkl from adj_mx_bay.pkl")

    if fetch_optional and not skip_download:
        download_file(
            DCRNN_RAW["graph_sensor_locations_bay.csv"],
            data_dir / "graph_sensor_locations_bay.csv",
        )

    if not (data_dir / "pems-bay.h5").exists():
        print(
            "  pems-bay.h5 still missing. Options:\n"
            f"  - Zenodo: {ZENODO_PEMS}\n"
            f"  - Google Drive: {DCRNN_GDRIVE}\n"
            f"  Target: {data_dir / 'pems-bay.h5'}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare real DCRNN dataset files")
    p.add_argument("--dataset", choices=["metr-la", "pems-bay", "both"], default="both")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Only backup synthetic files and remove speed.npy caches",
    )
    p.add_argument(
        "--no-optional",
        action="store_true",
        help="Skip optional CSV downloads (distances, sensor locations)",
    )
    args = p.parse_args()
    fetch_optional = not args.no_optional

    if args.dataset in ("metr-la", "both"):
        prepare_metr_la(args.data_root / "metr-la", args.skip_download, fetch_optional)
    if args.dataset in ("pems-bay", "both"):
        prepare_pems_bay(args.data_root / "pems-bay", args.skip_download, fetch_optional)

    print("\nAfter placing real .h5 files, validate before FULL:")
    print("  python scripts/validate_datasets.py --require-full")
    print("\nExpected:")
    print("  METR-LA  shape=[34272, 207]  synthetic=false  data_source=real_dcrnn")
    print("  PEMS-BAY shape=[52116, 325]  synthetic=false  data_source=real_dcrnn")


if __name__ == "__main__":
    main()
