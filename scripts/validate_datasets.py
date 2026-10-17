#!/usr/bin/env python3
"""Validate METR-LA / PEMS-BAY data before FULL experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparc_mfd.data_loader import (
    DATASET_CONFIG,
    format_validation_report,
    inspect_dataset,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Validate DCRNN datasets for FULL runs")
    p.add_argument(
        "--dataset",
        choices=["metr-la", "pems-bay", "both"],
        default="both",
    )
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--require-full", action="store_true", help="Exit 1 unless ok_for_full")
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--md-out", type=Path, default=None)
    args = p.parse_args()

    names = list(DATASET_CONFIG) if args.dataset == "both" else [args.dataset]
    reports = [inspect_dataset(args.data_root / name, name) for name in names]

    md = format_validation_report(reports)
    print(md)

    payload = {
        name: {
            **report.to_provenance(),
            "ok_for_full": report.ok_for_full(),
            "messages": report.messages,
        }
        for name, report in zip(names, reports)
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")

    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(md, encoding="utf-8")
        print(f"Wrote {args.md_out}")

    if args.require_full and not all(r.ok_for_full() for r in reports):
        print("\nVALIDATION FAILED: synthetic=true or real DCRNN files missing.", file=sys.stderr)
        sys.exit(1)

    if all(r.ok_for_full() for r in reports):
        print("\nVALIDATION OK: synthetic=false, data_source=real_dcrnn")


if __name__ == "__main__":
    main()
