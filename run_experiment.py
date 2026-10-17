#!/usr/bin/env python3
"""Run SPARC-OTS experiments on METR-LA and PEMS-BAY."""

from __future__ import annotations

import argparse
from pathlib import Path

from sparc_mfd.checkpointing import resolve_output_dir
from sparc_mfd.experiment import ExperimentConfig, run_all, run_dataset


def main() -> None:
    p = argparse.ArgumentParser(description="SPARC-OTS experiments")
    p.add_argument("--dataset", choices=["metr-la", "pems-bay", "both"], default="both")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--n-od-pairs", type=int, default=None, help="Total OD pairs for E3 routing")
    p.add_argument("--od-batch-size", type=int, default=None, help="OD pairs per batch (checkpoint after each)")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Resume from checkpoints")
    p.add_argument("--no-routing-checkpoint", action="store_true", help="Disable incremental checkpoints")
    p.add_argument("--no-profile", action="store_true", help="Disable performance profiling")
    p.add_argument("--benchmark-e1", action="store_true", help="Run E1 legacy vs KrigingSolver benchmark only")
    p.add_argument(
        "--validate-data",
        action="store_true",
        help="Validate datasets and exit (use --require-full to block synthetic)",
    )
    p.add_argument(
        "--require-real-data",
        action="store_true",
        help="With --validate-data: exit 1 unless real DCRNN files are present",
    )
    p.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow synthetic fallback even in FULL mode (dev only)",
    )
    p.add_argument(
        "--routing-backend",
        choices=["scipy", "python", "legacy"],
        default="scipy",
        help="E3 Dijkstra backend (default: scipy)",
    )
    p.add_argument(
        "--routing-topology",
        choices=["auto", "csv_direct", "adj_mx", "legacy_dense"],
        default="auto",
        help="E3 sparse graph source (default: auto from regression report)",
    )
    p.add_argument(
        "--od-workers",
        type=int,
        default=0,
        help="Parallel OD workers (0=auto, 1=serial)",
    )
    args = p.parse_args()

    if args.validate_data:
        import subprocess
        import sys

        cmd = [sys.executable, str(Path(__file__).parent / "scripts" / "validate_datasets.py")]
        if args.dataset != "both":
            cmd.extend(["--dataset", args.dataset])
        if args.require_real_data:
            cmd.append("--require-full")
        subprocess.check_call(cmd)
        return

    if not args.quick and not args.allow_synthetic:
        import subprocess
        import sys

        datasets = ["metr-la", "pems-bay"] if args.dataset == "both" else [args.dataset]
        for ds in datasets:
            subprocess.check_call(
                [
                    sys.executable,
                    str(Path(__file__).parent / "scripts" / "validate_datasets.py"),
                    "--dataset",
                    ds,
                    "--require-full",
                ]
            )

    if args.benchmark_e1:
        import subprocess
        import sys

        cmd = [sys.executable, str(Path(__file__).parent / "scripts" / "benchmark_e1_kriging.py")]
        if args.dataset != "both":
            cmd.extend(["--dataset", args.dataset])
        if args.quick:
            cmd.append("--quick")
        subprocess.check_call(cmd)
        return

    base_kwargs = dict(
        quick=args.quick,
        device=args.device,
        epochs=args.epochs,
        resume=args.resume,
        routing_checkpoint=not args.no_routing_checkpoint,
        profile=not args.no_profile,
        allow_synthetic=args.allow_synthetic,
        routing_backend=args.routing_backend,
        routing_topology=args.routing_topology,
        od_workers=args.od_workers,
    )
    if args.n_od_pairs is not None:
        base_kwargs["n_od_pairs"] = args.n_od_pairs
    if args.od_batch_size is not None:
        base_kwargs["od_batch_size"] = args.od_batch_size

    if args.dataset == "both":
        base = ExperimentConfig(**base_kwargs)
        results = run_all(base)
    else:
        cfg = ExperimentConfig(
            dataset=args.dataset,
            data_dir=Path(f"data/{args.dataset}"),
            output_dir=resolve_output_dir(args.dataset, args.quick),
            **base_kwargs,
        )
        results = {args.dataset: run_dataset(cfg)}

    mode = "quick" if args.quick else "full"
    for ds, r in results.items():
        e0 = r.get("E0_prediction", {})
        e3 = r.get("E3_routing", {})
        print(f"[{ds}/{mode}] OTS MAE={e0.get('mae', 0):.4f} | RII reduction={e3.get('risk_reduction_pct', 0):.1f}%")
    print(f"Results saved under outputs/{{dataset}}/{mode}/")


if __name__ == "__main__":
    main()
