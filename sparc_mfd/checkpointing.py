"""Multi-stage pipeline checkpoints with metadata validation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .experiment import ExperimentConfig, RoutingAccumulator


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def run_mode(quick: bool) -> str:
    return "quick" if quick else "full"


def resolve_output_dir(dataset: str, quick: bool) -> Path:
    return Path(f"outputs/{dataset}/{run_mode(quick)}")


def protocol_config_dict(cfg: "ExperimentConfig") -> Dict[str, Any]:
    return {
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "quick": cfg.quick,
        "sparsity_levels": list(cfg.sparsity_levels),
        "n_od_pairs": cfg.n_od_pairs,
        "od_batch_size": cfg.od_batch_size,
        "seq_len": cfg.seq_len,
        "epochs": cfg.epochs,
        "hidden_dim": cfg.hidden_dim,
        "batch_size": cfg.batch_size,
    }


def protocol_hash(cfg: "ExperimentConfig") -> str:
    payload = json.dumps(protocol_config_dict(cfg), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def checkpoint_envelope(
    cfg: "ExperimentConfig",
    stage: str,
    status: str,
    payload: Optional[Dict[str, Any]] = None,
    partial_metrics: Optional[Dict[str, Any]] = None,
    completed_batches: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "dataset": cfg.dataset,
        "run_mode": run_mode(cfg.quick),
        "configuration": protocol_config_dict(cfg),
        "seed": cfg.seed,
        "code_version": _git_sha(),
        "protocol_hash": protocol_hash(cfg),
        "stage": stage,
        "status": status,
        "completed_batches": completed_batches or {},
        "partial_metrics": partial_metrics or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload or {},
    }


class PipelineCheckpoints:
    def __init__(self, cfg: "ExperimentConfig"):
        self.cfg = cfg
        self.dir = cfg.output_dir / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "pipeline_state.json"

    def _stage_path(self, stage: str) -> Path:
        names = {
            "E0": "e0_prediction.json",
            "E1": "e1_imputation.json",
            "E2": "e2_mfd.json",
            "E3": "e3_routing.json",
            "E4": "e4_assignment.json",
            "E5": "e5_decision_support.json",
        }
        return self.dir / names[stage]

    def load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {"completed_stages": [], "protocol_hash": protocol_hash(self.cfg)}
        with open(self.state_path, encoding="utf-8") as f:
            return json.load(f)

    def _validate_envelope(self, data: Dict[str, Any]) -> bool:
        if data.get("protocol_hash") != protocol_hash(self.cfg):
            return False
        if data.get("seed") != self.cfg.seed:
            return False
        if data.get("dataset") != self.cfg.dataset:
            return False
        if data.get("run_mode") != run_mode(self.cfg.quick):
            return False
        return True

    def completed_stages(self) -> List[str]:
        if not self.cfg.resume:
            return []
        state = self.load_state()
        if state.get("protocol_hash") != protocol_hash(self.cfg):
            print("  [CKPT] protocol_hash mismatch - ignoring prior checkpoints")
            return []
        return list(state.get("completed_stages", []))

    def is_done(self, stage: str) -> bool:
        return stage in self.completed_stages()

    def save_stage(
        self,
        stage: str,
        payload: Dict[str, Any],
        partial_metrics: Optional[Dict[str, Any]] = None,
        completed_batches: Optional[Dict[str, Any]] = None,
        status: str = "completed",
        quiet: bool = False,
    ) -> None:
        envelope = checkpoint_envelope(
            self.cfg, stage, status, payload, partial_metrics, completed_batches
        )
        path = self._stage_path(stage)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(envelope, f, indent=2)
        state = self.load_state()
        done = set(state.get("completed_stages", []))
        if status == "completed":
            done.add(stage)
        state = {
            "dataset": self.cfg.dataset,
            "run_mode": run_mode(self.cfg.quick),
            "protocol_hash": protocol_hash(self.cfg),
            "completed_stages": sorted(done),
            "last_stage": stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        if not quiet:
            print(f"  [CKPT] {stage} saved -> {path.name}")

    def load_stage(self, stage: str) -> Optional[Dict[str, Any]]:
        path = self._stage_path(stage)
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not self._validate_envelope(data):
            print(f"  [CKPT] {stage} envelope invalid - skipping resume")
            return None
        return data.get("payload", data)

    def save_e0_artifacts(
        self,
        model: torch.nn.Module,
        mu: np.ndarray,
        sigma: np.ndarray,
        emb: np.ndarray,
        e0_metrics: Dict[str, float],
    ) -> None:
        torch.save(model.state_dict(), self.dir / "encoder.pt")
        np.save(self.dir / "mu.npy", mu)
        np.save(self.dir / "sigma.npy", sigma)
        np.save(self.dir / "emb.npy", emb)
        self.save_stage(
            "E0",
            {"metrics": e0_metrics, "artifacts": ["encoder.pt", "mu.npy", "sigma.npy", "emb.npy"]},
        )

    def load_e0_artifacts(self, model: torch.nn.Module, device: str) -> Optional[tuple]:
        if not (self.dir / "encoder.pt").exists():
            return None
        payload = self.load_stage("E0")
        if payload is None:
            return None
        model.load_state_dict(torch.load(self.dir / "encoder.pt", map_location=device, weights_only=True))
        mu = np.load(self.dir / "mu.npy")
        sigma = np.load(self.dir / "sigma.npy")
        emb = np.load(self.dir / "emb.npy")
        print("  [CKPT] E0 artifacts loaded")
        return mu, sigma, emb, payload.get("metrics", {})

    def save_od_pairs(self, od_pairs: List[tuple], meta: Optional[Dict] = None) -> None:
        payload = {"od_pairs": [list(p) for p in od_pairs], "meta": meta or {}}
        with open(self.dir / "od_pairs.json", "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _read_od_pairs_payload(self) -> Optional[Dict]:
        p = self.dir / "od_pairs.json"
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        # Pre-topology checkpoints stored a bare list of pairs.
        if isinstance(raw, list):
            return {"od_pairs": raw, "meta": {}}
        return raw

    def load_od_pairs(self, topology: Optional[str] = None) -> Optional[List[tuple]]:
        """Cached OD pairs, or None when they were drawn on another topology.

        Pairs sampled on a different graph may be unreachable on this one, which
        would silently poison the routing means, so a mismatch invalidates them.
        """
        payload = self._read_od_pairs_payload()
        if payload is None:
            return None
        if topology is not None:
            cached = payload.get("meta", {}).get("topology")
            if cached != topology:
                print(
                    f"  [CKPT] OD pairs were drawn on topology '{cached}', "
                    f"now running '{topology}' - resampling"
                )
                return None
        return [(int(o), int(d)) for o, d in payload.get("od_pairs", [])]

    def load_od_pairs_meta(self) -> Dict:
        payload = self._read_od_pairs_payload()
        return (payload or {}).get("meta", {})

    def save_e3_progress(
        self,
        acc: "RoutingAccumulator",
        batch_idx: int,
        batches_total: int,
        timestep_idx: Optional[int] = None,
        timesteps_per_batch: Optional[int] = None,
        quiet: bool = False,
    ) -> None:
        payload = acc.to_checkpoint_dict(
            self.cfg.dataset, batch_idx, batches_total, timestep_idx, timesteps_per_batch
        )
        done_batches = batch_idx + 1
        in_progress = timestep_idx is not None and (
            timestep_idx + 1 < (timesteps_per_batch or 0)
            or batch_idx + 1 < batches_total
        )
        status = "in_progress" if in_progress else "completed"
        self.save_stage(
            "E3",
            payload,
            partial_metrics={"n_samples": dict(acc.rii_count)},
            completed_batches={"E3": {"done": done_batches, "total": batches_total}},
            status=status,
            quiet=quiet,
        )
        if not quiet:
            if timestep_idx is not None and timesteps_per_batch:
                print(
                    f"  [CKPT] E3 batch {batch_idx + 1}/{batches_total} "
                    f"timestep {timestep_idx + 1}/{timesteps_per_batch} saved"
                )
            else:
                print(f"  [CKPT] E3 batch {batch_idx + 1}/{batches_total} saved")

    def load_e3_accumulator(self, method_names: List[str], timesteps_per_batch: int = 0) -> tuple:
        from .experiment import RoutingAccumulator

        data = self.load_stage("E3")
        if data is None:
            return RoutingAccumulator.init(method_names), 0, 0
        acc = RoutingAccumulator.from_checkpoint(data, method_names)
        last_batch = int(data.get("last_batch_idx", -1))
        last_ts = data.get("last_timestep_idx")
        n_ts = int(data.get("timesteps_per_batch", timesteps_per_batch or 0))
        total = int(data.get("batches_total", 1))

        if last_ts is None:
            start_batch = last_batch + 1
            start_timestep = 0
        elif n_ts > 0 and last_ts + 1 < n_ts:
            start_batch = last_batch
            start_timestep = last_ts + 1
        else:
            start_batch = last_batch + 1
            start_timestep = 0

        if start_batch < total:
            if start_timestep > 0:
                print(
                    f"  [E3] Resuming batch {start_batch + 1}/{total} "
                    f"from timestep {start_timestep + 1}/{n_ts}"
                )
            else:
                print(f"  [E3] Resuming routing from batch {start_batch + 1}/{total}")
        return acc, start_batch, start_timestep
