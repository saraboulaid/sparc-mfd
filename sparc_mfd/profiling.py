"""Pipeline stage profiling and performance reports."""

from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StageRecord:
    stage: str
    wall_seconds: float
    peak_ram_bytes: int
    sub_stages: Dict[str, float] = field(default_factory=dict)
    top_functions: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


class StageProfiler:
    """Wall-clock + tracemalloc peak per pipeline stage."""

    def __init__(self) -> None:
        self.records: List[StageRecord] = []
        self._stage: Optional[str] = None
        self._t0: float = 0.0
        self._sub: Dict[str, float] = {}
        self._sub_t0: float = 0.0
        self._sub_name: Optional[str] = None

    def start(self, stage: str) -> None:
        if self._stage is not None:
            self.end()
        self._stage = stage
        self._sub = {}
        self._t0 = time.perf_counter()
        tracemalloc.start()

    def end(self, extra: Optional[Dict[str, Any]] = None) -> StageRecord:
        if self._stage is None:
            raise RuntimeError("No active stage")
        if self._sub_name is not None:
            self.end_sub()
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        rec = StageRecord(
            stage=self._stage,
            wall_seconds=time.perf_counter() - self._t0,
            peak_ram_bytes=int(peak),
            sub_stages=dict(self._sub),
            extra=extra or {},
        )
        self.records.append(rec)
        self._stage = None
        return rec

    def start_sub(self, name: str) -> None:
        if self._sub_name is not None:
            self.end_sub()
        self._sub_name = name
        self._sub_t0 = time.perf_counter()

    def end_sub(self) -> None:
        if self._sub_name is None:
            return
        self._sub[self._sub_name] = time.perf_counter() - self._sub_t0
        self._sub_name = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"stages": []}
        for r in self.records:
            out["stages"].append(
                {
                    "stage": r.stage,
                    "wall_seconds": round(r.wall_seconds, 4),
                    "peak_ram_mb": round(r.peak_ram_bytes / (1024 * 1024), 2),
                    "sub_stages": {k: round(v, 4) for k, v in r.sub_stages.items()},
                    "extra": r.extra,
                }
            )
        if self.records:
            out["total_wall_seconds"] = round(sum(r.wall_seconds for r in self.records), 4)
            out["peak_ram_mb"] = round(max(r.peak_ram_bytes for r in self.records) / (1024 * 1024), 2)
        return out

    def write_reports(self, json_path: Path, md_path: Optional[Path] = None) -> None:
        data = self.to_dict()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        if md_path is not None:
            lines = ["# Performance Report\n"]
            for s in data.get("stages", []):
                lines.append(f"## {s['stage']}\n")
                lines.append(f"- **Temps:** {s['wall_seconds']:.2f} s")
                lines.append(f"- **RAM peak:** {s['peak_ram_mb']:.2f} MB")
                if s.get("sub_stages"):
                    lines.append("- **Sous-étapes:**")
                    for k, v in s["sub_stages"].items():
                        lines.append(f"  - {k}: {v:.2f} s")
                if s.get("extra"):
                    lines.append(f"- **Extra:** `{s['extra']}`")
                lines.append("")
            if "total_wall_seconds" in data:
                lines.append(f"**Total:** {data['total_wall_seconds']:.2f} s | peak RAM {data.get('peak_ram_mb', 0):.2f} MB")
            md_path.write_text("\n".join(lines), encoding="utf-8")
