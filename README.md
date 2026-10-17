# SPARC-OTS

**Learning Future Operational Traffic Stress with Graph–Kriging Fusion and MFD-Gated Decision Support for Stability-Aware Routing under Sparse Sensing**

Reproducible pipeline for forecasting short-term traffic instability from speed
measurements alone — no accident or incident labels — and for using that signal in
stability-aware routing, network assignment and operator decision support.

Standalone study, independent of [`stzitd-metrla-label`](https://github.com/saraboulaid/stzitd-metrla-label).

---

## What this is, and what it is not

**OTS (Operational Traffic Stress)** measures *operational degradation of the speed field*:
future volatility, shock magnitude, and persistence of low speeds. It is **not** a crash
probability. No accident data is ingested anywhere in this repository, and no safety
interpretation is supported by these results.

---

## Modules

| Module | Stage | Role |
|---|---|---|
| `sparc_mfd/data_loader.py` | ingest | Archive reading, provenance + detector-order verification |
| `sparc_mfd/features.py` | ingest | Hourly aggregation, dual-branch features, sliding windows |
| `sparc_mfd/graph.py` | ingest | Adjacency loading, distance matrices |
| `sparc_mfd/ots_encoder.py` | E0 | OTS target and dual-branch GRU + graph-attention encoder |
| `sparc_mfd/risk_field.py` | E0 | Proxy stress components (analysis + oracle) |
| `sparc_mfd/sparse_impute.py` | E1 | Spherical variogram, cached kriging solver, masking |
| `sparc_mfd/gkf_fusion.py` | E1 | Learned GNN–kriging fusion gate |
| `sparc_mfd/mfd_regime.py` | E2 | Accumulation proxy and MFD regime classification |
| `sparc_mfd/mgac.py` | E3 | MFD-gated adaptive cost weights |
| `sparc_mfd/routing.py` | E3 | Topologies, edge costs, Dijkstra backends, OD sampling |
| `sparc_mfd/routing_worker.py` | E3 | Per-OD evaluation of all methods + DQI |
| `sparc_mfd/assignment.py` | E4 | Frank–Wolfe network loading |
| `sparc_mfd/decision_support.py` | E5 | Pareto filtering, recommendation, DQI |
| `sparc_mfd/metrics.py` | all | MAE, RMSE, relative regret, Wilcoxon (defined, unused) |
| `sparc_mfd/checkpointing.py` | all | Stage state, resumable E3 accumulators |
| `sparc_mfd/profiling.py` | all | Wall time and peak memory per stage |

### Terminology

- **OTS** — Operational Traffic Stress (speed-only instability target)
- **GKF** — Graph–Kriging Fusion (sparse-sensor imputation)
- **MGAC** — MFD-Gated Adaptive Cost (routing edge cost)
- **RII** — Route Instability Index (sum of predicted stress along a path)
- **DQI** — Decision Quality Index (recovery of the oracle stability gap)

---

## Setup

```bash
pip install -r requirements.txt
```

### Data

The two public DCRNN archives are **not** tracked here (~190 MB). Expected layout:

```text
data/metr-la/   metr-la.h5, adj_mx.pkl, distances_la.csv
data/pems-bay/  pems-bay.h5, adj_mx.pkl, distances_bay.csv
```

```bash
python scripts/download_dcrnn_data.py          # fetch / prepare real archives
python scripts/validate_datasets.py --dataset both --require-full
```

Validation is **mandatory** before a `full` run and enforces four guards: timestep count
above the synthetic-generator signature, adjacency density below 0.5, distance CSVs using
DCRNN detector IDs, and empirical detector-order alignment between the HDF5 columns and the
adjacency rows (edge vs. random speed-correlation margin ≥ 0.03).

Both archives currently pass — see [`outputs/dataset_validation.md`](outputs/dataset_validation.md):

| | METR-LA | PEMS-BAY |
|---|---|---|
| Detectors × timesteps | 207 × 34,272 | 325 × 52,116 |
| Period | 2012-03-01 → 2012-06-27 | 2017-01-01 → 2017-06-30 |
| Adjacency density | 0.0402 | 0.0255 |
| Detector-order margin | 0.1465 | 0.1833 |
| `data_source` | `real_dcrnn` | `real_dcrnn` |

---

## Run

```bash
# Quick smoke test (~10 min CPU, synthetic fallback allowed)
python run_experiment.py --dataset metr-la --quick

# Full run, one dataset
python run_experiment.py --dataset metr-la --n-od-pairs 1000 --od-batch-size 50 --epochs 10

# Resume an interrupted run (default: --resume)
python run_experiment.py --dataset metr-la --resume

# Kriging solver benchmark only
python run_experiment.py --benchmark-e1 --dataset metr-la --quick
```

### Routing graph and OTS horizon

Two options dominate both the realism and the cost of a run:

```bash
# Sparse road graph (mean out-degree 7.3) with measured road distances, 3-step horizon.
# These are now the defaults; `python` is ~3x faster than the `scipy` backend here.
python run_experiment.py --dataset metr-la --routing-topology adj_real \
    --ots-horizon 3 --routing-backend python

# 15-minute analysis steps instead of hourly, for a sensitivity study
python run_experiment.py --dataset metr-la --steps-per-hour 3 --ots-horizon 3
```

| Topology | Arcs (METR-LA) | Mean out-degree | Edge lengths | Reachable OD pairs | s / OD pair |
|---|---|---|---|---|---|
| `adj_real` *(default)* | 1,515 | 7.3 | measured, median 2,758 m | 93.8 % | 0.25 |
| `csv_direct` | 11,546 | 55.8 | measured, median 7,433 m | 99.5 % | 2.48 |
| `adj_mx` | 1,515 | 7.3 | proxy `500/aᵢⱼ` | 93.8 % | — |
| `legacy_dense` | 42,642 | 206.0 | completed to 12 km | 100 % | — |

`--steps-per-hour` sets how many raw 5-minute steps form one analysis step, so it also sets
what `--ots-horizon` counts. With `--ots-horizon 1` the OTS volatility component is
identically zero and persistence is binary; E0 now prints a degeneracy flag per component.

A sparse graph is not strongly connected, so OD sampling rejects pairs with no directed path
and records the rejection count in `results.json` under `E3_routing.od_sampling`. Cached OD
pairs are tagged with the topology they were drawn on and resampled if it changes.

E3 processes OD pairs in batches and checkpoints after every batch, so an interrupted run
resumes without recomputation and without double counting.

Outputs land in `outputs/{dataset}/{quick,full}/`: `results.json`, `performance_report.{json,md}`,
`checkpoints/`, `figures/`.

---

## Pipeline (E0–E5)

| Stage | Processing |
|---|---|
| **E0** | Build OTS target; train the dual-branch encoder; predict `(μ, σ, embedding)` on the test split |
| **E1** | Mask 5/10/20/50 % of detectors; cached kriging; gated fusion; MAE on unobserved nodes |
| **E2** | Network mean speed and accumulation per test hour; MFD regime assignment |
| **E3** | Sample OD pairs; per (batch, hour) impute at 20 % masking, build MGAC costs, run Dijkstra for five methods; accumulate RII, travel time, DQI |
| **E4** | Frank–Wolfe loading of 50 OD pairs under three cost models |
| **E5** | Pareto filtering and DQI aggregation |

Routing methods compared: `time_only`, `ots_instant`, `ots_cvar_fixed`, `sparc_ots`,
`oracle` (sees the true stress field; an upper bound, not a competitor).

---

## Results (METR-LA `full`, verified real archive)

1,000 OD pairs × 21 test hours = 21,000 routing instances per method.
Source: [`outputs/metr-la/full/results.json`](outputs/metr-la/full/results.json).

### Routing (E3)

| Method | Mean RII | Std RII | Mean travel time (h) | RII relative regret |
|---|---|---|---|---|
| `time_only` | 3.106 | 2.101 | 0.665 | +1.820 |
| `ots_instant` | 1.154 | 0.763 | 0.708 | +0.001 |
| `ots_cvar_fixed` | 1.157 | 0.731 | 0.716 | +0.010 |
| `sparc_ots` | 1.157 | 0.727 | 0.710 | +0.009 |
| `oracle` | 1.155 | 0.765 | 0.712 | — |

**−63.4 % mean RII** and **−65 % dispersion** versus time-only routing, for **+6.8 %** travel time.

### Other stages

| Stage | Result |
|---|---|
| E0 | OTS MAE 0.706, RMSE 1.061 |
| E1 | Imputation MAE 0.107–0.133 from 5 % to 50 % masking; cached solver 9.6× faster with max ΔMAE = 0 |
| E2 | 82.6 % free flow, 7.2 % critical, 10.2 % congested |
| E4 | Risk exposure 186.2 → 77.6 (−58.3 %), travel cost 12.52 → 14.82 |
| E5 | Mean DQI 0.950 over 21,000 recommendations |
| Cost | E3 318,633 s (88.5 h), E4 1,470 s, peak extra RAM 8 MB |

---

## Known limitations

Reported honestly, because several of these bound the conclusions. The results table above
comes from the archived run, which predates the corrections marked **fixed in code**; those
numbers will change when the run is repeated.

1. **Degenerate target.** The archived run uses horizon `h=1`, so the volatility component is
   identically zero and persistence is binary. Only the shock component varies continuously.
   *Fixed in code:* `--ots-horizon` now defaults to 3 and E0 reports a per-component
   degeneracy flag. The archived results have not yet been regenerated.
2. **Weak node-level forecasting.** The encoder tracks the network-mean OTS trajectory but not
   per-node structure; residuals show systematic under-prediction of high-stress cells.
3. **Routing topology is not road-faithful.** The archived run used `legacy_dense`, mean
   out-degree 206 of a possible 206 — effectively a complete detector graph, in which many OD
   pairs are a single arc. This deflates RII magnitudes and plausibly explains limitation 4.
   *Fixed in code:* `adj_real` keeps the adjacency support with measured road distances
   (out-degree 7.3) and is the new default. The archived results have not yet been regenerated.
4. **MFD gating is not shown to help.** `sparc_ots`, `ots_instant`, `ots_cvar_fixed` and
   `oracle` sit within 0.5 % of each other. The evidence supports "using the stress field
   helps a lot"; it does not support "MFD gating or CVaR tails add measurable value". Whether
   this survives the topology fix is an open question, not a settled result.
5. **Statistics.** E3 means/medians use the first 5,000 of 21,000 observations per method, and
   those are consecutive rather than random. No paired significance test is run
   (`metrics.wilcoxon_paired` exists but is never called, and per-pair samples are not persisted).
6. **Single run, single seed** (42), one detector mask per sparsity level.
7. **No verified PEMS-BAY run.** `outputs/pems-bay/results.json` carries no `data_provenance`
   block and its regime fractions match a synthetic-era run, so those numbers are withheld
   rather than reported. The real archive is validated (`ok_for_full() is True`) and the run
   is pending.
8. **Ablations not executed.** `sparc_no_gkf` and `sparc_no_mgac` are defined in
   `routing.py` but absent from the archived runs, so per-component contributions are unknown.
9. **E5 Pareto statistics are not measured.** The E3 worker builds a single SPARC-OTS
   candidate per OD pair, so Pareto filtering is trivial and the mean size of 3.0 is a
   constant written by the aggregation code. The reported DQI median equals the mean because
   the accumulator keeps only a running sum and count.
10. **No real-time claim.** 88.5 h of single-machine CPU time for E3 on the dense graph.
    *Measured after the fix:* 0.25 s per OD pair on `adj_real` with the `python` backend,
    roughly 1.4 h for the same grid — still not real time.
11. **Uncertainty is now quantified, not asserted.** E0 reports PICP, interval width, Winkler
    interval score and Gaussian NLL at four nominal levels. A first `--quick` run gives
    PICP 87.9 % at a nominal 90 %, i.e. slightly over-confident intervals. The `full`
    calibration numbers are pending.

---

## Tests

```bash
pytest tests/ -v
```

- `test_kriging_equivalence.py` — cached solver vs. reference `Γ⁻¹` implementation
- `test_dataset_validation.py` — provenance guards and synthetic-fallback detection
- `test_topology.py` — `adj_real` sparsity and metric realism, connectivity report,
  reachability-filtered OD sampling
- `test_ots_horizon_and_calibration.py` — horizon degeneracy at `h=1`, recovery at `h=3`,
  and calibration metrics against a well-specified Gaussian

---

## Repository layout

```text
sparc_mfd/      pipeline modules (E0–E5)
baselines/      routing baseline registry
scripts/        data download, dataset validation, benchmarks, WSL2 helpers
tests/          unit tests
outputs/        archived run metrics, reports and figures (no large binaries)
run_experiment.py
```

Large artifacts (`*.npy`, `*.pt`), the `data/` archives and the LaTeX manuscript are not
tracked; everything reported above is regenerable from the committed code and the public
archives.

---

## Citation

```bibtex
@misc{sparcots2026code,
  author = {Boulaid, Sara and Bouhsissin, Soukaina and Banane, Mouad and Zakrani, Abdelali},
  title  = {{SPARC-OTS}: Operational Traffic Stress forecasting, graph--kriging fusion
            and {MFD}-gated routing under sparse sensing},
  year   = {2026},
  note   = {AIMCE Laboratory, ENSAM Casablanca, Hassan II University of Casablanca}
}
```

## Authors

Sara Boulaid, Soukaina Bouhsissin, Mouad Banane, Abdelali Zakrani —
AIMCE Laboratory, ENSAM Casablanca, Hassan II University of Casablanca, Morocco.
