# RESIS Pro — ERT geological interpretation (forward-hypothesis matching)

Reproducible code and data for the manuscript:

> **Forward-hypothesis matching for ERT-only structural dip ranking without
> external priors: validation under independent forward solvers and
> out-of-distribution control.**
> Junghoon Choi, GEOINNOVATION Co., Ltd. (submitted to *Journal of Applied Geophysics*).

This repository contains the RESIS Pro 2-D ERT analysis program, the geological
candidate library, the forward-hypothesis matching and pseudosection-diagnostic
code, the synthetic benchmarks, the inverse-crime-free validation scripts
(mesh separation and finite-element/finite-difference solver cross-validation),
the result tables, and the figure-generation scripts.

## Contents

| Path | Description |
|------|-------------|
| `RESIS_Pro.py` | 2.5-D ERT forward modeling + inversion program (FDM/FEM, L2/MGS, etc.) |
| `geo_library.py` | Geological candidate template library |
| `geo_hypothesis_matching.py`, `forward_matcher.py` | Forward-hypothesis matching |
| `dip_diagnostics.py`, `dip_adaptive.py`, `dip_calibration.py` | Pseudosection dip diagnostics (M1–M5) |
| `dip_ml_train.py`, `dip_ml_robust.py`, `dip_ml_predict.py` | ML dip estimation with out-of-distribution control |
| `geo_structure_interpreter.py` | Integrated interpreter |
| `geo_synthetic_benchmark.py`, `run_hypothesis_test.py` | Controlled synthetic benchmark |
| `run_occam_comparison.py` | Occam L2 dip-extraction baseline |
| `star_jag_benchmark.py` | Oracle-dip / anisotropic regularization negative-result benchmark |
| `forward_matching_mesh_separation_validation.py` | Mesh-separation inverse-crime control |
| `independent_forward_validation.py` | FEM→FDM solver cross-validation |
| `STAR_JAG_Benchmark/`, `IndependentForwardValidation/`, `ForwardMatchingMeshSeparated/` | Result CSVs |
| `GeoHypothesisMatching/`, `SyntheticGeoBenchmark/` | Template libraries and benchmark outputs |
| `make_figures.py`, `make_fig3.py` | Figure-generation scripts |
| `figures/` | Manuscript figures (PNG/PDF, EPS for line art) |
| `*.APV` | Synthetic example datasets |

## Requirements
Python 3 with `numpy`, `scipy`, `matplotlib` (optional `numba` for JIT).

## Run
```bash
python3 RESIS_Pro.py                 # GUI
python3 run_hypothesis_test.py       # controlled benchmark
python3 star_jag_benchmark.py        # oracle-dip negative result
python3 independent_forward_validation.py   # FEM->FDM cross-validation
```
Scripts use paths relative to the repository root; run them from the repo root.

## Notes
- Field datasets containing private site identifiers are **not** included; the
  benchmarks here are fully synthetic and reproducible.
- © GEOINNOVATION Co., Ltd. Licensing terms to be finalized; please contact the
  author (hoons9096@gmail.com) regarding reuse.
