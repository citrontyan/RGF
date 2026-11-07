# RGF: Recursive Generative Framework for Edge-Cut Separable Problems

RGF is a learning-based solver for Maximum Independent Set (MIS), Dominating Set (DS), and Vertex Cover (VC) on large graphs. It recursively partitions a graph, predicts the cut-related decisions with a neural predictor, and uses an exact solver at small leaves.

This repository contains the core implementation and the scripts used to reproduce the paper's experiments. It is the cleaned release version; internal campaign logs, result archives, and local paths are not included.

## Repository layout

```
rgf/                 Core package: ECSP definitions, CDSP, recursion, partitions,
                     predictors, baselines, data I/O, metrics
scripts/             Data preparation, training/evaluation entry points,
                     and summarization utilities
tests/               Self-checks on small graphs
third_party/         Placeholder and installation notes for external tools
requirements.txt     Python dependencies
pyproject.toml       Package/test configuration
LICENSE              Apache-2.0
```

## Components

- `rgf/problems/`: MIS, DS, VC definitions and feasibility checks.
- `rgf/cdsp.py`: cut-determination subproblem (CDSP) and its solution format.
- `rgf/framework.py`: recursive RGF solver and the exact-solver threshold `alpha`.
- `rgf/partition/`: METIS, size-balanced random, and single-vertex partitioners.
- `rgf/predictor/`: DIFUSCO, FastT2T, COExpander, GenSCO adapters, neighborhood subgraph construction, and the legality-repair hook.
- `rgf/baselines/`: KaMIS reference, DS MILP reference, and random-walk baseline.
- `rgf/datasets/`: graph loading and synthetic RB/ER generation.

## Installation

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build that matches your driver first, for example:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch
pip install -r requirements.txt
```

### External tools

The neural predictors and classical baselines are not vendored here. See `third_party/README.md` for:

- DIFUSCO and FastT2T predictor code and checkpoints;
- METIS (`gpmetis`) for graph partitioning;
- KaMIS / redumis for the reference MIS solution.

`scripts/build_metis_wsl.sh` and `scripts/build_kamis_wsl.sh` provide helper build recipes for a WSL environment; adapt them to your platform.

## Data and checkpoints

All datasets, labels, and checkpoints live outside the repository, under a configurable data root. The scripts default to a local data root and accept `--data-root`, `--data-dir`, or `--ckpt` arguments where applicable.

Recommended layout:

```
<DATA_ROOT>/
  rb/rb200_300/test/*.npz
  er/er700_800/test/*.npz
  dimacs10/
  external/
  checkpoints/
    fastt2t/
    difusco/
```

Generate the synthetic benchmarks:

```bash
python scripts/gen_rb_er.py --help
python scripts/label_dataset.py --help
```

Download real-world graphs:

```bash
python scripts/fetch_dimacs10.py --help
python scripts/fetch_snap.py --help
python scripts/fetch_suitesparse.py --help
```

## Quick start

Run RGF on the synthetic RB test set:

```bash
python scripts/run_rgf.py \
  --data-root /path/to/data \
  --problem mis --dataset rb --predictor fastt2t \
  --partitioner metis --alpha 50 --runs 5 --sampling 4
```

Run the large-graph table:

```bash
python scripts/run_large_table.py \
  --data-dir /path/to/data/external \
  --instances web-NotreDame,web-Stanford \
  --alpha 10000 --hidden 128
```

Evaluate a direct (non-recursive) neural baseline:

```bash
python scripts/eval_direct_large.py --help
```

Run the self-checks:

```bash
python -m pytest tests -q
```

## Reproducibility notes

- `alpha` is the graph-size threshold for invoking the exact solver; `beta` is the neighborhood radius used by the predictor.
- Synthetic experiments use 5 independent runs unless noted; large real-world experiments use 5 seeds where stated in the paper.
- The paper reports `drop`, i.e., the relative gap to the reference solution, with wall-clock time.
- Some scripts contain optional third-party integrations; missing external tools or checkpoints raise a clear error instead of silently falling back.

## License

Apache License 2.0. See `LICENSE`.
