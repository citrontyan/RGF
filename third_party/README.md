# External dependencies

The core RGF package is self-contained Python. The following external tools and model checkpoints are needed for the full experiments and are intentionally not vendored in this repository.

## Neural predictors

### DIFUSCO

- Paper: *DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization* (NeurIPS 2023).
- Upstream code and checkpoints: see the authors' public repository.
- Required for the `--predictor difusco` path.

### FastT2T

- Paper: *Fast T2T: Optimization Consistency Speeds Up Diffusion-Based Training-to-Testing Solving for Combinatorial Optimization* (NeurIPS 2024).
- Upstream code and checkpoints: see the authors' public repository.
- Required for the `--predictor fastt2t` path.

### COExpander / GenSCO (optional)

- COExpander: *COExpander: Adaptive Solution Expansion for Combinatorial Optimization* (ICML 2025).
- GenSCO: *Generation as Search Operator for Test-Time Scaling of Diffusion-Based Combinatorial Optimization* (NeurIPS 2025).
- Optional stronger MIS predictors used in the paper's comparison.

Place predictor code and checkpoints under `<DATA_ROOT>/checkpoints/` in the layout expected by `rgf/predictor/` and `scripts/run_rgf.py`.

## Graph partitioning

### METIS

- `gpmetis` is used by `rgf/partition/metis.py`.
- Set `GPMETIS` to the executable path, or place it on `PATH`.
- `scripts/build_metis_wsl.sh` is a helper recipe for WSL.

## Reference MIS solver

### KaMIS / redumis

- Used to compute the reference MIS labels and the KaMIS baseline.
- `scripts/build_kamis_wsl.sh` is a helper recipe for WSL.
- Reference labels are cached as `<graph>.label.npz`.

## DS reference

DS reference solutions are computed by `rgf/baselines/ds_reference.py` using SciPy's MILP solver (HiGHS). No separate binary is required.
