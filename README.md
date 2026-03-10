# RNA Diffusion Framework

Framework for RNA secondary structure prediction with diffusion and supervised models. Hydra defines experiments, `scripts/` triggers stages, `src/experiments/` turns config into workflows, and MLflow stores runs, checkpoints, samples, and metrics.

## Mental Model

This repository is a small experimentation framework.

- `configs/` defines experiments declaratively with Hydra.
- `scripts/` are thin operational entrypoints.
- `src/experiments/` orchestrates complete stages such as training, testing, and ensemble workflows.
- `src/data`, `src/models`, `src/training`, and `src/evaluation` contain reusable implementation pieces.
- `src/utils` contains only infrastructure helpers: config/data resolution, MLflow integration, and reporting.
- `src/core` contains the stable scientific logic of the framework.

A typical run flows like this:

1. Hydra composes `cfg` from `configs/`.
2. `scripts/run.py` selects a pipeline profile or a stage override.
3. `src/experiments/*` executes the workflow for that stage.
4. Lower layers build models, load data, train, evaluate, or generate ensembles.
5. MLflow stores checkpoints, generated samples, and derived metrics.

## Project Layout

- `configs/` — Hydra configs for data, network, and pipeline profiles.
- `src/core/` — stable scientific logic.
- `src/experiments/` — workflow orchestration by stage.
- `src/data/`, `src/models/`, `src/training/`, `src/evaluation/` — reusable building blocks.
- `src/utils/` — I/O, MLflow, and reporting helpers.
- `scripts/` — operational entrypoints and optional multirun shortcuts.
- `tests/` — unit, integration, and end-to-end coverage.
- `data/` — shared datasets and partition files.
- `logs/` — human-readable summaries and exported reports.
- `mlruns/` — MLflow tracking store.
- `archive/legacy/` — notebooks and historical material kept out of the active tree.

## Install

```bash
pip install -r requirements.txt
```

## Quick Start

Run the full pipeline controlled by `cfg.pipeline`:

```bash
python scripts/run.py
```

Run a specific stage through the same entrypoint:

```bash
python scripts/run.py pipeline=train
python scripts/run.py pipeline=test
python scripts/run.py pipeline=ensemble
python scripts/run.py pipeline=ensemble_analysis
```

Summarize a local MLflow experiment directory:

```bash
python scripts/summarize_experiment.py mlruns/0
```

## Configuration

Main config: `configs/config.yaml`. Typical sections:

- `data` — dataset paths, partitions, split, max length.
- `network` — denoiser/backbone selection plus wrapper config.
- `pipeline` — execution profile under `configs/pipeline/*`.
- `exp_name` / `run_name` — direct CLI inputs for experiment and run naming.
- `mlflow` — tracking uri plus the resolved MLflow names.

Example override:

```bash
python scripts/run.py pipeline=train_test training.epochs=1 network.wrapper.timesteps=5
```

That command will default to a run name similar to `train_epochs-1_timesteps-5_YYYYMMDD_HHMMSS`.

You can override names directly from the CLI:

```bash
python scripts/run.py pipeline=train exp_name=rna_debug run_name=epochs1_timesteps5
```

Inspect the composed config without running the workflow:

```bash
python scripts/run.py --cfg job
```

Available pipeline profiles:

- `pipeline=train`
- `pipeline=test`
- `pipeline=ensemble`
- `pipeline=ensemble_analysis`
- `pipeline=ensemble_pipeline`
- `pipeline=train_test`
- `pipeline=full_pipeline`

## MLflow Artifacts

Checkpoints and generated samples are stored under each run's `artifacts/` directory inside `mlruns/`.

## Hydra Multirun

Sweeps are done with Hydra, not with custom Python sweep logic.

Canonical multirun entrypoint:

```bash
python scripts/run.py -m pipeline=train network.wrapper.timesteps=5,10,15
```

Optional shortcuts:

```bash
python scripts/orchestrator.py data.split=sim60,sim70 network.wrapper.timesteps=5,10
python scripts/orchestrator_supervised.py data.split=sim70,sim80 +fold_number=0,1,2
```

## Tests

Run all tests:

```bash
pytest -q
```

Test layers:

- Unit tests: `tests/unit`
- Integration tests: `tests/integration`
- End-to-end tests: `tests/e2e`
