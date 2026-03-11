# RNADiffusion

RNADiffusion is a diffusion-based `seq2struct` model for RNA secondary structure prediction.

The active code is intentionally small and organized around one experimental flow:

`train -> ensemble_generation (test) -> ensemble_analysis`

## Core Layout

- `src/config.py`: YAML config loading, defaults, validation and experiment metadata
- `src/data.py`: partition loading, dataset, collate and Lightning `DataModule`
- `src/diffusion.py`: diffusion core
- `src/model.py`: model builder
- `src/train_module.py`: Lightning training module
- `src/io.py`: experiment directory, logging and checkpoint loading
- `src/ensemble.py`: raw sample generation and ensemble statistics
- `src/layers/simpleunet.py`: backbone network

CLI wrappers:

- `ensemble_generator.py`: thin wrapper over `src/ensemble.py::generate_raw_samples`
- `ensemble_stats.py`: thin wrapper over `src/ensemble.py::evaluate_samples_dir`

## Requirements

Install the runtime dependencies listed in [requirements.txt](/home/gkulemeyer/Documents/Repos/RNADiffusion/requirements.txt).

```bash
pip install -r requirements.txt
```

Main packages:

- `torch`
- `lightning`
- `pandas`
- `numpy`
- `scikit-learn`
- `PyYAML`

## Config

Training uses a single YAML config entrypoint:

```bash
python train.py --config configs/train.yaml
```

The config has five sections:

- `experiment`
- `data`
- `model`
- `training`
- `logging`

Example data contract:

- `data.base_path`: main dataset CSV
- `data.partition_path`: CSV with `id`, `partition` and optionally `fold_number`
- `data.fold`: fold number to filter

The active split contract is:

- `train`
- `valid`
- `test`

See [configs/train.yaml](/home/gkulemeyer/Documents/Repos/RNADiffusion/configs/train.yaml) and [configs/datasets/archiveii_simfold_128.yaml](/home/gkulemeyer/Documents/Repos/RNADiffusion/configs/datasets/archiveii_simfold_128.yaml).

## Minimum Contract

To keep the code path simple, these expectations are documented explicitly and validated only at I/O boundaries.

- Main dataset CSV must contain: `id`, `sequence`, and either `base_pairs` or a dot-bracket column (`structure` or `dotbracket`).
- Partition CSV must contain: `id`, `partition`; include `fold_number` when `data.fold` is not `null`.
- Valid partition values are expected to match the active flow: `train`, `valid`, `test`.
- `data.fold` must be an integer or `null`.
- Model invariants: `model.out_channels == model.num_classes`; `model.base_dim % 8 == 0`; `model.in_channels == 16 + model.num_classes` (16 comes from nucleotide-pair conditioning channels).
- Checkpoint loading expects a Lightning `.ckpt` file with `state_dict`.

## Training Pipeline

The official command is:

```bash
python train.py --config configs/train.yaml
```

`train.py` does all of the following:

1. loads and resolves the config
2. creates the experiment directory
3. trains with PyTorch Lightning
4. saves best and last checkpoints
5. exports training metrics
6. generates raw ensemble samples on the test partition
7. computes the final ensemble statistics

## Experiment Outputs

After training, each experiment directory contains:

- `config.yaml`
- `metrics.csv`
- `metrics_summary.csv` (one row per `epoch/step`, merged from sparse Lightning events)
- `ensemble.csv`
- `train.log`
- `checkpoints/best.ckpt`
- `checkpoints/last.ckpt`

`config.yaml` also stores experiment metadata such as:

- `experiment.uuid`
- `experiment.timestamp`
- resolved output paths under `logging`

`raw_samples/` may also be present as an intermediate artifact used to build `ensemble.csv`.

## Auxiliary Commands

Evaluate one checkpoint:

```bash
python test.py --config configs/train.yaml --checkpoint logs/RNADiffusion/<exp>/checkpoints/best.ckpt
```

Regenerate raw ensemble samples:

```bash
python ensemble_generator.py --config configs/train.yaml --checkpoint logs/RNADiffusion/<exp>/checkpoints/best.ckpt
```

Recompute ensemble statistics from saved samples:

```bash
python ensemble_stats.py --samples-dir logs/RNADiffusion/<exp>/raw_samples --output logs/RNADiffusion/<exp>/ensemble.csv
```

Run the `sim60/sim70/sim80/sim90` x `timesteps 5/10` sweep with `epochs=1`, `fold=0`, and final `consensus=1,3`:

```bash
./scripts/run_archiveii_sim_sweep.sh
```

Direct Python runner (includes `--resume` and `--dry-run`):

```bash
python scripts/run_archiveii_sim_sweep.py --resume --base-config configs/train.yaml
```

The runner loads one base config and overrides these fields per run:

- `experiment.name`
- `experiment.note`
- `data.partition_path`
- `data.fold`
- `model.timesteps`
- `training.max_epochs`
- `logging.save_dir`

These scripts are for reruns and debugging. The main workflow remains `train.py`.
Both commands are thin wrappers; ensemble logic lives in `src/ensemble.py`.

## Notes

- The active checkpoint format is Lightning `.ckpt`
- The active config format is YAML
- The active ensemble artifact is `ensemble.csv`
- The code is optimized for experimentation and small refactors, not for broad backward compatibility with older run layouts
