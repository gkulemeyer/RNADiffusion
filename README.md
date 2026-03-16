# RNADiffusion

RNADiffusion is an experimental repo for RNA secondary structure prediction with a diffusion model.

The main workflow is simple:

`train -> generate ensemble samples -> compute ensemble statistics`

## Install

```bash
pip install -r requirements.txt
```

## Quick Start

Run one experiment end-to-end:

```bash
python train.py --config configs/train/default.yaml
```

This does all of the following:
- loads the YAML config
- trains the model
- saves checkpoints and training metrics
- generates ensemble samples on the `test` partition
- computes `ensemble.csv` and `ensemble_stats.csv`

## Config

The main config is a YAML file.

Default files:
- [configs/train/default.yaml](/home/gkulemeyer/Documents/Repos/RNADiffusion/configs/train/default.yaml)
- [configs/ensemble/default.yaml](/home/gkulemeyer/Documents/Repos/RNADiffusion/configs/ensemble/default.yaml)

Main sections:
- `experiment`
- `data`
- `model`
- `training`
- `logging`
- `ensemble`

Important data fields:
- `data.base_path`: dataset CSV
- `data.partition_path`: partition CSV
- `data.fold`: fold number

The partition CSV is expected to define `train`, `valid`, and `test`.

## Sweep

For ablations, the intended workflow is to edit a Python script and run it.

Example:
- [scripts/run_archiveii_sim_sweep.py](/home/gkulemeyer/Documents/Repos/RNADiffusion/scripts/run_archiveii_sim_sweep.py)

Run it with:

```bash
python scripts/run_archiveii_sim_sweep.py
```

The setup block is at the top of the file. You edit:
- partitions
- timesteps
- total epochs
- fold
- resume / dry-run flags

## Resume Training

The sweep script supports continuing an existing run.

Typical use:
1. Run with `EPOCHS = 2` and `RESUME = False`
2. Change to `EPOCHS = 4` and `RESUME = True`
3. Run the same script again

This continues the same experiment up to 4 total epochs.

When a run is resumed:
- the previous `ensemble_stats.csv` is moved to `epoch_<n>/ensemble_stats.csv`
- the previous `checkpoints/last.ckpt` is moved to `epoch_<n>/last.ckpt`
- a new `last.ckpt` and a new `ensemble_stats.csv` are generated for the updated run

## Useful Commands

Evaluate a checkpoint:

```bash
python test.py --config <run>/config.yaml --checkpoint <run>/checkpoints/best.ckpt
```

Generate ensemble samples manually:

```bash
python ensemble_generator.py \
  --config <run>/config.yaml \
  --checkpoint <run>/checkpoints/best.ckpt \
  --samples-dir <run>/raw_samples_custom \
  --num-samples 5
```

Compute ensemble stats from a samples directory:

```bash
python ensemble_stats.py \
  --samples-dir <run>/raw_samples_custom \
  --output <run>/ensemble_custom.csv \
  --consensus 3
```

## What Gets Saved

Each experiment is saved under:

```text
<logging.save_dir>/<experiment.name>/
```

If that directory already exists, the runner creates a suffixed directory like `_1`, `_2`, unless the run is being resumed.

Main files:
- `config.yaml`: resolved config used for the run
- `metrics.csv`: clean training metrics
- `ensemble.csv`: per-sequence ensemble results
- `ensemble_stats.csv`: summary averages from `ensemble.csv`
- `ensemble_metadata.yaml`: metadata used to generate the ensemble outputs
- `train.log`: log of the full pipeline
- `checkpoints/best.ckpt`: best checkpoint by `val_f1`
- `checkpoints/last.ckpt`: latest checkpoint

Supporting files:
- `raw_samples/*.pt`: saved ensemble samples
- `raw_samples/samples_metadata.yaml`: metadata for those samples
- `lightning/metrics_raw.csv`: raw Lightning CSV log
- `lightning/hparams.yaml`
- `lightning/tensorboard/...` if TensorBoard logging is enabled

## Where To Look First

If you only want the most useful artifacts after a run, start here:
- `config.yaml`
- `metrics.csv`
- `ensemble_stats.csv`
- `checkpoints/best.ckpt`
- `train.log`
