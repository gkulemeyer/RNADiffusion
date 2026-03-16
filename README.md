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
- `src/io.py`: logging and checkpoint loading
- `src/ensemble.py`: raw sample generation and ensemble statistics
- `src/experiment.py`: shared end-to-end experiment runner
- `src/sweeps.py`: helpers for ablation scripts
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
- `ml-collections`

## Config

Training uses a single YAML config entrypoint:

```bash
python train.py --config configs/train/default.yaml
```

The config has six sections:

- `experiment`
- `data`
- `model`
- `training`
- `logging`
- `ensemble`

Example data contract:

- `data.base_path`: main dataset CSV
- `data.partition_path`: CSV with `id`, `partition` and optionally `fold_number`
- `data.fold`: fold number to filter

The active split contract is:

- `train`
- `valid`
- `test`

Default configs live in:

- [configs/train/default.yaml](/home/gkulemeyer/Documents/Repos/RNADiffusion/configs/train/default.yaml)
- [configs/ensemble/default.yaml](/home/gkulemeyer/Documents/Repos/RNADiffusion/configs/ensemble/default.yaml)

See also [configs/datasets/archiveii_simfold_128.yaml](/home/gkulemeyer/Documents/Repos/RNADiffusion/configs/datasets/archiveii_simfold_128.yaml).

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
python train.py --config configs/train/default.yaml
```

`train.py` does all of the following:

1. loads and resolves the config
2. creates the experiment directory
3. trains with PyTorch Lightning
4. saves best and last checkpoints
5. exports training metrics
6. generates raw ensemble samples on the test partition
7. computes the final ensemble statistics

The `ensemble` section defines:

- `num_samples`
- `base_seed`
- `trials`
- `consensus_sizes`
- `chunk_size`

## Como Se Utiliza El Repositorio

Hay dos modos de uso activos:

1. Un experimento individual end-to-end.
2. Un sweep editable desde un script Python.

### 1. Experimento individual

Entrenamiento completo:

```bash
python train.py --config configs/train/default.yaml
```

Evaluacion puntual de un checkpoint:

```bash
python test.py --config <run>/config.yaml --checkpoint <run>/checkpoints/best.ckpt
```

Regenerar muestras del ensemble:

```bash
python ensemble_generator.py \
  --config <run>/config.yaml \
  --checkpoint <run>/checkpoints/best.ckpt \
  --samples-dir <run>/raw_samples_custom \
  --num-samples 5
```

Recalcular estadisticas del ensemble:

```bash
python ensemble_stats.py \
  --samples-dir <run>/raw_samples_custom \
  --output <run>/ensemble_custom.csv \
  --consensus 3
```

`ensemble_generator.py` escribe tambien `samples_metadata.yaml` dentro del directorio de muestras.
`ensemble_stats.py` escribe metadata asociada al CSV de salida, por ejemplo `ensemble_metadata.yaml` o `<nombre>_metadata.yaml`.

### 2. Sweep editable

El flujo de ablaciones sigue siendo un script editable, por ejemplo [scripts/run_archiveii_sim_sweep.py](/home/gkulemeyer/Documents/Repos/RNADiffusion/scripts/run_archiveii_sim_sweep.py).

La idea es:

- abrir el script
- editar el bloque de setup del principio
- correrlo con Python

```bash
python scripts/run_archiveii_sim_sweep.py
```

El script carga un YAML base, aplica overrides en memoria y llama al mismo runner compartido que usa `train.py`.

## Experiment Outputs

After training, each experiment directory contains:

- `config.yaml`
- `metrics.csv`
- `ensemble.csv`
- `ensemble_stats.csv`
- `ensemble_metadata.yaml`
- `train.log`
- `checkpoints/best.ckpt`
- `checkpoints/last.ckpt`
- `raw_samples/*.pt`

`config.yaml` also stores experiment metadata such as:

- `experiment.uuid`
- `experiment.timestamp`
- resolved output paths under `logging`

Internal logger artifacts are stored under `lightning/`, for example:

- `lightning/metrics_raw.csv`
- `lightning/hparams.yaml`
- `lightning/tensorboard/...`

## Como Se Guardan Los Experimentos

Cada run vive en:

```text
<logging.save_dir>/<experiment.name o nombre_resuelto>/
```

Ejemplo:

```text
logs/RNADiffusion/exp_T5_E10_20260316_120000/
```

o si defines un nombre explicito:

```text
logs/RNADiffusion/prueba_0/
```

Si ese directorio ya existe, el runner crea un sufijo `_1`, `_2`, etc.

Archivos principales:

- `config.yaml`: config final resuelta, con `uuid`, `timestamp` y paths internos del run
- `metrics.csv`: tabla limpia y canonica, una fila por `epoch/step`
- `ensemble.csv`: estadisticas finales del ensemble sobre la particion de test
- `ensemble_stats.csv`: resumen agregado de `ensemble.csv` por tamaño de consenso
- `ensemble_metadata.yaml`: metadata exacta usada para generar `ensemble.csv`
- `train.log`: log operacional del experimento completo
- `checkpoints/best.ckpt`: mejor checkpoint segun `val_f1`
- `checkpoints/last.ckpt`: ultimo checkpoint del entrenamiento

Artefactos auxiliares:

- `raw_samples/`: muestras persistentes para reconstruir `ensemble.csv`
- `lightning/`: artefactos raw del logger de Lightning

## Smoke Test Verificado

Se verifico localmente el flujo completo con el entorno conda `seq2seq` usando un subset chico del dataset real, con:

- `experiment.name = prueba_0`
- `training.max_epochs = 1`
- `ensemble_generator.py --num-samples 5`
- `ensemble_stats.py --consensus 3`

Artefactos generados en la prueba:

- `config.yaml`
- `metrics.csv`
- `ensemble.csv`
- `ensemble_stats.csv`
- `ensemble_metadata.yaml`
- `train.log`
- `checkpoints/best.ckpt`
- `checkpoints/last.ckpt`
- `test_summary.csv`
- `ensemble_k3_5samples.csv`

Para una guia de debugging archivo por archivo, ver [docs/debugging_guide.md](/home/gkulemeyer/Documents/Repos/RNADiffusion/docs/debugging_guide.md).

## Auxiliary Commands

Evaluate one checkpoint:

```bash
python test.py --config configs/train/default.yaml --checkpoint logs/RNADiffusion/<exp>/checkpoints/best.ckpt
```

Regenerate raw ensemble samples:

```bash
python ensemble_generator.py --config configs/train/default.yaml --checkpoint logs/RNADiffusion/<exp>/checkpoints/best.ckpt
```

Recompute ensemble statistics from saved samples:

```bash
python ensemble_stats.py --samples-dir logs/RNADiffusion/<exp>/raw_samples --output logs/RNADiffusion/<exp>/ensemble.csv
```

Run the `sim60/sim70/sim80/sim90` x `timesteps 5/10` sweep with `epochs=1` and `fold=0`:

```bash
./scripts/run_archiveii_sim_sweep.sh
```

The runner loads one base config, converts it to `ml_collections.ConfigDict`, and overrides these fields per run:

- `experiment.name`
- `experiment.note`
- `data.partition_path`
- `data.fold`
- `model.timesteps`
- `training.max_epochs`
- `logging.save_dir`

The sweep script keeps the editable setup block at the top, but the heavy experiment logic is shared with `train.py`.
It does not generate intermediate per-run YAML files and it does not recompute `ensemble.csv` outside the main training pipeline.
`ml_collections` is only used in the sweep flow for in-memory config editing.

## Notes

- The active checkpoint format is Lightning `.ckpt`
- The active config format is YAML
- The active ensemble artifact is `ensemble.csv`
- The code is optimized for experimentation and small refactors, not for broad backward compatibility with older run layouts
