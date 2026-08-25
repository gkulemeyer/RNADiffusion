from pathlib import Path
import torch as tr
import torch.nn.functional as F

from src.config import load_config
from src.experiment import (
    build_dataloader,
    load_model_checkpoint,
)
from src.ensemble import generate_ensemble_db
from src.run_io import RunIO

def validate_contact_channel(
    model,
    loader,
    threshold=0.5,
    seed=42
):
    model.eval()
    device = next(model.parameters()).device

    batch = next(iter(loader))

    conditioning = batch["conditioning"].to(device)
    lengths = tr.as_tensor(
        batch["length"],
        dtype=tr.long,
        device=device
    )

    target = (
        batch["contact_oh"]
        .to(device)
        .argmax(dim=1)
    )

    tr.manual_seed(seed)

    with tr.inference_mode():
        _, logits = model.sample(
            conditioning,
            lengths=lengths,
            return_logits=True
        )

    # [B,2,L,L]
    probs = F.softmax(logits, dim=1)

    # Usar el mismo criterio simétrico que en mat2bp
    probs = 0.5 * (
        probs
        + probs.transpose(-1, -2)
    )

    B, _, L, _ = probs.shape

    indices = tr.arange(L, device=device)
    valid_nt = indices[None] < lengths[:, None]

    mask = (
        valid_nt[:, :, None]
        & valid_nt[:, None, :]
    )

    upper = tr.triu(
        tr.ones(L, L, dtype=tr.bool, device=device),
        diagonal=1
    )

    mask = mask & upper[None]

    results = {}

    print(f"\nContact-channel validation, threshold={threshold}")

    for channel in [0, 1]:
        scores = probs[:, channel]
        prediction = scores > threshold

        contact_mask = mask & (target == 1)
        noncontact_mask = mask & (target == 0)

        tp = (prediction & contact_mask).sum().item()
        fp = (prediction & noncontact_mask).sum().item()
        fn = (~prediction & contact_mask).sum().item()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)

        mean_contact = scores[contact_mask].mean().item()
        mean_noncontact = scores[noncontact_mask].mean().item()

        results[channel] = f1

        print(f"\nChannel {channel}")
        print(f"  mean probability on contacts:    {mean_contact:.4f}")
        print(f"  mean probability on noncontacts: {mean_noncontact:.4f}")
        print(f"  precision: {precision:.4f}")
        print(f"  recall:    {recall:.4f}")
        print(f"  F1:        {f1:.4f}")

    best_channel = max(results, key=results.get)

    print(f"\nRecommended contact channel: {best_channel}")

    return best_channel

RUN_DIR = Path(
    "/home/gkulemeyer/Documents/Repos/RNADiffusion/"
    "logs/ArchiveII_ff128/srp/t10/bd32/bs1_acc4/seed42"
) 
CONFIG_PATH = None
CHECKPOINT_PATH = None
OUTPUT_CSV = None

NUM_SAMPLES = 10
SEED = 42
CHUNK_SIZE = 1
PARTITION = "test"
SAVE_AS = "db"
thresholds = [0, 0.1,  0.5,  0.8,]


def resolve_inputs(threshold=0):
    if RUN_DIR is not None:
        run = RunIO(RUN_DIR)

        output_csv = (
            Path(OUTPUT_CSV)
            if OUTPUT_CSV is not None
            else run.best_eval_dir / f"generated_ensemble_db_thr{threshold}.csv"
        )

        return run.config_path, run.best_ckpt_path,output_csv  
    return Path(CONFIG_PATH), Path(CHECKPOINT_PATH), Path(OUTPUT_CSV)

def main():
    config_path, checkpoint_path, _ = resolve_inputs()

    config = load_config(config_path)

    model = load_model_checkpoint(
        config,
        checkpoint_path,
        eval_mode=True
    )

    loader = build_dataloader(
        config,
        partition=PARTITION,
        shuffle=False
    )

    print(f"Config:     {config_path}")
    print(f"Checkpoint: {checkpoint_path}")

    contact_channel = validate_contact_channel(
        model=model,
        loader=loader,
        threshold=0.5,
        seed=SEED
    )

    for threshold in thresholds:
        _, _, output_csv = resolve_inputs(
            threshold=threshold
        )

        print(f"\nThreshold: {threshold}")
        print(f"Output:    {output_csv}")

        ensemble = generate_ensemble_db(
            model=model,
            loader=loader,
            output_csv=output_csv,
            num_samples=NUM_SAMPLES,
            seed=SEED,
            chunk_size=CHUNK_SIZE,
            save_as=SAVE_AS,
            threshold=threshold,
        )

        print(f"Generated rows: {len(ensemble)}")
        print(f"Saved ensemble to: {output_csv}")


if __name__ == "__main__":
    main()