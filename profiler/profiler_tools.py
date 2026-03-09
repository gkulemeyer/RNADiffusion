from __future__ import annotations

import glob
import json
import os
import sys
import time
from bisect import bisect_left, bisect_right
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch as tr
from torch.cuda.amp import GradScaler, autocast
from torch.profiler import ProfilerActivity, profile, record_function, schedule, tensorboard_trace_handler
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.dataset import SeqDataset, pad_batch
from src.metrics import contact_f1
from src.utils import load_model, save_config


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def make_run_logger(run_dir: str | Path, run_name: str | None = None) -> Callable[[str, dict | None, str], None]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"
    resolved_run_name = run_name or run_dir.name

    def log_event(event: str, data: dict | None = None, level: str = "info") -> None:
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "event": str(event),
            "run_name": resolved_run_name,
            "level": str(level),
            "data": _json_safe(data or {}),
        }
        with events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    log_event.events_path = str(events_path)  # type: ignore[attr-defined]
    return log_event


def _iter_loader(loader, max_batches: int | None = None):
    if max_batches is None:
        yield from loader
        return
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        yield batch


def _safe_total_ms(key_avgs, name: str, use_cuda: bool = False, self_only: bool = True) -> float:
    for evt in key_avgs:
        if evt.key == name:
            if use_cuda:
                t = evt.self_cuda_time_total if self_only else evt.cuda_time_total
            else:
                t = evt.self_cpu_time_total if self_only else evt.cpu_time_total
            return float(t) / 1000.0
    return 0.0


def _keyavg_self_totals_ms(key_avgs) -> tuple[float, float]:
    cpu_ms = sum(float(getattr(evt, "self_cpu_time_total", 0.0)) for evt in key_avgs) / 1000.0
    cuda_ms = sum(float(getattr(evt, "self_cuda_time_total", 0.0)) for evt in key_avgs) / 1000.0
    return cpu_ms, cuda_ms


def _latest_trace_file(trace_dir: str | Path) -> str | None:
    traces = sorted(glob.glob(os.path.join(str(trace_dir), "*.pt.trace.json")))
    if not traces:
        return None
    return max(traces, key=os.path.getmtime)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def train_one_epoch_profiled(
    model,
    loader,
    optimizer,
    device,
    prof=None,
    max_batches: int | None = None,
    use_amp: bool = False,
    scaler: GradScaler | None = None,
    grad_accum_steps: int = 1,
    amp_dtype=tr.float16,
):
    model.train()
    amp_enabled = bool(use_amp and device.type == "cuda")
    grad_accum_steps = max(int(grad_accum_steps), 1)

    optimizer.zero_grad(set_to_none=True)
    micro_idx = 0

    for micro_idx, batch in enumerate(_iter_loader(loader, max_batches=max_batches), start=1):
        with record_function("train_batch"):
            cond = batch["outer"].to(device, non_blocking=True)
            target = batch["contact_oh"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            with record_function("train_forward"):
                with autocast(enabled=amp_enabled, dtype=amp_dtype):
                    loss = model.forward_all_timesteps(target, cond, mask=mask)
                    loss = loss / grad_accum_steps

            if amp_enabled and scaler is not None:
                with record_function("train_backward"):
                    scaler.scale(loss).backward()
            else:
                with record_function("train_backward"):
                    loss.backward()

            do_step = micro_idx % grad_accum_steps == 0
            if do_step:
                with record_function("train_step"):
                    if amp_enabled and scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if prof is not None:
            prof.step()

    # Flush remaining accumulated grads when the profiled microbatch count is not divisible.
    if micro_idx > 0 and micro_idx % grad_accum_steps != 0:
        with record_function("train_step"):
            if amp_enabled and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        optimizer.zero_grad(set_to_none=True)


@tr.no_grad()
def eval_one_epoch_profiled(
    model,
    loader,
    device,
    compute_f1: bool = True,
    prof=None,
    max_batches: int | None = None,
    use_amp: bool = False,
    amp_dtype=tr.float16,
    amp_sampling: bool = False,
):
    model.eval()
    amp_enabled = bool(use_amp and device.type == "cuda")

    for batch in _iter_loader(loader, max_batches=max_batches):
        with record_function("eval_batch"):
            cond = batch["outer"].to(device, non_blocking=True)
            target = batch["contact_oh"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            with record_function("eval_loss_forward"):
                with autocast(enabled=amp_enabled, dtype=amp_dtype):
                    _ = model.forward_all_timesteps(target, cond, mask=mask)

            if compute_f1:
                lens = batch["length"]
                with record_function("eval_sample"):
                    if amp_sampling and amp_enabled:
                        with autocast(enabled=True, dtype=amp_dtype):
                            samples = model._sample(cond)
                    else:
                        with autocast(enabled=False):
                            samples = model._sample(cond)
                with record_function("eval_contact_f1"):
                    _ = contact_f1(samples, target, lengths=lens, reduce=True)

        if prof is not None:
            prof.step()


def _phase_alloc_metric_keys() -> list[str]:
    phases = ["train_forward", "train_backward", "train_step"]
    keys: list[str] = []
    for phase in phases:
        for dev in ["gpu", "cpu"]:
            keys.append(f"{phase}_alloc_avg_mb_{dev}")
            keys.append(f"{phase}_alloc_peak_mb_{dev}")
            keys.append(f"{phase}_alloc_delta_avg_mb_{dev}")
            keys.append(f"{phase}_alloc_delta_peak_mb_{dev}")
            keys.append(f"{phase}_alloc_traffic_mb_{dev}")
            keys.append(f"{phase}_free_traffic_mb_{dev}")
    return keys


def empty_phase_alloc_metrics() -> dict[str, float]:
    return {k: np.nan for k in _phase_alloc_metric_keys()}


def _phase_device_alloc_mb(events: list[dict[str, Any]]) -> dict[str, float]:
    phase_names = ["train_forward", "train_backward", "train_step"]
    phase_spans: dict[str, list[tuple[float, float]]] = {p: [] for p in phase_names}
    mem_by_dev: dict[int, list[tuple[float, float, float]]] = {0: [], 1: []}

    for e in events:
        name = e.get("name", "")
        if name in phase_spans and e.get("ph") == "X":
            ts = e.get("ts")
            if ts is not None:
                start = float(ts)
                dur = float(e.get("dur", 0.0) or 0.0)
                phase_spans[name].append((start, start + max(dur, 0.0)))

        if name == "[memory]":
            args = e.get("args", {}) or {}
            ts = e.get("ts")
            dev = args.get("Device Type")
            total_alloc = args.get("Total Allocated", 0)
            bytes_change = args.get("Bytes", 0)
            if ts is None:
                continue
            try:
                dev = int(dev)
                alloc_b = float(total_alloc or 0.0)
                bytes_b = float(bytes_change or 0.0)
            except Exception:
                continue
            if dev in mem_by_dev:
                mem_by_dev[dev].append((float(ts), alloc_b, bytes_b))

    mem_arrays: dict[int, tuple[list[float], list[float], list[float]]] = {}
    for dev, points in mem_by_dev.items():
        points.sort(key=lambda x: x[0])
        ts_arr = [p[0] for p in points]
        alloc_arr = [p[1] for p in points]
        bytes_arr = [p[2] for p in points]
        mem_arrays[dev] = (ts_arr, alloc_arr, bytes_arr)

    out = empty_phase_alloc_metrics()
    for phase in phase_names:
        spans = phase_spans.get(phase, [])
        for dev_name, dev_type in [("gpu", 1), ("cpu", 0)]:
            avg_key = f"{phase}_alloc_avg_mb_{dev_name}"
            peak_key = f"{phase}_alloc_peak_mb_{dev_name}"
            delta_avg_key = f"{phase}_alloc_delta_avg_mb_{dev_name}"
            delta_peak_key = f"{phase}_alloc_delta_peak_mb_{dev_name}"
            alloc_traffic_key = f"{phase}_alloc_traffic_mb_{dev_name}"
            free_traffic_key = f"{phase}_free_traffic_mb_{dev_name}"

            if not spans:
                continue
            ts_arr, alloc_arr, bytes_arr = mem_arrays.get(dev_type, ([], [], []))
            if not ts_arr:
                continue

            span_avg_vals: list[float] = []
            span_peak_vals: list[float] = []
            span_delta_avg_vals: list[float] = []
            span_delta_peak_vals: list[float] = []
            span_alloc_traffic_vals: list[float] = []
            span_free_traffic_vals: list[float] = []

            for start, end in spans:
                i0 = bisect_left(ts_arr, start)
                i1 = bisect_right(ts_arr, end)
                if i1 <= i0:
                    continue

                vals = alloc_arr[i0:i1]
                bvals = bytes_arr[i0:i1]
                if not vals:
                    continue

                i_start = bisect_right(ts_arr, start) - 1
                baseline = alloc_arr[i_start] if i_start >= 0 else vals[0]

                delta_vals = [max(v - baseline, 0.0) for v in vals]
                alloc_traffic = sum(b for b in bvals if b > 0.0)
                free_traffic = sum(-b for b in bvals if b < 0.0)

                span_avg_vals.append(sum(vals) / len(vals))
                span_peak_vals.append(max(vals))
                span_delta_avg_vals.append(sum(delta_vals) / len(delta_vals))
                span_delta_peak_vals.append(max(delta_vals))
                span_alloc_traffic_vals.append(alloc_traffic)
                span_free_traffic_vals.append(free_traffic)

            if span_avg_vals:
                out[avg_key] = (sum(span_avg_vals) / len(span_avg_vals)) / (1024 ** 2)
                out[peak_key] = (sum(span_peak_vals) / len(span_peak_vals)) / (1024 ** 2)
                out[delta_avg_key] = (sum(span_delta_avg_vals) / len(span_delta_avg_vals)) / (1024 ** 2)
                out[delta_peak_key] = (sum(span_delta_peak_vals) / len(span_delta_peak_vals)) / (1024 ** 2)
                out[alloc_traffic_key] = (sum(span_alloc_traffic_vals) / len(span_alloc_traffic_vals)) / (1024 ** 2)
                out[free_traffic_key] = (sum(span_free_traffic_vals) / len(span_free_traffic_vals)) / (1024 ** 2)

    return out


def analyze_trace_file(trace_path: str | None) -> dict[str, Any]:
    if trace_path is None or not os.path.exists(trace_path):
        return {
            "trace_found": False,
            "trace_path": trace_path,
            **empty_phase_alloc_metrics(),
        }

    with open(trace_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    events = data.get("traceEvents", [])
    names = [
        "train_batch",
        "train_forward",
        "train_backward",
        "train_step",
        "eval_batch",
        "eval_loss_forward",
        "eval_sample",
        "eval_contact_f1",
        "epoch_train",
        "epoch_eval_no_f1",
        "epoch_eval_with_f1",
    ]

    stats = {k: {"count": 0, "dur_us": 0.0} for k in names}
    eval_batches: list[tuple[float, float]] = []

    oom_events = 0
    max_alloc = 0
    max_reserved = 0

    cat_dur_us = {
        "cpu_op": 0.0,
        "kernel": 0.0,
        "cuda_runtime": 0.0,
        "gpu_memcpy": 0.0,
        "gpu_memset": 0.0,
    }

    for e in events:
        n = e.get("name", "")

        if n in ("[memory]", "[OutOfMemory]"):
            args = e.get("args", {})
            max_alloc = max(max_alloc, int(args.get("Total Allocated", 0) or 0))
            max_reserved = max(max_reserved, int(args.get("Total Reserved", 0) or 0))
            if n == "[OutOfMemory]":
                oom_events += 1

        if e.get("ph") != "X":
            continue

        dur = float(e.get("dur", 0.0))
        cat = e.get("cat", "")
        if cat in cat_dur_us:
            cat_dur_us[cat] += dur

        if n in stats:
            stats[n]["count"] += 1
            stats[n]["dur_us"] += dur

        if n == "eval_batch":
            eval_batches.append((float(e.get("ts", 0.0)), dur))

    eval_batches.sort(key=lambda x: x[0])
    eval_batch_count = stats["eval_batch"]["count"]
    eval_sample_count = stats["eval_sample"]["count"]
    eval_no_f1_count = max(eval_batch_count - eval_sample_count, 0)

    eval_no_f1_us = 0.0
    eval_with_f1_us = 0.0

    if eval_batch_count and eval_no_f1_count <= eval_batch_count:
        eval_no_f1_us = sum(d for _, d in eval_batches[:eval_no_f1_count])
        eval_with_f1_us = sum(d for _, d in eval_batches[eval_no_f1_count:])

    device_name = None
    if data.get("deviceProperties"):
        device_name = data["deviceProperties"][0].get("name")

    out: dict[str, Any] = {
        "trace_found": True,
        "trace_path": trace_path,
        "trace_size_mb": os.path.getsize(trace_path) / (1024 ** 2),
        "trace_events": len(events),
        "trace_device": device_name,
        "oom_events": oom_events,
        "max_alloc_gb_trace": max_alloc / (1024 ** 3),
        "max_reserved_gb_trace": max_reserved / (1024 ** 3),
        "cpu_op_ms_trace": cat_dur_us["cpu_op"] / 1000.0,
        "gpu_kernel_ms_trace": cat_dur_us["kernel"] / 1000.0,
        "gpu_runtime_ms_trace": cat_dur_us["cuda_runtime"] / 1000.0,
        "gpu_memcpy_ms_trace": cat_dur_us["gpu_memcpy"] / 1000.0,
        "gpu_memset_ms_trace": cat_dur_us["gpu_memset"] / 1000.0,
        "gpu_total_ms_trace": (
            cat_dur_us["kernel"] + cat_dur_us["cuda_runtime"] + cat_dur_us["gpu_memcpy"] + cat_dur_us["gpu_memset"]
        )
        / 1000.0,
        "cpu_gpu_ratio_trace": _safe_div(
            cat_dur_us["cpu_op"],
            (cat_dur_us["kernel"] + cat_dur_us["cuda_runtime"] + cat_dur_us["gpu_memcpy"] + cat_dur_us["gpu_memset"]),
        ),
        "train_batch_count": stats["train_batch"]["count"],
        "eval_batch_count": stats["eval_batch"]["count"],
        "eval_sample_count": stats["eval_sample"]["count"],
        "train_batch_ms": _safe_div(stats["train_batch"]["dur_us"], stats["train_batch"]["count"]) / 1000.0,
        "train_forward_ms": _safe_div(stats["train_forward"]["dur_us"], stats["train_forward"]["count"]) / 1000.0,
        "train_backward_ms": _safe_div(stats["train_backward"]["dur_us"], stats["train_backward"]["count"]) / 1000.0,
        "train_step_ms": _safe_div(stats["train_step"]["dur_us"], stats["train_step"]["count"]) / 1000.0,
        "eval_loss_forward_ms": _safe_div(stats["eval_loss_forward"]["dur_us"], stats["eval_loss_forward"]["count"]) / 1000.0,
        "eval_sample_ms": _safe_div(stats["eval_sample"]["dur_us"], stats["eval_sample"]["count"]) / 1000.0,
        "eval_contact_f1_ms": _safe_div(stats["eval_contact_f1"]["dur_us"], stats["eval_contact_f1"]["count"]) / 1000.0,
        "eval_no_f1_ms": _safe_div(eval_no_f1_us, eval_no_f1_count) / 1000.0,
        "eval_with_f1_ms": _safe_div(eval_with_f1_us, eval_sample_count) / 1000.0,
        "f1_overhead_pct_from_trace": (
            100.0
            * _safe_div(
                (_safe_div(eval_with_f1_us, eval_sample_count) - _safe_div(eval_no_f1_us, eval_no_f1_count)),
                _safe_div(eval_no_f1_us, eval_no_f1_count),
            )
            if eval_no_f1_count and eval_sample_count
            else 0.0
        ),
    }
    out.update(_phase_device_alloc_mb(events))
    return out


def _build_run_paths(config: dict[str, Any], run_name: str | None = None):
    timestamp = get_timestamp()
    resolved_name = run_name or config.get("run_name", "diffusion_profile")
    append_timestamp = bool(config.get("append_timestamp", True))
    run_id = f"{resolved_name}_{timestamp}" if append_timestamp else resolved_name

    folder_name = config.get("folder_name")
    run_folder = folder_name if folder_name else run_id

    run_dir = os.path.join(config["log_path"], run_folder)
    trace_dir = os.path.join(run_dir, "profiler_traces")
    return resolved_name, run_id, run_folder, run_dir, trace_dir


def run_profiler(config: dict[str, Any], run_name: str | None = None) -> dict[str, Any]:
    start_wall = time.time()
    resolved_name, run_id, run_folder, run_dir, trace_dir = _build_run_paths(config, run_name=run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(trace_dir, exist_ok=True)

    log_event = make_run_logger(run_dir, run_name=resolved_name)

    config_to_save = dict(config)
    config_to_save.update({
        "resolved_run_name": resolved_name,
        "run_id": run_id,
        "run_folder": run_folder,
    })
    save_config(config_to_save, run_dir)

    device = tr.device("cuda" if tr.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    use_amp = bool(config.get("use_amp", False)) and use_cuda
    amp_sampling = bool(config.get("amp_sampling", False))
    grad_accum_steps = max(int(config.get("grad_accum_steps", 1)), 1)

    log_event(
        "run_started",
        {
            "run_id": run_id,
            "run_folder": run_folder,
            "timesteps": int(config["timesteps"]),
            "batch_size": int(config["batch_size"]),
            "epochs": int(config["epochs"]),
            "use_amp": use_amp,
            "grad_accum_steps": grad_accum_steps,
            "device": str(device),
        },
    )

    num_workers = int(config.get("num_workers", 2))
    persistent_workers = num_workers > 0

    train_ds = SeqDataset(config["train_path"])
    val_ds = SeqDataset(config["val_path"])

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=pad_batch,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=pad_batch,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=persistent_workers,
    )

    model = load_model(config=config, eval=False)
    optimizer = tr.optim.Adam(model.parameters(), lr=config["lr"])
    scaler = GradScaler(enabled=use_amp)

    profile_batches = int(config.get("profile_batches", 4))
    trace_wait = int(config.get("trace_wait", 0))
    trace_warmup = int(config.get("trace_warmup", 1))
    trace_active = int(config.get("trace_active", 3 * profile_batches))

    record_shapes = bool(config.get("record_shapes", False))
    profile_memory = bool(config.get("profile_memory", True))
    with_stack = bool(config.get("with_stack", False))

    planned_steps = int(config["epochs"]) * (3 * profile_batches)
    required_steps = trace_wait + trace_warmup + trace_active

    if use_cuda:
        tr.cuda.reset_peak_memory_stats()

    activities = [ProfilerActivity.CPU]
    if use_cuda:
        activities.append(ProfilerActivity.CUDA)

    status = "ok"
    error_message = ""
    prof = None

    try:
        with profile(
            activities=activities,
            schedule=schedule(wait=trace_wait, warmup=trace_warmup, active=trace_active, repeat=1),
            on_trace_ready=tensorboard_trace_handler(trace_dir),
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
        ) as prof:
            for _ in range(1, int(config["epochs"]) + 1):
                with record_function("epoch_train"):
                    train_one_epoch_profiled(
                        model,
                        train_loader,
                        optimizer,
                        device,
                        prof=prof,
                        max_batches=profile_batches,
                        use_amp=use_amp,
                        scaler=scaler,
                        grad_accum_steps=grad_accum_steps,
                        amp_dtype=tr.float16,
                    )

                with record_function("epoch_eval_no_f1"):
                    eval_one_epoch_profiled(
                        model,
                        val_loader,
                        device,
                        compute_f1=False,
                        prof=prof,
                        max_batches=profile_batches,
                        use_amp=use_amp,
                        amp_dtype=tr.float16,
                        amp_sampling=False,
                    )

                with record_function("epoch_eval_with_f1"):
                    eval_one_epoch_profiled(
                        model,
                        val_loader,
                        device,
                        compute_f1=True,
                        prof=prof,
                        max_batches=profile_batches,
                        use_amp=use_amp,
                        amp_dtype=tr.float16,
                        amp_sampling=amp_sampling,
                    )
    except RuntimeError as err:
        msg = str(err)
        if "out of memory" in msg.lower():
            status = "oom"
            error_message = msg
            log_event("run_error", {"kind": "oom", "message": msg}, level="error")
            if use_cuda:
                tr.cuda.empty_cache()
        else:
            log_event("run_error", {"kind": "runtime", "message": msg}, level="error")
            raise

    peak_alloc_gb = tr.cuda.max_memory_allocated() / (1024 ** 3) if use_cuda else 0.0
    peak_reserved_gb = tr.cuda.max_memory_reserved() / (1024 ** 3) if use_cuda else 0.0

    top_summary = {
        "eval_no_f1_ms_keyavg": 0.0,
        "eval_with_f1_ms_keyavg": 0.0,
        "eval_sample_ms_keyavg": 0.0,
        "eval_contact_f1_ms_keyavg": 0.0,
        "prof_cpu_self_ms_keyavg": 0.0,
        "prof_cuda_self_ms_keyavg": 0.0,
        "train_forward_cpu_ms_keyavg": 0.0,
        "train_forward_cuda_ms_keyavg": 0.0,
        "train_backward_cpu_ms_keyavg": 0.0,
        "train_backward_cuda_ms_keyavg": 0.0,
        "eval_loss_forward_cpu_ms_keyavg": 0.0,
        "eval_loss_forward_cuda_ms_keyavg": 0.0,
        "eval_sample_cpu_ms_keyavg": 0.0,
        "eval_sample_cuda_ms_keyavg": 0.0,
    }

    if prof is not None:
        events = prof.key_averages()
        cpu_self_ms, cuda_self_ms = _keyavg_self_totals_ms(events)
        top_summary = {
            "eval_no_f1_ms_keyavg": _safe_total_ms(events, "epoch_eval_no_f1", self_only=False),
            "eval_with_f1_ms_keyavg": _safe_total_ms(events, "epoch_eval_with_f1", self_only=False),
            "eval_sample_ms_keyavg": _safe_total_ms(events, "eval_sample", self_only=False),
            "eval_contact_f1_ms_keyavg": _safe_total_ms(events, "eval_contact_f1", self_only=False),
            "prof_cpu_self_ms_keyavg": cpu_self_ms,
            "prof_cuda_self_ms_keyavg": cuda_self_ms,
            "train_forward_cpu_ms_keyavg": _safe_total_ms(events, "train_forward", use_cuda=False, self_only=False),
            "train_forward_cuda_ms_keyavg": _safe_total_ms(events, "train_forward", use_cuda=True, self_only=False),
            "train_backward_cpu_ms_keyavg": _safe_total_ms(events, "train_backward", use_cuda=False, self_only=False),
            "train_backward_cuda_ms_keyavg": _safe_total_ms(events, "train_backward", use_cuda=True, self_only=False),
            "eval_loss_forward_cpu_ms_keyavg": _safe_total_ms(events, "eval_loss_forward", use_cuda=False, self_only=False),
            "eval_loss_forward_cuda_ms_keyavg": _safe_total_ms(events, "eval_loss_forward", use_cuda=True, self_only=False),
            "eval_sample_cpu_ms_keyavg": _safe_total_ms(events, "eval_sample", use_cuda=False, self_only=False),
            "eval_sample_cuda_ms_keyavg": _safe_total_ms(events, "eval_sample", use_cuda=True, self_only=False),
        }

    trace_path = _latest_trace_file(trace_dir)
    if trace_path:
        log_event("trace_saved", {"trace_path": trace_path, "trace_size_mb": os.path.getsize(trace_path) / (1024 ** 2)})

    trace_metrics = analyze_trace_file(trace_path)
    wall_time_s = time.time() - start_wall

    result = {
        "status": status,
        "error": error_message,
        "run_name": resolved_name,
        "run_id": run_id,
        "run_folder": run_folder,
        "run_dir": run_dir,
        "trace_dir": trace_dir,
        "trace_path": trace_path,
        "timesteps": int(config["timesteps"]),
        "batch_size": int(config["batch_size"]),
        "epochs": int(config["epochs"]),
        "profile_batches": profile_batches,
        "use_amp": use_amp,
        "grad_accum_steps": grad_accum_steps,
        "amp_sampling": amp_sampling,
        "planned_steps": planned_steps,
        "required_steps": required_steps,
        "wall_time_s": wall_time_s,
        "peak_alloc_gb_runtime": peak_alloc_gb,
        "peak_reserved_gb_runtime": peak_reserved_gb,
        **top_summary,
        **trace_metrics,
    }

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(result), f, indent=2)

    log_event("summary_saved", {"summary_path": summary_path})
    log_event(
        "run_finished",
        {
            "status": status,
            "wall_time_s": wall_time_s,
            "trace_found": bool(trace_metrics.get("trace_found")),
            "oom_events": int(trace_metrics.get("oom_events", 0) or 0),
        },
        level="error" if status != "ok" else "info",
    )

    return result


def run_profiler_grad_acc(
    config: dict[str, Any],
    grad_accum_steps: int = 1,
    run_name: str | None = None,
) -> dict[str, Any]:
    cfg = dict(config)
    cfg["grad_accum_steps"] = max(int(grad_accum_steps), 1)
    return run_profiler(cfg, run_name=run_name)


def run_resource_sweep(
    base_conf: dict[str, Any],
    timesteps_list,
    batch_sizes=(4,),
    amp_modes=(False, True),
) -> pd.DataFrame:
    results = []
    for ts in timesteps_list:
        for bs in batch_sizes:
            for amp in amp_modes:
                cfg = dict(base_conf)
                cfg["timesteps"] = int(ts)
                cfg["batch_size"] = int(bs)
                cfg["use_amp"] = bool(amp)
                suffix = "amp" if amp else "fp32"
                pb = int(cfg.get("profile_batches", 4))
                run_name = f"{cfg.get('sim_tag', 'sim')}_bs{bs}_pb{pb}_{ts}ts_{suffix}"
                cfg["run_name"] = run_name
                cfg["folder_name"] = run_name
                cfg["append_timestamp"] = False
                try:
                    results.append(run_profiler(cfg))
                except Exception as err:  # noqa: PERF203
                    results.append(
                        {
                            "status": "error",
                            "error": str(err),
                            "timesteps": int(ts),
                            "batch_size": int(bs),
                            "use_amp": bool(amp),
                            "run_name": run_name,
                        }
                    )
    return pd.DataFrame(results)


def run_resource_sweep_grad_acc(
    base_conf: dict[str, Any],
    timesteps_list,
    batch_sizes=(4,),
    grad_accum_steps_list=(1,),
    amp_modes=(False, True),
    output_log_path: str | Path | None = None,
    run_name_prefix: str | None = None,
) -> pd.DataFrame:
    """
    Controlled sweep over timesteps, batch_size and grad_accum_steps.
    Use output_log_path to store results in a separate logs folder.
    """
    results = []
    target_log_path = str(output_log_path) if output_log_path is not None else str(base_conf.get("log_path", "logs/diffusion_profiler_light"))
    os.makedirs(target_log_path, exist_ok=True)

    for ts in timesteps_list:
        for bs in batch_sizes:
            for ga in grad_accum_steps_list:
                for amp in amp_modes:
                    cfg = dict(base_conf)
                    cfg["timesteps"] = int(ts)
                    cfg["batch_size"] = int(bs)
                    cfg["grad_accum_steps"] = max(int(ga), 1)
                    cfg["use_amp"] = bool(amp)
                    cfg["log_path"] = target_log_path

                    suffix = "amp" if amp else "fp32"
                    pb = int(cfg.get("profile_batches", 4))
                    sim_tag = str(cfg.get("sim_tag", "sim"))
                    prefix = f"{run_name_prefix}_" if run_name_prefix else ""
                    run_name = f"{prefix}{sim_tag}_bs{bs}_ga{cfg['grad_accum_steps']}_pb{pb}_{ts}ts_{suffix}"
                    cfg["run_name"] = run_name
                    cfg["folder_name"] = run_name
                    cfg["append_timestamp"] = False

                    try:
                        results.append(run_profiler(cfg))
                    except Exception as err:  # noqa: PERF203
                        results.append(
                            {
                                "status": "error",
                                "error": str(err),
                                "timesteps": int(ts),
                                "batch_size": int(bs),
                                "grad_accum_steps": max(int(ga), 1),
                                "use_amp": bool(amp),
                                "run_name": run_name,
                                "log_path": target_log_path,
                            }
                        )
    return pd.DataFrame(results)


def _resolve_trace_path(run_dir: str, trace_path: str | None, trace_dir: str | None = None) -> str | None:
    if isinstance(trace_path, float) and pd.isna(trace_path):
        trace_path = None
    if trace_path:
        p = Path(str(trace_path))
        candidates: list[Path] = []

        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(Path.cwd() / p)
            candidates.append(Path(run_dir) / p)
            base = Path(run_dir)
            for depth, parent in enumerate(base.parents):
                if depth >= 5:
                    break
                candidates.append(parent / p)

        for cand in candidates:
            if cand.exists():
                return str(cand)

    if trace_dir:
        latest = _latest_trace_file(trace_dir)
        if latest:
            return latest
    return trace_path


def _needs_phase_alloc_refresh(row: dict[str, Any]) -> bool:
    keys = _phase_alloc_metric_keys()
    if any(k not in row for k in keys):
        return True
    vals = [row.get(k) for k in keys]
    return all(pd.isna(v) for v in vals)


def collect_saved_runs(log_root: str | Path) -> pd.DataFrame:
    log_root = str(log_root)
    rows = []

    for cfg_path in sorted(glob.glob(os.path.join(log_root, "*", "config.json"))):
        run_dir = os.path.dirname(cfg_path)
        summary_path = os.path.join(run_dir, "summary.json")

        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                row = json.load(f)

            trace_dir = os.path.join(run_dir, "profiler_traces")
            trace_path = _resolve_trace_path(run_dir, row.get("trace_path"), trace_dir=trace_dir)
            row["trace_path"] = trace_path

            if _needs_phase_alloc_refresh(row):
                row.update(analyze_trace_file(trace_path))
            rows.append(row)
            continue

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        trace_dir = os.path.join(run_dir, "profiler_traces")
        trace_path = _latest_trace_file(trace_dir)
        trace_metrics = analyze_trace_file(trace_path)

        row = {
            "status": "unknown",
            "run_dir": run_dir,
            "trace_dir": trace_dir,
            "trace_path": trace_path,
            "timesteps": cfg.get("timesteps"),
            "batch_size": cfg.get("batch_size"),
            "use_amp": cfg.get("use_amp", str(cfg.get("run_name", "")).endswith("_amp")),
            "run_name": cfg.get("run_name", os.path.basename(run_dir)),
            **trace_metrics,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "timesteps" in df.columns:
        df = df.sort_values(["timesteps", "use_amp"], ascending=[True, True]).reset_index(drop=True)
    return df


def build_core_summary(df: pd.DataFrame) -> pd.DataFrame:
    core_cols = [
        "run_name",
        "status",
        "timesteps",
        "batch_size",
        "grad_accum_steps",
        "use_amp",
        "oom_events",
        "trace_path",
        "trace_size_mb",
        "train_batch_ms",
        "train_forward_ms",
        "train_backward_ms",
        "train_step_ms",
        "eval_no_f1_ms",
        "eval_with_f1_ms",
        "max_alloc_gb_trace",
        "max_reserved_gb_trace",
        "cpu_op_ms_trace",
        "gpu_total_ms_trace",
        "train_forward_alloc_delta_avg_mb_gpu",
        "train_forward_alloc_delta_peak_mb_gpu",
        "train_backward_alloc_delta_avg_mb_gpu",
        "train_backward_alloc_delta_peak_mb_gpu",
        "train_step_alloc_delta_avg_mb_gpu",
        "train_step_alloc_delta_peak_mb_gpu",
        "train_forward_alloc_delta_avg_mb_cpu",
        "train_forward_alloc_delta_peak_mb_cpu",
        "train_backward_alloc_delta_avg_mb_cpu",
        "train_backward_alloc_delta_peak_mb_cpu",
        "train_step_alloc_delta_avg_mb_cpu",
        "train_step_alloc_delta_peak_mb_cpu",
        "train_forward_alloc_traffic_mb_gpu",
        "train_forward_free_traffic_mb_gpu",
        "train_backward_alloc_traffic_mb_gpu",
        "train_backward_free_traffic_mb_gpu",
        "train_step_alloc_traffic_mb_gpu",
        "train_step_free_traffic_mb_gpu",
        "train_forward_alloc_traffic_mb_cpu",
        "train_forward_free_traffic_mb_cpu",
        "train_backward_alloc_traffic_mb_cpu",
        "train_backward_free_traffic_mb_cpu",
        "train_step_alloc_traffic_mb_cpu",
        "train_step_free_traffic_mb_cpu",
    ]
    cols = [c for c in core_cols if c in df.columns]
    return df[cols].copy()


def export_core_summary(
    log_root: str | Path,
    output_csv: str | Path,
    output_full_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    saved_df = collect_saved_runs(log_root)
    if len(saved_df) == 0:
        return saved_df, pd.DataFrame()

    core_df = build_core_summary(saved_df)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    core_df.to_csv(output_csv, index=False)

    if output_full_csv is not None:
        Path(output_full_csv).parent.mkdir(parents=True, exist_ok=True)
        saved_df.to_csv(output_full_csv, index=False)

    return saved_df, core_df
