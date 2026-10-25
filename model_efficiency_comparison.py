#!/usr/bin/env python
"""Train and compare ExDM/ExDM with forecasting baselines.

The default experiments are:
  * NP:     seq_len=168, pred_len=24, MS forecasting (target=OT)
  * Energy: seq_len=720, pred_len=360, MS forecasting (last column target)

For every model this script records test MSE, mean training seconds per epoch,
peak allocated CUDA memory, trainable parameters, and forward-pass latency.  It
also creates a two-panel bubble chart plus Markdown/LaTeX tables.

Examples (run from ExDM7.8 or any other directory):
  python model_efficiency_comparison.py
  python model_efficiency_comparison.py --models ExDM DLinear --epochs 2
  python model_efficiency_comparison.py --plot-only

The eight baselines available in GCGNet are imported from
../GCGNet-main/ts_benchmark/baselines.  That checkout does not contain an
XLinear adapter, so XLinear is loaded from ../XLinear-main/models/XLinear.py.
No model implementation is copied into this file.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
GCG_ROOT = WORKSPACE_DIR / "GCGNet-main"
XLINEAR_FILE = WORKSPACE_DIR / "XLinear-main" / "models" / "XLinear.py"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(GCG_ROOT) not in sys.path:
    sys.path.insert(0, str(GCG_ROOT))


MODEL_NAMES = (
    "ExDM",
    "XLinear",
    "TimeXer",
    "TimeMixer",
    "iTransformer",
    "PatchTST",
    "Crossformer",
    "TiDE",
    "TimesNet",
    "DLinear",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    data: str
    root_path: Path
    data_path: str
    seq_len: int
    pred_len: int
    channels: int
    batch_size: int
    learning_rate: float
    exdm_d_model: int
    exdm_dff: int


DATASETS: Dict[str, DatasetSpec] = {
    "NP": DatasetSpec(
        name="NP",
        data="custom",
        root_path=SCRIPT_DIR / "data",
        data_path="NP_exdm.csv",
        seq_len=168,
        pred_len=24,
        channels=3,
        batch_size=32,
        learning_rate=5e-4,
        exdm_d_model=64,
        exdm_dff=64,
    ),
    "Energy": DatasetSpec(
        name="Energy",
        data="custom",
        root_path=SCRIPT_DIR / "data",
        data_path="Energy_exdm.csv",
        seq_len=720,
        pred_len=360,
        channels=6,
        batch_size=32,
        learning_rate=3.8e-4,
        exdm_d_model=512,
        exdm_dff=384,
    ),
}


@dataclass
class Result:
    dataset: str
    model: str
    status: str
    seq_len: int
    pred_len: int
    batch_size: int
    epochs_requested: int
    epochs_trained: int
    best_val_mse: float
    mse: float
    train_time_s_epoch: float
    peak_gpu_gb: float
    parameters: int
    parameters_million: float
    inference_ms_batch: float
    inference_ms_sample: float
    device: str
    source: str
    error: str = ""


class UnifiedForecaster(nn.Module):
    """Give all referenced implementations one target-only forward API."""

    def __init__(self, model: nn.Module, kind: str, pred_len: int, channels: int):
        super().__init__()
        self.model = model
        self.kind = kind
        self.pred_len = pred_len
        self.channels = channels

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_mark: torch.Tensor,
        y_mark: torch.Tensor,
        cycle: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.kind == "ExDM":
            out = self.model(x, batch_cycle=cycle)
        elif self.kind in {"XLinear", "DLinear"}:
            out = self.model(x)
        elif self.kind == "TimeXer":
            # GCGNet TimeXer expects predicted series first and exogenous
            # channels after it; ExDM datasets keep OT as the last channel.
            reordered = torch.cat([x[:, :, -1:], x[:, :, :-1]], dim=-1)
            unused_future_exog = x.new_zeros(
                x.shape[0], self.pred_len, max(self.channels - 1, 0)
            )
            out = self.model(reordered, unused_future_exog)
        else:
            zeros = torch.zeros_like(y[:, -self.pred_len :, :])
            dec = torch.cat([y[:, : y.shape[1] - self.pred_len, :], zeros], dim=1)
            out = self.model(x, x_mark, dec, y_mark)

        if isinstance(out, tuple):
            out = out[0]
        out = out[:, -self.pred_len :, :]
        if out.shape[-1] == self.channels:
            out = out[:, :, -1:]
        if out.shape[-1] != 1:
            raise RuntimeError(
                f"{self.kind} returned {tuple(out.shape)}; expected one target channel"
            )
        return out


def _namespace(defaults: Dict, spec: DatasetSpec, **overrides) -> SimpleNamespace:
    values = dict(defaults)
    values.update(
        seq_len=spec.seq_len,
        pred_len=spec.pred_len,
        horizon=spec.pred_len,
        label_len=min(48, spec.seq_len),
        enc_in=spec.channels,
        dec_in=spec.channels,
        c_out=spec.channels,
        task_name="long_term_forecast",
        freq="h",
        output_attention=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _load_xlinear_class():
    if not XLINEAR_FILE.exists():
        raise FileNotFoundError(
            f"XLinear was not found at {XLINEAR_FILE}. Clone/place XLinear-main "
            "beside ExDM7.8 or omit XLinear with --models."
        )
    module_spec = importlib.util.spec_from_file_location(
        "efficiency_xlinear_model", XLINEAR_FILE
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Cannot load XLinear module from {XLINEAR_FILE}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.Model


def build_model(name: str, spec: DatasetSpec, device: torch.device) -> Tuple[nn.Module, str]:
    """Instantiate the original model class, using its benchmark defaults."""
    common = dict(
        dropout=0.1,
        activation="gelu",
        moving_avg=25,
        individual=False,
        d_model=512,
        d_ff=2048,
        e_layers=2,
        d_layers=1,
        n_heads=8,
        factor=1,
        embed="timeF",
        patch_len=16,
        stride=8,
        top_k=5,
        num_kernels=6,
        use_norm=1,
        use_future_exog=False,
    )

    if name == "ExDM":
        from models.ExDM import Model

        cfg = _namespace(
            common,
            spec,
            feature_num=spec.channels,
            input_dim=spec.channels - 1,
            output_dim=1,
            fused_exogenous=spec.channels - 1,
            fused_endogenous=1,
            revin=1,
            ch_ind=1,
            residual=1,
            d_model=spec.exdm_d_model,
            dff=spec.exdm_dff,
            d_state=256,
            dconv=2,
            e_fact=1,
            dropout=0.02 if spec.name == "NP" else 0.05,
            direct_weight=0.95 if spec.name == "NP" else 0.72,
            direct_dropout=0.0 if spec.name == "NP" else 0.05,
            direct_loss_weight=0.24 if spec.name == "NP" else 0.10,
            sample_alpha=1.0 if spec.name == "NP" else 0.5,
            sample_p=0.1,
            ma_type="ema" if spec.name == "NP" else "dema",
            alpha=0.3 if spec.name == "NP" else 0.25,
            beta=0.3,
            endl_t_ff=spec.exdm_d_model,
            endl_c_ff=16,
            endl_t_dropout=0.0,
            endl_c_dropout=0.0,
            endl_embed_dropout=0.02 if spec.name == "NP" else 0.05,
            endl_head_dropout=0.0 if spec.name == "NP" else 0.02,
            invnorm_p=1.0 if spec.name == "NP" else 0.8,
            invnorm_sigma=0.1 if spec.name == "NP" else 0.15,
            exodiff_steps=100 if spec.name == "NP" else 50,
            exodiff_hidden_dim=128,
            device=str(device),
        )
        raw_model = Model(cfg)
        source = "ExDM7.8/models/ExDM.py"

    elif name == "XLinear":
        Model = _load_xlinear_class()
        cfg = _namespace(
            common,
            spec,
            features="MS",
            d_model=512,
            t_ff=128,
            c_ff=spec.channels,
            usenorm=1,
            embed_dropout=0.0,
            head_dropout=0.2,
            t_dropout=0.0,
            c_dropout=0.0,
        )
        raw_model = Model(cfg)
        source = "XLinear-main/models/XLinear.py"

    elif name == "TimeXer":
        from ts_benchmark.baselines.timexer.timexer import MODEL_HYPER_PARAMS
        from ts_benchmark.baselines.timexer.model.timexer_model import timexer_model

        cfg = _namespace(
            MODEL_HYPER_PARAMS,
            spec,
            series_dim=1,
            input_dim=spec.channels,
            output_dim=1,
            use_future=0,
        )
        raw_model = timexer_model(cfg)
        source = "GCGNet-main/ts_benchmark/baselines/timexer"

    elif name == "TimeMixer":
        from ts_benchmark.baselines.timemixer.timemixer import MODEL_HYPER_PARAMS
        from ts_benchmark.baselines.timemixer.models.timemixer_model import Model

        cfg = _namespace(MODEL_HYPER_PARAMS, spec)
        raw_model = Model(cfg)
        source = "GCGNet-main/ts_benchmark/baselines/timemixer"

    elif name == "iTransformer":
        from ts_benchmark.baselines.itransformer.itransformer import MODEL_HYPER_PARAMS
        from ts_benchmark.baselines.itransformer.models.itransformer_model import Model

        cfg = _namespace(MODEL_HYPER_PARAMS, spec)
        raw_model = Model(cfg)
        source = "GCGNet-main/ts_benchmark/baselines/itransformer"

    elif name == "TimesNet":
        from ts_benchmark.baselines.timesnet.timesnet import MODEL_HYPER_PARAMS
        from ts_benchmark.baselines.timesnet.models.timesnet_model import Model

        cfg = _namespace(MODEL_HYPER_PARAMS, spec)
        raw_model = Model(cfg)
        source = "GCGNet-main/ts_benchmark/baselines/timesnet"

    elif name == "Crossformer":
        from ts_benchmark.baselines.crossformer.crossformer import MODEL_HYPER_PARAMS
        from ts_benchmark.baselines.time_series_library.models.Crossformer import Crossformer

        cfg = _namespace(MODEL_HYPER_PARAMS, spec)
        raw_model = Crossformer(cfg)
        source = "GCGNet-main/ts_benchmark/baselines/crossformer"

    elif name in {"PatchTST", "TiDE"}:
        from ts_benchmark.baselines.time_series_library.adapters_for_transformers import (
            MODEL_HYPER_PARAMS,
        )

        cfg = _namespace(MODEL_HYPER_PARAMS, spec, use_future_exog=False)
        if name == "PatchTST":
            from ts_benchmark.baselines.time_series_library.models.PatchTST import PatchTST

            raw_model = PatchTST(cfg)
        else:
            from ts_benchmark.baselines.time_series_library.models.TiDE import TiDE

            raw_model = TiDE(cfg)
        source = f"GCGNet-main/ts_benchmark/baselines/time_series_library/models/{name}.py"

    elif name == "DLinear":
        from ts_benchmark.baselines.dlinear.dlinear import MODEL_HYPER_PARAMS
        from ts_benchmark.baselines.dlinear.models.dlinear_model import DLinearModel

        cfg = _namespace(MODEL_HYPER_PARAMS, spec)
        raw_model = DLinearModel(cfg)
        source = "GCGNet-main/ts_benchmark/baselines/dlinear"
    else:
        raise ValueError(f"Unknown model: {name}")

    return UnifiedForecaster(raw_model, name, spec.pred_len, spec.channels), source


def make_data_args(spec: DatasetSpec, batch_size: int, workers: int) -> SimpleNamespace:
    return SimpleNamespace(
        data=spec.data,
        root_path=str(spec.root_path),
        data_path=spec.data_path,
        seq_len=spec.seq_len,
        label_len=min(48, spec.seq_len),
        pred_len=spec.pred_len,
        features="MS",
        target="OT",
        embed="timeF",
        freq="h",
        cycle=24,
        batch_size=batch_size,
        num_workers=workers,
    )


def make_loaders(spec: DatasetSpec, batch_size: int, workers: int):
    from data_provider.data_factory import data_provider

    if not (spec.root_path / spec.data_path).exists():
        raise FileNotFoundError(f"Dataset file not found: {spec.root_path / spec.data_path}")
    args = make_data_args(spec, batch_size, workers)
    return tuple(data_provider(args, split)[1] for split in ("train", "val", "test"))


def unpack_batch(batch, device: torch.device):
    x, y, x_mark, y_mark = batch[:4]
    cycle = batch[4] if len(batch) > 4 else None
    x = x.float().to(device, non_blocking=True)
    y = y.float().to(device, non_blocking=True)
    x_mark = x_mark.float().to(device, non_blocking=True)
    y_mark = y_mark.float().to(device, non_blocking=True)
    if cycle is not None:
        cycle = cycle.long().to(device, non_blocking=True)
    return x, y, x_mark, y_mark, cycle


def limited(loader: Iterable, max_batches: Optional[int]):
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        yield batch


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    model.eval()
    squared_error = 0.0
    element_count = 0
    for batch in limited(loader, max_batches):
        x, y, x_mark, y_mark, cycle = unpack_batch(batch, device)
        pred = model(x, y, x_mark, y_mark, cycle)
        target = y[:, -pred.shape[1] :, -1:]
        squared_error += torch.sum((pred - target) ** 2).item()
        element_count += target.numel()
    return squared_error / max(element_count, 1)


def clone_state_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    patience: int,
    learning_rate: float,
    amp: bool,
    max_train_batches: Optional[int],
    max_eval_batches: Optional[int],
) -> Tuple[List[float], int, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_val = math.inf
    best_state = None
    stale_epochs = 0
    epoch_times: List[float] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        total_loss = 0.0
        batches = 0
        for batch in limited(train_loader, max_train_batches):
            x, y, x_mark, y_mark, cycle = unpack_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                pred = model(x, y, x_mark, y_mark, cycle)
                target = y[:, -pred.shape[1] :, -1:]
                loss = criterion(pred, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach().item()
            batches += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_times.append(time.perf_counter() - start)

        val_mse = evaluate(model, val_loader, device, max_eval_batches)
        print(
            f"    epoch {epoch + 1:>3}/{epochs}: "
            f"train_mse={total_loss / max(batches, 1):.6f}, "
            f"val_mse={val_mse:.6f}, time={epoch_times[-1]:.2f}s"
        )
        if val_mse < best_val:
            best_val = val_mse
            best_state = clone_state_to_cpu(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if patience > 0 and stale_epochs >= patience:
                print(f"    early stopping after {epoch + 1} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return epoch_times, len(epoch_times), best_val


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    loader,
    device: torch.device,
    warmup: int,
    runs: int,
) -> Tuple[float, float]:
    model.eval()
    batch = next(iter(loader))
    x, y, x_mark, y_mark, cycle = unpack_batch(batch, device)
    for _ in range(warmup):
        model(x, y, x_mark, y_mark, cycle)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    samples: List[float] = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start = time.perf_counter()
        model(x, y, x_mark, y_mark, cycle)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        samples.append((time.perf_counter() - start) * 1000.0)
    latency_batch = float(np.mean(samples))
    return latency_batch, latency_batch / x.shape[0]


def result_key(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("model", "")),
        str(row.get("seq_len", "")),
        str(row.get("pred_len", "")),
    )


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(Result("", "", "", 0, 0, 0, 0, 0, math.nan,
                                     math.nan, math.nan, math.nan, 0, math.nan,
                                     math.nan, math.nan, "", "")).keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def upsert_result(path: Path, result: Result):
    rows = read_rows(path)
    key = (
        result.dataset,
        result.model,
        str(result.seq_len),
        str(result.pred_len),
    )
    rows = [row for row in rows if result_key(row) != key]
    rows.append(asdict(result))
    order = {name: i for i, name in enumerate(MODEL_NAMES)}
    dataset_order = {name: i for i, name in enumerate(DATASETS)}
    rows.sort(key=lambda row: (dataset_order.get(row["dataset"], 99), order.get(row["model"], 99)))
    write_rows(path, rows)


def clean_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_one(spec: DatasetSpec, name: str, args, device: torch.device) -> Result:
    batch_size = args.batch_size or spec.batch_size
    print(f"\n[{spec.name}] {name}: L={spec.seq_len}, H={spec.pred_len}, batch={batch_size}")
    train_loader, val_loader, test_loader = make_loaders(spec, batch_size, args.workers)
    model, source = build_model(name, spec, device)
    model = model.to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    epoch_times, epochs_trained, best_val = train_model(
        model,
        train_loader,
        val_loader,
        device,
        args.epochs,
        args.patience,
        args.learning_rate or spec.learning_rate,
        args.amp and device.type == "cuda",
        args.max_train_batches,
        args.max_eval_batches,
    )
    peak_gpu_gb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if device.type == "cuda"
        else math.nan
    )
    test_mse = evaluate(model, test_loader, device, args.max_eval_batches)
    latency_batch, latency_sample = measure_latency(
        model, test_loader, device, args.latency_warmup, args.latency_runs
    )
    print(
        f"    test_mse={test_mse:.6f}, params={parameters / 1e6:.3f}M, "
        f"time={np.mean(epoch_times):.2f}s/epoch, peak_gpu={peak_gpu_gb:.3f}GB, "
        f"latency={latency_batch:.3f}ms/batch"
    )
    return Result(
        dataset=spec.name,
        model=name,
        status="ok",
        seq_len=spec.seq_len,
        pred_len=spec.pred_len,
        batch_size=batch_size,
        epochs_requested=args.epochs,
        epochs_trained=epochs_trained,
        best_val_mse=best_val,
        mse=test_mse,
        train_time_s_epoch=float(np.mean(epoch_times)),
        peak_gpu_gb=peak_gpu_gb,
        parameters=parameters,
        parameters_million=parameters / 1e6,
        inference_ms_batch=latency_batch,
        inference_ms_sample=latency_sample,
        device=str(device),
        source=source,
    )


def _float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def successful_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    selected = []
    for row in rows:
        spec = DATASETS.get(row.get("dataset", ""))
        if spec is None:
            continue
        if str(row.get("seq_len", "")) != str(spec.seq_len):
            continue
        if str(row.get("pred_len", "")) != str(spec.pred_len):
            continue
        if row.get("status") == "ok" and math.isfinite(_float(row, "mse")):
            selected.append(row)
    return selected


def create_bubble_plot(rows: Sequence[Dict[str, str]], output_dir: Path):
    import matplotlib.pyplot as plt

    good = successful_rows(rows)
    if not good:
        print("No successful rows available for plotting.")
        return
    colors = dict(zip(MODEL_NAMES, plt.cm.tab10(np.linspace(0, 1, len(MODEL_NAMES)))))
    datasets = [name for name in DATASETS if any(row["dataset"] == name for row in good)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.2 * len(datasets), 5.4), squeeze=False)

    for axis, dataset in zip(axes[0], datasets):
        subset = [row for row in good if row["dataset"] == dataset]
        memories = [_float(row, "peak_gpu_gb") for row in subset]
        finite_memory = [value for value in memories if math.isfinite(value) and value > 0]
        fallback_memory = min(finite_memory) if finite_memory else 0.25
        for index, row in enumerate(subset):
            train_time = _float(row, "train_time_s_epoch")
            mse = _float(row, "mse")
            memory = _float(row, "peak_gpu_gb")
            bubble_memory = memory if math.isfinite(memory) and memory > 0 else fallback_memory
            size = max(
                80.0,
                600.0 * bubble_memory / max(finite_memory or [fallback_memory]),
            )
            axis.scatter(
                train_time,
                mse,
                s=size,
                color=colors.get(row["model"], "gray"),
                alpha=0.72,
                edgecolor="white",
                linewidth=1.1,
                zorder=3,
            )
            memory_text = f"{memory:.2f}GB" if math.isfinite(memory) else "CPU"
            x_offset = 8 if index % 2 == 0 else -8
            y_offset = 12 if (index // 2) % 2 == 0 else -20
            axis.annotate(
                f"{row['model']}\n{memory_text}, {train_time:.2f}s",
                (train_time, mse),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha="left" if x_offset > 0 else "right",
                fontsize=8.3,
                color=colors.get(row["model"], "black"),
            )
        spec = DATASETS[dataset]
        axis.set_title(f"{dataset} (L/H = {spec.seq_len}/{spec.pred_len})")
        axis.set_xlabel("Training time (s/epoch)")
        axis.set_ylabel("MSE")
        axis.grid(True, linestyle="--", alpha=0.28, zorder=0)
        axis.margins(x=0.18, y=0.18)

    fig.suptitle("Model efficiency comparison (bubble area = peak GPU memory)", fontsize=13)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "model_efficiency_bubble.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "model_efficiency_bubble.pdf", bbox_inches="tight")
    plt.close(fig)


def create_tables(rows: Sequence[Dict[str, str]], output_dir: Path):
    good = successful_rows(rows)
    if not good:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    headers = ["Dataset", "Model", "Params (M)", "Latency (ms/batch)", "MSE", "s/epoch", "GPU (GB)"]
    table_rows = []
    for row in good:
        memory = _float(row, "peak_gpu_gb")
        table_rows.append(
            [
                row["dataset"],
                row["model"],
                f"{_float(row, 'parameters_million'):.3f}",
                f"{_float(row, 'inference_ms_batch'):.3f}",
                f"{_float(row, 'mse'):.6f}",
                f"{_float(row, 'train_time_s_epoch'):.2f}",
                f"{memory:.3f}" if math.isfinite(memory) else "CPU",
            ]
        )

    widths = [max(len(headers[i]), *(len(row[i]) for row in table_rows)) for i in range(len(headers))]
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    md_lines.extend("| " + " | ".join(row) + " |" for row in table_rows)
    (output_dir / "model_efficiency_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Model efficiency on NP and Energy. Latency is measured per batch.}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        "Dataset & Model & Params (M) & Latency (ms) & MSE & s/epoch & GPU (GB) \\\\",
        r"\midrule",
    ]
    previous_dataset = None
    for row in table_rows:
        if previous_dataset is not None and row[0] != previous_dataset:
            latex_lines.append(r"\midrule")
        escaped = [value.replace("_", r"\_") for value in row]
        latex_lines.append(" & ".join(escaped) + r" \\")
        previous_dataset = row[0]
    latex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (output_dir / "model_efficiency_table.tex").write_text(
        "\n".join(latex_lines) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--epochs", type=int, default=10, help="maximum epochs for every model")
    parser.add_argument("--patience", type=int, default=3, help="early-stopping patience; 0 disables it")
    parser.add_argument("--batch-size", type=int, default=None, help="override dataset batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="override dataset learning rate")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--amp", action="store_true", help="use CUDA automatic mixed precision")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-runs", type=int, default=50)
    parser.add_argument("--max-train-batches", type=int, default=None, help="debug/profiling subset")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="debug/profiling subset")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "efficiency_results")
    parser.add_argument("--plot-only", action="store_true", help="regenerate figures/tables from CSV")
    parser.add_argument("--overwrite", action="store_true", help="rerun completed dataset/model pairs")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def validate_args(args):
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.latency_runs <= 0 or args.latency_warmup < 0:
        raise ValueError("latency runs must be positive and warmup non-negative")


def main():
    args = parse_args()
    validate_args(args)
    args.output_dir = args.output_dir.resolve()
    results_path = args.output_dir / "model_efficiency_results.csv"

    if args.plot_only:
        rows = read_rows(results_path)
        create_bubble_plot(rows, args.output_dir)
        create_tables(rows, args.output_dir)
        print(f"Outputs regenerated in {args.output_dir}")
        return

    device = choose_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        # Keep CUDA timing and model execution on the explicitly selected GPU.
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(device)
        print(f"Device: {device} ({gpu_name})")
    else:
        print("Device: CPU (GPU memory will be reported as NaN/CPU)")

    existing = {result_key(row): row for row in read_rows(results_path)}
    for dataset_name in args.datasets:
        spec = DATASETS[dataset_name]
        for model_name in args.models:
            key = (
                dataset_name,
                model_name,
                str(spec.seq_len),
                str(spec.pred_len),
            )
            if not args.overwrite and existing.get(key, {}).get("status") == "ok":
                print(f"Skipping completed pair {dataset_name}/{model_name}; use --overwrite to rerun.")
                continue
            try:
                result = run_one(spec, model_name, args, device)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                if device.type == "cuda" and "out of memory" in str(exc).lower():
                    error = "CUDA out of memory; reduce --batch-size"
                else:
                    error = f"{type(exc).__name__}: {exc}"
                print(f"    FAILED: {error}")
                traceback.print_exc()
                result = Result(
                    dataset=dataset_name,
                    model=model_name,
                    status="failed",
                    seq_len=spec.seq_len,
                    pred_len=spec.pred_len,
                    batch_size=args.batch_size or spec.batch_size,
                    epochs_requested=args.epochs,
                    epochs_trained=0,
                    best_val_mse=math.nan,
                    mse=math.nan,
                    train_time_s_epoch=math.nan,
                    peak_gpu_gb=math.nan,
                    parameters=0,
                    parameters_million=math.nan,
                    inference_ms_batch=math.nan,
                    inference_ms_sample=math.nan,
                    device=str(device),
                    source="",
                    error=error,
                )
            upsert_result(results_path, result)
            existing[key] = {k: str(v) for k, v in asdict(result).items()}
            clean_cuda()

    rows = read_rows(results_path)
    create_bubble_plot(rows, args.output_dir)
    create_tables(rows, args.output_dir)
    metadata = {
        "seed": args.seed,
        "torch": torch.__version__,
        "device": str(device),
        "cuda": torch.version.cuda,
        "bubble_area": "peak allocated GPU memory in GB",
        "latency": "mean model forward time for one test batch; data transfer excluded",
        "mse": "test MSE in the dataset's train-standardized scale",
        "training_objective": "common MSE loss on the OT target for every model",
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nResults, figure, and tables: {args.output_dir}")


if __name__ == "__main__":
    main()
