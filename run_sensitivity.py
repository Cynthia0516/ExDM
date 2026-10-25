"""Four-parameter, one-factor sensitivity runs through the ExDM entrypoint."""

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import uuid


PROJECT_ROOT = Path(__file__).resolve().parent
CSV_RESULTS_DIR = PROJECT_ROOT / "csv_results"
# Keep the original ablation CSV columns and order, then add sweep metadata.
RESULT_COLUMNS = (
    "Seed", "Model", "ablation_mode", "seq_len", "label_len", "pred_len",
    "train_epochs", "batch", "patience", "LR", "dropout", "ch_ind", "revin",
    "e_fact", "dconv", "MSE", "MAE", "residual", "d_state", "checkpoint_path",
)
LEGACY_SENSITIVITY_COLUMNS = (
    "dataset", "run_id", "case_id", "sensitivity_parameter", "sensitivity_value",
    "baseline_value", "sample_alpha", "exodiff_steps", "exodiff_hidden_dim",
    "sample_m", "sample_n", "same_sampling_as_baseline", "elapsed_seconds",
    "run_dir", "log_path",
)
SENSITIVITY_COLUMNS = LEGACY_SENSITIVITY_COLUMNS + (
    "endl_t_ff", "effective_endl_t_ff", "enable_endl_t_ff",
)
PARAMETERS = ("sample_alpha", "exodiff_steps", "exodiff_hidden_dim", "endl_t_ff")
COMMON = dict(
    model="ExDM", is_training=1, features="MS", target="OT", itr=1,
    output_dim=1, fused_endogenous=1, des="Exp",
    endl_t_dropout=0, endl_c_dropout=0, exodiff_hidden_dim=128,
)
# Snapshots of the requested branches in scripts/ExDM/{dataset}.sh.
# Colbun uses p10_base. baseline_config opts into the requested t_ff=256 baseline.
PROFILES = {
    "etth1": dict(
        data="ETTh1", data_path="ETTh1.csv", seq_len=96, pred_len=336,
        random_seed=2024, enc_in=7, feature_num=7, input_dim=6, fused_exogenous=6,
        d_model=128, endl_t_ff=768, endl_c_ff=7,
        endl_embed_dropout=0.05, endl_head_dropout=0.06,
        sample_alpha=0.5, sample_p=0.2, exodiff_steps=100,
        alpha=0.3, beta=0.3, patience=22, dropout=0.1, dff=96,
        loss="mse", mse_weight=0.7, mae_weight=0.3,
        direct_weight=0.92, direct_dropout=0.02, direct_loss_weight=0.04,
        train_epochs=180, batch_size=128, learning_rate=0.00025,
    ),
    "PJM": dict(
        data="custom", data_path="PJM_exdm.csv", seq_len=168, pred_len=24,
        random_seed=2025, enc_in=3, feature_num=3, input_dim=2, fused_exogenous=2,
        d_model=512, endl_t_ff=512, endl_c_ff=16,
        endl_embed_dropout=0, endl_head_dropout=0,
        sample_alpha=0.3, sample_p=0.1, exodiff_steps=50,
        patience=18, dropout=0, dff=1536,
        loss="mae", mse_weight=0.5, mae_weight=-1.0,
        direct_weight=0.95, direct_dropout=0.02, direct_loss_weight=0.36,
        train_epochs=110, batch_size=64, learning_rate=0.00065,
    ),
    "Energy": dict(
        data="custom", data_path="Energy_exdm.csv", seq_len=168, pred_len=24,
        random_seed=2025, enc_in=6, feature_num=6, input_dim=5, fused_exogenous=5,
        d_model=256, endl_t_ff=256, endl_c_ff=16,
        endl_embed_dropout=0.05, endl_head_dropout=0.03,
        sample_alpha=0.5, sample_p=0.1, exodiff_steps=50,
        ma_type="dema", alpha=0.25, beta=0.3, invnorm_p=0.8, invnorm_sigma=0.15,
        patience=14, dropout=0.05, dff=384, lradj="type3",
        loss="huber", mse_weight=0.5, mae_weight=0.5,
        direct_weight=0.78, direct_dropout=0.04, direct_loss_weight=0.12,
        train_epochs=100, batch_size=64, learning_rate=0.0005,
    ),
    "Colbun": dict(
        data="custom", data_path="Colbun_exdm.csv", seq_len=60, pred_len=10,
        random_seed=2025, enc_in=3, feature_num=3, input_dim=2, fused_exogenous=2,
        d_model=256, endl_t_ff=256, endl_c_ff=16,
        endl_embed_dropout=0, endl_head_dropout=0,
        sample_alpha=0.5, sample_p=0.1, exodiff_steps=50,
        freq="d", revin=1, residual=1, d_state=256, dconv=2,
        ma_type="ema", alpha=0.3, beta=0.3, invnorm_p=1.0, invnorm_sigma=0.1,
        patience=10, dropout=0, dff=256, lradj="type3",
        loss="mixed", mse_weight=0.7, mae_weight=0.3,
        direct_weight=0.88, direct_dropout=0.02, direct_loss_weight=0.12,
        train_epochs=95, batch_size=64, learning_rate=0.0007,
    ),
}


def baseline_config(dataset):
    return {**COMMON, **PROFILES[dataset], "endl_t_ff": 256, "enable_endl_t_ff": 1}


def candidate_values(config, parameter):
    if parameter == "sample_alpha":
        return [0.1, 0.5, 1.0] if config[parameter] == 0.3 else [0.1, 0.3, 1.0]
    if parameter == "exodiff_steps":
        return [value for value in (25, 50, 100, 200) if value != config[parameter]]
    if parameter == "exodiff_hidden_dim":
        return [64, 256, 512]
    if parameter == "endl_t_ff":
        return [64, 128, 512]
    raise ValueError(parameter)


def effective_endl_t_ff(config):
    if config.get("enable_endl_t_ff", 0):
        return config.get("endl_t_ff") or 256
    return 128


def sampling_shape(config):
    """Match EndL's ceil/clamp rules, including the all-channels special case."""
    channels = config["input_dim"]
    m = max(1, min(math.ceil(config["sample_alpha"] * channels), channels))
    p = min(max(config["sample_p"], 1e-8), 1 - 1e-8)
    n = 1 if m == channels else max(1, math.ceil(math.log(p) / math.log(1 - m / channels)))
    return m, n


def experiment_cases(dataset, parameter="all", include_baseline=True):
    baseline = baseline_config(dataset)
    cases = []
    if include_baseline:
        cases.append(("baseline", "baseline", None, baseline.copy()))
    selected = PARAMETERS if parameter == "all" else (parameter,)
    for name in selected:
        for value in candidate_values(baseline, name):
            config = {**baseline, name: value}
            case_id = name + "_" + str(value).replace(".", "p")
            cases.append((case_id, name, value, config))
    return cases


def command_for(dataset, case_id, config, root_path):
    model_id = "{}_sensitivity_{}_{}_{}".format(
        dataset, config["seq_len"], config["pred_len"], case_id)
    args = {**config, "root_path": str(root_path), "model_id": model_id}
    command = [sys.executable, "-u", str(PROJECT_ROOT / "run_longExp.py")]
    for name, value in args.items():
        command.extend(["--" + name, str(value)])
    return command


def prepare_data(dataset, root_path):
    data_path = root_path / PROFILES[dataset]["data_path"]
    if data_path.is_file():
        return
    raw_path = root_path / (dataset + ".csv")
    if dataset == "etth1" or not raw_path.is_file():
        raise FileNotFoundError("Missing data: {}. Set --root-path to your data directory.".format(data_path))
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "preprocess_data.py"), str(raw_path), str(data_path)],
        cwd=str(PROJECT_ROOT), check=True,
    )


def read_result(case_dir, config):
    result_csv = case_dir / "csv_results" / ("result_" + config["data_path"])
    with result_csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError("Expected one result row in {}, got {}".format(result_csv, len(rows)))
    row = rows[0]
    missing = set(RESULT_COLUMNS) - row.keys()
    if missing:
        raise ValueError("Missing result columns in {}: {}".format(result_csv, sorted(missing)))
    if any(int(row[key]) != config[key] for key in ("seq_len", "pred_len")):
        raise ValueError("Result sequence lengths do not match the requested experiment")
    metrics = {name: float(row[name]) for name in ("MSE", "MAE")}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Non-finite evaluation metrics")
    return row


def dataset_result_path(dataset):
    name = "ETTh1" if dataset == "etth1" else dataset
    return CSV_RESULTS_DIR / ("result_" + name + "_sensitivity.csv")


@contextmanager
def result_lock(path):
    """Serialize parallel sweeps/backfills writing the same dataset CSV."""
    with path.with_suffix(path.suffix + ".lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            if lock.tell() == 0:
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def save_dataset_result(dataset, run_dir, item, result, elapsed_seconds=""):
    """Append one completed experiment; repeated collection is idempotent."""
    path = dataset_result_path(dataset)
    case_dir = run_dir / item["case_id"]
    row = {name: result[name] for name in RESULT_COLUMNS}
    row.update(
        dataset="ETTh1" if dataset == "etth1" else dataset,
        run_id=run_dir.name, case_id=item["case_id"],
        sensitivity_parameter=item["parameter"], sensitivity_value=item["value"],
        baseline_value=item["baseline_value"],
        **{p: item["config"].get(p, "") for p in PARAMETERS},
        effective_endl_t_ff=effective_endl_t_ff(item["config"]),
        enable_endl_t_ff=item["config"].get("enable_endl_t_ff", 0),
        sample_m=item["sample_m"], sample_n=item["sample_n"],
        same_sampling_as_baseline=item["same_sampling_as_baseline"],
        elapsed_seconds=elapsed_seconds, run_dir=str(run_dir.resolve()),
        log_path=str((case_dir / "train.log").resolve()),
    )
    columns = RESULT_COLUMNS + SENSITIVITY_COLUMNS
    path.parent.mkdir(parents=True, exist_ok=True)
    with result_lock(path):
        previous_rows, header = [], None
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                header = reader.fieldnames
                previous_rows = list(reader)
        legacy_header = list(RESULT_COLUMNS + LEGACY_SENSITIVITY_COLUMNS)
        if header not in (None, list(columns), legacy_header):
            raise ValueError("Existing sensitivity CSV has an incompatible header: {}".format(path))
        if header == legacy_header:
            for previous in previous_rows:
                # These rows predate the opt-in: their actual t_ff was 128.
                previous.update(endl_t_ff="", effective_endl_t_ff=128, enable_endl_t_ff=0)
                config_path = Path(previous["run_dir"]) / previous["case_id"] / "config.json"
                if config_path.is_file():
                    saved = json.loads(config_path.read_text(encoding="utf-8"))
                    previous["endl_t_ff"] = saved["config"].get("endl_t_ff", "")
        exists = any(previous["run_id"] == row["run_id"] and previous["case_id"] == row["case_id"]
                     for previous in previous_rows)
        if exists and header != legacy_header:
            return False
        if not exists:
            previous_rows.append(row)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                             dir=path.parent, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(previous_rows)
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return not exists


def collect_results(dataset, output_root):
    """Export completed runs made by either version of the sensitivity runner."""
    added = 0
    failed = False
    for manifest_path in sorted((output_root / dataset).glob("*/manifest.json")):
        run_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["dataset"] != dataset:
            raise ValueError("Dataset mismatch in {}".format(manifest_path))
        summary_path = run_dir / "summary.csv"
        summaries = {}
        if summary_path.is_file():
            with summary_path.open(encoding="utf-8-sig", newline="") as stream:
                summaries = {row["case_id"]: row for row in csv.DictReader(stream)}
        for item in manifest["experiments"]:
            summary = summaries.get(item["case_id"], {})
            if summary.get("status") != "completed":
                continue
            try:
                result = read_result(run_dir / item["case_id"], item["config"])
                added += save_dataset_result(dataset, run_dir, item, result, summary.get("elapsed_seconds", ""))
            except (OSError, ValueError, KeyError) as exc:
                failed = True
                print("Cannot collect {}: {}".format(run_dir / item["case_id"], exc), flush=True)
    print("Added {} completed results: {}".format(added, dataset_result_path(dataset)), flush=True)
    return 1 if failed else 0


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_summary(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=tuple(PROFILES))
    parser.add_argument("--parameter", choices=("all",) + PARAMETERS, default="all")
    parser.add_argument("--root-path", type=Path, help="Data directory; defaults to the original .sh path")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "sensitivity_results")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Run candidates only (three per parameter; twelve when --parameter all)")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without training or writing files")
    parser.add_argument("--collect-only", action="store_true",
                        help="Export existing completed runs into csv_results without training")
    args = parser.parse_args(argv)
    if args.collect_only:
        if args.dry_run or args.skip_baseline or args.parameter != "all":
            parser.error("--collect-only cannot be combined with --dry-run, --skip-baseline or --parameter")
        return collect_results(args.dataset, args.output_root.expanduser().resolve())

    default_root = Path("/home/u/sunqj/ExDM6.6/data")
    if args.dataset == "etth1":
        default_root = default_root / "ETT-small"
    root_path = (args.root_path or default_root).expanduser().resolve()
    baseline = baseline_config(args.dataset)
    base_m, base_n = sampling_shape(baseline)
    plan = []
    for case_id, parameter, value, config in experiment_cases(
            args.dataset, args.parameter, not args.skip_baseline):
        m, n = sampling_shape(config)
        plan.append(dict(
            case_id=case_id, parameter=parameter, value=value,
            baseline_value=baseline.get(parameter), config=config,
            sample_m=m, sample_n=n,
            same_sampling_as_baseline=(m, n) == (base_m, base_n),
            command=command_for(args.dataset, case_id, config, root_path),
        ))

    print("{}: seq_len={}, pred_len={}, {} runs".format(
        args.dataset, baseline["seq_len"], baseline["pred_len"], len(plan)), flush=True)
    if args.dataset == "Colbun":
        print("Baseline variant: Colbun.sh p10_base", flush=True)
    if baseline["input_dim"] == 2:
        print("NOTE: Only two exogenous channels: alpha alternatives reduce to two effective sampling configurations.", flush=True)
    for item in plan:
        print("{}: alpha={}, steps={}, hidden={}, t_ff={}, m={}, n={}".format(
            item["case_id"], *(item["config"][p] for p in PARAMETERS),
            item["sample_m"], item["sample_n"]), flush=True)
        if args.dry_run:
            print(shlex.join(item["command"]))
    if args.dry_run:
        return 0

    prepare_data(args.dataset, root_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output_dir = args.output_root.expanduser().resolve() / args.dataset / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "manifest.json", dict(
        dataset=args.dataset, source_script="scripts/ExDM/" + args.dataset + ".sh",
        baseline_variant="p10_base" if args.dataset == "Colbun" else "default",
        root_path=str(root_path), baseline=baseline, experiments=plan,
        note="EndL t_ff is forwarded with enable_endl_t_ff=1; baseline is 256. Other EndL forwarding is unchanged.",
    ))
    print("Output: {}".format(output_dir), flush=True)
    print("Dataset results: {}".format(dataset_result_path(args.dataset)), flush=True)
    rows = []
    failed = False
    for index, item in enumerate(plan, 1):
        case_dir = output_dir / item["case_id"]
        case_dir.mkdir()
        write_json(case_dir / "config.json", item)
        row = dict(
            dataset=args.dataset, case_id=item["case_id"], parameter=item["parameter"],
            value=item["value"], baseline_value=item["baseline_value"],
            seq_len=baseline["seq_len"], pred_len=baseline["pred_len"],
            random_seed=baseline["random_seed"],
            **{p: item["config"][p] for p in PARAMETERS},
            effective_endl_t_ff=effective_endl_t_ff(item["config"]),
            enable_endl_t_ff=item["config"]["enable_endl_t_ff"],
            sample_m=item["sample_m"], sample_n=item["sample_n"],
            same_sampling_as_baseline=item["same_sampling_as_baseline"],
            status="running", MSE="", MAE="", elapsed_seconds="", error="",
            log_path=str(case_dir / "train.log"),
        )
        rows.append(row)
        write_summary(output_dir / "summary.csv", rows)
        print("[{}/{}] {}".format(index, len(plan), item["case_id"]), flush=True)
        started = time.monotonic()
        try:
            # Isolate all legacy relative output paths without modifying exp_main.py.
            with (case_dir / "train.log").open("w", encoding="utf-8") as log:
                subprocess.run(item["command"], cwd=str(case_dir), stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            result = read_result(case_dir, item["config"])
            row.update({name: float(result[name]) for name in ("MSE", "MAE")})
            row["status"] = "completed"
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            failed = True
            print("FAILED: {} (see {})".format(exc, row["log_path"]), flush=True)
        except KeyboardInterrupt:
            row["status"] = "interrupted"
            raise
        finally:
            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
            write_summary(output_dir / "summary.csv", rows)
        if row["status"] == "completed":
            try:
                save_dataset_result(args.dataset, output_dir, item, result, row["elapsed_seconds"])
            except (OSError, ValueError, KeyError) as exc:
                failed = True
                print("CSV export failed: {}. Retry with --collect-only.".format(exc), flush=True)
    print("Summary: {}".format(output_dir / "summary.csv"), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
