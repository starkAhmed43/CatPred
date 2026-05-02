import argparse
import json
import os
import subprocess
import sys
import time
import threading
from collections import deque
from pathlib import Path

import optuna
import pandas as pd

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from common import REPO_ROOT, default_cache_dir, discover_threshold_dirs, ensure_split_triplet


BUILD_SCRIPT = REPO_ROOT / "emulator_bench" / "build_tvt_data.py"
TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def _stream_worker_output(pipe, log_file, prefix: str):
    try:
        for line in iter(pipe.readline, ""):
            if line == "":
                break
            log_file.write(line)
            log_file.flush()
            sys.stdout.write(f"{prefix} {line}")
            sys.stdout.flush()
    finally:
        pipe.close()


def resolve_value_root(base_dir: str, value_type: str) -> Path:
    base = Path(base_dir)
    nested = base / value_type
    return nested if nested.exists() else base


def load_best_hparams(storage: str | None, study_name: str | None, best_hparams_json: str | None):
    if best_hparams_json is not None:
        with open(best_hparams_json) as f:
            return json.load(f)
    if storage is None or study_name is None:
        raise ValueError("Provide either --best_hparams_json or both --storage and --study_name.")
    study = optuna.load_study(study_name=study_name, storage=storage)
    return dict(study.best_trial.params)


def stage_job_data(job, args):
    split_group, threshold_name, threshold_dir = job
    train_path, val_path, test_path = ensure_split_triplet(threshold_dir)
    if not (train_path and val_path and test_path):
        return None

    data_dir = threshold_dir / "catpred_data"
    manifest = data_dir / f"{args.dataset_name}_manifest.json"
    if not manifest.exists() or args.overwrite_stage:
        cmd = [
            sys.executable,
            str(BUILD_SCRIPT),
            "--train_csv", str(train_path),
            "--val_csv", str(val_path),
            "--test_csv", str(test_path),
            "--output_root", str(data_dir),
            "--dataset_name", args.dataset_name,
            "--sequence_col", args.sequence_col,
            "--uniprot_id_col", args.uniprot_id_col,
            "--cache_dir", args.cache_dir,
        ]
        if args.warm_esm_cache:
            cmd.append("--warm_esm_cache")
        if args.overwrite_esm_cache:
            cmd.append("--overwrite_esm_cache")
        if args.require_cached_esm:
            cmd.append("--require_cached_esm")
        if args.overwrite_stage:
            cmd.append("--overwrite")
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    with open(manifest) as f:
        payload = json.load(f)
    payload["split_group"] = split_group
    payload["threshold"] = threshold_name
    payload["threshold_dir"] = str(threshold_dir)
    return payload


def launch_train(job, seed, gpu_id, hp, run_root: Path, args):
    split_group = job["split_group"]
    threshold = job["threshold"]
    out_dir = run_root / split_group / threshold / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    train_csv = job["train_csv"]
    val_csv = job["val_csv"]
    test_csv = job["test_csv"]

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train_csv", train_csv,
        "--val_csv", val_csv,
        "--test_csv", test_csv,
        "--out_dir", str(out_dir),
        "--task_name", args.task_name,
        "--dataset_type", args.dataset_type,
        "--sequence_col", args.sequence_col,
        "--uniprot_id_col", args.uniprot_id_col,
        "--smiles_columns", *args.smiles_columns,
        "--target_columns", *args.target_columns,
        "--extra_metrics", "mae", "mse", "r2",
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--batch_size", str(int(hp.get("batch_size", args.batch_size))),
        "--init_lr", str(float(hp.get("init_lr", args.init_lr))),
        "--max_lr", str(float(hp.get("max_lr", args.max_lr))),
        "--final_lr", str(float(hp.get("final_lr", args.final_lr))),
        "--warmup_epochs", str(float(hp.get("warmup_epochs", args.warmup_epochs))),
        "--dropout", str(float(hp.get("dropout", args.dropout))),
        "--ensemble_size", str(args.ensemble_size),
        "--num_workers", str(args.num_workers),
        "--grad_accum_steps", str(args.grad_accum_steps),
        "--cache_cutoff", str(args.cache_cutoff),
        "--device", f"cuda:{gpu_id}" if args.use_cuda else "cpu",
        "--cache_dir", args.cache_dir,
        "--prefetch_factor", str(args.prefetch_factor),
        "--mixed_precision", args.mixed_precision,
        "--optimizer_fused", args.optimizer_fused,
        "--lr_scheduler", args.lr_scheduler,
        "--val_every_n_epochs", str(args.val_every_n_epochs),
        "--early_stopping_patience", str(args.early_stopping_patience),
        "--early_stopping_min_delta", str(args.early_stopping_min_delta),
        "--metric", args.metric_name,
        "--resume_if_complete",
    ]
    if args.final_epoch_metrics_only:
        cmd.append("--final_epoch_metrics_only")
    if args.cpu_threads is not None:
        cmd.extend(["--cpu_threads", str(args.cpu_threads)])
    if args.interop_threads is not None:
        cmd.extend(["--interop_threads", str(args.interop_threads)])
    if args.smart_batching:
        cmd.append("--smart_batching")
    if args.disable_pin_memory:
        cmd.append("--disable_pin_memory")
    if args.disable_persistent_workers:
        cmd.append("--disable_persistent_workers")
    if args.disable_tf32:
        cmd.append("--disable_tf32")
    if args.overwrite_esm_cache:
        cmd.append("--overwrite_esm_cache")
    if args.require_cached_esm:
        cmd.append("--require_cached_esm")
    if args.passthrough:
        cmd.extend(args.passthrough)

    log_file = open(log_path, "a", buffering=1)
    stream_thread = None
    if args.verbose_workers:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        prefix = f"[gpu={gpu_id} seed={seed} {split_group}/{threshold}]"
        stream_thread = threading.Thread(
            target=_stream_worker_output,
            args=(proc.stdout, log_file, prefix),
            daemon=True,
        )
        stream_thread.start()
    else:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=log_file, stderr=subprocess.STDOUT, text=True)
    return proc, log_file, stream_thread, out_dir, split_group, threshold, seed, gpu_id


def collect_metrics(out_dir: Path):
    result = {"out_dir": str(out_dir)}
    for split in ("train", "val", "test"):
        path = out_dir / f"final_results_{split}.csv"
        if path.exists():
            row = pd.read_csv(path).iloc[0].to_dict()
            for k, v in row.items():
                result[f"{split}_{k}"] = v
    return result


def main():
    parser = argparse.ArgumentParser(description="Retrain CatPred bench runs from best Optuna hparams across splits/seeds with GPU parallelism.")
    parser.add_argument("--base_dir", required=True, type=str)
    parser.add_argument("--value_type", default="custom", type=str)
    parser.add_argument("--split_groups", nargs="+", required=True)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--best_hparams_json", default=None, type=str)
    parser.add_argument("--storage", default=None, type=str)
    parser.add_argument("--study_name", default=None, type=str)
    parser.add_argument("--output_root", default=None, type=str)
    parser.add_argument("--run_name", default=None, type=str)
    parser.add_argument("--task_name", default="custom", type=str)
    parser.add_argument("--dataset_name", default="custom", type=str)
    parser.add_argument("--dataset_type", default="regression", type=str)
    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--uniprot_id_col", default="uniprot_id", type=str)
    parser.add_argument("--smiles_columns", nargs="+", default=["smiles"])
    parser.add_argument("--target_columns", nargs="+", default=["log10_value"])
    parser.add_argument("--cache_dir", default=default_cache_dir(), type=str)
    parser.add_argument("--warm_esm_cache", action="store_true")
    parser.add_argument("--overwrite_esm_cache", action="store_true")
    parser.add_argument("--require_cached_esm", action="store_true")
    parser.add_argument("--overwrite_stage", action="store_true")
    parser.add_argument("--use_cuda", action="store_true", default=True)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--init_lr", default=1e-4, type=float)
    parser.add_argument("--max_lr", default=1e-3, type=float)
    parser.add_argument("--final_lr", default=1e-4, type=float)
    parser.add_argument("--warmup_epochs", default=2.0, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--ensemble_size", default=10, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--grad_accum_steps", default=1, type=int)
    parser.add_argument("--cache_cutoff", default="inf", type=str)
    parser.add_argument("--prefetch_factor", default=4, type=int)
    parser.add_argument("--cpu_threads", default=None, type=int)
    parser.add_argument("--interop_threads", default=None, type=int)
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--disable_persistent_workers", action="store_true")
    parser.add_argument("--smart_batching", action="store_true")
    parser.add_argument("--mixed_precision", choices=["auto", "none", "bf16", "fp16"], default="auto")
    parser.add_argument("--optimizer_fused", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--lr_scheduler", choices=["cosine_warmup", "noam"], default="cosine_warmup")
    parser.add_argument("--val_every_n_epochs", default=1, type=int)
    parser.add_argument("--final_epoch_metrics_only", action="store_true")
    parser.add_argument("--early_stopping_patience", default=0, type=int)
    parser.add_argument("--early_stopping_min_delta", default=0.0, type=float)
    parser.add_argument("--metric_name", default="rmse", type=str)
    parser.add_argument("--disable_tf32", action="store_true")
    parser.add_argument("--verbose_workers", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args, passthrough = parser.parse_known_args()
    args.passthrough = passthrough

    hp = load_best_hparams(args.storage, args.study_name, args.best_hparams_json)

    value_root = resolve_value_root(args.base_dir, args.value_type)
    jobs = discover_threshold_dirs(value_root, args.split_groups, args.thresholds)
    if not jobs:
        raise RuntimeError("No matching split jobs discovered.")

    prepared = []
    for job in jobs:
        one = stage_job_data(job, args)
        if one is not None:
            prepared.append(one)

    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root is not None else Path(args.base_dir) / "runs"
    run_root = output_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    with open(run_root / "best_hparams.json", "w") as f:
        json.dump(hp, f, indent=2)

    queue = deque()
    for job in prepared:
        for seed in args.seeds:
            queue.append((job, seed))

    if args.dry_run:
        print(f"Planned {len(queue)} runs")
        for job, seed in list(queue)[:20]:
            print(f"- {job['split_group']}/{job['threshold']} seed={seed}")
        return

    free_gpus = deque(args.gpus)
    running = []
    rows = []

    while queue or running:
        while queue and free_gpus:
            job, seed = queue.popleft()
            gpu_id = free_gpus.popleft()
            proc_info = launch_train(job, seed, gpu_id, hp, run_root, args)
            running.append(proc_info)
            print(f"[launch] gpu={gpu_id} split={job['split_group']} threshold={job['threshold']} seed={seed}")

        still_running = []
        for proc, log_file, stream_thread, out_dir, split_group, threshold, seed, gpu_id in running:
            code = proc.poll()
            if code is None:
                still_running.append((proc, log_file, stream_thread, out_dir, split_group, threshold, seed, gpu_id))
                continue

            if stream_thread is not None:
                stream_thread.join(timeout=5)
            log_file.close()
            free_gpus.append(gpu_id)
            row = {
                "split_group": split_group,
                "threshold": threshold,
                "seed": seed,
                "gpu_id": gpu_id,
                "exit_code": code,
                "out_dir": str(out_dir),
                "log_path": str(out_dir / "train.log"),
            }
            if code == 0:
                row.update(collect_metrics(out_dir))
                print(f"[done] gpu={gpu_id} split={split_group} threshold={threshold} seed={seed}")
            else:
                print(f"[fail] gpu={gpu_id} split={split_group} threshold={threshold} seed={seed} code={code}")
            rows.append(row)

        running = still_running
        if running:
            time.sleep(2)

    summary_path = run_root / "summary_runs.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
