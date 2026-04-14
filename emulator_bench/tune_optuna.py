import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import optuna
import pandas as pd
from tqdm.auto import tqdm

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from common import REPO_ROOT, default_cache_dir, discover_threshold_dirs, ensure_split_triplet, materialize_tabular_as_csv, write_json


BUILD_SCRIPT = REPO_ROOT / "emulator_bench" / "build_tvt_data.py"
TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def _subprocess_env(cache_dir: str | None) -> dict:
    env = os.environ.copy()
    if cache_dir:
        resolved = str(Path(cache_dir).expanduser().resolve())
        env["CATPRED_CACHE_PATH"] = resolved
        env["CATPRED_BENCH_CACHE_DIR"] = resolved
        env["CATPRED_BENCH_TABULAR_CACHE_DIR"] = str((Path(resolved) / "tabular_csv").resolve())
    return env


def resolve_value_root(base_dir: str, value_type: str) -> Path:
    base = Path(base_dir)
    nested = base / value_type
    return nested if nested.exists() else base


def parse_visible_cuda_slots(explicit_device_ids: str | None):
    if explicit_device_ids:
        ids = [x.strip() for x in str(explicit_device_ids).split(",") if x.strip() != ""]
        return list(range(len(ids))) if len(ids) > 0 else []

    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() == "" or visible.strip() == "-1":
        return []

    ids = [x.strip() for x in visible.split(",") if x.strip() != ""]
    return list(range(len(ids))) if len(ids) > 0 else []


def discover_jobs(args):
    value_root = resolve_value_root(args.base_dir, args.value_type)
    jobs = discover_threshold_dirs(value_root, args.split_groups, args.thresholds)
    if args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]
    return jobs


def maybe_build(job, args):
    split_group, threshold_name, threshold_dir = job
    train_csv, val_csv, test_csv = ensure_split_triplet(threshold_dir)
    if not (train_csv and val_csv and test_csv):
        return None

    data_dir = threshold_dir / "catpred_data"
    manifest = data_dir / f"{args.dataset_name}_manifest.json"
    auto_warm_esm_now = bool(args.add_esm_feats and args.require_cached_esm and args.auto_warm_esm_cache)
    explicit_warm_esm_now = bool(args.warm_esm_cache)
    should_run_build = (not manifest.exists()) or args.overwrite_data or auto_warm_esm_now or explicit_warm_esm_now

    if should_run_build:
        cmd = [
            sys.executable,
            str(BUILD_SCRIPT),
            "--train_csv", str(train_csv),
            "--val_csv", str(val_csv),
            "--test_csv", str(test_csv),
            "--output_root", str(data_dir),
            "--dataset_name", args.dataset_name,
            "--sequence_col", args.sequence_col,
            "--uniprot_id_col", args.uniprot_id_col,
        ]
        if args.cache_dir:
            cmd.extend(["--cache_dir", args.cache_dir])
        if explicit_warm_esm_now or auto_warm_esm_now:
            cmd.append("--warm_esm_cache")
            cmd.extend(["--esm_warm_batch_size", str(args.esm_warm_batch_size)])
        cmd.extend(["--sequence_max_length", str(args.sequence_max_length)])
        if args.overwrite_esm_cache:
            cmd.append("--overwrite_esm_cache")
        if args.require_cached_esm and not (explicit_warm_esm_now or auto_warm_esm_now):
            cmd.append("--require_cached_esm")
        if args.overwrite_data:
            cmd.append("--overwrite")
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=_subprocess_env(args.cache_dir))
    with open(manifest) as f:
        payload = json.load(f)

    # If cached tabular CSVs were deleted, regenerate them from original split files.
    tabular_cache_dir = str((Path(args.cache_dir).expanduser().resolve() / "tabular_csv")) if args.cache_dir else None
    repaired_manifest = False
    for split_name, split_source in (("train", train_csv), ("val", val_csv), ("test", test_csv)):
        csv_key = f"{split_name}_csv"
        input_key = f"{split_name}_input_path"
        cached_path = payload.get(csv_key)
        if cached_path and Path(cached_path).exists():
            continue

        candidates = [payload.get(input_key), str(split_source)]
        rebuilt_path = None
        for candidate in candidates:
            if not candidate:
                continue
            candidate_path = Path(candidate)
            if candidate_path.exists():
                rebuilt_path = materialize_tabular_as_csv(candidate_path, cache_dir=tabular_cache_dir)
                break

        if rebuilt_path is None:
            raise FileNotFoundError(
                f"Unable to recreate missing cached split file for '{split_name}'. "
                f"Manifest path: {cached_path}. Candidates tried: {candidates}"
            )

        payload[csv_key] = str(Path(rebuilt_path).resolve())
        repaired_manifest = True

    if repaired_manifest:
        write_json(manifest, payload)

    payload["split_group"] = split_group
    payload["threshold"] = threshold_name
    payload["threshold_dir"] = str(threshold_dir)
    return payload


def train_one(job, args, hp, seed, trial_number, device, passthrough):
    out_dir = (
        Path(job["threshold_dir"]) / "catpred_optuna_runs" / f"trial_{trial_number}" / job["split_group"] / job["threshold"] / f"seed_{seed}"
    )
    metric_file = out_dir / f"final_results_{args.eval_split}.csv"
    if not metric_file.exists():
        print(
            "[optuna] launch "
            f"trial={trial_number} seed={seed} device={device} "
            f"batch_size={hp['batch_size']} num_workers={args.num_workers} "
            f"val_every_n_epochs={args.val_every_n_epochs} "
            f"final_epoch_metrics_only={int(args.final_epoch_metrics_only)} "
            f"early_stopping_patience={args.early_stopping_patience} "
            f"smart_batching={int(args.smart_batching)}",
            flush=True,
        )

        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--train_csv", job["train_csv"],
            "--val_csv", job["val_csv"],
            "--test_csv", job["test_csv"],
            "--out_dir", str(out_dir),
            "--task_name", args.value_type,
            "--dataset_type", args.dataset_type,
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--batch_size", str(hp["batch_size"]),
            "--init_lr", str(hp["init_lr"]),
            "--max_lr", str(hp["max_lr"]),
            "--final_lr", str(hp["final_lr"]),
            "--warmup_epochs", str(hp["warmup_epochs"]),
            "--dropout", str(hp["dropout"]),
            "--ensemble_size", str(args.ensemble_size),
            "--num_workers", str(args.num_workers),
            "--grad_accum_steps", str(args.grad_accum_steps),
            "--cache_cutoff", str(args.cache_cutoff),
            "--device", device,
            "--prefetch_factor", str(args.prefetch_factor),
            "--mixed_precision", args.mixed_precision,
            "--optimizer_fused", args.optimizer_fused,
            "--lr_scheduler", args.lr_scheduler,
            "--val_every_n_epochs", str(args.val_every_n_epochs),
            "--early_stopping_patience", str(args.early_stopping_patience),
            "--early_stopping_min_delta", str(args.early_stopping_min_delta),
            "--metric", args.metric_name,
            "--sequence_max_length", str(args.sequence_max_length),
            "--resume_if_complete",
            "--resume_marker_split", args.eval_split,
            "--ram_budget_gb", str(args.ram_budget_gb),
        ]
        if args.add_esm_feats:
            cmd.append("--add_esm_feats")
            cmd.extend(["--seq_embed_dim", str(args.seq_embed_dim)])
            cmd.extend(["--seq_self_attn_nheads", str(args.seq_self_attn_nheads)])
        cmd.append("--final_epoch_metrics_only" if args.final_epoch_metrics_only else "--no-final_epoch_metrics_only")
        if args.overwrite_esm_cache:
            cmd.append("--overwrite_esm_cache")
        if args.require_cached_esm:
            cmd.append("--require_cached_esm")
        cmd.append("--smart_batching" if args.smart_batching else "--no-smart_batching")
        cmd.append("--cache_batch_graphs" if args.cache_batch_graphs else "--no-cache_batch_graphs")
        cmd.append("--low_ram_mode" if args.low_ram_mode else "--no-low_ram_mode")
        cmd.append("--skip_native_test_eval" if args.skip_native_test_eval else "--no-skip_native_test_eval")
        if args.disable_pin_memory:
            cmd.append("--disable_pin_memory")
        if args.disable_persistent_workers:
            cmd.append("--disable_persistent_workers")
        if args.disable_lazy_esm:
            cmd.append("--disable_lazy_esm")
        if args.disable_dataset_cache:
            cmd.append("--disable_dataset_cache")
        if args.disable_low_ram_mode:
            cmd.append("--disable_low_ram_mode")
        if args.disable_tf32:
            cmd.append("--disable_tf32")
        if args.disable_molgraph_disk_cache:
            cmd.append("--disable_molgraph_disk_cache")
        if args.esm_mem_cache_max is not None:
            cmd.extend(["--esm_mem_cache_max", str(args.esm_mem_cache_max)])
        cmd.extend(["--bucket_multiplier", str(args.bucket_multiplier)])

        if args.sweep_fast_mode:
            if args.eval_split == "val":
                cmd.extend(["--skip_train_eval_after_fit", "--skip_test_eval_after_fit"])
            else:
                cmd.extend(["--skip_train_eval_after_fit", "--skip_val_eval_after_fit"])

        cmd.extend(["--smiles_columns", *args.smiles_columns])
        cmd.extend(["--target_columns", *args.target_columns])
        cmd.extend(["--extra_metrics", *args.extra_metrics])
        if args.cache_dir:
            cmd.extend(["--cache_dir", args.cache_dir])
        cmd.extend(passthrough)
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=_subprocess_env(args.cache_dir))
    df = pd.read_csv(metric_file)
    return float(df.iloc[0][args.metric])


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for the CatPred emulator benchmark.")
    parser.add_argument("--base_dir", default="/home/ubuntu/adhil/EMULaToR/data/processed/baselines/catpred", type=str)
    parser.add_argument("--value_type", default="custom", type=str)
    parser.add_argument("--split_groups", nargs="+", default=["enzyme_sequence_splits", "substrate_splits", "random_splits"])
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--max_jobs", default=4, type=int)
    parser.add_argument("--dataset_name", default="custom", type=str)
    parser.add_argument("--dataset_type", default="regression", type=str)
    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--uniprot_id_col", default="uniprot_id", type=str)
    parser.add_argument("--smiles_columns", nargs="+", default=["smiles"])
    parser.add_argument("--target_columns", nargs="+", default=["log10_value"])
    parser.add_argument("--warm_esm_cache", action="store_true")
    parser.add_argument("--esm_warm_batch_size", default=64, type=int)
    parser.add_argument("--auto_warm_esm_cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite_esm_cache", action="store_true")
    parser.add_argument("--require_cached_esm", action="store_true")
    parser.add_argument("--add_esm_feats", action="store_true")
    parser.add_argument("--seq_embed_dim", default=300, type=int)
    parser.add_argument("--seq_self_attn_nheads", default=6, type=int)
    parser.add_argument("--cache_dir", default=default_cache_dir(), type=str)
    parser.add_argument("--metric", default="MSE", choices=["PCC", "SCC", "R2", "RMSE", "MSE", "MAE"])
    parser.add_argument("--metric_name", default=None, type=str)
    parser.add_argument("--eval_split", default="val", choices=["val", "test"])
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--cuda_device_ids", default=None, type=str)
    parser.add_argument("--n_parallel_trials", default=1, type=int)
    parser.add_argument("--parallel_trials_per_gpu", default=None, type=int)
    parser.add_argument("--mixed_precision", choices=["auto", "none", "bf16", "fp16"], default="auto")
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--warmup_epochs", default=2.0, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--ensemble_size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--grad_accum_steps", default=1, type=int)
    parser.add_argument("--cache_cutoff", default="inf", type=str)
    parser.add_argument("--prefetch_factor", default=4, type=int)
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--disable_persistent_workers", action="store_true")
    parser.add_argument("--disable_lazy_esm", action="store_true")
    parser.add_argument("--disable_dataset_cache", action="store_true")
    parser.add_argument("--low_ram_mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable_low_ram_mode", action="store_true")
    parser.add_argument("--esm_mem_cache_max", default=None, type=int)
    parser.add_argument("--ram_budget_gb", default=90.0, type=float)
    parser.add_argument("--disable_tf32", action="store_true")
    parser.add_argument("--disable_molgraph_disk_cache", action="store_true")
    parser.add_argument("--smart_batching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache_batch_graphs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bucket_multiplier", default=50, type=int)
    parser.add_argument("--optimizer_fused", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--lr_scheduler", choices=["cosine_warmup", "noam"], default="cosine_warmup")
    parser.add_argument("--val_every_n_epochs", default=1, type=int)
    parser.add_argument("--final_epoch_metrics_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early_stopping_patience", default=4, type=int)
    parser.add_argument("--early_stopping_min_delta", default=0.0, type=float)
    parser.add_argument("--sequence_max_length", default=2048, type=int)
    parser.add_argument("--extra_metrics", nargs="+", default=["mae", "mse", "r2"])
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--sweep_batch_size", action="store_true")
    parser.add_argument("--batch_size_candidates", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--sweep_dropout", action="store_true")
    parser.add_argument("--dropout_candidates", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2])
    parser.add_argument("--sweep_warmup_epochs", action="store_true")
    parser.add_argument("--warmup_epochs_min", default=1.0, type=float)
    parser.add_argument("--warmup_epochs_max", default=4.0, type=float)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--n_trials", default=20, type=int)
    parser.add_argument("--optuna_startup_trials", default=5, type=int)
    parser.add_argument("--sampler_seed", default=42, type=int)
    parser.add_argument("--study_name", default=None, type=str)
    parser.add_argument("--storage", default=None, type=str)
    parser.add_argument("--skip_native_test_eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sweep_fast_mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite_data", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args, passthrough = parser.parse_known_args()
    if args.metric_name is None:
        args.metric_name = args.metric.lower()
    if args.require_cached_esm and not args.add_esm_feats:
        raise ValueError(
            "--require_cached_esm was set but --add_esm_feats is not enabled. "
            "Enable --add_esm_feats to actually use ESM features."
        )
    if args.disable_low_ram_mode:
        args.low_ram_mode = False

    visible_cuda_slots = parse_visible_cuda_slots(args.cuda_device_ids)
    if str(args.device).lower().startswith("cpu"):
        visible_cuda_slots = []

    if args.parallel_trials_per_gpu is not None:
        if args.parallel_trials_per_gpu <= 0:
            raise ValueError("--parallel_trials_per_gpu must be >= 1.")

        if len(visible_cuda_slots) == 0 and str(args.device).lower().startswith("cuda"):
            # When CUDA_VISIBLE_DEVICES is not set, default to one addressable CUDA device.
            visible_cuda_slots = [0]

        if len(visible_cuda_slots) == 0:
            raise ValueError(
                "--parallel_trials_per_gpu requires at least one CUDA device. "
                "Set --device cuda:0 or export CUDA_VISIBLE_DEVICES."
            )

        args.n_parallel_trials = len(visible_cuda_slots) * args.parallel_trials_per_gpu
    elif args.n_parallel_trials <= 0:
        args.n_parallel_trials = max(1, len(visible_cuda_slots)) if len(visible_cuda_slots) > 0 else 1

    jobs = discover_jobs(args)
    if not jobs:
        raise RuntimeError("No threshold jobs discovered for tuning.")
    if args.dry_run:
        for split_group, threshold_name, threshold_dir in jobs:
            print(f"- {split_group}/{threshold_name}: {threshold_dir}")
        return

    prepared = [maybe_build(job, args) for job in tqdm(jobs, desc="Preparing jobs", unit="job")]
    prepared = [job for job in prepared if job is not None]
    if not prepared:
        raise RuntimeError("No valid train/val/test split triplets found for tuning.")
    direction = "maximize" if args.metric in {"PCC", "SCC", "R2"} else "minimize"
    study_name = args.study_name or f"catpred_{args.value_type}_{args.metric.lower()}"
    sampler = optuna.samplers.TPESampler(n_startup_trials=args.optuna_startup_trials, seed=args.sampler_seed)
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        storage=args.storage,
        load_if_exists=True,
    )

    def objective(trial):
        if len(visible_cuda_slots) == 0:
            trial_device = args.device
        else:
            trial_device = f"cuda:{visible_cuda_slots[trial.number % len(visible_cuda_slots)]}"

        hp = {
            "batch_size": trial.suggest_categorical("batch_size", args.batch_size_candidates) if args.sweep_batch_size else args.batch_size,
            "init_lr": trial.suggest_float("init_lr", 1e-5, 5e-4, log=True),
            "max_lr": trial.suggest_float("max_lr", 5e-5, 5e-3, log=True),
            "final_lr": trial.suggest_float("final_lr", 1e-6, 5e-4, log=True),
            "warmup_epochs": trial.suggest_float("warmup_epochs", args.warmup_epochs_min, args.warmup_epochs_max) if args.sweep_warmup_epochs else args.warmup_epochs,
            "dropout": trial.suggest_categorical("dropout", args.dropout_candidates) if args.sweep_dropout else args.dropout,
        }
        if hp["final_lr"] > hp["max_lr"]:
            hp["final_lr"] = hp["max_lr"]

        scores = []
        for job in prepared:
            for seed in args.seeds:
                scores.append(train_one(job, args, hp, seed, trial.number, trial_device, passthrough))
        return float(sum(scores) / len(scores))

    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_parallel_trials)

    out_dir = resolve_value_root(args.base_dir, args.value_type) / "optuna_studies"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"{study_name}_best_hparams.json"
    trials_path = out_dir / f"{study_name}_trials.csv"
    with open(best_path, "w") as f:
        json.dump(study.best_trial.params, f, indent=2)
    pd.DataFrame(study.trials_dataframe()).to_csv(trials_path, index=False)
    print(f"Saved best params to: {best_path}")
    print(f"Saved trials to: {trials_path}")


if __name__ == "__main__":
    main()
