import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from common import (
    REPO_ROOT,
    default_cache_dir,
    discover_threshold_dirs,
    ensure_split_triplet,
    get_split_meta,
    threshold_to_float,
)


BUILD_SCRIPT = REPO_ROOT / "emulator_bench" / "build_tvt_data.py"
TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def resolve_value_root(base_dir: str, value_type: str) -> Path:
    base = Path(base_dir)
    nested = base / value_type
    return nested if nested.exists() else base


def maybe_build_data(job, args):
    split_group, threshold_name, threshold_dir = job
    train_csv, val_csv, test_csv = ensure_split_triplet(threshold_dir)
    if not (train_csv and val_csv and test_csv):
        return None
    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        return None

    data_dir = threshold_dir / "catpred_data"
    manifest = data_dir / f"{args.dataset_name}_manifest.json"
    if not manifest.exists() or args.overwrite:
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
        if args.warm_esm_cache:
            cmd.append("--warm_esm_cache")
        if args.overwrite_esm_cache:
            cmd.append("--overwrite_esm_cache")
        if args.require_cached_esm:
            cmd.append("--require_cached_esm")
        if args.overwrite:
            cmd.append("--overwrite")
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    with open(manifest) as f:
        manifest_payload = json.load(f)
    manifest_payload["split_group"] = split_group
    manifest_payload["threshold"] = threshold_name
    manifest_payload["threshold_dir"] = str(threshold_dir)
    return manifest_payload


def run_training(job, args, seed, passthrough):
    threshold_dir = Path(job["threshold_dir"])
    out_dir = threshold_dir / "catpred_results" / f"seed_{seed}"
    metric_file = out_dir / "final_results_test.csv"
    if metric_file.exists() and not args.overwrite:
        return out_dir

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
        "--batch_size", str(args.batch_size),
        "--init_lr", str(args.init_lr),
        "--max_lr", str(args.max_lr),
        "--final_lr", str(args.final_lr),
        "--warmup_epochs", str(args.warmup_epochs),
        "--dropout", str(args.dropout),
        "--ensemble_size", str(args.ensemble_size),
        "--num_workers", str(args.num_workers),
        "--grad_accum_steps", str(args.grad_accum_steps),
        "--cache_cutoff", str(args.cache_cutoff),
        "--device", args.device,
        "--prefetch_factor", str(args.prefetch_factor),
        "--mixed_precision", args.mixed_precision,
        "--optimizer_fused", args.optimizer_fused,
        "--lr_scheduler", args.lr_scheduler,
        "--val_every_n_epochs", str(args.val_every_n_epochs),
        "--early_stopping_patience", str(args.early_stopping_patience),
        "--early_stopping_min_delta", str(args.early_stopping_min_delta),
        "--metric", args.metric,
        "--resume_if_complete",
    ]
    if args.final_epoch_metrics_only:
        cmd.append("--final_epoch_metrics_only")
    if args.overwrite_esm_cache:
        cmd.append("--overwrite_esm_cache")
    if args.require_cached_esm:
        cmd.append("--require_cached_esm")
    if args.smart_batching:
        cmd.append("--smart_batching")
    if args.disable_pin_memory:
        cmd.append("--disable_pin_memory")
    if args.disable_persistent_workers:
        cmd.append("--disable_persistent_workers")
    cmd.extend(["--bucket_multiplier", str(args.bucket_multiplier)])
    cmd.extend(["--smiles_columns", *args.smiles_columns])
    cmd.extend(["--target_columns", *args.target_columns])
    cmd.extend(["--extra_metrics", *args.extra_metrics])
    if args.cache_dir:
        cmd.extend(["--cache_dir", args.cache_dir])
    cmd.extend(passthrough)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    return out_dir


def maybe_load_hparams(args):
    if not args.hparams_json:
        return args
    with open(args.hparams_json) as f:
        hp = json.load(f)
    for key in [
        "batch_size",
        "epochs",
        "init_lr",
        "max_lr",
        "final_lr",
        "warmup_epochs",
        "dropout",
        "ensemble_size",
        "num_workers",
        "grad_accum_steps",
        "cache_cutoff",
    ]:
        if key in hp:
            setattr(args, key, hp[key])
    print(f"Loaded hyperparameters from: {args.hparams_json}")
    return args


def main():
    parser = argparse.ArgumentParser(description="Run CatPred TVT benchmark across split families and thresholds.")
    parser.add_argument("--base_dir", default="/home/ubuntu/adhil/EMULaToR/data/processed/baselines/catpred", type=str)
    parser.add_argument("--value_type", default="custom", type=str)
    parser.add_argument("--split_groups", nargs="+", default=["enzyme_sequence_splits", "substrate_splits", "random_splits"])
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--dataset_name", default="custom", type=str)
    parser.add_argument("--dataset_type", default="regression", type=str)
    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--uniprot_id_col", default="uniprot_id", type=str)
    parser.add_argument("--smiles_columns", nargs="+", default=["smiles"])
    parser.add_argument("--target_columns", nargs="+", default=["log10_value"])
    parser.add_argument("--warm_esm_cache", action="store_true")
    parser.add_argument("--cache_dir", default=default_cache_dir(), type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--init_lr", default=1e-4, type=float)
    parser.add_argument("--max_lr", default=1e-3, type=float)
    parser.add_argument("--final_lr", default=1e-4, type=float)
    parser.add_argument("--warmup_epochs", default=2.0, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--ensemble_size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--grad_accum_steps", default=1, type=int)
    parser.add_argument("--cache_cutoff", default="inf", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--prefetch_factor", default=4, type=int)
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--disable_persistent_workers", action="store_true")
    parser.add_argument("--smart_batching", action="store_true")
    parser.add_argument("--bucket_multiplier", default=50, type=int)
    parser.add_argument("--mixed_precision", choices=["auto", "none", "bf16", "fp16"], default="auto")
    parser.add_argument("--optimizer_fused", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--lr_scheduler", choices=["cosine_warmup", "noam"], default="cosine_warmup")
    parser.add_argument("--val_every_n_epochs", default=1, type=int)
    parser.add_argument("--final_epoch_metrics_only", action="store_true")
    parser.add_argument("--early_stopping_patience", default=0, type=int)
    parser.add_argument("--early_stopping_min_delta", default=0.0, type=float)
    parser.add_argument("--overwrite_esm_cache", action="store_true")
    parser.add_argument("--require_cached_esm", action="store_true")
    parser.add_argument("--metric", default="rmse", type=str)
    parser.add_argument("--extra_metrics", nargs="+", default=["mae", "mse", "r2"])
    parser.add_argument("--hparams_json", default=None, type=str)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--ratio_tolerance", default=0.02, type=float)
    parser.add_argument("--primary_metric", default="MSE", choices=["PCC", "SCC", "R2", "RMSE", "MSE", "MAE"], type=str)
    parser.add_argument("--higher_is_better", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args, passthrough = parser.parse_known_args()
    args = maybe_load_hparams(args)

    value_root = resolve_value_root(args.base_dir, args.value_type)
    if not value_root.exists():
        raise FileNotFoundError(f"Value type directory not found: {value_root}")

    jobs = discover_threshold_dirs(value_root, args.split_groups, args.thresholds)
    if not jobs:
        raise RuntimeError("No threshold jobs discovered.")

    print(f"Discovered {len(jobs)} jobs for value_type={args.value_type}")
    if args.dry_run:
        for split_group, threshold_name, threshold_dir in jobs:
            print(f"- {split_group}/{threshold_name}: {threshold_dir}")
        return

    prepared = []
    for job in tqdm(jobs, desc=f"Preparing {args.value_type}", unit="job"):
        one = maybe_build_data(job, args)
        if one is not None:
            prepared.append(one)

    run_rows = []
    for job in tqdm(prepared, desc=f"Benchmark {args.value_type}", unit="job"):
        threshold_dir = Path(job["threshold_dir"])
        train_csv, val_csv, test_csv = ensure_split_triplet(threshold_dir)
        if not (train_csv and val_csv and test_csv):
            continue
        split_meta = get_split_meta(train_csv, val_csv, test_csv, args.ratio_tolerance)

        for seed in args.seeds:
            out_dir = run_training(job, args, seed, passthrough)
            test_metrics = pd.read_csv(out_dir / "final_results_test.csv").iloc[0].to_dict()
            val_metrics = pd.read_csv(out_dir / "final_results_val.csv").iloc[0].to_dict()
            run_rows.append(
                {
                    "value_type": args.value_type,
                    "split_group": job["split_group"],
                    "threshold": job["threshold"],
                    "threshold_value": threshold_to_float(job["threshold"]),
                    "seed": seed,
                    "out_dir": str(out_dir),
                    **split_meta,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                }
            )

    runs_df = pd.DataFrame(run_rows)
    summary_root = value_root
    runs_path = summary_root / "catpred_summary_runs.csv"
    runs_df.to_csv(runs_path, index=False)

    group_cols = ["value_type", "split_group", "threshold", "threshold_value"]
    test_metric = f"test_{args.primary_metric}"
    threshold_summary = (
        runs_df.groupby(group_cols, as_index=False)
        .agg(
            **{
                test_metric: (test_metric, "mean"),
                f"{test_metric}_std": (test_metric, "std"),
                "train_size": ("train_size", "first"),
                "val_size": ("val_size", "first"),
                "test_size": ("test_size", "first"),
                "small_split_flag": ("small_split_flag", "first"),
            }
        )
        .sort_values(test_metric, ascending=not args.higher_is_better)
    )
    threshold_summary.to_csv(summary_root / "catpred_summary_thresholds.csv", index=False)

    by_split = (
        runs_df.groupby(["value_type", "split_group"], as_index=False)
        .agg(**{test_metric: (test_metric, "mean"), f"{test_metric}_std": (test_metric, "std")})
        .sort_values(test_metric, ascending=not args.higher_is_better)
    )
    by_split.to_csv(summary_root / "catpred_summary_by_split_group.csv", index=False)

    ranked = threshold_summary.sort_values(test_metric, ascending=not args.higher_is_better).reset_index(drop=True)
    ranked.to_csv(summary_root / "catpred_summary_ranked.csv", index=False)
    ranked.to_csv(summary_root / "catpred_summary.csv", index=False)


if __name__ == "__main__":
    main()
