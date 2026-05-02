import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from common import REPO_ROOT, default_cache_dir


TUNE_SCRIPT = REPO_ROOT / "emulator_bench" / "tune_optuna.py"


def normalize_metric(metric: str) -> str:
    m = str(metric).strip().upper()
    aliases = {
        "PEARSON": "PCC",
        "SPEARMAN": "SCC",
    }
    return aliases.get(m, m)


def main():
    parser = argparse.ArgumentParser(description="Launch parallel Optuna sweeps for CatPred emulator bench.")
    parser.add_argument("--gpus", nargs="+", type=int, required=True, help="GPU ids, e.g. --gpus 0 1 2 3")
    parser.add_argument("--base_dir", required=True, type=str)
    parser.add_argument("--split_groups", nargs="+", required=True)
    parser.add_argument("--threshold", default=None, type=str, help="Single threshold alias, e.g. threshold_0.09")
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--metric", default="rmse", type=str)
    parser.add_argument("--eval_split", default="val", choices=["val", "test"])
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--val_every", default=2, type=int)
    parser.add_argument("--n_trials", default=30, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--storage", default=None, type=str)
    parser.add_argument("--study_name", default=None, type=str)
    parser.add_argument("--value_type", default="custom", type=str)
    parser.add_argument("--dataset_type", default="regression", type=str)
    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--uniprot_id_col", default="uniprot_id", type=str)
    parser.add_argument("--smiles_columns", nargs="+", default=["smiles"])
    parser.add_argument("--target_columns", nargs="+", default=["log10_value"])
    parser.add_argument("--cache_dir", default=default_cache_dir(), type=str)
    parser.add_argument("--mixed_precision", default="auto", choices=["auto", "none", "bf16", "fp16"])
    parser.add_argument("--optimizer_fused", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--lr_scheduler", default="cosine_warmup", choices=["cosine_warmup", "noam"])
    parser.add_argument("--early_stopping_patience", default=0, type=int)
    parser.add_argument("--early_stopping_min_delta", default=0.0, type=float)
    parser.add_argument("--warm_esm_cache", action="store_true")
    parser.add_argument("--overwrite_esm_cache", action="store_true")
    parser.add_argument("--require_cached_esm", action="store_true")
    parser.add_argument("--sampler_seed", default=42, type=int)
    parser.add_argument("--optuna_startup_trials", default=5, type=int)
    parser.add_argument("--dry_run", action="store_true")
    args, passthrough = parser.parse_known_args()

    thresholds = args.thresholds
    if thresholds is None and args.threshold is not None:
        thresholds = [args.threshold]

    metric_upper = normalize_metric(args.metric)
    cuda_device_ids = ",".join(str(x) for x in args.gpus)

    cmd = [
        sys.executable,
        str(TUNE_SCRIPT),
        "--base_dir", args.base_dir,
        "--value_type", args.value_type,
        "--dataset_type", args.dataset_type,
        "--split_groups", *args.split_groups,
        "--sequence_col", args.sequence_col,
        "--uniprot_id_col", args.uniprot_id_col,
        "--smiles_columns", *args.smiles_columns,
        "--target_columns", *args.target_columns,
        "--metric", metric_upper,
        "--metric_name", str(args.metric).lower(),
        "--eval_split", args.eval_split,
        "--epochs", str(args.epochs),
        "--val_every_n_epochs", str(args.val_every),
        "--n_trials", str(args.n_trials),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--cache_dir", args.cache_dir,
        "--mixed_precision", args.mixed_precision,
        "--optimizer_fused", args.optimizer_fused,
        "--lr_scheduler", args.lr_scheduler,
        "--early_stopping_patience", str(args.early_stopping_patience),
        "--early_stopping_min_delta", str(args.early_stopping_min_delta),
        "--cuda_device_ids", cuda_device_ids,
        "--n_parallel_trials", str(max(1, len(args.gpus))),
        "--sampler_seed", str(args.sampler_seed),
        "--optuna_startup_trials", str(args.optuna_startup_trials),
    ]
    if thresholds is not None:
        cmd.extend(["--thresholds", *thresholds])
    if args.storage:
        cmd.extend(["--storage", args.storage])
    if args.study_name:
        cmd.extend(["--study_name", args.study_name])
    if args.pin_memory:
        # tune_optuna uses disable flag, so no-op when pin_memory requested
        pass
    else:
        cmd.append("--disable_pin_memory")
    if args.warm_esm_cache:
        cmd.append("--warm_esm_cache")
    if args.overwrite_esm_cache:
        cmd.append("--overwrite_esm_cache")
    if args.require_cached_esm:
        cmd.append("--require_cached_esm")
    if args.dry_run:
        cmd.append("--dry_run")
    cmd.extend(passthrough)

    print("Launching:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
