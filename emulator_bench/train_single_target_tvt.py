import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from bench_amp import install_amp_patches
from bench_dataloader import install_dataloader_patches
from bench_fast_loader import INLINE_PROTEIN_SENTINEL, install_bench_patches
from bench_optimizer import install_optimizer_patches
from bench_scheduler import install_scheduler_patches
from bench_speedups import install_model_speed_patches
from bench_training_control import install_training_control_patches
from common import (
    configure_cpu_runtime,
    configure_esm_cache_policy,
    configure_torch_runtime,
    default_cache_dir,
    ensure_repo_on_path,
    materialize_tabular_as_csv,
    maybe_set_cache_env,
    load_molgraph_cache,
    molgraph_cache_path,
    save_molgraph_cache,
    warm_molgraph_cache,
    write_json,
)
from predict_single_target import evaluate_predictions_csv


def _checkpoint_dir(out_dir: Path) -> Path:
    return out_dir


def _first_model_ckpt(out_dir: Path) -> Path:
    return out_dir / "model_0" / "model.pt"


def _copy_single_model_alias(out_dir: Path):
    src = _first_model_ckpt(out_dir)
    dst = out_dir / "bestmodel.pth"
    if src.exists():
        shutil.copy2(src, dst)


def build_catpred_train_argv(args):
    argv = [
        "--data_path", args.train_csv,
        "--separate_val_path", args.val_csv,
        "--separate_test_path", args.test_csv,
        "--save_dir", args.out_dir,
        "--dataset_type", args.dataset_type,
        "--protein_records_path", args.protein_records_path,
        "--seed", str(args.seed),
        "--pytorch_seed", str(args.seed),
        "--metric", args.metric,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--init_lr", str(args.init_lr),
        "--max_lr", str(args.max_lr),
        "--final_lr", str(args.final_lr),
        "--warmup_epochs", str(args.warmup_epochs),
        "--dropout", str(args.dropout),
        "--ensemble_size", str(args.ensemble_size),
        "--num_workers", str(args.num_workers),
        "--cache_cutoff", str(args.cache_cutoff),
    ]
    device_text = str(args.device).lower()
    if device_text.startswith("cuda"):
        gpu_index = 0
        if ":" in device_text:
            try:
                gpu_index = int(device_text.split(":", 1)[1])
            except Exception:
                gpu_index = 0
        argv.extend(["--gpu", str(gpu_index)])
    else:
        argv.append("--no_cuda")
    argv.extend(["--smiles_columns", *args.smiles_columns])
    argv.extend(["--target_columns", *args.target_columns])
    argv.extend(["--extra_metrics", *args.extra_metrics])
    return argv


def write_metrics_bundle(out_dir: Path, split_name: str, pred_label_df: pd.DataFrame, metrics: dict):
    pred_label_df.to_csv(out_dir / f"pred_label_{split_name}.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / f"results_{split_name}.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / f"final_results_{split_name}.csv", index=False)


def _auto_esm_mem_cache_max(ram_budget_gb: float) -> int:
    # Keep ESM in-memory cache large enough to reduce disk churn while staying within host RAM headroom.
    safe_budget_gb = max(8.0, float(ram_budget_gb))
    cache_budget_mb = safe_budget_gb * 1024.0 * 0.35
    per_entry_mb = 2.0
    estimated_entries = int(cache_budget_mb / per_entry_mb)
    return max(512, min(32768, estimated_entries))


def _effective_low_ram_mode(args) -> bool:
    if bool(getattr(args, "disable_low_ram_mode", False)):
        return False
    return bool(getattr(args, "low_ram_mode", False))


def _run_single_training(parsed_args, run_training):
    from catpred.data import get_data, get_task_names, validate_dataset_type
    from catpred.features import (
        reset_featurization_parameters,
        set_explicit_h,
        set_adding_hs,
        set_keeping_atom_map,
        set_reaction,
        set_extra_atom_fdim,
        set_extra_bond_fdim,
    )

    reset_featurization_parameters()
    set_explicit_h(parsed_args.explicit_h)
    set_adding_hs(parsed_args.adding_h)
    set_keeping_atom_map(parsed_args.keeping_atom_map)
    if parsed_args.reaction:
        set_reaction(parsed_args.reaction, parsed_args.reaction_mode)
    elif parsed_args.reaction_solvent:
        set_reaction(True, parsed_args.reaction_mode)

    parsed_args.task_names = get_task_names(
        path=parsed_args.data_path,
        smiles_columns=parsed_args.smiles_columns,
        target_columns=parsed_args.target_columns,
        ignore_columns=parsed_args.ignore_columns,
    )

    data = get_data(
        path=parsed_args.data_path,
        protein_records_path=parsed_args.protein_records_path,
        vocabulary_path=parsed_args.vocabulary_path,
        args=parsed_args,
        skip_none_targets=True,
        data_weights_path=parsed_args.data_weights_path,
    )
    validate_dataset_type(data, dataset_type=parsed_args.dataset_type)

    parsed_args.features_size = data.features_size()
    if parsed_args.atom_descriptors == "descriptor":
        parsed_args.atom_descriptors_size = data.atom_descriptors_size()
    elif parsed_args.atom_descriptors == "feature":
        parsed_args.atom_features_size = data.atom_features_size()
        set_extra_atom_fdim(parsed_args.atom_features_size)

    if parsed_args.bond_descriptors == "descriptor":
        parsed_args.bond_descriptors_size = data.bond_descriptors_size()
    elif parsed_args.bond_descriptors == "feature":
        parsed_args.bond_features_size = data.bond_features_size()
        set_extra_bond_fdim(parsed_args.bond_features_size)

    run_training(args=parsed_args, data=data, logger=None)


def main():
    parser = argparse.ArgumentParser(description="Run CatPred on explicit train/val/test splits without modifying CatPred core code.")
    parser.add_argument("--train_csv", required=True, type=str)
    parser.add_argument("--val_csv", required=True, type=str)
    parser.add_argument("--test_csv", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--task_name", default="affinity", type=str)
    parser.add_argument("--dataset_type", default="regression", type=str)
    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--uniprot_id_col", default="uniprot_id", type=str)
    parser.add_argument("--smiles_columns", nargs="+", default=["smiles"])
    parser.add_argument("--target_columns", nargs="+", default=["log10_value"])
    parser.add_argument("--metric", default="rmse", type=str)
    parser.add_argument("--extra_metrics", nargs="+", default=["mae", "mse", "r2"])
    parser.add_argument("--seed", default=42, type=int)
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
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--cache_dir", default=default_cache_dir(), type=str)
    parser.add_argument("--prefetch_factor", default=4, type=int)
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--disable_persistent_workers", action="store_true")
    parser.add_argument("--smart_batching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bucket_multiplier", default=50, type=int)
    parser.add_argument("--cache_batch_graphs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mixed_precision", choices=["auto", "none", "bf16", "fp16"], default="auto")
    parser.add_argument("--optimizer_fused", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--lr_scheduler", choices=["cosine_warmup", "noam"], default="cosine_warmup")
    parser.add_argument("--val_every_n_epochs", default=1, type=int)
    parser.add_argument("--final_epoch_metrics_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early_stopping_patience", default=4, type=int)
    parser.add_argument("--early_stopping_min_delta", default=0.0, type=float)
    parser.add_argument("--overwrite_esm_cache", action="store_true")
    parser.add_argument("--require_cached_esm", action="store_true")
    parser.add_argument("--disable_lazy_esm", action="store_true")
    parser.add_argument("--disable_dataset_cache", action="store_true")
    parser.add_argument("--low_ram_mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable_low_ram_mode", action="store_true")
    parser.add_argument("--esm_mem_cache_max", default=None, type=int)
    parser.add_argument("--ram_budget_gb", default=90.0, type=float)
    parser.add_argument("--cpu_threads", default=None, type=int)
    parser.add_argument("--interop_threads", default=None, type=int)
    parser.add_argument("--disable_tf32", action="store_true")
    parser.add_argument("--resume_if_complete", action="store_true")
    parser.add_argument("--resume_marker_split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--skip_train_eval_after_fit", action="store_true")
    parser.add_argument("--skip_val_eval_after_fit", action="store_true")
    parser.add_argument("--skip_test_eval_after_fit", action="store_true")
    parser.add_argument("--skip_native_test_eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable_molgraph_disk_cache", action="store_true")
    args, passthrough = parser.parse_known_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker_file = out_dir / f"final_results_{args.resume_marker_split}.csv"
    if args.resume_if_complete and marker_file.exists():
        print(f"Skipping completed run: {out_dir}")
        return

    maybe_set_cache_env(args.cache_dir)
    configure_cpu_runtime(
        num_threads=args.cpu_threads,
        interop_threads=args.interop_threads,
    )
    effective_esm_mem_cache_max = args.esm_mem_cache_max
    if effective_esm_mem_cache_max is None:
        effective_esm_mem_cache_max = _auto_esm_mem_cache_max(args.ram_budget_gb)
    effective_low_ram_mode = _effective_low_ram_mode(args)

    os.environ["CATPRED_BENCH_LAZY_ESM"] = "0" if args.disable_lazy_esm else "1"
    os.environ["CATPRED_BENCH_DATASET_CACHE"] = "0" if args.disable_dataset_cache else "1"
    os.environ["CATPRED_BENCH_ESM_MEM_CACHE_MAX"] = str(max(1, int(effective_esm_mem_cache_max)))
    os.environ["CATPRED_BENCH_RAM_BUDGET_GB"] = str(max(1.0, float(args.ram_budget_gb)))
    configure_esm_cache_policy(
        overwrite_esm_cache=args.overwrite_esm_cache,
        require_cached_esm=args.require_cached_esm,
    )
    ensure_repo_on_path()
    configure_torch_runtime(args.device, enable_tf32=not args.disable_tf32)

    train_csv = materialize_tabular_as_csv(args.train_csv)
    val_csv = materialize_tabular_as_csv(args.val_csv)
    test_csv = materialize_tabular_as_csv(args.test_csv)
    protein_records_path = INLINE_PROTEIN_SENTINEL
    install_bench_patches(sequence_col=args.sequence_col, uniprot_id_col=args.uniprot_id_col)
    install_model_speed_patches()
    install_amp_patches(args.mixed_precision)
    install_optimizer_patches(args.optimizer_fused)
    install_scheduler_patches(args.lr_scheduler)
    install_training_control_patches(
        val_every_n_epochs=args.val_every_n_epochs,
        final_epoch_metrics_only=args.final_epoch_metrics_only,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        skip_native_test_eval=args.skip_native_test_eval,
        cache_batch_graphs=args.cache_batch_graphs,
    )
    install_dataloader_patches(
        pin_memory=not args.disable_pin_memory and "cuda" in str(args.device).lower(),
        persistent_workers=not args.disable_persistent_workers,
        prefetch_factor=args.prefetch_factor,
        smart_batching=args.smart_batching,
        bucket_multiplier=args.bucket_multiplier,
        low_ram_mode=effective_low_ram_mode,
        cache_batch_graphs=args.cache_batch_graphs,
    )

    from catpred.args import TrainArgs
    from catpred.args import PredictArgs
    from catpred.train.make_predictions import load_model, set_features
    from catpred.train.run_training import run_training

    args.train_csv = train_csv
    args.val_csv = val_csv
    args.test_csv = test_csv
    args.protein_records_path = protein_records_path

    if not args.disable_molgraph_disk_cache:
        train_cache_file = molgraph_cache_path(train_csv, args.cache_dir)
        val_cache_file = molgraph_cache_path(val_csv, args.cache_dir)
        test_cache_file = molgraph_cache_path(test_csv, args.cache_dir)

        loaded_train = load_molgraph_cache(train_csv, args.smiles_columns, args.cache_dir)
        loaded_val = load_molgraph_cache(val_csv, args.smiles_columns, args.cache_dir)
        loaded_test = load_molgraph_cache(test_csv, args.smiles_columns, args.cache_dir)
        print(
            "[bench] molgraph_cache "
            f"files train={'hit' if train_cache_file.exists() else 'miss'} "
            f"val={'hit' if val_cache_file.exists() else 'miss'} "
            f"test={'hit' if test_cache_file.exists() else 'miss'} | "
            f"loaded train={loaded_train} val={loaded_val} test={loaded_test}",
            flush=True,
        )

        if (loaded_train + loaded_val + loaded_test) == 0:
            warmed = warm_molgraph_cache(
                [train_csv, val_csv, test_csv],
                args.smiles_columns,
                args.cache_dir,
            )
            print(f"[bench] molgraph_cache warmup created={warmed}", flush=True)

    train_argv = build_catpred_train_argv(args) + passthrough
    old_argv = sys.argv[:]
    sys.argv = [__file__] + train_argv
    start = time.time()
    try:
        parsed_args = TrainArgs().parse_args(train_argv)
        parsed_args.grad_accum_steps = args.grad_accum_steps
        parsed_args.num_folds = 1
        _run_single_training(parsed_args, run_training)
    finally:
        sys.argv = old_argv
        if not args.disable_molgraph_disk_cache:
            saved_train = save_molgraph_cache(train_csv, args.smiles_columns, args.cache_dir)
            saved_val = save_molgraph_cache(val_csv, args.smiles_columns, args.cache_dir)
            saved_test = save_molgraph_cache(test_csv, args.smiles_columns, args.cache_dir)
            print(
                "[bench] molgraph_cache "
                f"saved train={saved_train} val={saved_val} test={saved_test}",
                flush=True,
            )

    ckpt_dir = _checkpoint_dir(out_dir)
    write_json(out_dir / "bestmodel_dir.json", {"checkpoint_dir": str(ckpt_dir)})
    if args.ensemble_size == 1:
        _copy_single_model_alias(out_dir)

    eval_plan = [
        ("train", train_csv, args.skip_train_eval_after_fit),
        ("val", val_csv, args.skip_val_eval_after_fit),
        ("test", test_csv, args.skip_test_eval_after_fit),
    ]
    eval_plan = [(split_name, split_csv) for split_name, split_csv, skip in eval_plan if not skip]
    if len(eval_plan) == 0:
        raise ValueError("All post-fit splits were skipped. Enable at least one of train/val/test evaluation.")

    pred_argv = [
        "--test_path", eval_plan[0][1],
        "--preds_path", str(out_dir / "_tmp_preds.csv"),
        "--checkpoint_dir", str(ckpt_dir),
        "--protein_records_path", protein_records_path,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--drop_extra_columns",
    ]
    device_text = str(args.device).lower()
    if device_text.startswith("cuda"):
        gpu_index = 0
        if ":" in device_text:
            try:
                gpu_index = int(device_text.split(":", 1)[1])
            except Exception:
                gpu_index = 0
        pred_argv.extend(["--gpu", str(gpu_index)])
    else:
        pred_argv.append("--no_cuda")
    pred_argv.extend(["--smiles_columns", *args.smiles_columns])
    pred_args = PredictArgs().parse_args(pred_argv)
    pred_bundle = load_model(pred_args, generator=False)
    set_features(pred_bundle[0], pred_bundle[1])

    split_metrics = {}
    for split_name, split_csv in eval_plan:
        pred_label_df, metrics, _ = evaluate_predictions_csv(
            input_csv=split_csv,
            protein_records_path=protein_records_path,
            checkpoint_dir=str(ckpt_dir),
            smiles_columns=args.smiles_columns,
            target_columns=args.target_columns,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            cache_dir=args.cache_dir,
            loaded_bundle=pred_bundle,
        )
        write_metrics_bundle(out_dir, split_name, pred_label_df, metrics)
        split_metrics[split_name] = metrics

    summary_row = {
        "task_name": args.task_name,
        "seed": args.seed,
        "checkpoint_dir": str(ckpt_dir),
        "elapsed_seconds": time.time() - start,
    }
    for split_name, metrics in split_metrics.items():
        for metric_name, metric_value in metrics.items():
            summary_row[f"{split_name}_{metric_name}"] = metric_value

    pd.DataFrame(
        [
            summary_row
        ]
    ).to_csv(out_dir / "run_summary.csv", index=False)
    with open(out_dir / "time_running.dat", "w") as f:
        f.write(f"{time.time() - start:.4f}\n")


if __name__ == "__main__":
    main()
