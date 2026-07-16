import argparse
import concurrent.futures
import gc
import os
import signal
import sys
import types
from pathlib import Path

import pandas as pd
try:
    from src.utils.rich_progress import progress, write
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.utils.rich_progress import progress, write

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bench_fast_loader import INLINE_PROTEIN_SENTINEL, install_bench_patches  # noqa: E402
from bench_dataloader import clear_batch_graph_memory_cache, install_dataloader_patches  # noqa: E402
from common import (  # noqa: E402
    DEFAULT_SPLIT_GROUPS,
    DEFAULT_VALUE_TYPES,
    default_base_dir,
    default_cache_dir,
    default_features_dir,
    discover_split_jobs,
    ensure_repo_on_path,
    materialize_tabular_as_csv,
    maybe_set_cache_env,
    molgraph_cache_path,
    write_json,
    warm_molgraph_cache,
)


def _install_import_shims() -> None:
    if "ipdb" not in sys.modules:
        shim = types.ModuleType("ipdb")
        shim.set_trace = lambda *args, **kwargs: None
        shim.post_mortem = lambda *args, **kwargs: None
        sys.modules["ipdb"] = shim


def _limited_copy(path: str, limit_rows: int | None, feature_root: Path) -> str:
    if not limit_rows:
        return path
    source = Path(path).expanduser().resolve()
    out_dir = feature_root / "limited_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source.stem}.limit_{limit_rows}.csv"
    if out_path.exists():
        return str(out_path)
    frame = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    frame.head(int(limit_rows)).to_csv(out_path, index=False)
    return str(out_path)


def _train_argv(args, train_csv: str, val_csv: str, test_csv: str, seed: int) -> list[str]:
    argv = [
        "--data_path", train_csv,
        "--separate_val_path", val_csv,
        "--separate_test_path", test_csv,
        "--save_dir", str(Path(args.feature_root) / "_precompute_dummy"),
        "--dataset_type", args.dataset_type,
        "--protein_records_path", INLINE_PROTEIN_SENTINEL,
        "--seed", str(seed),
        "--pytorch_seed", str(seed),
        "--metric", args.metric,
        "--epochs", "1",
        "--batch_size", str(args.batch_size),
        "--init_lr", str(args.init_lr),
        "--max_lr", str(args.max_lr),
        "--final_lr", str(args.final_lr),
        "--warmup_epochs", str(args.warmup_epochs),
        "--dropout", str(args.dropout),
        "--ensemble_size", "1",
        "--num_workers", "0",
        "--cache_cutoff", "inf",
        "--no_cuda",
        "--loss_function", args.loss_function,
        "--seq_embed_dim", str(args.seq_embed_dim),
        "--seq_self_attn_nheads", str(args.seq_self_attn_nheads),
        "--sequence_max_length", str(args.sequence_max_length),
    ]
    argv.extend(["--smiles_columns", *args.smiles_columns])
    argv.extend(["--target_columns", *args.target_columns])
    if args.add_esm_feats:
        argv.append("--add_esm_feats")
    if args.add_pretrained_egnn_feats and args.pretrained_egnn_feats_path:
        argv.extend(["--add_pretrained_egnn_feats", "--pretrained_egnn_feats_path", args.pretrained_egnn_feats_path])
    return argv


def _load_dataset(path: str, parsed_args, skip_none_targets: bool):
    from catpred.data import get_data

    return get_data(
        path=path,
        protein_records_path=INLINE_PROTEIN_SENTINEL,
        vocabulary_path=parsed_args.vocabulary_path,
        args=parsed_args,
        skip_none_targets=skip_none_targets,
        smiles_columns=parsed_args.smiles_columns,
        target_columns=parsed_args.target_columns,
        loss_function=parsed_args.loss_function,
        store_row=True,
    )


def _build_batch_cache(dataset, batch_size: int, seed: int, shuffle: bool) -> int:
    from catpred.data import MoleculeDataLoader

    loader = MoleculeDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0,
        class_balance=False,
        shuffle=shuffle,
        seed=seed,
    )
    count = 0
    for _batch in loader:
        count += 1
    del loader
    clear_batch_graph_memory_cache()
    gc.collect()
    return count


def _release_precompute_memory() -> None:
    clear_batch_graph_memory_cache()
    try:
        import catpred.data.data as data_module

        data_module.empty_cache()
    except Exception:
        pass
    gc.collect()


def _handle_stop_signal(signum, _frame) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")


def _terminate_process_pool(executor: concurrent.futures.ProcessPoolExecutor, grace_seconds: float = 5.0) -> None:
    processes = list((getattr(executor, "_processes", None) or {}).values())
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        executor.shutdown(wait=False)

    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=grace_seconds)
    for process in processes:
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
    for process in processes:
        process.join(timeout=1.0)


def _precompute_one(payload: dict) -> dict:
    try:
        _install_import_shims()
        ensure_repo_on_path()
        maybe_set_cache_env(payload["cache_dir"])
        os.environ["CATPRED_BENCH_DATASET_CACHE"] = "1"
        os.environ["CATPRED_BENCH_STRICT_PRECOMPUTE"] = "0"
        os.environ["CATPRED_BENCH_REQUIRE_CACHED_ESM"] = "1" if payload["require_cached_esm"] else "0"
        os.environ["CATPRED_BENCH_RECOVER_MISSING_ESM"] = "0"
        os.environ["CATPRED_BENCH_LAZY_ESM"] = "1"
        os.environ["CATPRED_BENCH_KEEP_BATCH_CACHE_IN_MEMORY"] = "0"

        install_bench_patches(sequence_col=payload["sequence_col"], uniprot_id_col=payload["uniprot_id_col"])
        install_dataloader_patches(
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2,
            smart_batching=payload["smart_batching"],
            bucket_multiplier=payload["bucket_multiplier"],
            low_ram_mode=True,
            cache_batch_graphs=payload["cache_batch_graphs"],
        )

        feature_root = Path(payload["feature_root"])
        train_input = _limited_copy(payload["train_path"], payload["limit_rows"], feature_root)
        val_input = _limited_copy(payload["val_path"], payload["limit_rows"], feature_root)
        test_input = _limited_copy(payload["test_path"], payload["limit_rows"], feature_root)
        train_csv = materialize_tabular_as_csv(train_input)
        val_csv = materialize_tabular_as_csv(val_input)
        test_csv = materialize_tabular_as_csv(test_input)
        csvs = [train_csv, val_csv, test_csv]

        created_graphs = warm_molgraph_cache(csvs, payload["smiles_columns"], payload["cache_dir"])

        from catpred.args import TrainArgs

        parsed_args = TrainArgs().parse_args(_train_argv(argparse.Namespace(**payload), train_csv, val_csv, test_csv, payload["seed"]))
        parsed_args.grad_accum_steps = 1
        parsed_args.num_folds = 1

        train_raw = _load_dataset(train_csv, parsed_args, False)
        val_data = _load_dataset(val_csv, parsed_args, False)
        test_data = _load_dataset(test_csv, parsed_args, False)

        batch_counts = {}
        if payload["cache_batch_graphs"]:
            batch_counts["train_eval"] = _build_batch_cache(train_raw, payload["batch_size"], 0, False)
            batch_counts["val"] = _build_batch_cache(val_data, payload["batch_size"], 0, False)
            batch_counts["test"] = _build_batch_cache(test_data, payload["batch_size"], 0, False)

            train_norm = _load_dataset(train_csv, parsed_args, True)
            train_norm.normalize_targets()
            batch_counts["train_shuffle"] = _build_batch_cache(train_norm, payload["batch_size"], payload["seed"], True)

        cache_files = {
            "train_molgraph": str(molgraph_cache_path(train_csv, payload["cache_dir"])),
            "val_molgraph": str(molgraph_cache_path(val_csv, payload["cache_dir"])),
            "test_molgraph": str(molgraph_cache_path(test_csv, payload["cache_dir"])),
        }
        return {
            "value_type": payload["value_type"],
            "split_group": payload["split_group"],
            "threshold": payload["threshold"],
            "seed": payload["seed"],
            "train_csv": train_csv,
            "val_csv": val_csv,
            "test_csv": test_csv,
            "created_molgraphs": created_graphs,
            "batch_counts": batch_counts,
            "cache_files": cache_files,
        }
    finally:
        _release_precompute_memory()


def _payloads(args) -> list[dict]:
    jobs = discover_split_jobs(
        base_dir=args.base_dir,
        value_types=args.value_types,
        split_groups=args.split_groups,
        thresholds=args.thresholds,
    )
    if not jobs:
        raise RuntimeError(f"No split jobs discovered under {args.base_dir}")

    out = []
    for job in jobs:
        for seed in args.seeds:
            payload = {
                **job,
                "seed": int(seed),
                "cache_dir": args.cache_dir,
                "feature_root": str(args.feature_root),
                "dataset_type": args.dataset_type,
                "sequence_col": args.sequence_col,
                "uniprot_id_col": args.uniprot_id_col,
                "smiles_columns": args.smiles_columns,
                "target_columns": args.target_columns,
                "metric": args.metric,
                "batch_size": args.batch_size,
                "init_lr": args.init_lr,
                "max_lr": args.max_lr,
                "final_lr": args.final_lr,
                "warmup_epochs": args.warmup_epochs,
                "dropout": args.dropout,
                "loss_function": args.loss_function,
                "seq_embed_dim": args.seq_embed_dim,
                "seq_self_attn_nheads": args.seq_self_attn_nheads,
                "sequence_max_length": args.sequence_max_length,
                "add_esm_feats": args.add_esm_feats,
                "add_pretrained_egnn_feats": args.add_pretrained_egnn_feats,
                "pretrained_egnn_feats_path": args.pretrained_egnn_feats_path,
                "require_cached_esm": args.require_cached_esm,
                "cache_batch_graphs": args.cache_batch_graphs,
                "smart_batching": args.smart_batching,
                "bucket_multiplier": args.bucket_multiplier,
                "limit_rows": args.limit_rows,
            }
            out.append(payload)
    return out


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_stop_signal)

    parser = argparse.ArgumentParser(description="Precompute CatPred CPU feature caches before GPU training.")
    parser.add_argument("--base_dir", default=str(default_base_dir()), type=str)
    parser.add_argument("--value_types", nargs="+", default=list(DEFAULT_VALUE_TYPES))
    parser.add_argument("--split_groups", nargs="+", default=list(DEFAULT_SPLIT_GROUPS))
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", default=[42], type=int)
    parser.add_argument("--cache_dir", default=default_cache_dir(), type=str)
    parser.add_argument("--feature_root", default=str(default_features_dir()), type=str)
    parser.add_argument("--dataset_type", default="regression", type=str)
    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--uniprot_id_col", default="catpred_structure_id", type=str)
    parser.add_argument("--smiles_columns", nargs="+", default=["smiles"])
    parser.add_argument("--target_columns", nargs="+", default=["log10_value"])
    parser.add_argument("--metric", default="rmse", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--init_lr", default=1e-4, type=float)
    parser.add_argument("--max_lr", default=1e-3, type=float)
    parser.add_argument("--final_lr", default=1e-4, type=float)
    parser.add_argument("--warmup_epochs", default=2.0, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--loss_function", default="mve", type=str)
    parser.add_argument("--seq_embed_dim", default=36, type=int)
    parser.add_argument("--seq_self_attn_nheads", default=6, type=int)
    parser.add_argument("--sequence_max_length", default=2048, type=int)
    parser.add_argument("--add_esm_feats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add_pretrained_egnn_feats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained_egnn_feats_path", default=None, type=str)
    parser.add_argument("--require_cached_esm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache_batch_graphs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smart_batching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bucket_multiplier", default=50, type=int)
    parser.add_argument("--limit_rows", default=None, type=int)
    parser.add_argument("--jobs", default=max(1, min(4, (os.cpu_count() or 4) // 4)), type=int)
    parser.add_argument(
        "--max_tasks_per_child",
        default=1,
        type=int,
        help="Recycle each precompute worker after this many split/seed jobs. Use 0 to disable.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.pretrained_egnn_feats_path is None:
        args.pretrained_egnn_feats_path = str(Path(args.cache_dir) / "progres" / "progres_egnn_by_structure_id.pt")
    args.feature_root = Path(args.feature_root).expanduser().resolve()
    args.feature_root.mkdir(parents=True, exist_ok=True)

    payloads = _payloads(args)
    print(
        f"[precompute] jobs={len(payloads)} workers={args.jobs} "
        f"max_tasks_per_child={args.max_tasks_per_child} cache_dir={args.cache_dir}",
        flush=True,
    )
    if args.dry_run:
        for payload in payloads:
            print(f"- {payload['value_type']}/{payload['split_group']}/{payload['threshold']} seed={payload['seed']}")
        return

    manifest_rows = []
    manifest_path = args.feature_root / "precompute_manifest.json"
    partial_manifest_path = args.feature_root / "precompute_manifest.partial.json"
    executor_kwargs = {"max_workers": max(1, args.jobs)}
    if args.max_tasks_per_child and args.max_tasks_per_child > 0:
        executor_kwargs["max_tasks_per_child"] = int(args.max_tasks_per_child)
    futures = []
    executor = concurrent.futures.ProcessPoolExecutor(**executor_kwargs)
    try:
        futures = [executor.submit(_precompute_one, payload) for payload in payloads]
        for future in progress(concurrent.futures.as_completed(futures), total=len(futures), desc="Precompute", unit="job"):
            manifest_rows.append(future.result())
        executor.shutdown(wait=True)
    except KeyboardInterrupt:
        print("\n[precompute] stop requested; terminating worker processes...", flush=True)
        for future in futures:
            future.cancel()
        _terminate_process_pool(executor)
        if manifest_rows:
            write_json(
                partial_manifest_path,
                {
                    "interrupted": True,
                    "completed_jobs": len(manifest_rows),
                    "total_jobs": len(payloads),
                    "jobs": manifest_rows,
                },
            )
            print(f"[precompute] wrote partial manifest: {partial_manifest_path}", flush=True)
        raise SystemExit(130)

    write_json(manifest_path, {"jobs": manifest_rows})
    print(f"[precompute] wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
