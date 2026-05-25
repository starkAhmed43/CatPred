import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

from tqdm.auto import tqdm

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    DEFAULT_SPLIT_GROUPS,
    DEFAULT_VALUE_TYPES,
    REPO_ROOT,
    default_base_dir,
    default_cache_dir,
    discover_split_jobs,
    load_json,
)


ALIGN_SCRIPT = HERE / "align_structures.py"
CACHE_SCRIPT = HERE / "cache_embeddings.py"
PRECOMPUTE_SCRIPT = HERE / "precompute_features.py"
TRAIN_SCRIPT = HERE / "train_single_target_tvt.py"
DEFAULT_HPARAMS = HERE / "original_catpred_retrain_hparams.json"
_ACTIVE_PROCESSES: dict[int, tuple[subprocess.Popen, str]] = {}
_ACTIVE_LOCK = threading.Lock()
_STOP_REQUESTED = threading.Event()


def _parse_gpus(raw) -> list[str]:
    if raw is None:
        return ["1"]
    if isinstance(raw, (list, tuple)):
        items = []
        for item in raw:
            items.extend(str(item).split(","))
    else:
        items = str(raw).split(",")
    out = [item.strip() for item in items if item.strip()]
    return out or ["1"]


def _parse_gpu_arg(items) -> list[str]:
    if items is None:
        return ["1"]
    out = []
    for item in items:
        out.extend(part.strip() for part in str(item).split(",") if part.strip())
    return out or ["1"]


def _load_hparams(path: str | None) -> dict:
    if not path:
        path = str(DEFAULT_HPARAMS)
    hparams = load_json(Path(path)) if Path(path).exists() else {}
    return hparams


def _cli_supplied(argv: list[str], flag: str) -> bool:
    return any(item == flag or item.startswith(f"{flag}=") for item in argv)


def _apply_cli_hparam_overrides(args, hparams: dict, argv: list[str]) -> dict:
    merged = dict(hparams)
    override_fields = {
        "epochs": "--epochs",
        "batch_size": "--batch_size",
        "init_lr": "--init_lr",
        "max_lr": "--max_lr",
        "final_lr": "--final_lr",
        "warmup_epochs": "--warmup_epochs",
        "dropout": "--dropout",
        "loss_function": "--loss_function",
        "ensemble_size": "--ensemble_size",
        "seq_embed_dim": "--seq_embed_dim",
        "seq_self_attn_nheads": "--seq_self_attn_nheads",
        "sequence_max_length": "--sequence_max_length",
    }
    applied = {}
    for field, flag in override_fields.items():
        if _cli_supplied(argv, flag):
            value = getattr(args, field)
            merged[field] = value
            applied[field] = value
    if applied:
        print(f"[launch] CLI hparam overrides: {json.dumps(applied, sort_keys=True)}", flush=True)
    return merged


def _handle_stop_signal(signum, _frame) -> None:
    _STOP_REQUESTED.set()
    raise KeyboardInterrupt(f"received signal {signum}")


def _register_process(process: subprocess.Popen, label: str) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_PROCESSES[process.pid] = (process, label)


def _unregister_process(process: subprocess.Popen) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_PROCESSES.pop(process.pid, None)


def _terminate_process(process: subprocess.Popen, label: str, grace_seconds: float = 15.0) -> None:
    if process.poll() is not None:
        return
    print(f"[stop] terminating {label} pid={process.pid}", flush=True)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        process.terminate()
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass

    if process.poll() is None:
        print(f"[stop] killing {label} pid={process.pid}", flush=True)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except Exception:
            process.kill()
        process.wait(timeout=5.0)


def _terminate_active_processes() -> None:
    with _ACTIVE_LOCK:
        processes = list(_ACTIVE_PROCESSES.values())
    for process, label in processes:
        _terminate_process(process, label)


def _run_subprocess(cmd: list[str], env: dict, label: str) -> None:
    process = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        start_new_session=True,
    )
    _register_process(process, label)
    try:
        returncode = process.wait()
    except KeyboardInterrupt:
        _STOP_REQUESTED.set()
        _terminate_process(process, label)
        raise
    finally:
        _unregister_process(process)

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


def _run_step(cmd: list[str], env: dict, dry_run: bool, label: str) -> None:
    print(f"[{label}] " + " ".join(cmd), flush=True)
    if dry_run:
        return
    _run_subprocess(cmd, env, label)


def _common_scope_args(args) -> list[str]:
    out = ["--base_dir", args.base_dir]
    if args.value_types:
        out.extend(["--value_types", *args.value_types])
    if args.split_groups:
        out.extend(["--split_groups", *args.split_groups])
    if args.thresholds:
        out.extend(["--thresholds", *args.thresholds])
    return out


def _run_prepare_steps(args, hparams: dict, gpus: list[str]) -> None:
    env = os.environ.copy()
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = gpus[0]

    if not args.skip_align:
        cmd = [
            sys.executable,
            str(ALIGN_SCRIPT),
            *_common_scope_args(args),
            "--workers",
            str(args.align_workers),
            "--identity_threshold",
            str(args.identity_threshold),
        ]
        if args.overwrite_align:
            cmd.append("--overwrite")
        _run_step(cmd, env, args.dry_run, "align")

    pretrained_egnn = str(Path(args.cache_dir) / "progres" / "progres_egnn_by_structure_id.pt")
    if not args.skip_embeddings:
        cmd = [
            sys.executable,
            str(CACHE_SCRIPT),
            *_common_scope_args(args),
            "--cache_dir",
            args.cache_dir,
            "--sequence_max_length",
            str(hparams.get("sequence_max_length", args.sequence_max_length)),
            "--esm_batch_size",
            str(args.esm_batch_size),
            "--progres_device",
            "cuda:0",
            "--esm_gpus",
            "0",
        ]
        if args.overwrite_embeddings:
            cmd.append("--overwrite")
        if not args.add_esm_feats:
            cmd.append("--no-warm_esm")
        if not args.add_pretrained_egnn_feats:
            cmd.append("--no-warm_progres")
        _run_step(cmd, env, args.dry_run, "embeddings")

    if not args.skip_precompute:
        cmd = [
            sys.executable,
            str(PRECOMPUTE_SCRIPT),
            *_common_scope_args(args),
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--cache_dir",
            args.cache_dir,
            "--feature_root",
            args.feature_root,
            "--batch_size",
            str(hparams.get("batch_size", args.batch_size)),
            "--init_lr",
            str(hparams.get("init_lr", args.init_lr)),
            "--max_lr",
            str(hparams.get("max_lr", args.max_lr)),
            "--final_lr",
            str(hparams.get("final_lr", args.final_lr)),
            "--warmup_epochs",
            str(hparams.get("warmup_epochs", args.warmup_epochs)),
            "--dropout",
            str(hparams.get("dropout", args.dropout)),
            "--loss_function",
            str(hparams.get("loss_function", args.loss_function)),
            "--seq_embed_dim",
            str(hparams.get("seq_embed_dim", args.seq_embed_dim)),
            "--seq_self_attn_nheads",
            str(hparams.get("seq_self_attn_nheads", args.seq_self_attn_nheads)),
            "--sequence_max_length",
            str(hparams.get("sequence_max_length", args.sequence_max_length)),
            "--pretrained_egnn_feats_path",
            pretrained_egnn,
            "--jobs",
            str(args.precompute_jobs),
            "--max_tasks_per_child",
            str(args.precompute_max_tasks_per_child),
            "--bucket_multiplier",
            str(args.bucket_multiplier),
        ]
        if not args.add_esm_feats:
            cmd.append("--no-add_esm_feats")
        if not args.add_pretrained_egnn_feats:
            cmd.append("--no-add_pretrained_egnn_feats")
        if not args.cache_batch_graphs:
            cmd.append("--no-cache_batch_graphs")
        if not args.smart_batching:
            cmd.append("--no-smart_batching")
        if args.limit_rows:
            cmd.extend(["--limit_rows", str(args.limit_rows)])
        _run_step(cmd, env, args.dry_run, "precompute")


def _train_command(job: dict, seed: int, args, hparams: dict) -> tuple[list[str], Path]:
    threshold_part = job["threshold"] if job["threshold"] != "default" else "root"
    out_dir = (
        Path(job["threshold_dir"])
        / "catpred_results"
        / args.run_name
        / f"seed_{seed}"
    )
    pretrained_egnn = str(Path(args.cache_dir) / "progres" / "progres_egnn_by_structure_id.pt")
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train_csv",
        job["train_path"],
        "--val_csv",
        job["val_path"],
        "--test_csv",
        job["test_path"],
        "--out_dir",
        str(out_dir),
        "--task_name",
        job["value_type"],
        "--seed",
        str(seed),
        "--epochs",
        str(hparams.get("epochs", args.epochs)),
        "--batch_size",
        str(hparams.get("batch_size", args.batch_size)),
        "--init_lr",
        str(hparams.get("init_lr", args.init_lr)),
        "--max_lr",
        str(hparams.get("max_lr", args.max_lr)),
        "--final_lr",
        str(hparams.get("final_lr", args.final_lr)),
        "--warmup_epochs",
        str(hparams.get("warmup_epochs", args.warmup_epochs)),
        "--dropout",
        str(hparams.get("dropout", args.dropout)),
        "--loss_function",
        str(hparams.get("loss_function", args.loss_function)),
        "--ensemble_size",
        str(hparams.get("ensemble_size", args.ensemble_size)),
        "--seq_embed_dim",
        str(hparams.get("seq_embed_dim", args.seq_embed_dim)),
        "--seq_self_attn_nheads",
        str(hparams.get("seq_self_attn_nheads", args.seq_self_attn_nheads)),
        "--sequence_max_length",
        str(hparams.get("sequence_max_length", args.sequence_max_length)),
        "--num_workers",
        str(args.num_workers),
        "--grad_accum_steps",
        str(args.grad_accum_steps),
        "--cache_cutoff",
        str(args.cache_cutoff),
        "--device",
        "cuda:0",
        "--cache_dir",
        args.cache_dir,
        "--prefetch_factor",
        str(args.prefetch_factor),
        "--mixed_precision",
        args.mixed_precision,
        "--optimizer_fused",
        args.optimizer_fused,
        "--lr_scheduler",
        args.lr_scheduler,
        "--val_every_n_epochs",
        str(args.val_every_n_epochs),
        "--early_stopping_patience",
        str(args.early_stopping_patience),
        "--early_stopping_min_delta",
        str(args.early_stopping_min_delta),
        "--metric",
        args.metric,
        "--resume_if_complete",
        "--require_cached_esm",
        "--strict_precompute",
        "--uniprot_id_col",
        args.uniprot_id_col,
        "--bucket_multiplier",
        str(args.bucket_multiplier),
        "--smiles_columns",
        *args.smiles_columns,
        "--target_columns",
        *args.target_columns,
        "--extra_metrics",
        *args.extra_metrics,
    ]
    if args.add_esm_feats:
        cmd.append("--add_esm_feats")
    else:
        cmd.append("--no-add_esm_feats")
    if args.add_pretrained_egnn_feats:
        cmd.extend(["--add_pretrained_egnn_feats", "--pretrained_egnn_feats_path", pretrained_egnn])
    else:
        cmd.append("--no-add_pretrained_egnn_feats")
    if args.cache_batch_graphs:
        cmd.append("--cache_batch_graphs")
    else:
        cmd.append("--no-cache_batch_graphs")
    if args.smart_batching:
        cmd.append("--smart_batching")
    else:
        cmd.append("--no-smart_batching")
    if args.final_epoch_metrics_only:
        cmd.append("--final_epoch_metrics_only")
    if args.disable_pin_memory:
        cmd.append("--disable_pin_memory")
    if args.disable_persistent_workers:
        cmd.append("--disable_persistent_workers")
    if args.low_ram_mode:
        cmd.append("--low_ram_mode")
    if args.cpu_threads is not None:
        cmd.extend(["--cpu_threads", str(args.cpu_threads)])
    if args.interop_threads is not None:
        cmd.extend(["--interop_threads", str(args.interop_threads)])
    return cmd, out_dir


def _apply_cpu_thread_env(env: dict, cpu_threads: int | None, interop_threads: int | None) -> None:
    if cpu_threads is not None:
        value = str(max(1, int(cpu_threads)))
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            env[name] = value
    if interop_threads is not None:
        env["TORCH_NUM_INTEROP_THREADS"] = str(max(1, int(interop_threads)))


def _run_train_job(item) -> dict:
    cmd, out_dir, gpu, env = item
    if _STOP_REQUESTED.is_set():
        return {"out_dir": str(out_dir), "gpu": gpu, "status": "stopped_before_start"}
    marker = out_dir / "final_results_test.csv"
    if marker.exists():
        return {"out_dir": str(out_dir), "gpu": gpu, "status": "skipped_complete"}
    env = dict(env)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    label = f"train gpu={gpu} out={out_dir}"
    _run_subprocess(cmd, env, label)
    return {"out_dir": str(out_dir), "gpu": gpu, "status": "completed"}


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_stop_signal)

    parser = argparse.ArgumentParser(description="Retrain CatPred across all EMULaToR splits with resumable multi-GPU workers.")
    parser.add_argument("--base_dir", default=str(default_base_dir()), type=str)
    parser.add_argument("--value_types", nargs="+", default=list(DEFAULT_VALUE_TYPES))
    parser.add_argument("--split_groups", nargs="+", default=list(DEFAULT_SPLIT_GROUPS))
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--gpus", nargs="+", default=["1"], help="Physical GPU ids, e.g. --gpus 0 1 or --gpus 0,1")
    parser.add_argument("--max_parallel_per_gpu", default=1, type=int)
    parser.add_argument("--seeds", nargs="+", default=[42], type=int)
    parser.add_argument("--hparams_json", default=str(DEFAULT_HPARAMS), type=str)
    parser.add_argument("--cache_dir", default=default_cache_dir(), type=str)
    parser.add_argument("--feature_root", default=str(Path(default_base_dir()) / "features"), type=str)
    parser.add_argument("--run_name", default="paper_defaults", type=str)
    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--uniprot_id_col", default="catpred_structure_id", type=str)
    parser.add_argument("--smiles_columns", nargs="+", default=["smiles"])
    parser.add_argument("--target_columns", nargs="+", default=["log10_value"])
    parser.add_argument("--metric", default="rmse", type=str)
    parser.add_argument("--extra_metrics", nargs="+", default=["mae", "mse", "r2"])
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--init_lr", default=1e-4, type=float)
    parser.add_argument("--max_lr", default=1e-3, type=float)
    parser.add_argument("--final_lr", default=1e-4, type=float)
    parser.add_argument("--warmup_epochs", default=2.0, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--loss_function", default="mve", type=str)
    parser.add_argument("--ensemble_size", default=10, type=int)
    parser.add_argument("--seq_embed_dim", default=36, type=int)
    parser.add_argument("--seq_self_attn_nheads", default=6, type=int)
    parser.add_argument("--sequence_max_length", default=2048, type=int)
    parser.add_argument("--add_esm_feats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add_pretrained_egnn_feats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache_batch_graphs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smart_batching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bucket_multiplier", default=50, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--grad_accum_steps", default=1, type=int)
    parser.add_argument("--cache_cutoff", default="inf", type=str)
    parser.add_argument("--prefetch_factor", default=2, type=int)
    parser.add_argument("--cpu_threads", default=2, type=int)
    parser.add_argument("--interop_threads", default=1, type=int)
    parser.add_argument("--mixed_precision", choices=["auto", "none", "bf16", "fp16"], default="auto")
    parser.add_argument("--optimizer_fused", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--lr_scheduler", choices=["cosine_warmup", "noam"], default="cosine_warmup")
    parser.add_argument("--val_every_n_epochs", default=1, type=int)
    parser.add_argument("--final_epoch_metrics_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early_stopping_patience", default=0, type=int)
    parser.add_argument("--early_stopping_min_delta", default=0.0, type=float)
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--disable_persistent_workers", action="store_true")
    parser.add_argument("--low_ram_mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_align", action="store_true")
    parser.add_argument("--skip_embeddings", action="store_true")
    parser.add_argument("--skip_precompute", action="store_true")
    parser.add_argument("--overwrite_align", action="store_true")
    parser.add_argument("--overwrite_embeddings", action="store_true")
    parser.add_argument("--align_workers", default=max(1, (os.cpu_count() or 8) // 2), type=int)
    parser.add_argument("--identity_threshold", default=90.0, type=float)
    parser.add_argument("--esm_batch_size", default=8, type=int)
    parser.add_argument("--precompute_jobs", default=max(1, min(4, (os.cpu_count() or 4) // 4)), type=int)
    parser.add_argument("--precompute_max_tasks_per_child", default=1, type=int)
    parser.add_argument("--limit_rows", default=None, type=int)
    parser.add_argument("--dry_run", action="store_true")
    argv = sys.argv[1:]
    args = parser.parse_args(argv)

    gpus = _parse_gpu_arg(args.gpus)
    hparams = _apply_cli_hparam_overrides(args, _load_hparams(args.hparams_json), argv)
    args.base_dir = str(Path(args.base_dir).expanduser().resolve())
    args.cache_dir = str(Path(args.cache_dir).expanduser().resolve())
    args.feature_root = str(Path(args.feature_root).expanduser().resolve())

    _run_prepare_steps(args, hparams, gpus)

    jobs = discover_split_jobs(
        base_dir=args.base_dir,
        value_types=args.value_types,
        split_groups=args.split_groups,
        thresholds=args.thresholds,
    )
    train_items = []
    base_env = os.environ.copy()
    _apply_cpu_thread_env(base_env, args.cpu_threads, args.interop_threads)
    slots = []
    for gpu in gpus:
        slots.extend([gpu] * max(1, args.max_parallel_per_gpu))
    for idx, job in enumerate(jobs):
        for seed in args.seeds:
            cmd, out_dir = _train_command(job, seed, args, hparams)
            train_items.append((cmd, out_dir, slots[len(train_items) % len(slots)], base_env))

    print(f"[launch] train_jobs={len(train_items)} gpu_slots={slots}", flush=True)
    if args.dry_run:
        for cmd, out_dir, gpu, _env in train_items:
            print(f"[gpu {gpu}] {' '.join(cmd)}")
        return

    results = []
    stopped = False
    max_workers = len(slots)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    futures = []
    try:
        futures = [executor.submit(_run_train_job, item) for item in train_items]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Retrain", unit="job"):
            results.append(future.result())
        executor.shutdown(wait=True)
    except KeyboardInterrupt:
        stopped = True
        _STOP_REQUESTED.set()
        print("\n[launch] stop requested; terminating active training subprocesses...", flush=True)
        for future in futures:
            future.cancel()
        _terminate_active_processes()
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        _STOP_REQUESTED.set()
        _terminate_active_processes()
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    manifest_path = Path(args.base_dir) / "catpred_parallel_retrain_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump({"interrupted": stopped, "jobs": results}, handle, indent=2, sort_keys=True)
    print(f"[launch] wrote manifest: {manifest_path}", flush=True)
    if stopped:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
