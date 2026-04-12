import argparse
import os
from pathlib import Path

import torch

from bench_feature_cache import load_cached_esm, save_cached_esm
from common import (
    configure_esm_cache_policy,
    default_cache_dir,
    ensure_repo_on_path,
    iter_tabular_rows,
    materialize_tabular_as_csv,
    maybe_set_cache_env,
    tabular_columns,
    write_json,
)


def iter_unique_proteins(paths, sequence_col: str, uniprot_id_col: str, sequence_max_length: int = 2048):
    seen = set()
    max_seq_len = int(sequence_max_length or 0)
    for path in paths:
        for row in iter_tabular_rows(path):
            seq = str(row[sequence_col]).strip()
            if max_seq_len > 0 and len(seq) > max_seq_len:
                seq = seq[:max_seq_len]
            protein_id = str(row[uniprot_id_col]).strip()
            key = (protein_id, seq)
            if key not in seen:
                seen.add(key)
                yield protein_id, seq


def _embed_esm_batch(sequences: list[str]) -> list[torch.Tensor]:
    ensure_repo_on_path()
    from catpred.data import esm_utils as esm_mod

    esm_mod.init_esm()
    model, batch_converter = esm_mod.GLOBAL_VARIABLES["model"]

    data = [("protein", seq) for seq in sequences]
    _labels, _strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens[:, : esm_mod.ESM_MAX_LENGTH]

    if not esm_mod.PROTEIN_EMBED_USE_CPU:
        batch_tokens = batch_tokens.to(next(model.parameters()).device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])

    reps = results["representations"][33]
    out = []
    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), esm_mod.ESM_MAX_LENGTH)
        out.append(reps[i][1 : seq_len + 1].detach().cpu().to(dtype=torch.float32))
    return out


def warm_esm_cache(
    paths,
    sequence_col: str,
    uniprot_id_col: str,
    cache_dir: str,
    batch_size: int = 64,
    sequence_max_length: int = 2048,
):
    ensure_repo_on_path()

    proteins = list(iter_unique_proteins(paths, sequence_col, uniprot_id_col, sequence_max_length=sequence_max_length))
    total = len(proteins)
    if total == 0:
        print("[cache] no proteins found for ESM warmup", flush=True)
        return

    missing = [(pid, seq) for pid, seq in proteins if load_cached_esm(seq, cache_dir=cache_dir) is None]
    if len(missing) == 0:
        print(f"[cache] all {total} protein ESM entries already cached", flush=True)
        return

    current_batch = max(1, int(batch_size))
    warmed = 0
    idx = 0
    missing_total = len(missing)
    print(
        f"[cache] warming missing ESM entries: {missing_total}/{total} "
        f"(batch_size={current_batch}, sequence_max_length={sequence_max_length})",
        flush=True,
    )

    while idx < missing_total:
        chunk = missing[idx : idx + current_batch]
        seqs = [seq for _pid, seq in chunk]
        try:
            batch_embeddings = _embed_esm_batch(seqs)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg and current_batch > 1:
                current_batch = max(1, current_batch // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[cache] OOM during warmup, reducing batch_size to {current_batch}", flush=True)
                continue
            raise

        for (_pid, seq), emb in zip(chunk, batch_embeddings):
            save_cached_esm(seq, emb, cache_dir=cache_dir)

        idx += len(chunk)
        warmed += len(chunk)
        if warmed % 100 == 0 or warmed == missing_total:
            print(f"[cache] warmed {warmed}/{missing_total} missing ({warmed}/{total} total)", flush=True)


def warm_esm_cache_allow_missing(
    paths,
    sequence_col: str,
    uniprot_id_col: str,
    cache_dir: str,
    batch_size: int = 64,
    sequence_max_length: int = 2048,
):
    """Warm cache by computing missing entries regardless of strict require-cached mode."""
    previous = os.getenv("CATPRED_BENCH_REQUIRE_CACHED_ESM")
    os.environ["CATPRED_BENCH_REQUIRE_CACHED_ESM"] = "0"
    try:
        warm_esm_cache(
            paths,
            sequence_col,
            uniprot_id_col,
            cache_dir,
            batch_size=batch_size,
            sequence_max_length=sequence_max_length,
        )
    finally:
        if previous is None:
            os.environ.pop("CATPRED_BENCH_REQUIRE_CACHED_ESM", None)
        else:
            os.environ["CATPRED_BENCH_REQUIRE_CACHED_ESM"] = previous


def main():
    parser = argparse.ArgumentParser(description="Validate CatPred TVT tabular files (.csv/.parquet) and optionally warm the ESM cache.")
    parser.add_argument("--train_csv", required=True, type=str)
    parser.add_argument("--val_csv", required=True, type=str)
    parser.add_argument("--test_csv", required=True, type=str)
    parser.add_argument("--output_root", required=True, type=str)
    parser.add_argument("--dataset_name", default="custom", type=str)
    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--uniprot_id_col", default="uniprot_id", type=str)
    parser.add_argument("--cache_dir", default=default_cache_dir(), type=str)
    parser.add_argument("--warm_esm_cache", action="store_true")
    parser.add_argument("--esm_warm_batch_size", default=64, type=int)
    parser.add_argument("--sequence_max_length", default=2048, type=int)
    parser.add_argument("--overwrite_esm_cache", action="store_true")
    parser.add_argument("--require_cached_esm", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    maybe_set_cache_env(args.cache_dir)
    configure_esm_cache_policy(
        overwrite_esm_cache=args.overwrite_esm_cache,
        require_cached_esm=args.require_cached_esm,
    )

    output_root = Path(args.output_root)
    manifest_path = output_root / f"{args.dataset_name}_manifest.json"

    if manifest_path.exists() and not args.overwrite:
        if args.warm_esm_cache:
            warm_esm_cache_allow_missing(
                [args.train_csv, args.val_csv, args.test_csv],
                args.sequence_col,
                args.uniprot_id_col,
                args.cache_dir,
                batch_size=args.esm_warm_batch_size,
                sequence_max_length=args.sequence_max_length,
            )
            try:
                existing = {}
                if manifest_path.exists():
                    import json

                    with open(manifest_path) as f:
                        existing = json.load(f)
                existing["warm_esm_cache"] = True
                existing["cache_dir"] = args.cache_dir
                write_json(manifest_path, existing)
            except Exception:
                pass
        print(f"Using existing manifest: {manifest_path}")
        return

    for path in [args.train_csv, args.val_csv, args.test_csv]:
        fieldnames = tabular_columns(path)
        if args.sequence_col not in fieldnames:
            raise ValueError(f"Missing sequence column '{args.sequence_col}' in {path}")
        if args.uniprot_id_col not in fieldnames:
            raise ValueError(f"Missing uniprot id column '{args.uniprot_id_col}' in {path}")

    train_csv = materialize_tabular_as_csv(args.train_csv)
    val_csv = materialize_tabular_as_csv(args.val_csv)
    test_csv = materialize_tabular_as_csv(args.test_csv)

    if args.warm_esm_cache:
        warm_esm_cache_allow_missing(
            [args.train_csv, args.val_csv, args.test_csv],
            args.sequence_col,
            args.uniprot_id_col,
            args.cache_dir,
            batch_size=args.esm_warm_batch_size,
            sequence_max_length=args.sequence_max_length,
        )

    manifest = {
        "dataset_name": args.dataset_name,
        "train_csv": str(Path(train_csv).resolve()),
        "val_csv": str(Path(val_csv).resolve()),
        "test_csv": str(Path(test_csv).resolve()),
        "train_input_path": str(Path(args.train_csv).resolve()),
        "val_input_path": str(Path(args.val_csv).resolve()),
        "test_input_path": str(Path(args.test_csv).resolve()),
        "sequence_col": args.sequence_col,
        "uniprot_id_col": args.uniprot_id_col,
        "cache_dir": args.cache_dir,
        "sequence_max_length": int(args.sequence_max_length),
        "warm_esm_cache": bool(args.warm_esm_cache),
        "overwrite_esm_cache": bool(args.overwrite_esm_cache),
        "require_cached_esm": bool(args.require_cached_esm),
    }
    write_json(manifest_path, manifest)
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
