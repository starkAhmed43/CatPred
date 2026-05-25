import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_tvt_data import warm_esm_cache_allow_missing  # noqa: E402
from common import (  # noqa: E402
    DEFAULT_SPLIT_GROUPS,
    DEFAULT_VALUE_TYPES,
    STRUCTURE_COLUMNS,
    default_base_dir,
    default_cache_dir,
    discover_split_jobs,
    maybe_set_cache_env,
    read_table,
    write_json,
)


def _norm(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "nat"} else text


def _collect_split_paths(args) -> list[str]:
    jobs = discover_split_jobs(
        base_dir=args.base_dir,
        value_types=args.value_types,
        split_groups=args.split_groups,
        thresholds=args.thresholds,
    )
    paths = []
    for job in jobs:
        paths.extend([job["train_path"], job["val_path"], job["test_path"]])
    if not paths:
        raise RuntimeError(f"No split files discovered under {args.base_dir}")
    return sorted(set(paths))


def _collect_structures(paths: list[str]) -> dict[str, str]:
    structure_by_id: dict[str, str] = {}
    for path in tqdm(paths, desc="Scan structure columns", unit="file"):
        frame = read_table(path)
        missing = [column for column in ("structure_path", "catpred_structure_id") if column not in frame.columns]
        if missing:
            raise ValueError(
                f"Missing aligned structure columns {missing} in {path}. "
                "Run emulator_bench/align_structures.py first."
            )
        for row in frame[["structure_path", "catpred_structure_id"]].dropna().to_dict("records"):
            structure_path = _norm(row.get("structure_path"))
            structure_id = _norm(row.get("catpred_structure_id"))
            if not structure_path or not structure_id:
                continue
            if not Path(structure_path).exists():
                continue
            structure_by_id.setdefault(structure_id, structure_path)
    return dict(sorted(structure_by_id.items()))


def _write_progres_list(structure_by_id: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{os.getpid()}.txt")
    with tmp.open("w", encoding="utf-8") as handle:
        for structure_id, structure_path in structure_by_id.items():
            handle.write(f"{structure_path}\t{structure_id}\t-\n")
    os.replace(tmp, path)


def _run_progres(filepaths: Path, output: Path, device: str, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        print(f"[progres] using existing searchdb: {output}", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "progres",
        "embed",
        "-l",
        str(filepaths),
        "-o",
        str(output),
        "-f",
        "pdb",
        "-d",
        device,
    ]
    env = os.environ.copy()
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    print("[progres] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _convert_searchdb(searchdb: Path, output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        print(f"[progres] using existing CatPred EGNN dict: {output}", flush=True)
        return
    raw = torch.load(searchdb, map_location="cpu")
    ids = list(raw["ids"])
    embeddings = raw["embeddings"]
    if len(ids) != int(embeddings.shape[0]):
        raise ValueError(f"proGRES searchdb id/embedding length mismatch in {searchdb}")
    payload = {
        str(structure_id): embeddings[index].detach().to(dtype=torch.float32, device="cpu")
        for index, structure_id in enumerate(ids)
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(f".tmp.{os.getpid()}.pt")
    torch.save(payload, tmp)
    os.replace(tmp, output)
    print(f"[progres] wrote CatPred EGNN dict: {output} ({len(payload)} structures)", flush=True)


def _warm_esm(paths: list[str], args) -> None:
    if not args.warm_esm:
        return
    maybe_set_cache_env(args.cache_dir)
    warm_esm_cache_allow_missing(
        paths,
        sequence_col=args.sequence_col,
        uniprot_id_col=args.uniprot_id_col,
        cache_dir=args.cache_dir,
        batch_size=args.esm_batch_size,
        sequence_max_length=args.sequence_max_length,
        esm_warm_gpu_ids=args.esm_gpus,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one-time CatPred ESM2 and proGRES embedding caches.")
    parser.add_argument("--base_dir", default=str(default_base_dir()), type=str)
    parser.add_argument("--value_types", nargs="+", default=list(DEFAULT_VALUE_TYPES))
    parser.add_argument("--split_groups", nargs="+", default=list(DEFAULT_SPLIT_GROUPS))
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--cache_dir", default=default_cache_dir(), type=str)
    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--uniprot_id_col", default="catpred_structure_id", type=str)
    parser.add_argument("--sequence_max_length", default=2048, type=int)
    parser.add_argument("--warm_esm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--esm_batch_size", default=8, type=int)
    parser.add_argument("--esm_gpus", nargs="*", default=[0], type=int)
    parser.add_argument("--warm_progres", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progres_device", default="cuda:0", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    progres_dir = cache_dir / "progres"
    filepaths = progres_dir / "filepaths.txt"
    searchdb = progres_dir / "searchdb.pt"
    catpred_egnn = progres_dir / "progres_egnn_by_structure_id.pt"

    paths = _collect_split_paths(args)
    if args.dry_run:
        print(f"[cache] split_files={len(paths)} cache_dir={cache_dir}", flush=True)
        print(f"[cache] would write {filepaths}")
        print(f"[cache] would write {catpred_egnn}")
        return

    structures = _collect_structures(paths)
    print(f"[cache] split_files={len(paths)} unique_structures={len(structures)} cache_dir={cache_dir}", flush=True)

    if args.warm_progres:
        _write_progres_list(structures, filepaths)
        _run_progres(filepaths, searchdb, args.progres_device, args.overwrite)
        _convert_searchdb(searchdb, catpred_egnn, args.overwrite)

    _warm_esm(paths, args)

    write_json(
        cache_dir / "embedding_manifest.json",
        {
            "split_files": paths,
            "unique_structures": len(structures),
            "progres_filepaths": str(filepaths),
            "progres_searchdb": str(searchdb),
            "pretrained_egnn_feats_path": str(catpred_egnn),
            "esm_cache_dir": str(cache_dir / "esm2"),
            "structure_columns": list(STRUCTURE_COLUMNS),
        },
    )


if __name__ == "__main__":
    main()
