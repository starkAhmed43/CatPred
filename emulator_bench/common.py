import json
import os
import hashlib
import shutil
import sys
import csv
import pickle
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_repo_on_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def default_cache_dir() -> str:
    # Keep all benchmark caches under the EMULaToR processed baseline tree when available.
    env_override = os.getenv("CATPRED_BENCH_CACHE_DIR") or os.getenv("CATPRED_CACHE_PATH")
    if env_override:
        return str(Path(env_override).expanduser().resolve())

    external_cache = (
        REPO_ROOT.parent
        / "EMULaToR"
        / "data"
        / "processed"
        / "baselines"
        / "catpred"
        / "embeddings"
    )
    if external_cache.parent.exists():
        return str(external_cache.resolve())

    return str((REPO_ROOT / "emulator_bench" / ".cache_embeddings").resolve())


def discover_threshold_dirs(value_root: Path, split_groups, explicit_thresholds=None):
    jobs = []
    seen = set()
    for split_group in split_groups:
        split_root = value_root / split_group
        if not split_root.exists():
            continue

        if explicit_thresholds:
            threshold_dirs = []
            for threshold_name in explicit_thresholds:
                if threshold_name in {"default", "root", "all"}:
                    threshold_dirs.append(split_root)
                else:
                    threshold_dirs.append(split_root / threshold_name)
        else:
            threshold_dirs = [split_root]
            threshold_dirs.extend(
                [
                    p
                    for p in sorted(split_root.iterdir())
                    if p.is_dir() and (p.name.startswith("threshold_") or has_split_triplet(p))
                ]
            )

        for threshold_dir in threshold_dirs:
            if not threshold_dir.exists() or not has_split_triplet(threshold_dir):
                continue
            threshold_name = "default" if threshold_dir == split_root else threshold_dir.name
            key = (split_group, threshold_name, str(threshold_dir.resolve()))
            if key in seen:
                continue
            seen.add(key)
            jobs.append((split_group, threshold_name, threshold_dir))

    return jobs


def ensure_csv_triplet(threshold_dir: Path):
    return ensure_split_triplet(threshold_dir)


def split_file(threshold_dir: Path, split_name: str):
    for ext in (".csv", ".parquet"):
        candidate = threshold_dir / f"{split_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def has_split_triplet(threshold_dir: Path) -> bool:
    return all(split_file(threshold_dir, name) is not None for name in ("train", "val", "test"))


def ensure_split_triplet(threshold_dir: Path):
    train_path = split_file(threshold_dir, "train")
    val_path = split_file(threshold_dir, "val")
    test_path = split_file(threshold_dir, "test")
    return train_path, val_path, test_path


def threshold_to_float(name: str):
    try:
        return float(str(name).split("threshold_")[-1])
    except Exception:
        return float("inf")


def slug(text: str) -> str:
    return str(text).replace("/", "_").replace(" ", "_")


def get_split_meta(train_csv: Path, val_csv: Path, test_csv: Path, ratio_tolerance: float):
    train_size = len(load_tabular_dataframe(train_csv))
    val_size = len(load_tabular_dataframe(val_csv))
    test_size = len(load_tabular_dataframe(test_csv))
    total = train_size + val_size + test_size

    if total == 0:
        train_ratio = val_ratio = test_ratio = 0.0
    else:
        train_ratio = train_size / total
        val_ratio = val_size / total
        test_ratio = test_size / total

    target = (0.8, 0.1, 0.1)
    small_split_flag = int(
        abs(train_ratio - target[0]) > ratio_tolerance
        or abs(val_ratio - target[1]) > ratio_tolerance
        or abs(test_ratio - target[2]) > ratio_tolerance
    )

    return {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "small_split_flag": small_split_flag,
    }


def maybe_set_cache_env(cache_dir: str | None):
    resolved = str(Path(cache_dir or default_cache_dir()).expanduser().resolve())
    os.environ["CATPRED_CACHE_PATH"] = resolved
    os.environ["CATPRED_BENCH_CACHE_DIR"] = resolved
    os.environ["CATPRED_BENCH_TABULAR_CACHE_DIR"] = str((Path(resolved) / "tabular_csv").resolve())


def configure_esm_cache_policy(overwrite_esm_cache: bool = False, require_cached_esm: bool = False) -> None:
    os.environ["CATPRED_BENCH_OVERWRITE_ESM"] = "1" if overwrite_esm_cache else "0"
    os.environ["CATPRED_BENCH_REQUIRE_CACHED_ESM"] = "1" if require_cached_esm else "0"


def _tabular_cache_root(cache_dir: str | None = None) -> Path:
    root = cache_dir or os.getenv("CATPRED_BENCH_TABULAR_CACHE_DIR")
    if root is None:
        root = Path(default_cache_dir()) / "tabular_csv"
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def materialize_tabular_as_csv(path: str | Path, cache_dir: str | None = None) -> str:
    input_path = Path(path).expanduser().resolve()
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return str(input_path)
    if suffix != ".parquet":
        raise ValueError(f"Unsupported tabular format for {input_path}. Expected .csv or .parquet.")

    stat = input_path.stat()
    signature = f"{input_path}:{stat.st_size}:{stat.st_mtime_ns}"
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
    out_path = _tabular_cache_root(cache_dir) / f"{input_path.stem}.{digest}.csv"
    if out_path.exists():
        return str(out_path)

    df = pd.read_parquet(input_path)
    tmp_path = out_path.with_suffix(".tmp.csv")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(out_path)
    return str(out_path)


def load_tabular_dataframe(path: str | Path, usecols=None) -> pd.DataFrame:
    input_path = Path(path).expanduser().resolve()
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path, usecols=usecols)
    if suffix == ".parquet":
        return pd.read_parquet(input_path, columns=usecols)
    raise ValueError(f"Unsupported tabular format for {input_path}. Expected .csv or .parquet.")


def tabular_columns(path: str | Path):
    input_path = Path(path).expanduser().resolve()
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        with open(input_path) as f:
            reader = csv.DictReader(f)
            return list(reader.fieldnames or [])
    if suffix == ".parquet":
        return list(pd.read_parquet(input_path).columns)
    raise ValueError(f"Unsupported tabular format for {input_path}. Expected .csv or .parquet.")


def iter_tabular_rows(path: str | Path):
    input_path = Path(path).expanduser().resolve()
    if input_path.suffix.lower() == ".csv":
        with open(input_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row
        return

    df = load_tabular_dataframe(input_path)
    for row in df.to_dict(orient="records"):
        yield row


def configure_torch_runtime(device: str, enable_tf32: bool = True) -> None:
    if "cuda" not in str(device).lower():
        return

    import torch

    if enable_tf32:
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def copy_if_missing(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def _file_signature(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    signature = f"{resolved}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]


def _molgraph_cache_root(cache_dir: str | None = None) -> Path:
    root = Path(cache_dir or default_cache_dir()).expanduser().resolve() / "molgraph"
    root.mkdir(parents=True, exist_ok=True)
    return root


def molgraph_cache_path(csv_path: str | Path, cache_dir: str | None = None) -> Path:
    """Return the expected on-disk MolGraph cache path for a split CSV."""
    return _molgraph_cache_root(cache_dir) / f"{Path(csv_path).stem}.{_file_signature(csv_path)}.pkl"


def _collect_smiles_from_csv(csv_path: str | Path, smiles_columns: list[str]) -> set[str]:
    smiles_set: set[str] = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in smiles_columns:
                value = row.get(col)
                if value is not None and value != "":
                    smiles_set.add(str(value))
    return smiles_set


def load_molgraph_cache(csv_path: str | Path, smiles_columns: list[str], cache_dir: str | None = None) -> int:
    """Load cached MolGraph objects for this CSV into catpred's in-memory graph cache."""
    try:
        cache_path = _molgraph_cache_root(cache_dir) / f"{Path(csv_path).stem}.{_file_signature(csv_path)}.pkl"
        if not cache_path.exists():
            return 0

        with open(cache_path, "rb") as f:
            payload = pickle.load(f)

        graph_map = payload.get("graphs", payload)
        if not isinstance(graph_map, dict):
            return 0

        import catpred.data.data as data_module

        loaded = 0
        valid_smiles = _collect_smiles_from_csv(csv_path, smiles_columns)
        for smiles, graph in graph_map.items():
            if smiles in valid_smiles and smiles not in data_module.SMILES_TO_GRAPH:
                data_module.SMILES_TO_GRAPH[smiles] = graph
                loaded += 1
        return loaded
    except Exception:
        return 0


def save_molgraph_cache(csv_path: str | Path, smiles_columns: list[str], cache_dir: str | None = None) -> int:
    """Persist MolGraph objects used by this CSV from catpred's in-memory graph cache."""
    try:
        import catpred.data.data as data_module

        smiles_set = _collect_smiles_from_csv(csv_path, smiles_columns)
        subset = {s: data_module.SMILES_TO_GRAPH[s] for s in smiles_set if s in data_module.SMILES_TO_GRAPH}
        if len(subset) == 0:
            return 0

        cache_path = _molgraph_cache_root(cache_dir) / f"{Path(csv_path).stem}.{_file_signature(csv_path)}.pkl"
        tmp_path = cache_path.with_suffix(".tmp.pkl")
        with open(tmp_path, "wb") as f:
            pickle.dump({"graphs": subset}, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)
        return len(subset)
    except Exception:
        return 0


def warm_molgraph_cache(csv_paths: list[str], smiles_columns: list[str], cache_dir: str | None = None) -> int:
    """Precompute MolGraph cache for given split files and persist to disk cache."""
    try:
        ensure_repo_on_path()
        import catpred.data.data as data_module
        from catpred.features import MolGraph
        from catpred.rdkit import make_mol

        created = 0
        for csv_path in csv_paths:
            for row in iter_tabular_rows(csv_path):
                for col in smiles_columns:
                    smiles = row.get(col)
                    if smiles is None:
                        continue
                    smiles = str(smiles).strip()
                    if smiles == "" or smiles in data_module.SMILES_TO_GRAPH:
                        continue

                    try:
                        mol = make_mol(smiles, explicit_h=False, adding_h=False, keep_h=False)
                    except TypeError:
                        # Older signatures may use keeping_atom_map keyword.
                        mol = make_mol(smiles, False, False, False)
                    if mol is None:
                        continue

                    data_module.SMILES_TO_GRAPH[smiles] = MolGraph(mol)
                    created += 1

        for csv_path in csv_paths:
            save_molgraph_cache(csv_path, smiles_columns, cache_dir)

        return created
    except Exception:
        return 0
