import hashlib
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from common import default_cache_dir


_MEM_CACHE = OrderedDict()
_CACHE_VERSION = "esm2_t33_650M_UR50D_v1"
_WARNED_RECOVERY_KEYS = set()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _mem_cache_limit(default: int = 128) -> int:
    raw = os.getenv("CATPRED_BENCH_ESM_MEM_CACHE_MAX")
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except Exception:
        return default


def _cache_root(cache_dir: str | None = None) -> Path:
    root = cache_dir or os.getenv("CATPRED_BENCH_CACHE_DIR") or os.getenv("CATPRED_CACHE_PATH")
    if root is None:
        root = default_cache_dir()
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def esm_cache_dir(cache_dir: str | None = None) -> Path:
    path = _cache_root(cache_dir) / "esm2"
    path.mkdir(parents=True, exist_ok=True)
    return path


def esm_cache_key(sequence: str) -> str:
    return hashlib.sha256(f"{_CACHE_VERSION}|{sequence}".encode("utf-8")).hexdigest()


def esm_cache_path(sequence: str, cache_dir: str | None = None) -> Path:
    return esm_cache_dir(cache_dir) / f"{esm_cache_key(sequence)}.npy"


def load_cached_esm(sequence: str, cache_dir: str | None = None):
    key = esm_cache_key(sequence)
    cached = _MEM_CACHE.get(key)
    if cached is not None:
        _MEM_CACHE.move_to_end(key)
        return cached

    path = esm_cache_path(sequence, cache_dir)
    if not path.exists():
        return None

    arr = np.load(path, allow_pickle=False)
    tensor = torch.from_numpy(np.asarray(arr, dtype=np.float32))
    _MEM_CACHE[key] = tensor
    if len(_MEM_CACHE) > _mem_cache_limit():
        _MEM_CACHE.popitem(last=False)
    return tensor


def save_cached_esm(sequence: str, tensor, cache_dir: str | None = None):
    array = tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
    array = np.asarray(array, dtype=np.float32)
    path = esm_cache_path(sequence, cache_dir)
    tmp = path.with_suffix(f".tmp.{os.getpid()}.npy")
    with open(tmp, "wb") as f:
        np.save(f, array)
    os.replace(tmp, path)

    key = esm_cache_key(sequence)
    cached = torch.from_numpy(array)
    _MEM_CACHE[key] = cached
    if len(_MEM_CACHE) > _mem_cache_limit():
        _MEM_CACHE.popitem(last=False)
    return cached


def get_or_compute_esm(sequence: str, protein_name: str, sequence_feat_getter, cache_dir: str | None = None):
    overwrite_esm = _env_flag("CATPRED_BENCH_OVERWRITE_ESM", default=False)
    require_cached = _env_flag("CATPRED_BENCH_REQUIRE_CACHED_ESM", default=False)
    recover_missing = _env_flag("CATPRED_BENCH_RECOVER_MISSING_ESM", default=True)
    if not overwrite_esm:
        cached = load_cached_esm(sequence, cache_dir)
        if cached is not None:
            return cached
    elif require_cached:
        require_cached = False

    if require_cached:
        if not recover_missing:
            raise FileNotFoundError(
                "Missing cached ESM embedding while CATPRED_BENCH_REQUIRE_CACHED_ESM=1. "
                "Run build_tvt_data.py with --warm_esm_cache first or disable require_cached_esm. "
                "Common causes: sequence_max_length mismatch between warmup and training, "
                "cache root mismatch, or cache version change."
            )

        cache_key = esm_cache_key(sequence)
        if cache_key not in _WARNED_RECOVERY_KEYS:
            resolved_cache_dir = str(_cache_root(cache_dir))
            print(
                "[cache] missing cached ESM entry encountered under require_cached mode; "
                f"recomputing once for protein '{protein_name}' (seq_len={len(sequence)}, cache_dir={resolved_cache_dir}). "
                "Set CATPRED_BENCH_RECOVER_MISSING_ESM=0 to enforce strict failure.",
                flush=True,
            )
            _WARNED_RECOVERY_KEYS.add(cache_key)

    sequence_features, _ = sequence_feat_getter(sequence, name=protein_name, device="cpu")
    tensor = sequence_features[0].detach().cpu().to(dtype=torch.float32)
    return save_cached_esm(sequence, tensor, cache_dir)
