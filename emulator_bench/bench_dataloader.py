import os
import json
import hashlib
import pickle
import threading
from pathlib import Path
from random import Random

from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from common import default_cache_dir


_CACHED_BATCHES_BY_KEY = {}
_BATCH_CACHE_SCHEMA_VERSION = 1


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _batch_cache_root() -> Path:
    root = os.getenv("CATPRED_BENCH_CACHE_DIR") or os.getenv("CATPRED_CACHE_PATH")
    if root is None:
        root_path = Path(default_cache_dir())
    else:
        root_path = Path(root).expanduser().resolve()
    path = root_path / "batch_graph_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _batch_cache_path(meta: dict) -> Path:
    digest = hashlib.sha256(json.dumps(meta, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:24]
    return _batch_cache_root() / f"{digest}.pkl"


def _load_disk_batch_cache(path: Path):
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            return None
        if payload.get("schema_version") != _BATCH_CACHE_SCHEMA_VERSION:
            return None
        batches = payload.get("batches")
        if not isinstance(batches, list):
            return None
        return batches
    except Exception:
        return None


def _save_disk_batch_cache(path: Path, batches):
    try:
        payload = {
            "schema_version": _BATCH_CACHE_SCHEMA_VERSION,
            "batches": batches,
        }
        tmp_path = path.with_suffix(f".tmp.{os.getpid()}.pkl")
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
    except Exception:
        pass


class LengthBucketBatchSampler(Sampler):
    def __init__(self, sampler, length_keys, batch_size, bucket_multiplier=50, drop_last=False, seed=None):
        self.sampler = sampler
        self.length_keys = length_keys
        self.batch_size = batch_size
        self.bucket_size = max(batch_size, batch_size * bucket_multiplier)
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

    def __iter__(self):
        pool = []
        rng = Random() if self.seed is None else Random(self.seed + self._epoch)
        self._epoch += 1

        def emit_batches(indices):
            sorted_indices = sorted(indices, key=lambda idx: self.length_keys[idx][0] + self.length_keys[idx][1])
            batches = [
                sorted_indices[i : i + self.batch_size]
                for i in range(0, len(sorted_indices), self.batch_size)
            ]
            if self.drop_last and batches and len(batches[-1]) < self.batch_size:
                batches = batches[:-1]
            rng.shuffle(batches)
            for batch in batches:
                yield batch

        for idx in self.sampler:
            pool.append(idx)
            if len(pool) >= self.bucket_size:
                yield from emit_batches(pool)
                pool = []

        if pool:
            yield from emit_batches(pool)

    def __len__(self):
        sampler_len = len(self.sampler)
        if self.drop_last:
            return sampler_len // self.batch_size
        return (sampler_len + self.batch_size - 1) // self.batch_size


def _length_keys(dataset):
    keys = []
    for datapoint in dataset:
        protein_len = len(datapoint.protein_record["seq"]) if getattr(datapoint, "protein_record", None) else 0
        smiles_len = sum(len(smiles) for smiles in datapoint.smiles) if getattr(datapoint, "smiles", None) else 0
        keys.append((protein_len, smiles_len))
    return keys


def install_dataloader_patches(
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int = 4,
    smart_batching: bool = False,
    bucket_multiplier: int = 50,
    low_ram_mode: bool = True,
    cache_batch_graphs: bool = False,
) -> None:
    if pin_memory is not None:
        os.environ["CATPRED_BENCH_PIN_MEMORY"] = "1" if pin_memory else "0"
    if persistent_workers is not None:
        os.environ["CATPRED_BENCH_PERSISTENT_WORKERS"] = "1" if persistent_workers else "0"
    os.environ["CATPRED_BENCH_PREFETCH_FACTOR"] = str(prefetch_factor)
    os.environ["CATPRED_BENCH_SMART_BATCHING"] = "1" if smart_batching else "0"
    os.environ["CATPRED_BENCH_BUCKET_MULTIPLIER"] = str(bucket_multiplier)
    os.environ["CATPRED_BENCH_LOW_RAM_MODE"] = "1" if low_ram_mode else "0"
    os.environ["CATPRED_BENCH_CACHE_BATCH_GRAPHS"] = "1" if cache_batch_graphs else "0"

    import catpred.data as data_pkg
    import catpred.data.data as data_module

    if getattr(data_module, "_bench_loader_patched", False):
        return

    MoleculeSampler = data_module.MoleculeSampler
    construct_molecule_batch = data_module.construct_molecule_batch

    class BenchMoleculeDataLoader(DataLoader):
        def __init__(
            self,
            dataset,
            batch_size: int = 50,
            num_workers: int = 8,
            class_balance: bool = False,
            shuffle: bool = False,
            seed: int = 0,
        ):
            self._dataset = dataset
            self._batch_size = batch_size
            self._num_workers = num_workers
            self._class_balance = class_balance
            self._shuffle = shuffle
            self._seed = seed
            self._context = None
            self._timeout = 0
            is_main_thread = threading.current_thread() is threading.main_thread()
            if not is_main_thread and self._num_workers > 0:
                self._context = "forkserver"
                self._timeout = 3600

            self._sampler = MoleculeSampler(
                dataset=self._dataset,
                class_balance=self._class_balance,
                shuffle=self._shuffle,
                seed=self._seed,
            )

            pin_memory_now = _env_flag("CATPRED_BENCH_PIN_MEMORY", default=False)
            persistent_workers_now = _env_flag("CATPRED_BENCH_PERSISTENT_WORKERS", default=True) and self._num_workers > 0
            prefetch_factor_now = _env_int("CATPRED_BENCH_PREFETCH_FACTOR", 4)
            smart_batching_now = (
                _env_flag("CATPRED_BENCH_SMART_BATCHING", default=False)
                and self._shuffle
                and len(self._dataset) > 0
            )
            bucket_multiplier_now = _env_int("CATPRED_BENCH_BUCKET_MULTIPLIER", 50)
            low_ram_mode_now = _env_flag("CATPRED_BENCH_LOW_RAM_MODE", default=True)
            cache_batch_graphs_now = _env_flag("CATPRED_BENCH_CACHE_BATCH_GRAPHS", default=False)
            self._cache_batch_graphs = cache_batch_graphs_now
            self._cached_batches = None
            self._cached_batch_order_rng = Random(self._seed)
            self._cache_epoch_counter = 0

            has_protein_records = len(self._dataset) > 0 and getattr(self._dataset[0], "protein_record", None) is not None
            if low_ram_mode_now and has_protein_records and self._num_workers > 0:
                # Protein records can be very large; avoid keeping bulky worker state resident.
                persistent_workers_now = False
                prefetch_factor_now = min(prefetch_factor_now, 2)

            if cache_batch_graphs_now:
                # Building batches once happens in the main process, so worker collation is unnecessary.
                self._num_workers = 0
                persistent_workers_now = False

            loader_kwargs = dict(
                dataset=self._dataset,
                num_workers=self._num_workers,
                collate_fn=construct_molecule_batch,
                multiprocessing_context=self._context,
                timeout=self._timeout,
                pin_memory=pin_memory_now,
            )
            if self._num_workers > 0:
                loader_kwargs["persistent_workers"] = persistent_workers_now
                loader_kwargs["prefetch_factor"] = prefetch_factor_now

            if smart_batching_now:
                batch_sampler = LengthBucketBatchSampler(
                    self._sampler,
                    _length_keys(self._dataset),
                    batch_size=self._batch_size,
                    bucket_multiplier=bucket_multiplier_now,
                    drop_last=False,
                    seed=self._seed,
                )
                super().__init__(batch_sampler=batch_sampler, **loader_kwargs)
            else:
                super().__init__(
                    batch_size=self._batch_size,
                    sampler=self._sampler,
                    **loader_kwargs,
                )

            if cache_batch_graphs_now:
                inmem_cache_key = (
                    id(self._dataset),
                    int(self._batch_size),
                    int(self._seed),
                    bool(self._shuffle),
                    bool(self._class_balance),
                    bool(smart_batching_now),
                    int(bucket_multiplier_now),
                )
                dataset_signature = getattr(self._dataset, "_bench_dataset_signature", None)
                disk_cache_meta = None
                disk_cache_path = None
                if dataset_signature is not None:
                    disk_cache_meta = {
                        "schema_version": _BATCH_CACHE_SCHEMA_VERSION,
                        "dataset_signature": str(dataset_signature),
                        "batch_size": int(self._batch_size),
                        "seed": int(self._seed),
                        "shuffle": bool(self._shuffle),
                        "class_balance": bool(self._class_balance),
                        "smart_batching": bool(smart_batching_now),
                        "bucket_multiplier": int(bucket_multiplier_now),
                    }
                    disk_cache_path = _batch_cache_path(disk_cache_meta)

                cached_batches = _CACHED_BATCHES_BY_KEY.get(inmem_cache_key)
                if cached_batches is None and disk_cache_path is not None and disk_cache_path.exists():
                    loaded_batches = _load_disk_batch_cache(disk_cache_path)
                    if loaded_batches is not None:
                        cached_batches = loaded_batches
                        _CACHED_BATCHES_BY_KEY[inmem_cache_key] = cached_batches
                        print(
                            "[bench] batch_graph_cache "
                            f"loaded_from_disk batches={len(cached_batches)} seed={self._seed}",
                            flush=True,
                        )

                if cached_batches is None:
                    print(
                        "[bench] batch_graph_cache "
                        f"building one-time batches for seed={self._seed} (this can take time)",
                        flush=True,
                    )
                    if smart_batching_now:
                        batch_sampler = LengthBucketBatchSampler(
                            self._sampler,
                            _length_keys(self._dataset),
                            batch_size=self._batch_size,
                            bucket_multiplier=bucket_multiplier_now,
                            drop_last=False,
                            seed=self._seed,
                        )
                        index_batches = [list(batch_indices) for batch_indices in batch_sampler]
                    else:
                        indices = list(self._sampler)
                        index_batches = [
                            indices[i : i + self._batch_size]
                            for i in range(0, len(indices), self._batch_size)
                        ]

                    cached_batches = []
                    for batch_indices in tqdm(
                        index_batches,
                        desc="[bench] batch_graph_cache build",
                        leave=False,
                    ):
                        cached_batches.append(
                            construct_molecule_batch([self._dataset[idx] for idx in batch_indices])
                        )
                    _CACHED_BATCHES_BY_KEY[inmem_cache_key] = cached_batches
                    if disk_cache_path is not None:
                        _save_disk_batch_cache(disk_cache_path, cached_batches)
                        print(
                            "[bench] batch_graph_cache "
                            f"saved_to_disk batches={len(cached_batches)} seed={self._seed}",
                            flush=True,
                        )
                    print(
                        "[bench] batch_graph_cache "
                        f"built batches={len(cached_batches)} seed={self._seed} shuffle={self._shuffle}",
                        flush=True,
                    )
                else:
                    print(
                        "[bench] batch_graph_cache "
                        f"reused batches={len(cached_batches)} seed={self._seed} shuffle={self._shuffle}",
                        flush=True,
                    )

                self._cached_batches = cached_batches

            print(
                "[bench] dataloader "
                f"shuffle={self._shuffle} batch_size={self._batch_size} num_workers={self._num_workers} "
                f"pin_memory={pin_memory_now} persistent_workers={persistent_workers_now} "
                f"prefetch_factor={prefetch_factor_now if self._num_workers > 0 else 'n/a'} "
                f"smart_batching={smart_batching_now} low_ram_mode={low_ram_mode_now} "
                f"cache_batch_graphs={cache_batch_graphs_now}",
                flush=True,
            )

        @property
        def targets(self):
            if self._class_balance or self._shuffle:
                raise ValueError("Cannot safely extract targets when class balance or shuffle are enabled.")
            return [self._dataset[index].targets for index in self._sampler]

        @property
        def gt_targets(self):
            if self._class_balance or self._shuffle:
                raise ValueError("Cannot safely extract targets when class balance or shuffle are enabled.")
            if not hasattr(self._dataset[0], "gt_targets"):
                return None
            return [self._dataset[index].gt_targets for index in self._sampler]

        @property
        def lt_targets(self):
            if self._class_balance or self._shuffle:
                raise ValueError("Cannot safely extract targets when class balance or shuffle are enabled.")
            if not hasattr(self._dataset[0], "lt_targets"):
                return None
            return [self._dataset[index].lt_targets for index in self._sampler]

        @property
        def iter_size(self):
            return len(self._sampler)

        def __len__(self):
            if self._cache_batch_graphs and self._cached_batches is not None:
                return len(self._cached_batches)
            return super().__len__()

        def __iter__(self):
            if self._cache_batch_graphs and self._cached_batches is not None:
                self._cache_epoch_counter += 1
                print(
                    "[bench] batch_graph_cache "
                    f"epoch={self._cache_epoch_counter} reusing prebuilt_batches={len(self._cached_batches)}",
                    flush=True,
                )
                order = list(range(len(self._cached_batches)))
                if self._shuffle and len(order) > 1:
                    self._cached_batch_order_rng.shuffle(order)
                for index in order:
                    yield self._cached_batches[index]
                return

            for batch in super().__iter__():
                yield batch

    data_module.MoleculeDataLoader = BenchMoleculeDataLoader
    data_pkg.MoleculeDataLoader = BenchMoleculeDataLoader
    data_module._bench_loader_patched = True
