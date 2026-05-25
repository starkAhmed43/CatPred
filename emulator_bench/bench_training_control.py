import json
import os
import random
import sys
import types
from contextlib import nullcontext
from logging import Logger
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import trange


def _install_early_shims() -> None:
    if "ipdb" not in sys.modules:
        ipdb_shim = types.ModuleType("ipdb")

        def _noop(*args, **kwargs):
            return None

        ipdb_shim.set_trace = _noop
        ipdb_shim.post_mortem = _noop
        sys.modules["ipdb"] = ipdb_shim

    if "tensorboardX" not in sys.modules:
        try:
            import tensorboardX  # type: ignore  # noqa: F401
        except Exception:
            try:
                from torch.utils.tensorboard import SummaryWriter as _SummaryWriter  # type: ignore
            except Exception:
                class _SummaryWriter:  # type: ignore
                    def __init__(self, *args, **kwargs):
                        pass

                    def add_scalar(self, *args, **kwargs):
                        pass

                    def add_scalars(self, *args, **kwargs):
                        pass

                    def add_histogram(self, *args, **kwargs):
                        pass

                    def flush(self):
                        pass

                    def close(self):
                        pass

            tbx_shim = types.ModuleType("tensorboardX")
            tbx_shim.SummaryWriter = _SummaryWriter
            sys.modules["tensorboardX"] = tbx_shim


_install_early_shims()

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from catpred.constants import MODEL_FILE_NAME
from catpred.data import MoleculeDataLoader, MoleculeDataset, get_class_sizes, get_data, set_cache_graph, split_data
from catpred.models import MoleculeModel
from catpred.nn_utils import param_count, param_count_all
from catpred.train.evaluate import evaluate, evaluate_predictions
from catpred.train.loss_functions import get_loss_func
from catpred.train.predict import predict
from catpred.train.train import train
from catpred.utils import (
    load_checkpoint,
    load_frzn_model,
    makedirs,
    multitask_mean,
    save_checkpoint,
    save_smiles_splits,
)


def _resolve_summary_writer():
    try:
        from tensorboardX import SummaryWriter as _SummaryWriter  # type: ignore
        return _SummaryWriter
    except Exception:
        try:
            from torch.utils.tensorboard import SummaryWriter as _SummaryWriter  # type: ignore
            return _SummaryWriter
        except Exception:
            class _SummaryWriter:  # type: ignore
                """Minimal no-op writer used when no tensorboard backend is installed."""

                def __init__(self, *args, **kwargs):
                    pass

                def add_scalar(self, *args, **kwargs):
                    pass

                def add_scalars(self, *args, **kwargs):
                    pass

                def add_histogram(self, *args, **kwargs):
                    pass

                def flush(self):
                    pass

                def close(self):
                    pass

            return _SummaryWriter


SummaryWriter = _resolve_summary_writer()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _cuda_empty_cache_interval() -> int:
    raw = os.getenv("CATPRED_BENCH_CUDA_EMPTY_CACHE_INTERVAL", "0")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _maybe_empty_cuda_cache(batch_idx: int, interval: int) -> None:
    if interval > 0 and torch.cuda.is_available() and (batch_idx + 1) % interval == 0:
        torch.cuda.empty_cache()


def _resolve_autocast_dtype(args):
    if str(getattr(args, "device", "")).lower().find("cuda") < 0 or not torch.cuda.is_available():
        return None
    mode = getattr(args, "mixed_precision", "auto")
    if mode == "none":
        return None
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    if mode != "auto":
        return None
    device = torch.device(args.device)
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    major, _minor = torch.cuda.get_device_capability(device_index)
    return torch.bfloat16 if major >= 8 else torch.float16


def _autocast_context(args):
    dtype = _resolve_autocast_dtype(args)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _training_state_path(save_dir: str) -> str:
    return os.path.join(save_dir, "training_state.pt")


def _training_complete_path(save_dir: str) -> str:
    return os.path.join(save_dir, "training_complete.json")


def _capture_rng_state() -> dict:
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    try:
        torch.set_rng_state(state["torch"])
    except Exception:
        pass
    try:
        np.random.set_state(state["numpy"])
    except Exception:
        pass
    try:
        random.setstate(state["python"])
    except Exception:
        pass
    if torch.cuda.is_available() and "cuda" in state:
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except Exception:
            pass


def _load_training_state(path: str, device):
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _save_training_state(
    path: str,
    *,
    model,
    optimizer,
    scheduler,
    epoch: int,
    n_iter: int,
    best_val_loss: float,
    best_epoch: int,
    epochs_since_improvement: int,
) -> None:
    scaler = getattr(model, "_bench_grad_scaler", None)
    payload = {
        "epoch": int(epoch),
        "n_iter": int(n_iter),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "epochs_since_improvement": int(epochs_since_improvement),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
        "grad_scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "rng_state": _capture_rng_state(),
    }
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _write_complete_marker(path: str, payload: dict) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _ensure_tensorboardx_shim() -> None:
    """Provide tensorboardX.SummaryWriter for modules that hard-import tensorboardX."""
    if "tensorboardX" in sys.modules:
        return

    shim = types.ModuleType("tensorboardX")
    shim.SummaryWriter = SummaryWriter
    sys.modules["tensorboardX"] = shim


def _compute_regression_val_loss(model, data_loader, loss_func, args) -> float:
    model.eval()
    losses = []
    empty_cache_interval = _cuda_empty_cache_interval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            mol_batch, features_batch, target_batch, mask_batch, atom_descriptors_batch, atom_features_batch, bond_descriptors_batch, bond_features_batch, _constraints_batch, data_weights_batch = (
                batch.batch_graph(),
                batch.features(),
                batch.targets(),
                batch.mask(),
                batch.atom_descriptors(),
                batch.atom_features(),
                batch.bond_descriptors(),
                batch.bond_features(),
                batch.constraints(),
                batch.data_weights(),
            )

            mask_batch = np.transpose(mask_batch).tolist()
            masks = torch.tensor(mask_batch, dtype=torch.bool)
            targets = torch.tensor([[0 if x is None else x for x in tb] for tb in target_batch])

            if args.target_weights is not None:
                target_weights = torch.tensor(args.target_weights).unsqueeze(0)
            else:
                target_weights = torch.ones(targets.shape[1]).unsqueeze(0)
            data_weights = torch.tensor(data_weights_batch).unsqueeze(1)

            if args.loss_function == "bounded_mse":
                lt_target_batch = torch.tensor(batch.lt_targets())
                gt_target_batch = torch.tensor(batch.gt_targets())

            torch_device = args.device
            masks = masks.to(torch_device)
            targets = targets.to(torch_device)
            target_weights = target_weights.to(torch_device)
            data_weights = data_weights.to(torch_device)
            if args.loss_function == "bounded_mse":
                lt_target_batch = lt_target_batch.to(torch_device)
                gt_target_batch = gt_target_batch.to(torch_device)

            with _autocast_context(args):
                preds = model(
                    mol_batch,
                    features_batch,
                    atom_descriptors_batch,
                    atom_features_batch,
                    bond_descriptors_batch,
                    bond_features_batch,
                    None,
                    None,
                )

                if args.loss_function == "bounded_mse":
                    loss = loss_func(preds, targets, lt_target_batch, gt_target_batch) * target_weights * data_weights * masks
                elif args.loss_function == "evidential":
                    loss = loss_func(preds, targets, args.evidential_regularization) * target_weights * data_weights * masks
                elif args.loss_function == "dirichlet":
                    loss = loss_func(preds, targets, args.evidential_regularization) * target_weights * data_weights * masks
                else:
                    loss = loss_func(preds, targets) * target_weights * data_weights * masks

                loss = loss.sum() / masks.sum()
            losses.append(float(loss.item()))
            _maybe_empty_cuda_cache(batch_idx, empty_cache_interval)

    model.train()
    if len(losses) == 0:
        return float("inf")
    return float(np.nanmean(losses))


def install_training_control_patches(
    val_every_n_epochs: int = 1,
    final_epoch_metrics_only: bool = False,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
    skip_native_test_eval: bool = False,
    cache_batch_graphs: bool = False,
) -> None:
    _ensure_tensorboardx_shim()
    import catpred.train.run_training as run_training_module

    run_training_module._bench_val_every_n_epochs = max(1, int(val_every_n_epochs))
    run_training_module._bench_final_epoch_metrics_only = bool(final_epoch_metrics_only)
    run_training_module._bench_early_stopping_patience = max(0, int(early_stopping_patience))
    run_training_module._bench_early_stopping_min_delta = float(early_stopping_min_delta)
    run_training_module._bench_skip_native_test_eval = bool(skip_native_test_eval)
    run_training_module._bench_cache_batch_graphs = bool(cache_batch_graphs)

    if getattr(run_training_module, "_bench_training_control_patched", False):
        return

    original_run_training = run_training_module.run_training

    def patched_run_training(args, data, logger: Logger = None) -> Dict[str, List[float]]:
        import catpred.data as data_pkg
        import catpred.train.train as train_module

        # Keep native behavior for settings this lightweight patch does not target.
        if args.dataset_type != "regression" or args.is_atom_bond_targets:
            return original_run_training(args, data, logger)

        if logger is not None:
            debug, info = logger.debug, logger.info
        else:
            debug = info = print

        torch.manual_seed(args.pytorch_seed)

        #debug(f"Splitting data with seed {args.seed}")
        if args.separate_test_path:
            test_data = data_pkg.get_data(
                path=args.separate_test_path,
                protein_records_path=args.protein_records_path,
                vocabulary_path=args.vocabulary_path,
                args=args,
                features_path=args.separate_test_features_path,
                atom_descriptors_path=args.separate_test_atom_descriptors_path,
                bond_descriptors_path=args.separate_test_bond_descriptors_path,
                phase_features_path=args.separate_test_phase_features_path,
                constraints_path=args.separate_test_constraints_path,
                smiles_columns=args.smiles_columns,
                loss_function=args.loss_function,
                logger=logger,
            )
        if args.separate_val_path:
            val_data = data_pkg.get_data(
                path=args.separate_val_path,
                protein_records_path=args.protein_records_path,
                vocabulary_path=args.vocabulary_path,
                args=args,
                features_path=args.separate_val_features_path,
                atom_descriptors_path=args.separate_val_atom_descriptors_path,
                bond_descriptors_path=args.separate_val_bond_descriptors_path,
                phase_features_path=args.separate_val_phase_features_path,
                constraints_path=args.separate_val_constraints_path,
                smiles_columns=args.smiles_columns,
                loss_function=args.loss_function,
                logger=logger,
            )

        if args.separate_val_path and args.separate_test_path:
            train_data = data
        elif args.separate_val_path:
            train_data, _, test_data = split_data(
                data=data,
                split_type=args.split_type,
                sizes=args.split_sizes,
                key_molecule_index=args.split_key_molecule,
                seed=args.seed,
                num_folds=args.num_folds,
                args=args,
                logger=logger,
            )
        elif args.separate_test_path:
            train_data, val_data, _ = split_data(
                data=data,
                split_type=args.split_type,
                sizes=args.split_sizes,
                key_molecule_index=args.split_key_molecule,
                seed=args.seed,
                num_folds=args.num_folds,
                args=args,
                logger=logger,
            )
        else:
            train_data, val_data, test_data = split_data(
                data=data,
                split_type=args.split_type,
                sizes=args.split_sizes,
                key_molecule_index=args.split_key_molecule,
                seed=args.seed,
                num_folds=args.num_folds,
                args=args,
                logger=logger,
            )

        if args.dataset_type == "classification":
            class_sizes = get_class_sizes(data)
            #debug("Class sizes")
            # for i, task_class_sizes in enumerate(class_sizes):
            #     debug(f'{args.task_names[i]} {", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')
            train_class_sizes = get_class_sizes(train_data, proportion=False)
            args.train_class_sizes = train_class_sizes

        if args.save_smiles_splits:
            save_smiles_splits(
                data_path=args.data_path,
                save_dir=args.save_dir,
                task_names=args.task_names,
                features_path=args.features_path,
                constraints_path=args.constraints_path,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                smiles_columns=args.smiles_columns,
                logger=logger,
            )

        if args.features_scaling:
            features_scaler = train_data.normalize_features(replace_nan_token=0)
            val_data.normalize_features(features_scaler)
            test_data.normalize_features(features_scaler)
        else:
            features_scaler = None

        if args.atom_descriptor_scaling and args.atom_descriptors is not None:
            atom_descriptor_scaler = train_data.normalize_features(replace_nan_token=0, scale_atom_descriptors=True)
            val_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
            test_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
        else:
            atom_descriptor_scaler = None

        if args.bond_descriptor_scaling and args.bond_descriptors is not None:
            bond_descriptor_scaler = train_data.normalize_features(replace_nan_token=0, scale_bond_descriptors=True)
            val_data.normalize_features(bond_descriptor_scaler, scale_bond_descriptors=True)
            test_data.normalize_features(bond_descriptor_scaler, scale_bond_descriptors=True)
        else:
            bond_descriptor_scaler = None

        args.train_data_size = len(train_data)

        debug(
            f"Total size = {len(data):,} | "
            f"train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}"
        )

        if len(val_data) == 0:
            raise ValueError("The validation data split is empty.")

        if len(test_data) == 0:
            debug("The test data split is empty. Performance on the test set will not be evaluated.")
            empty_test_set = True
        else:
            empty_test_set = False

        debug("Fitting scaler")
        scaler = train_data.normalize_targets()
        atom_bond_scaler = None
        args.spectra_phase_mask = None

        loss_func = get_loss_func(args)

        test_smiles, test_targets = test_data.smiles(), test_data.targets()
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

        cache_batch_graphs_now = bool(getattr(run_training_module, "_bench_cache_batch_graphs", False))

        # Batch-graph cache mode prebuilds BatchMolGraph objects once and reuses them each epoch.
        # Keep graph cache enabled and avoid worker collation to prevent repeated graph assembly.
        if cache_batch_graphs_now:
            set_cache_graph(True)
            num_workers = 0
            info("[bench] batch_graph_cache_mode active; forcing num_workers=0 and cache_graph=True")
        # If user requested workers, honor that request and disable dynamic graph writes.
        # Prewarmed MolGraph entries can still be reused from SMILES_TO_GRAPH.
        elif int(getattr(args, "num_workers", 0)) > 0:
            set_cache_graph(False)
            num_workers = int(args.num_workers)
        elif len(data) <= args.cache_cutoff:
            set_cache_graph(True)
            num_workers = 0
        else:
            set_cache_graph(False)
            num_workers = 0

        low_ram_mode = _env_flag("CATPRED_BENCH_LOW_RAM_MODE", default=True)
        uses_protein_esm = bool(getattr(args, "add_esm_feats", False)) and not bool(getattr(args, "skip_protein", False))
        force_single_worker = _env_flag("CATPRED_BENCH_FORCE_SINGLE_WORKER", default=False)
        if low_ram_mode and uses_protein_esm and num_workers > 1 and force_single_worker:
            info(
                "[bench] force_single_worker active for protein+ESM; "
                f"reducing num_workers {num_workers} -> 1"
            )
            num_workers = 1

        train_data_loader = data_pkg.MoleculeDataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            num_workers=num_workers,
            class_balance=args.class_balance,
            shuffle=True,
            seed=args.seed,
        )
        val_data_loader = data_pkg.MoleculeDataLoader(dataset=val_data, batch_size=args.batch_size, num_workers=num_workers)
        test_data_loader = data_pkg.MoleculeDataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=num_workers)

        val_every_n_epochs = int(getattr(run_training_module, "_bench_val_every_n_epochs", 1))
        final_epoch_metrics_only = bool(getattr(run_training_module, "_bench_final_epoch_metrics_only", False))
        early_stopping_patience = int(getattr(run_training_module, "_bench_early_stopping_patience", 0))
        early_stopping_min_delta = float(getattr(run_training_module, "_bench_early_stopping_min_delta", 0.0))
        skip_native_test_eval = bool(getattr(run_training_module, "_bench_skip_native_test_eval", False))

        info(
            "[bench] validation_control "
            f"val_every_n_epochs={val_every_n_epochs} "
            f"final_epoch_metrics_only={int(final_epoch_metrics_only)} "
            f"early_stopping_patience={early_stopping_patience} "
            f"early_stopping_min_delta={early_stopping_min_delta}"
        )

        for model_idx in range(args.ensemble_size):
            save_dir = os.path.join(args.save_dir, f"model_{model_idx}")
            makedirs(save_dir)
            try:
                writer = SummaryWriter(log_dir=save_dir)
            except TypeError:
                writer = SummaryWriter(logdir=save_dir)

            if args.checkpoint_paths is not None:
                debug(f"Loading model {model_idx} from {args.checkpoint_paths[model_idx]}")
                model = load_checkpoint(args.checkpoint_paths[model_idx], logger=logger)
            else:
                #debug(f"Building model {model_idx}")
                model = MoleculeModel(args)

            if args.checkpoint_frzn is not None:
                #debug(f"Loading and freezing parameters from {args.checkpoint_frzn}.")
                model = load_frzn_model(model=model, path=args.checkpoint_frzn, current_args=args, logger=logger)

            if args.checkpoint_frzn is not None:
                if args.unfreeze_all:
                    for param in model.parameters():
                        param.requires_grad = True
                #debug(f"Number of unfrozen parameters = {param_count(model):,}")
                #debug(f"Total number of parameters = {param_count_all(model):,}")
            # else:
                #debug(f"Number of parameters = {param_count_all(model):,}")

            model = model.to(args.device)

            model_file = os.path.join(save_dir, MODEL_FILE_NAME)
            state_file = _training_state_path(save_dir)
            complete_file = _training_complete_path(save_dir)
            if not os.path.exists(model_file):
                save_checkpoint(
                    model_file,
                    model,
                    scaler,
                    features_scaler,
                    atom_descriptor_scaler,
                    bond_descriptor_scaler,
                    atom_bond_scaler,
                    args,
                )

            optimizer = run_training_module.build_optimizer(model, args)
            scheduler = run_training_module.build_lr_scheduler(optimizer, args)

            best_val_loss = float("inf")
            best_epoch, n_iter = 0, 0
            epochs_since_improvement = 0
            start_epoch = 0
            training_complete = os.path.exists(complete_file) and os.path.exists(model_file)
            if training_complete:
                try:
                    with open(complete_file) as handle:
                        complete_payload = json.load(handle)
                    best_val_loss = float(complete_payload.get("best_val_loss", best_val_loss))
                    best_epoch = int(complete_payload.get("best_epoch", best_epoch))
                    info(f"[bench] resume model_{model_idx}: already complete; reusing {model_file}")
                except Exception:
                    pass
            else:
                state = _load_training_state(state_file, args.device)
                if state is not None:
                    model.load_state_dict(state["model_state_dict"])
                    optimizer.load_state_dict(state["optimizer_state_dict"])
                    if state.get("scheduler_state_dict") is not None and hasattr(scheduler, "load_state_dict"):
                        scheduler.load_state_dict(state["scheduler_state_dict"])
                    if state.get("grad_scaler_state_dict") is not None:
                        model._bench_pending_grad_scaler_state = state["grad_scaler_state_dict"]
                    best_val_loss = float(state.get("best_val_loss", best_val_loss))
                    best_epoch = int(state.get("best_epoch", best_epoch))
                    n_iter = int(state.get("n_iter", n_iter))
                    epochs_since_improvement = int(state.get("epochs_since_improvement", epochs_since_improvement))
                    start_epoch = int(state.get("epoch", -1)) + 1
                    _restore_rng_state(state.get("rng_state"))
                    info(f"[bench] resume model_{model_idx}: starting at epoch {start_epoch + 1}/{args.epochs}")

            for epoch in trange(start_epoch, args.epochs):
                # debug(f"Epoch {epoch}")
                n_iter = train_module.train(
                    model=model,
                    data_loader=train_data_loader,
                    loss_func=loss_func,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    n_iter=n_iter,
                    atom_bond_scaler=atom_bond_scaler,
                    logger=logger,
                    writer=writer,
                )

                is_final_epoch = epoch == args.epochs - 1
                should_validate = is_final_epoch or ((epoch + 1) % val_every_n_epochs == 0)
                if not should_validate:
                    _save_training_state(
                        state_file,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        n_iter=n_iter,
                        best_val_loss=best_val_loss,
                        best_epoch=best_epoch,
                        epochs_since_improvement=epochs_since_improvement,
                    )
                    continue

                val_loss = _compute_regression_val_loss(model, val_data_loader, loss_func, args)
                info(f"[bench] validation epoch={epoch + 1} loss={val_loss:.6f}")
                writer.add_scalar("validation_loss", val_loss, n_iter)

                should_compute_metrics = is_final_epoch or (not final_epoch_metrics_only)
                if should_compute_metrics:
                    val_scores = evaluate(
                        model=model,
                        data_loader=val_data_loader,
                        num_tasks=args.num_tasks,
                        metrics=args.metrics,
                        dataset_type=args.dataset_type,
                        scaler=scaler,
                        atom_bond_scaler=atom_bond_scaler,
                        logger=logger,
                    )
                    for metric, scores in val_scores.items():
                        mean_val_score = multitask_mean(scores, metric=metric)
                        debug(f"Validation {metric} = {mean_val_score:.6f}")
                        writer.add_scalar(f"validation_{metric}", mean_val_score, n_iter)

                improved = val_loss < (best_val_loss - early_stopping_min_delta)
                if improved:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    epochs_since_improvement = 0
                    save_checkpoint(
                        os.path.join(save_dir, MODEL_FILE_NAME),
                        model,
                        scaler,
                        features_scaler,
                        atom_descriptor_scaler,
                        bond_descriptor_scaler,
                        atom_bond_scaler,
                        args,
                    )
                else:
                    epochs_since_improvement += 1

                if early_stopping_patience > 0 and epochs_since_improvement >= early_stopping_patience:
                    info(
                        f"[bench] Early stopping at epoch {epoch} after {epochs_since_improvement} "
                        "validation checks without loss improvement."
                    )
                    _save_training_state(
                        state_file,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        n_iter=n_iter,
                        best_val_loss=best_val_loss,
                        best_epoch=best_epoch,
                        epochs_since_improvement=epochs_since_improvement,
                    )
                    break

                _save_training_state(
                    state_file,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    n_iter=n_iter,
                    best_val_loss=best_val_loss,
                    best_epoch=best_epoch,
                    epochs_since_improvement=epochs_since_improvement,
                )

            if not training_complete:
                _write_complete_marker(
                    complete_file,
                    {
                        "model_idx": model_idx,
                        "best_val_loss": best_val_loss,
                        "best_epoch": best_epoch,
                        "epochs": args.epochs,
                    },
                )

            info(f"Model {model_idx} best validation loss = {best_val_loss:.6f} on epoch {best_epoch}")
            model = load_checkpoint(model_file, device=args.device, logger=logger)

            if empty_test_set:
                info(f"Model {model_idx} provided with no test set, no metric evaluation will be performed.")
            elif skip_native_test_eval:
                info("[bench] native_test_eval skipped by flag")
            else:
                test_preds = predict(model=model, data_loader=test_data_loader, scaler=scaler, atom_bond_scaler=atom_bond_scaler)
                test_scores = evaluate_predictions(
                    preds=test_preds,
                    targets=test_targets,
                    num_tasks=args.num_tasks,
                    metrics=args.metrics,
                    dataset_type=args.dataset_type,
                    is_atom_bond_targets=args.is_atom_bond_targets,
                    gt_targets=test_data.gt_targets(),
                    lt_targets=test_data.lt_targets(),
                    logger=logger,
                )

                if len(test_preds) != 0:
                    sum_test_preds += np.array(test_preds)

                for metric, scores in test_scores.items():
                    avg_test_score = np.nanmean(scores)
                    info(f"Model {model_idx} test {metric} = {avg_test_score:.6f}")
                    writer.add_scalar(f"test_{metric}", avg_test_score, 0)
            writer.close()

        if empty_test_set:
            ensemble_scores = {metric: [np.nan for _task in args.task_names] for metric in args.metrics}
        elif skip_native_test_eval:
            ensemble_scores = {metric: [np.nan for _task in args.task_names] for metric in args.metrics}
        else:
            avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()
            ensemble_scores = evaluate_predictions(
                preds=avg_test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metrics=args.metrics,
                dataset_type=args.dataset_type,
                is_atom_bond_targets=args.is_atom_bond_targets,
                gt_targets=test_data.gt_targets(),
                lt_targets=test_data.lt_targets(),
                logger=logger,
            )

        for metric, scores in ensemble_scores.items():
            mean_ensemble_test_score = multitask_mean(scores, metric=metric)
            info(f"Ensemble test {metric} = {mean_ensemble_test_score:.6f}")

        with open(os.path.join(args.save_dir, "test_scores.json"), "w") as f:
            json.dump(ensemble_scores, f, indent=4, sort_keys=True)

        if args.save_preds and not empty_test_set:
            test_preds_dataframe = pd.DataFrame(data={"smiles": test_data.smiles()})
            for i, task_name in enumerate(args.task_names):
                test_preds_dataframe[task_name] = [pred[i] for pred in avg_test_preds]
            test_preds_dataframe.to_csv(os.path.join(args.save_dir, "test_preds.csv"), index=False)

        return ensemble_scores

    run_training_module.run_training = patched_run_training
    run_training_module._bench_training_control_patched = True
