from contextlib import nullcontext
import logging
import os
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm


def resolve_mixed_precision(device: torch.device, mode: str = "auto"):
    if mode == "none" or device.type != "cuda" or not torch.cuda.is_available():
        return None, "fp32", None

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    if mode == "bf16":
        return torch.bfloat16, "bf16-mixed", device_index
    if mode == "fp16":
        return torch.float16, "fp16-mixed", device_index
    if mode != "auto":
        raise ValueError(f"Unsupported mixed precision mode: {mode}")

    major, _minor = torch.cuda.get_device_capability(device_index)
    if major >= 8:
        return torch.bfloat16, "bf16-mixed", device_index
    return torch.float16, "fp16-mixed", device_index


def _autocast_context(device: torch.device, autocast_dtype=None):
    if autocast_dtype is not None and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=autocast_dtype)
    return nullcontext()


def install_amp_patches(mixed_precision: str = "auto") -> None:
    import catpred.train.evaluate as evaluate_module
    import catpred.train.run_training as run_training_module
    import catpred.train.predict as predict_module
    import catpred.train.train as train_module

    train_module._bench_mixed_precision_mode = mixed_precision
    predict_module._bench_mixed_precision_mode = mixed_precision

    if getattr(train_module, "_bench_amp_patched", False) and getattr(predict_module, "_bench_amp_patched", False):
        return

    def patched_train(
        model,
        data_loader,
        loss_func: Callable,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        args,
        n_iter: int = 0,
        atom_bond_scaler = None,
        logger: logging.Logger = None,
        writer = None,
    ) -> int:
        from catpred.nn_utils import NoamLR

        verbose_batch_logs = os.getenv("CATPRED_BENCH_VERBOSE_BATCH_LOGS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        show_loss_postfix = os.getenv("CATPRED_BENCH_TQDM_LOSS", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        debug = logger.debug if (logger is not None and verbose_batch_logs) else (lambda *args, **kwargs: None)

        model.train()
        if model.is_atom_bond_targets:
            loss_sum, iter_count = [0] * (len(args.atom_targets) + len(args.bond_targets)), 0
        else:
            loss_sum = iter_count = 0

        device = torch.device(args.device)
        precision_mode = getattr(train_module, "_bench_mixed_precision_mode", "auto")
        autocast_dtype, precision_label, precision_device_index = resolve_mixed_precision(device, precision_mode)

        scaler = getattr(model, "_bench_grad_scaler", None)
        if scaler is None or getattr(model, "_bench_grad_scaler_mode", None) != precision_mode:
            scaler = torch.amp.GradScaler("cuda", enabled=(autocast_dtype == torch.float16))
            model._bench_grad_scaler = scaler
            model._bench_grad_scaler_mode = precision_mode
        if not getattr(model, "_bench_amp_logged", False):
            if precision_device_index is not None:
                gpu_name = torch.cuda.get_device_name(precision_device_index)
                print(
                    f"[bench] mixed_precision={precision_mode} -> {precision_label} on {gpu_name} "
                    f"(grad_scaler={'on' if scaler.is_enabled() else 'off'})",
                    flush=True,
                )
            else:
                print(f"[bench] mixed_precision={precision_mode} -> {precision_label}", flush=True)
            model._bench_amp_logged = True
        grad_accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(data_loader, total=len(data_loader), leave=False)
        for batch_idx, batch in enumerate(pbar):
            mol_batch, features_batch, target_batch, mask_batch, atom_descriptors_batch, atom_features_batch, bond_descriptors_batch, bond_features_batch, constraints_batch, data_weights_batch = \
                batch.batch_graph(), batch.features(), batch.targets(), batch.mask(), batch.atom_descriptors(), \
                batch.atom_features(), batch.bond_descriptors(), batch.bond_features(), batch.constraints(), batch.data_weights()

            if model.is_atom_bond_targets:
                targets = []
                for dt in zip(*target_batch):
                    dt = np.concatenate(dt)
                    targets.append(torch.tensor([0 if x is None else x for x in dt], dtype=torch.float))
                masks = [torch.tensor(mask, dtype=torch.bool) for mask in mask_batch]
                if args.target_weights is not None:
                    target_weights = [torch.ones(1, 1) * i for i in args.target_weights]
                else:
                    target_weights = [torch.ones(1, 1) for _i in targets]
                data_weights = batch.atom_bond_data_weights()
                data_weights = [torch.tensor(x).unsqueeze(1) for x in data_weights]

                natoms, nbonds = batch.number_of_atoms, batch.number_of_bonds
                natoms, nbonds = np.array(natoms).flatten(), np.array(nbonds).flatten()
                constraints_batch = np.transpose(constraints_batch).tolist()
                ind = 0
                for i in range(len(args.atom_targets)):
                    if not args.atom_constraints[i]:
                        constraints_batch[ind] = None
                    else:
                        mean, std = atom_bond_scaler.means[ind][0], atom_bond_scaler.stds[ind][0]
                        for j, natom in enumerate(natoms):
                            constraints_batch[ind][j] = (constraints_batch[ind][j] - natom * mean) / std
                        constraints_batch[ind] = torch.tensor(constraints_batch[ind]).to(args.device)
                    ind += 1
                for i in range(len(args.bond_targets)):
                    if not args.bond_constraints[i]:
                        constraints_batch[ind] = None
                    else:
                        mean, std = atom_bond_scaler.means[ind][0], atom_bond_scaler.stds[ind][0]
                        for j, nbond in enumerate(nbonds):
                            constraints_batch[ind][j] = (constraints_batch[ind][j] - nbond * mean) / std
                        constraints_batch[ind] = torch.tensor(constraints_batch[ind]).to(args.device)
                    ind += 1
                bond_types_batch = []
                for _i in range(len(args.atom_targets)):
                    bond_types_batch.append(None)
                for i in range(len(args.bond_targets)):
                    if args.adding_bond_types and atom_bond_scaler is not None:
                        mean, std = atom_bond_scaler.means[i + len(args.atom_targets)][0], atom_bond_scaler.stds[i + len(args.atom_targets)][0]
                        bond_types = [(b.GetBondTypeAsDouble() - mean) / std for d in batch for b in d.mol[0].GetBonds()]
                        bond_types = torch.FloatTensor(bond_types).to(args.device)
                        bond_types_batch.append(bond_types)
                    else:
                        bond_types_batch.append(None)
            else:
                mask_batch = np.transpose(mask_batch).tolist()
                masks = torch.tensor(mask_batch, dtype=torch.bool)
                targets = torch.tensor([[0 if x is None else x for x in tb] for tb in target_batch])

                if args.target_weights is not None:
                    target_weights = torch.tensor(args.target_weights).unsqueeze(0)
                else:
                    target_weights = torch.ones(targets.shape[1]).unsqueeze(0)
                data_weights = torch.tensor(data_weights_batch).unsqueeze(1)

                constraints_batch = None
                bond_types_batch = None

                if args.loss_function == "bounded_mse":
                    lt_target_batch = torch.tensor(batch.lt_targets())
                    gt_target_batch = torch.tensor(batch.gt_targets())

            if model.is_atom_bond_targets:
                masks = [x.to(args.device).reshape([-1, 1]) for x in masks]
                targets = [x.to(args.device).reshape([-1, 1]) for x in targets]
                target_weights = [x.to(args.device) for x in target_weights]
                data_weights = [x.to(args.device) for x in data_weights]
            else:
                masks = masks.to(args.device)
                targets = targets.to(args.device)
                target_weights = target_weights.to(args.device)
                data_weights = data_weights.to(args.device)
                if args.loss_function == "bounded_mse":
                    lt_target_batch = lt_target_batch.to(args.device)
                    gt_target_batch = gt_target_batch.to(args.device)

            with _autocast_context(device, autocast_dtype=autocast_dtype):
                preds = model(
                    mol_batch,
                    features_batch,
                    atom_descriptors_batch,
                    atom_features_batch,
                    bond_descriptors_batch,
                    bond_features_batch,
                    constraints_batch,
                    bond_types_batch,
                )

                if model.is_atom_bond_targets:
                    loss_multi_task = []
                    for target, pred, target_weight, data_weight, mask in zip(targets, preds, target_weights, data_weights, masks):
                        if args.loss_function == "mcc" and args.dataset_type == "classification":
                            loss = loss_func(pred, target, data_weight, mask) * target_weight.squeeze(0)
                        elif args.loss_function == "bounded_mse":
                            raise ValueError(f'Loss function "{args.loss_function}" is not supported with dataset type {args.dataset_type} in atomic/bond properties prediction.')
                        elif args.loss_function in ["binary_cross_entropy", "mse", "mve"]:
                            loss = loss_func(pred, target) * target_weight * data_weight * mask
                        elif args.loss_function == "evidential":
                            loss = loss_func(pred, target, args.evidential_regularization) * target_weight * data_weight * mask
                        elif args.loss_function == "dirichlet" and args.dataset_type == "classification":
                            loss = loss_func(pred, target, args.evidential_regularization) * target_weight * data_weight * mask
                        else:
                            raise ValueError(f'Dataset type "{args.dataset_type}" is not supported.')
                        loss = loss.sum() / mask.sum()
                        loss_multi_task.append(loss)
                    loss_total = sum(loss_multi_task)
                else:
                    if args.loss_function == "mcc" and args.dataset_type == "classification":
                        loss = loss_func(preds, targets, data_weights, masks) * target_weights.squeeze(0)
                    elif args.loss_function == "mcc":
                        targets = targets.long()
                        target_losses = []
                        for target_index in range(preds.size(1)):
                            target_loss = loss_func(preds[:, target_index, :], targets[:, target_index], data_weights, masks[:, target_index]).unsqueeze(0)
                            target_losses.append(target_loss)
                        loss = torch.cat(target_losses) * target_weights.squeeze(0)
                    elif args.dataset_type == "multiclass":
                        targets = targets.long()
                        if args.loss_function == "dirichlet":
                            loss = loss_func(preds, targets, args.evidential_regularization) * target_weights * data_weights * masks
                        else:
                            target_losses = []
                            for target_index in range(preds.size(1)):
                                target_loss = loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1)
                                target_losses.append(target_loss)
                            loss = torch.cat(target_losses, dim=1).to(args.device) * target_weights * data_weights * masks
                    elif args.dataset_type == "spectra":
                        loss = loss_func(preds, targets, masks) * target_weights * data_weights * masks
                    elif args.loss_function == "bounded_mse":
                        loss = loss_func(preds, targets, lt_target_batch, gt_target_batch) * target_weights * data_weights * masks
                    elif args.loss_function == "evidential":
                        loss = loss_func(preds, targets, args.evidential_regularization) * target_weights * data_weights * masks
                    elif args.loss_function == "dirichlet":
                        loss = loss_func(preds, targets, args.evidential_regularization) * target_weights * data_weights * masks
                    else:
                        loss = loss_func(preds, targets) * target_weights * data_weights * masks

                    if args.loss_function == "mcc":
                        loss_total = loss.mean()
                    else:
                        loss_total = loss.sum() / masks.sum()

                loss_for_backward = loss_total / grad_accum_steps

            if model.is_atom_bond_targets:
                loss_sum = [x + y.item() for x, y in zip(loss_sum, loss_multi_task)]
                iter_count += 1
                if scaler.is_enabled():
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()
            else:
                loss_sum += loss_total.item()
                iter_count += 1
                if scaler.is_enabled():
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

            should_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(data_loader))

            if should_step and args.grad_clip:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if should_step:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if isinstance(scheduler, NoamLR) or getattr(scheduler, "_step_per_batch", False):
                    scheduler.step()

            n_iter += len(batch)

            if show_loss_postfix:
                pbar.set_postfix({"loss": f"{loss_total.item():.4f}"}, refresh=False)

            if (n_iter // args.batch_size) % args.log_frequency == 0:
                if model.is_atom_bond_targets:
                    loss_avg = sum(loss_sum) / iter_count
                    loss_sum, iter_count = [0] * (len(args.atom_targets) + len(args.bond_targets)), 0
                else:
                    loss_avg = loss_sum / iter_count
                    loss_sum = iter_count = 0

                if verbose_batch_logs:
                    lrs = scheduler.get_lr()
                    lrs_str = ", ".join(f"lr_{i} = {lr:.4e}" for i, lr in enumerate(lrs))
                    debug(f"Loss = {loss_avg:.4e}, {lrs_str}")

                if writer is not None:
                    writer.add_scalar("train_loss", loss_avg, n_iter)
                    if verbose_batch_logs:
                        for i, lr in enumerate(scheduler.get_lr()):
                            writer.add_scalar(f"learning_rate_{i}", lr, n_iter)

        return n_iter

    def patched_predict(
        model,
        data_loader,
        disable_progress_bar: bool = False,
        scaler=None,
        atom_bond_scaler = None,
        return_unc_parameters: bool = False,
        dropout_prob: float = 0.0,
    ):
        model.eval()

        if dropout_prob > 0.0:
            from catpred.nn_utils import activate_dropout

            def activate_dropout_(submodel):
                return activate_dropout(submodel, dropout_prob)

            model.apply(activate_dropout_)

        preds = []
        var, lambdas, alphas, betas = [], [], [], []

        device = next(model.parameters()).device
        precision_mode = getattr(predict_module, "_bench_mixed_precision_mode", "auto")
        autocast_dtype, _precision_label, _precision_device_index = resolve_mixed_precision(device, precision_mode)

        for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
            mol_batch = batch.batch_graph()
            features_batch = batch.features()
            atom_descriptors_batch = batch.atom_descriptors()
            atom_features_batch = batch.atom_features()
            bond_descriptors_batch = batch.bond_descriptors()
            bond_features_batch = batch.bond_features()
            constraints_batch = batch.constraints()

            if model.is_atom_bond_targets:
                natoms, nbonds = batch.number_of_atoms, batch.number_of_bonds
                natoms, nbonds = np.array(natoms).flatten(), np.array(nbonds).flatten()
                constraints_batch = np.transpose(constraints_batch).tolist()

                if constraints_batch == []:
                    for _ in batch._data:
                        natom_targets = len(model.atom_targets)
                        nbond_targets = len(model.bond_targets)
                        ntargets = natom_targets + nbond_targets
                        constraints_batch.append([None] * ntargets)

                ind = 0
                for i in range(len(model.atom_targets)):
                    if not model.atom_constraints[i]:
                        constraints_batch[ind] = None
                    else:
                        mean, std = atom_bond_scaler.means[ind][0], atom_bond_scaler.stds[ind][0]
                        for j, natom in enumerate(natoms):
                            constraints_batch[ind][j] = (constraints_batch[ind][j] - natom * mean) / std
                        constraints_batch[ind] = torch.tensor(constraints_batch[ind]).to(device)
                    ind += 1
                for i in range(len(model.bond_targets)):
                    if not model.bond_constraints[i]:
                        constraints_batch[ind] = None
                    else:
                        mean, std = atom_bond_scaler.means[ind][0], atom_bond_scaler.stds[ind][0]
                        for j, nbond in enumerate(nbonds):
                            constraints_batch[ind][j] = (constraints_batch[ind][j] - nbond * mean) / std
                        constraints_batch[ind] = torch.tensor(constraints_batch[ind]).to(device)
                    ind += 1
                bond_types_batch = []
                for _i in range(len(model.atom_targets)):
                    bond_types_batch.append(None)
                for i in range(len(model.bond_targets)):
                    if model.adding_bond_types and atom_bond_scaler is not None:
                        mean, std = atom_bond_scaler.means[i + len(model.atom_targets)][0], atom_bond_scaler.stds[i + len(model.atom_targets)][0]
                        bond_types = [(b.GetBondTypeAsDouble() - mean) / std for d in batch for b in d.mol[0].GetBonds()]
                        bond_types = torch.FloatTensor(bond_types).to(device)
                        bond_types_batch.append(bond_types)
                    else:
                        bond_types_batch.append(None)
            else:
                bond_types_batch = None

            with torch.no_grad():
                with _autocast_context(device, autocast_dtype=autocast_dtype):
                    batch_preds = model(
                        mol_batch,
                        features_batch,
                        atom_descriptors_batch,
                        atom_features_batch,
                        bond_descriptors_batch,
                        bond_features_batch,
                        constraints_batch,
                        bond_types_batch,
                    )

            if model.is_atom_bond_targets:
                batch_preds = [x.data.float().cpu().numpy() for x in batch_preds]
                batch_vars, batch_lambdas, batch_alphas, batch_betas = [], [], [], []

                for i, batch_pred in enumerate(batch_preds):
                    if model.loss_function == "mve":
                        batch_pred, batch_var = np.split(batch_pred, 2, axis=1)
                        batch_vars.append(batch_var)
                    elif model.loss_function == "dirichlet":
                        if model.classification:
                            batch_alpha = np.reshape(batch_pred, [batch_pred.shape[0], batch_pred.shape[1] // 2, 2])
                            batch_pred = batch_alpha[:, :, 1] / np.sum(batch_alpha, axis=2)
                            batch_alphas.append(batch_alpha)
                        elif model.multiclass:
                            raise ValueError(f"In atomic/bond properties prediction, {model.multiclass} is not supported.")
                    elif model.loss_function == "evidential":
                        batch_pred, batch_lambda, batch_alpha, batch_beta = np.split(batch_pred, 4, axis=1)
                        batch_alphas.append(batch_alpha)
                        batch_lambdas.append(batch_lambda)
                        batch_betas.append(batch_beta)
                    batch_preds[i] = batch_pred

                if atom_bond_scaler is not None:
                    batch_preds = atom_bond_scaler.inverse_transform(batch_preds)
                    for i, stds in enumerate(atom_bond_scaler.stds):
                        if model.loss_function == "mve":
                            batch_vars[i] = batch_vars[i] * stds ** 2
                        elif model.loss_function == "evidential":
                            batch_betas[i] = batch_betas[i] * stds ** 2

                preds.append(batch_preds)
                if model.loss_function == "mve":
                    var.append(batch_vars)
                elif model.loss_function == "dirichlet" and model.classification:
                    alphas.append(batch_alphas)
                elif model.loss_function == "evidential":
                    lambdas.append(batch_lambdas)
                    alphas.append(batch_alphas)
                    betas.append(batch_betas)
            else:
                batch_preds = batch_preds.data.float().cpu().numpy()

                if model.loss_function == "mve":
                    batch_preds, batch_var = np.split(batch_preds, 2, axis=1)
                elif model.loss_function == "dirichlet":
                    if model.classification:
                        batch_alphas = np.reshape(batch_preds, [batch_preds.shape[0], batch_preds.shape[1] // 2, 2])
                        batch_preds = batch_alphas[:, :, 1] / np.sum(batch_alphas, axis=2)
                    elif model.multiclass:
                        batch_alphas = batch_preds
                        batch_preds = batch_preds / np.sum(batch_alphas, axis=2, keepdims=True)
                elif model.loss_function == "evidential":
                    batch_preds, batch_lambdas, batch_alphas, batch_betas = np.split(batch_preds, 4, axis=1)

                if scaler is not None:
                    batch_preds = scaler.inverse_transform(batch_preds)
                    if model.loss_function == "mve":
                        batch_var = batch_var * scaler.stds ** 2
                    elif model.loss_function == "evidential":
                        batch_betas = batch_betas * scaler.stds ** 2

                preds.extend(batch_preds.tolist())
                if model.loss_function == "mve":
                    var.extend(batch_var.tolist())
                elif model.loss_function == "dirichlet" and model.classification:
                    alphas.extend(batch_alphas.tolist())
                elif model.loss_function == "evidential":
                    lambdas.extend(batch_lambdas.tolist())
                    alphas.extend(batch_alphas.tolist())
                    betas.extend(batch_betas.tolist())

        if model.is_atom_bond_targets:
            preds = [np.concatenate(x) for x in zip(*preds)]
            var = [np.concatenate(x) for x in zip(*var)]
            alphas = [np.concatenate(x) for x in zip(*alphas)]
            betas = [np.concatenate(x) for x in zip(*betas)]
            lambdas = [np.concatenate(x) for x in zip(*lambdas)]

        if return_unc_parameters:
            if model.loss_function == "mve":
                return preds, var
            if model.loss_function == "dirichlet":
                return preds, alphas
            if model.loss_function == "evidential":
                return preds, lambdas, alphas, betas

        return preds

    train_module.train = patched_train
    predict_module.predict = patched_predict
    evaluate_module.predict = patched_predict
    run_training_module.train = patched_train
    train_module._bench_amp_patched = True
    predict_module._bench_amp_patched = True
