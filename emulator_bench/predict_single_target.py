import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bench_amp import install_amp_patches
from bench_dataloader import install_dataloader_patches
from bench_fast_loader import INLINE_PROTEIN_SENTINEL, install_bench_patches
from bench_speedups import install_model_speed_patches
from common import (
    configure_esm_cache_policy,
    configure_torch_runtime,
    default_cache_dir,
    ensure_repo_on_path,
    materialize_tabular_as_csv,
    maybe_set_cache_env,
)


def average_ensemble_predictions(predictions):
    if len(predictions) == 1:
        return predictions[0]
    acc = np.array(predictions[0], dtype=float)
    for pred in predictions[1:]:
        acc += np.array(pred, dtype=float)
    return (acc / len(predictions)).tolist()


def make_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pcc = float(np.corrcoef(y_true, y_pred)[0, 1])
        yt_rank = pd.Series(y_true).rank(method="average").to_numpy()
        yp_rank = pd.Series(y_pred).rank(method="average").to_numpy()
        scc = float(np.corrcoef(yt_rank, yp_rank)[0, 1])
    else:
        pcc = 0.0
        scc = 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"PCC": pcc, "SCC": scc, "R2": r2, "RMSE": rmse, "MSE": mse, "MAE": mae}


def evaluate_predictions_csv(
    input_csv: str,
    protein_records_path: str,
    checkpoint_dir: str,
    smiles_columns,
    target_columns,
    batch_size: int,
    num_workers: int,
    device: str,
    cache_dir: str | None,
    loaded_bundle=None,
):
    maybe_set_cache_env(cache_dir)
    ensure_repo_on_path()

    from catpred.args import PredictArgs
    from catpred.train.evaluate import evaluate_predictions
    from catpred.data import MoleculeDataLoader, get_data
    from catpred.train.make_predictions import load_model, set_features
    from catpred.train.predict import predict

    device_text = str(device).lower()
    use_cuda = device_text.startswith("cuda")
    gpu_index = 0
    if use_cuda and ":" in device_text:
        try:
            gpu_index = int(device_text.split(":", 1)[1])
        except Exception:
            gpu_index = 0

    if loaded_bundle is None:
        pred_argv = [
            "--test_path", input_csv,
            "--preds_path", str(Path(input_csv).with_suffix(".preds.tmp.csv")),
            "--checkpoint_dir", checkpoint_dir,
            "--protein_records_path", protein_records_path,
            "--batch_size", str(batch_size),
            "--num_workers", str(num_workers),
            "--drop_extra_columns",
        ]
        if use_cuda:
            pred_argv.extend(["--gpu", str(gpu_index)])
        else:
            pred_argv.append("--no_cuda")
        pred_argv.extend(["--smiles_columns", *smiles_columns])
        pred_args = PredictArgs().parse_args(pred_argv)
        pred_args, train_args, models, scaler_sets, num_tasks, task_names = load_model(pred_args, generator=False)
        set_features(pred_args, train_args)
    else:
        pred_args, train_args, models, scaler_sets, num_tasks, task_names = loaded_bundle

    pred_args.test_path = input_csv
    pred_args.batch_size = batch_size
    pred_args.num_workers = num_workers
    pred_args.device = torch.device("cuda", gpu_index) if use_cuda else torch.device("cpu")

    test_data = get_data(
        path=input_csv,
        protein_records_path=protein_records_path,
        smiles_columns=smiles_columns,
        target_columns=task_names if task_names else target_columns,
        skip_invalid_smiles=True,
        args=pred_args,
        store_row=False,
        loss_function=train_args.loss_function,
    )
    test_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=pred_args.batch_size,
        num_workers=pred_args.num_workers,
    )

    model_preds = []
    for model, scalers in zip(models, scaler_sets):
        scaler, _, _, _, atom_bond_scaler = scalers
        preds = predict(model=model, data_loader=test_loader, scaler=scaler, atom_bond_scaler=atom_bond_scaler)
        model_preds.append(preds)

    avg_preds = average_ensemble_predictions(model_preds)
    targets = test_data.targets()
    metric_names = ["rmse", "mae", "mse", "r2"]
    raw_metrics = evaluate_predictions(
        preds=avg_preds,
        targets=targets,
        num_tasks=num_tasks,
        metrics=metric_names,
        dataset_type=train_args.dataset_type,
        is_atom_bond_targets=False,
    )

    rows = []
    for pred_row, tgt_row in zip(avg_preds, targets):
        row = {}
        for idx, task in enumerate(task_names):
            row[f"{task}_pred"] = pred_row[idx]
            row[f"{task}_label"] = tgt_row[idx]
        rows.append(row)
    pred_label_df = pd.DataFrame(rows)

    metrics = {}
    for metric_name, values in raw_metrics.items():
        metrics[metric_name.upper()] = float(np.nanmean(values)) if len(values) else float("nan")

    if len(task_names) == 1:
        single_task = task_names[0]
        scalar_metrics = make_metrics(pred_label_df[f"{single_task}_label"], pred_label_df[f"{single_task}_pred"])
        metrics.update(scalar_metrics)

    return pred_label_df, metrics, task_names


def main():
    parser = argparse.ArgumentParser(description="Predict and score a CatPred split using a trained checkpoint directory.")
    parser.add_argument("--input_csv", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--out_csv", required=True, type=str)
    parser.add_argument("--metrics_csv", default=None, type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--cache_dir", default=default_cache_dir(), type=str)
    parser.add_argument("--prefetch_factor", default=4, type=int)
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--disable_persistent_workers", action="store_true")
    parser.add_argument("--mixed_precision", choices=["auto", "none", "bf16", "fp16"], default="auto")
    parser.add_argument("--overwrite_esm_cache", action="store_true")
    parser.add_argument("--require_cached_esm", action="store_true")
    parser.add_argument("--disable_tf32", action="store_true")
    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--uniprot_id_col", default="uniprot_id", type=str)
    parser.add_argument("--smiles_columns", nargs="+", default=["smiles"])
    parser.add_argument("--target_columns", nargs="+", default=["log10_value"])
    args = parser.parse_args()

    input_csv = materialize_tabular_as_csv(args.input_csv)
    protein_records_path = INLINE_PROTEIN_SENTINEL
    configure_esm_cache_policy(
        overwrite_esm_cache=args.overwrite_esm_cache,
        require_cached_esm=args.require_cached_esm,
    )
    configure_torch_runtime(args.device, enable_tf32=not args.disable_tf32)
    install_bench_patches(sequence_col=args.sequence_col, uniprot_id_col=args.uniprot_id_col)
    install_model_speed_patches()
    install_amp_patches(args.mixed_precision)
    install_dataloader_patches(
        pin_memory=not args.disable_pin_memory and "cuda" in str(args.device).lower(),
        persistent_workers=not args.disable_persistent_workers,
        prefetch_factor=args.prefetch_factor,
        smart_batching=False,
        bucket_multiplier=50,
    )

    pred_label_df, metrics, _task_names = evaluate_predictions_csv(
        input_csv=input_csv,
        protein_records_path=protein_records_path,
        checkpoint_dir=args.checkpoint_dir,
        smiles_columns=args.smiles_columns,
        target_columns=args.target_columns,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        cache_dir=args.cache_dir,
        loaded_bundle=None,
    )
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_label_df.to_csv(out_csv, index=False)
    print(f"Saved predictions to: {out_csv}")

    if args.metrics_csv:
        metrics_path = Path(args.metrics_csv)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
