# emulator_bench

This folder ports the `DeepDTAGen_regression/emulator_bench` style workflow into CatPred without editing CatPred core files in place.

What this adds:
- Direct train/val/test flow from `.csv` or `.parquet` using `smiles`, `sequence`, and `uniprot_id`
- Bench-local inline protein loader that bypasses CatPred's JSON sidecar requirement
- Persistent ESM disk cache under `emulator_bench/.cache_embeddings/esm2`
- Optional ESM cache warmup to reduce repeated preprocessing cost across thresholds and seeds
- TVT training wrapper that reuses CatPred's native training path
- Split benchmark runner and Optuna tuner

What stays intact:
- CatPred source files under `catpred/*` are left unedited
- CatPred featurization, losses, checkpoint format, and train loop remain the base implementation
- Bench-only runtime patches are applied at wrapper startup for CSV protein loading and a few speed-focused execution fixes

## Important integration note

Original CatPred expects:
- a CSV with a `pdbpath` column
- a `protein_records_path` gzip JSON keyed by `basename(pdbpath)`

The bench does not. Your benchmark CSVs should provide:
- `smiles`
- `sequence`
- `uniprot_id`

The bench patches CatPred's loader to read sequence directly from the CSV and uses `uniprot_id` as the protein key for EGNN embedding lookup.

This is additive only. The original split CSVs are not modified.

## 1) Stage TVT data and warm cache

`--train_csv`, `--val_csv`, and `--test_csv` accept both `.csv` and `.parquet` paths.

```bash
python emulator_bench/build_tvt_data.py \
  --train_csv /path/to/train.csv \
  --val_csv /path/to/val.csv \
  --test_csv /path/to/test.csv \
  --output_root /path/to/workdir \
  --dataset_name myset \
  --sequence_col sequence \
  --uniprot_id_col uniprot_id \
  --warm_esm_cache \
  --cache_dir emulator_bench/.cache_embeddings
```

Outputs:
- `myset_manifest.json`

## 2) Train with explicit TVT

`--train_csv`, `--val_csv`, and `--test_csv` accept both `.csv` and `.parquet` paths.

```bash
python emulator_bench/train_single_target_tvt.py \
  --train_csv /path/to/train.csv \
  --val_csv /path/to/val.csv \
  --test_csv /path/to/test.csv \
  --sequence_col sequence \
  --uniprot_id_col uniprot_id \
  --out_dir /path/to/workdir/results \
  --smiles_columns smiles \
  --target_columns log10_value \
  --device cuda:0 \
  --grad_accum_steps 2 \
  --smart_batching \
  --optimizer_fused auto \
  --add_esm_feats --loss_function mve --seq_embed_dim 36 --seq_self_attn_nheads 6
```

Anything after the wrapper args is passed through to CatPred unchanged.

Speed-oriented defaults in the bench:
- inline CSV protein loading instead of sidecar JSON generation
- in-memory reuse of duplicate protein entries within a split load
- sequence tokenization cached once per unique protein instead of per batch
- pretrained EGNN `.pt` loaded once per process and reused across model reloads
- ESM embeddings are only computed when `--add_esm_feats` is actually enabled and persisted on disk under `emulator_bench/.cache_embeddings/esm2`
- graph cache forced on by default via `--cache_cutoff inf`
- mixed precision follows the DeepDTAGen/ProSmith policy via `--mixed_precision auto`:
  BF16 on Ampere/Hopper-class GPUs, otherwise FP16, with grad scaling only for FP16
- learning rate scheduler defaults to cosine annealing with warmup (no restarts) via `--lr_scheduler cosine_warmup`
- fused Adam is available via `--optimizer_fused {auto,on,off}` and defaults to `auto`
- gradient accumulation is available via `--grad_accum_steps`
- pinned-memory / persistent-worker / prefetch tuning are exposed in the wrapper
- optional length-bucketed smart batching is available via `--smart_batching`
- TF32 and cuDNN autotuning enabled by default on CUDA; pass `--disable_tf32` if you want stricter float32 behavior
- optional ESM cache warmup across train/val/test before repeated seed runs
- ESM cache policy flags:
  - `--require_cached_esm` fails fast if a needed embedding is missing (ensures no on-the-fly ESM compute)
  - `--overwrite_esm_cache` recomputes and overwrites ESM cache entries
- validation cadence and metric cost controls:
  - `--val_every_n_epochs N`
  - `--final_epoch_metrics_only` (validation checkpoints compute loss only; full metrics computed on final epoch)
  - `--early_stopping_patience` and `--early_stopping_min_delta`
- redundant native `test_preds.csv` writes disabled during training

Outputs in `out_dir`:
- `fold_0/...` native CatPred training outputs
- `bestmodel.pth` for single-model runs
- `bestmodel_dir.json`
- `pred_label_val.csv`
- `pred_label_test.csv`
- `results_val.csv`
- `results_test.csv`
- `final_results_val.csv`
- `final_results_test.csv`
- `run_summary.csv`

## 3) Predict a trained split

```bash
python emulator_bench/predict_single_target.py \
  --input_csv /path/to/test.csv \
  --sequence_col sequence \
  --uniprot_id_col uniprot_id \
  --checkpoint_dir /path/to/workdir/results/fold_0 \
  --out_csv /path/to/workdir/predictions.csv \
  --metrics_csv /path/to/workdir/predictions_metrics.csv \
  --smiles_columns smiles \
  --target_columns log10_value \
  --device cuda:0 \
  --num_workers 4
```

## 4) Run threshold benchmarks

Expected layout (both supported):
- `<base_dir>/<value_type>/<split_group>/threshold_x/train.{csv,parquet}`
- `<base_dir>/<value_type>/<split_group>/threshold_x/val.{csv,parquet}`
- `<base_dir>/<value_type>/<split_group>/threshold_x/test.{csv,parquet}`
- `<base_dir>/<split_group>/threshold_x/train.{csv,parquet}` (value type omitted)
- `<base_dir>/<split_group>/train.{csv,parquet}` (direct split roots, e.g. `random_splits`)

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python emulator_bench/run_split_benchmarks.py \
  --base_dir /home/ubuntu/adhil/EMULaToR/data/processed/baselines/catpred \
  --sequence_col sequence \
  --uniprot_id_col uniprot_id \
  --smiles_columns smiles \
  --target_columns log10_value \
  --device cuda:0 \
  --grad_accum_steps 2 \
  --smart_batching \
  --warm_esm_cache \
  --cache_dir emulator_bench/.cache_embeddings \
  --add_esm_feats --loss_function mve --seq_embed_dim 36 --seq_self_attn_nheads 6
```

Per-threshold artifacts:
- `catpred_data/*`
- `catpred_results/seed_<seed>/*`

Aggregate outputs:
- `catpred_summary_runs.csv`
- `catpred_summary_thresholds.csv`
- `catpred_summary_by_split_group.csv`
- `catpred_summary_ranked.csv`
- `catpred_summary.csv`

## 5) Tune generic training hyperparameters

```bash
CUDA_VISIBLE_DEVICES=0 python emulator_bench/tune_optuna.py \
  --base_dir /home/ubuntu/adhil/EMULaToR/data/processed/baselines/catpred \
  --sequence_col sequence \
  --uniprot_id_col uniprot_id \
  --metric MSE \
  --eval_split val \
  --n_trials 20 \
  --device cuda:0 \
  --grad_accum_steps 2 \
  --smart_batching \
  --warm_esm_cache \
  --cache_dir emulator_bench/.cache_embeddings \
  --add_esm_feats --loss_function mve --seq_embed_dim 36 --seq_self_attn_nheads 6
```

Artifacts:
- `<base_dir>/<value_type>/optuna_studies/*_best_hparams.json`
- `<base_dir>/<value_type>/optuna_studies/*_trials.csv`

## Notebook note

I checked the shipped notebooks. They contain demo and packaging flow, not hidden CatPred training architecture or loss logic, so no model-specific notebook logic needed to be copied into this folder.
