# CatPred EMULaToR Bench

This directory adds wrapper code for retraining CatPred on EMULaToR train/val/test split parquets under:

`/home/adhil/github/EMULaToR/data/processed/baselines/CatPred`

The CatPred model architecture is not changed. The bench patches data loading, mixed precision, optimizer setup, and checkpoint control at wrapper startup so CPU-heavy work is done before GPU training.

## Inputs

Each split table must provide:

- `smiles`: substrate SMILES for CatPred/RDKit D-MPNN features.
- `sequence`: enzyme sequence.
- `log10_value`: regression target.
- aligned structure columns produced by `align_structures.py`, especially `structure_path` and `catpred_structure_id`.

The data tree is discovered as:

- `kcat`, `km`, `ki`
- direct split roots such as `random_splits_grouped_sequence`, `random_splits_grouped_smiles`, `uniprot_time_splits`
- thresholded roots such as `enzyme_sequence_splits/threshold_*`, `substrate_splits/threshold_*`, `enzyme_structure_splits/threshold_*`, `conformer_cosine_splits/threshold_*`

## Structure Alignment

`align_structures.py` updates split parquet/CSV files in place and first backs up originals under:

`/home/adhil/github/EMULaToR/data/processed/baselines/CatPred/_aligned_pdb_backups`

It selects one structure per row:

1. best sequence-aligned experimental PDBe PDB from `/home/adhil/github/EMULaToR/data/intermediate/processed_exp_pdb`
2. AlphaFold fallback from `/home/adhil/github/EMULaToR/data/intermediate/alphafold`
3. ESM fallback from `/home/adhil/github/EMULaToR/data/intermediate/esm`

Added columns include `pdbs`, `pdb_source`, `pdb_type`, `structure_path`, `chain_id`, `catpred_structure_id`, and audit columns.

## Embedding Caches

Reusable caches live under:

- `.../CatPred/embeddings/esm2`: ESM2 tensors keyed by normalized sequence hash.
- `.../CatPred/embeddings/fair_esm`: isolated Meta fair-esm install used only by CatPred ESM2.
- `.../CatPred/embeddings/progres/filepaths.txt`: deduplicated proGRES structure list.
- `.../CatPred/embeddings/progres/searchdb.pt`: raw proGRES output.
- `.../CatPred/embeddings/progres/progres_egnn_by_structure_id.pt`: CatPred EGNN feature dict keyed by `catpred_structure_id`.

proGRES is run once with:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 progres embed \
  -l .../CatPred/embeddings/progres/filepaths.txt \
  -o .../CatPred/embeddings/progres/searchdb.pt \
  -f pdb \
  -d cuda:0
```

ESM2 embeddings are computed once per unique sequence and reused across `kcat`, `km`, `ki`, splits, seeds, and ensembles.

The `esm` import name conflicts between ESM3 and Meta fair-esm. CatPred uses an isolated fair-esm vendor path for ESM2, so the global `esm` package can remain ESM3:

```bash
python -m pip install fair-esm \
  -t /home/adhil/github/EMULaToR/data/processed/baselines/CatPred/embeddings/fair_esm
```

Override this location with `CATPRED_FAIR_ESM_PATH` if needed.

## CPU Precompute

`precompute_features.py` uses `ProcessPoolExecutor` across split/seed jobs. It materializes parquet to stable CSV views, builds inline CatPred dataset caches, warms RDKit `MolGraph` caches, and optionally builds per-seed `BatchMolGraph` caches. During batch-cache precompute, worker-local batch and RDKit globals are cleared after each job; `--max_tasks_per_child 1` is the safest setting when RAM pressure is high.

Training defaults to strict precompute mode. If a required dataset, ESM, MolGraph, or BatchMolGraph cache is missing, `train_single_target_tvt.py` fails before GPU training instead of doing CPU featurization on the training path.

## Training Defaults

Default paper retraining settings are in `original_catpred_retrain_hparams.json`:

- `batch_size=32`
- `seq_embed_dim=36`
- `seq_self_attn_nheads=6`
- `ensemble_size=10`
- `loss_function=mve`
- `max_lr=0.001`
- `epochs=30`

`launch_parallel_retrain.py` runs alignment, embedding cache, CPU precompute, and training across all discovered `kcat`, `km`, and `ki` splits. It uses physical GPU `1` by default and maps each worker process to `CUDA_VISIBLE_DEVICES=<gpu>` with in-process `--device cuda:0`.

The default JSON records the paper ensemble setting (`ensemble_size=10`). For one independent model per seed, pass `--ensemble_size 1`; launcher CLI hyperparameter flags override values from the JSON.

The launcher defaults to `--cpu_threads 2 --interop_threads 1` for each training subprocess and exports the matching OpenMP/BLAS environment variables before Python starts. This prevents six parallel GPU runs from each spawning very large CPU thread pools. With batch-graph caches enabled, training collation forces `num_workers=0`; keep `--num_workers 0` unless you intentionally want worker processes during post-fit evaluation.

Resumability:

- alignment, ESM, proGRES, dataset, MolGraph, and BatchMolGraph caches skip completed artifacts
- completed split runs are skipped by `final_results_test.csv`
- each ensemble member writes `training_state.pt` after epochs and `training_complete.json` when done
- restart resumes unfinished ensemble members from the last completed epoch
- Ctrl+C or SIGTERM cancels pending work and terminates active precompute workers or training subprocess groups; partial manifests are written where possible
- `kill_bench.py` lists matching bench processes by default and terminates them only with `--yes`

## Optuna

Optuna remains separate from default retraining. It tunes retraining-safe optimization settings such as learning rate schedule values, dropout, warmup, and optionally batch size. It does not change model architecture defaults unless explicitly requested.
