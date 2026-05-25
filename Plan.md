# CatPred EMULaToR Bench Plan

Implemented under `emulator_bench/`.

- Align one structure per split row and update original split parquet/CSV files after backup.
- Cache one ESM2 embedding per unique sequence under the CatPred baseline `embeddings/` tree.
- Run proGRES once over deduplicated structure paths and convert `searchdb.pt` to CatPred's EGNN feature dict.
- Precompute CPU-heavy CatPred dataset, RDKit MolGraph, and optional BatchMolGraph artifacts with `ProcessPoolExecutor`.
- Train in strict cache mode so GPU training does not silently do missing CPU featurization.
- Resume training by split run and by ensemble member epoch state.
- Keep Optuna separate from default retraining and limited to retraining-safe optimization settings.
