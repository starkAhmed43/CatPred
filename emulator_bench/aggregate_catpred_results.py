"""
Aggregate CatPred EMULaToR results across kcat, km, ki, splits, and seeds.

The split parquet files are treated as the source of truth, so missing result
directories or partially written seed runs are reported instead of silently
disappearing from the benchmark table.

Default input:
  ~/github/EMULaToR/data/processed/baselines/CatPred

Default outputs:
  ~/github/EMULaToR/data/processed/baselines/CatPred/aggregate_catpred_results.csv
  ~/github/EMULaToR/data/processed/baselines/CatPred/catpred_seed_results.csv
  ~/github/EMULaToR/data/processed/baselines/CatPred/catpred_missing_results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import DEFAULT_SPLIT_GROUPS, DEFAULT_VALUE_TYPES, default_base_dir  # noqa: E402


TVT_SPLITS = ("train", "val", "test")
DEFAULT_RUN_NAMES = ("paper_defaults_single_model",)
METRIC_ALIASES = {
    "rmse": ("RMSE", "rmse"),
    "mae": ("MAE", "mae"),
    "mse": ("MSE", "mse"),
    "r2": ("R2", "r2", "r2_score", "R2_score"),
    "pcc": ("PCC", "pcc", "pearson"),
    "scc": ("SCC", "scc", "spearman"),
    "loss": ("Loss", "loss"),
    "p1mag": ("p1mag", "P1MAG", "P1mag"),
}
METRICS = tuple(METRIC_ALIASES)


def _split_sort_key(item: str) -> tuple[int, str]:
    try:
        return (list(DEFAULT_SPLIT_GROUPS).index(item), item)
    except ValueError:
        return (len(DEFAULT_SPLIT_GROUPS), item)


def _seed_sort_key(seed: str) -> tuple[int, str]:
    if seed.startswith("seed_"):
        suffix = seed.removeprefix("seed_")
        if suffix.isdigit():
            return (int(suffix), seed)
    return (10**12, seed)


def _seed_int(seed: str) -> int | None:
    suffix = seed.removeprefix("seed_")
    return int(suffix) if suffix.isdigit() else None


def discover_split_dirs(
    base_dir: Path,
    value_types: Iterable[str],
    split_groups: Iterable[str],
) -> list[dict]:
    selected_values = set(value_types)
    selected_groups = set(split_groups)
    rows = []

    for value_type in sorted(selected_values):
        value_root = base_dir / value_type
        if not value_root.exists():
            print(f"[missing value_type] {value_root}")
            continue

        for train_path in sorted(value_root.rglob("train.parquet")):
            split_dir = train_path.parent
            if not all((split_dir / f"{name}.parquet").exists() for name in TVT_SPLITS):
                continue

            rel_parts = split_dir.relative_to(value_root).parts
            if not rel_parts:
                continue
            split_group = rel_parts[0]
            if split_group not in selected_groups:
                continue

            split_name = rel_parts[-1]
            threshold = split_name if split_name.startswith("threshold_") else None
            rows.append(
                {
                    "value_type": value_type,
                    "split_group": split_group,
                    "threshold": threshold,
                    "split_name": split_name,
                    "split_dir": split_dir,
                }
            )

    return sorted(rows, key=lambda row: (row["value_type"], _split_sort_key(row["split_group"]), row["split_name"]))


def discover_run_names(split_rows: list[dict]) -> list[str]:
    names = set()
    for row in split_rows:
        results_dir = row["split_dir"] / "catpred_results"
        if not results_dir.exists():
            continue
        names.update(path.name for path in results_dir.iterdir() if path.is_dir())
    return sorted(names)


def discover_expected_seeds(split_rows: list[dict], run_names: Iterable[str]) -> dict[str, list[str]]:
    expected: dict[str, set[str]] = {run_name: set() for run_name in run_names}
    for row in split_rows:
        for run_name in run_names:
            run_dir = row["split_dir"] / "catpred_results" / run_name
            if not run_dir.exists():
                continue
            expected[run_name].update(path.name for path in run_dir.glob("seed_*") if path.is_dir())
    return {run_name: sorted(seeds, key=_seed_sort_key) for run_name, seeds in expected.items()}


def _first_existing(series: pd.Series, aliases: tuple[str, ...]) -> float | None:
    for alias in aliases:
        if alias in series.index and pd.notna(series[alias]):
            return float(series[alias])
    return None


def _prediction_columns(df: pd.DataFrame) -> tuple[str, str] | None:
    pred_cols = [col for col in df.columns if col.endswith("_pred") or col == "pred"]
    label_cols = [col for col in df.columns if col.endswith("_label") or col == "label"]
    if pred_cols and label_cols:
        return pred_cols[0], label_cols[0]
    return None


def _metrics_from_predictions(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    cols = _prediction_columns(df)
    if cols is None:
        return {}

    pred_col, label_col = cols
    pred = pd.to_numeric(df[pred_col], errors="coerce")
    label = pd.to_numeric(df[label_col], errors="coerce")
    valid = pred.notna() & label.notna()
    pred = pred[valid]
    label = label[valid]
    if len(pred) == 0:
        return {}

    err = pred - label
    mse = float(np.mean(np.square(err)))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    denom = float(np.sum(np.square(label - np.mean(label))))
    r2 = float(1.0 - np.sum(np.square(err)) / denom) if denom > 0 else np.nan
    pcc = float(pred.corr(label, method="pearson")) if len(pred) > 1 else np.nan
    scc = float(pred.corr(label, method="spearman")) if len(pred) > 1 else np.nan
    p1mag = float(np.mean(np.abs(err) <= 1.0))
    return {"rmse": rmse, "mae": mae, "mse": mse, "r2": r2, "pcc": pcc, "scc": scc, "p1mag": p1mag}


def load_split_metrics(seed_dir: Path, tvt_split: str) -> tuple[dict[str, float], list[str]]:
    metrics: dict[str, float] = {}
    missing = []

    final_path = seed_dir / f"final_results_{tvt_split}.csv"
    if final_path.exists():
        df = pd.read_csv(final_path)
        if not df.empty:
            series = df.iloc[0]
            for metric, aliases in METRIC_ALIASES.items():
                value = _first_existing(series, aliases)
                if value is not None:
                    metrics[metric] = value
        else:
            missing.append(f"empty:{final_path.name}")
    else:
        missing.append(final_path.name)

    pred_metrics = _metrics_from_predictions(seed_dir / f"pred_label_{tvt_split}.csv")
    for metric, value in pred_metrics.items():
        metrics.setdefault(metric, value)

    if not metrics:
        missing.append(f"no_metrics:{tvt_split}")
    return metrics, missing


def build_rows(
    split_rows: list[dict],
    run_names: list[str],
    expected_seeds: dict[str, list[str]],
    allow_partial_seeds: bool,
) -> tuple[list[dict], list[dict]]:
    result_rows = []
    missing_rows = []

    for split_row in split_rows:
        split_dir = split_row["split_dir"]
        for run_name in run_names:
            run_dir = split_dir / "catpred_results" / run_name
            seeds = expected_seeds.get(run_name, [])
            if not seeds and run_dir.exists():
                seeds = sorted((path.name for path in run_dir.glob("seed_*") if path.is_dir()), key=_seed_sort_key)

            if not seeds:
                missing_rows.append(
                    {
                        **{key: value for key, value in split_row.items() if key != "split_dir"},
                        "run_name": run_name,
                        "seed": None,
                        "seed_int": None,
                        "tvt_split": None,
                        "status": "missing_seed_dirs",
                        "missing_files": "seed_*",
                        "seed_dir": str(run_dir),
                    }
                )
                continue

            for seed in seeds:
                seed_dir = run_dir / seed
                if not seed_dir.exists():
                    missing_rows.append(
                        {
                            **{key: value for key, value in split_row.items() if key != "split_dir"},
                            "run_name": run_name,
                            "seed": seed,
                            "seed_int": _seed_int(seed),
                            "tvt_split": None,
                            "status": "missing_seed_dir",
                            "missing_files": "seed_dir",
                            "seed_dir": str(seed_dir),
                        }
                    )
                    continue

                seed_split_metrics = {}
                seed_missing = {}
                for tvt_split in TVT_SPLITS:
                    metrics, missing = load_split_metrics(seed_dir, tvt_split)
                    if metrics:
                        seed_split_metrics[tvt_split] = metrics
                    if missing:
                        seed_missing[tvt_split] = missing

                is_complete = set(seed_split_metrics) == set(TVT_SPLITS) and not seed_missing
                if seed_missing:
                    for tvt_split, missing in seed_missing.items():
                        missing_rows.append(
                            {
                                **{key: value for key, value in split_row.items() if key != "split_dir"},
                                "run_name": run_name,
                                "seed": seed,
                                "seed_int": _seed_int(seed),
                                "tvt_split": tvt_split,
                                "status": "incomplete_seed",
                                "missing_files": ";".join(missing),
                                "seed_dir": str(seed_dir),
                            }
                        )

                if not is_complete and not allow_partial_seeds:
                    continue

                for tvt_split, metrics in seed_split_metrics.items():
                    row = {
                        **{key: value for key, value in split_row.items() if key != "split_dir"},
                        "run_name": run_name,
                        "seed": seed,
                        "seed_int": _seed_int(seed),
                        "tvt_split": tvt_split,
                        "seed_dir": str(seed_dir),
                    }
                    row.update(metrics)
                    result_rows.append(row)

    return result_rows, missing_rows


def aggregate(result_rows: list[dict]) -> pd.DataFrame:
    if not result_rows:
        return pd.DataFrame()

    df = pd.DataFrame(result_rows)
    group_keys = ["value_type", "run_name", "split_group", "threshold", "split_name", "tvt_split"]
    present_metrics = [metric for metric in METRICS if metric in df.columns]
    agg = df.groupby(group_keys, dropna=False)[present_metrics].agg(["mean", "var"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    grouped = df.groupby(group_keys, dropna=False)
    agg["n_seeds"] = grouped["seed"].nunique()
    agg["seeds"] = grouped["seed"].apply(lambda values: " ".join(sorted(set(map(str, values)), key=_seed_sort_key)))
    return agg.reset_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", type=Path, default=default_base_dir())
    parser.add_argument("--value_types", nargs="+", default=list(DEFAULT_VALUE_TYPES))
    parser.add_argument("--split_groups", nargs="+", default=list(DEFAULT_SPLIT_GROUPS))
    parser.add_argument("--run_names", nargs="+", default=list(DEFAULT_RUN_NAMES))
    parser.add_argument("--all_run_names", action="store_true", help="Aggregate every run name found under catpred_results.")
    parser.add_argument("--seeds", nargs="+", default=None, help="Expected seed names or integers, e.g. 666 777 888.")
    parser.add_argument("--allow_partial_seeds", action="store_true", help="Include available TVT files even when a seed is missing another TVT file.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed_output", type=Path, default=None)
    parser.add_argument("--missing_output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.expanduser().resolve()
    output = (args.output or base_dir / "aggregate_catpred_results.csv").expanduser().resolve()
    seed_output = (args.seed_output or base_dir / "catpred_seed_results.csv").expanduser().resolve()
    missing_output = (args.missing_output or base_dir / "catpred_missing_results.csv").expanduser().resolve()

    if not base_dir.exists():
        raise FileNotFoundError(f"CatPred base directory does not exist: {base_dir}")

    split_rows = discover_split_dirs(base_dir, args.value_types, args.split_groups)
    if not split_rows:
        raise FileNotFoundError(f"No CatPred split directories found under {base_dir}")

    run_names = discover_run_names(split_rows) if args.all_run_names else list(args.run_names)
    if not run_names:
        print("No CatPred run names found.")
        return

    if args.seeds:
        expected_seeds = {
            run_name: sorted(
                {seed if str(seed).startswith("seed_") else f"seed_{seed}" for seed in args.seeds},
                key=_seed_sort_key,
            )
            for run_name in run_names
        }
    else:
        expected_seeds = discover_expected_seeds(split_rows, run_names)

    result_rows, missing_rows = build_rows(
        split_rows=split_rows,
        run_names=run_names,
        expected_seeds=expected_seeds,
        allow_partial_seeds=args.allow_partial_seeds,
    )

    summary = aggregate(result_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    seed_output.parent.mkdir(parents=True, exist_ok=True)
    missing_output.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(output, index=False)
    pd.DataFrame(result_rows).to_csv(seed_output, index=False)
    pd.DataFrame(missing_rows).to_csv(missing_output, index=False)

    print(
        f"Saved {len(summary)} aggregate rows from {len(result_rows)} seed/TVT rows "
        f"across {len(split_rows)} split dirs to {output}"
    )
    print(f"Saved raw seed rows to {seed_output}")
    print(f"Saved {len(missing_rows)} missing/incomplete rows to {missing_output}")


if __name__ == "__main__":
    main()
