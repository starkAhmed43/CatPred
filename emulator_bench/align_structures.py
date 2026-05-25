import argparse
import ast
import concurrent.futures
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1
from tqdm.auto import tqdm

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    DEFAULT_SPLIT_GROUPS,
    DEFAULT_VALUE_TYPES,
    KEY_COLUMNS,
    STRUCTURE_COLUMNS,
    default_base_dir,
    discover_split_jobs,
    load_json,
    read_table,
    stable_hash,
    tabular_columns,
    write_json,
    write_table_atomic,
)


DEFAULT_EXPERIMENTAL_PDB_DIR = Path("/home/adhil/github/EMULaToR/data/intermediate/processed_exp_pdb")
DEFAULT_ALPHAFOLD_PDB_DIR = Path("/home/adhil/github/EMULaToR/data/intermediate/alphafold")
DEFAULT_ESM_PDB_DIR = Path("/home/adhil/github/EMULaToR/data/intermediate/esm")
PDB_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9])([0-9][A-Za-z0-9]{3})(?![A-Za-z0-9])")


def _norm_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "nat"} else text


def _norm_seq(value) -> str:
    return _norm_text(value).upper().replace("*", "")


def _row_key(row: dict) -> str:
    payload = {column: _norm_text(row.get(column, "")) for column in KEY_COLUMNS if column in row}
    return stable_hash(payload, length=24)


def _sequence_key(row: dict) -> str:
    return "sequence:" + stable_hash(_norm_seq(row.get("sequence", "")), length=24)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")


def _load_json_default(path: Path, default):
    if not path.exists():
        return default
    return load_json(path)


def _extract_candidates(value) -> List[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        text = _norm_text(value)
        if not text:
            return []
        if text[0] in "[(" and text[-1] in "])":
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = None
            raw_items = list(parsed) if isinstance(parsed, (list, tuple, set)) else re.split(r"[,\s;]+", text)
        else:
            raw_items = re.split(r"[,\s;]+", text)

    out, seen = [], set()
    for item in raw_items:
        token = Path(_norm_text(item)).stem
        if not token:
            continue
        if "|" in token:
            if token not in seen:
                seen.add(token)
                out.append(token)
            continue
        upper = token.upper()
        if re.fullmatch(r"[0-9A-Z]{4}", upper):
            if upper not in seen:
                seen.add(upper)
                out.append(upper)
            continue
        for match in PDB_ID_PATTERN.findall(upper):
            if match not in seen:
                seen.add(match)
                out.append(match)
    return out


def _base_accession(value: str) -> str:
    text = _norm_text(value)
    if "|" in text:
        text = text.split("|", 1)[0]
    return text.upper()


def _build_experimental_index(directory: Path) -> Dict[str, str]:
    directory = Path(directory).expanduser()
    if not directory.exists():
        return {}
    return {path.stem.upper(): str(path.resolve()) for path in directory.glob("*.pdb")}


def _build_alphafold_index(directory: Path) -> Dict[str, str]:
    directory = Path(directory).expanduser()
    if not directory.exists():
        return {}
    versioned: Dict[str, tuple[int, str]] = {}
    for path in directory.glob("AF-*-F1-model_v*.pdb"):
        stem = path.stem
        accession, _, version_text = stem[len("AF-") :].partition("-F1-model_v")
        try:
            version = int(version_text)
        except ValueError:
            version = -1
        current = versioned.get(accession.upper())
        if current is None or version > current[0]:
            versioned[accession.upper()] = (version, str(path.resolve()))
    return {key: path for key, (_version, path) in versioned.items()}


def _build_esm_index(directory: Path) -> Dict[str, str]:
    directory = Path(directory).expanduser()
    if not directory.exists():
        return {}
    out = {}
    prefix = "ESM3-open-small-"
    for path in directory.glob(f"{prefix}*.pdb"):
        key = path.name[len(prefix) : -len(".pdb")]
        out[key] = str(path.resolve())
        out[key.upper()] = str(path.resolve())
    return out


def _extract_pdb_sequences(path: str) -> List[dict]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(Path(path).stem, path)
    entries = []
    for model in structure:
        for chain in model:
            residues = []
            for residue in chain:
                if is_aa(residue, standard=False):
                    residues.append(seq1(residue.get_resname(), custom_map={"MSE": "M"}, undef_code="X"))
            if residues:
                entries.append({"chain_id": str(chain.id), "sequence": "".join(residues)})
        break
    return entries


def _parse_pdb_worker(item: tuple[str, str]):
    pdb_id, path = item
    try:
        return pdb_id, _extract_pdb_sequences(path), None
    except Exception as exc:
        return pdb_id, [], str(exc)


def _make_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner(mode="global")
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = 0.0
    aligner.extend_gap_score = 0.0
    return aligner


def _identity_pct(aligner: PairwiseAligner, query: str, subject: str) -> float:
    if not query or not subject:
        return 0.0
    return 100.0 * float(aligner.score(query, subject)) / max(len(query), len(subject))


def _candidate_lookup(value_root: Path, needed_keys: set[str], needed_sequence_keys: set[str]) -> dict:
    path = value_root / f"{value_root.name}_kinetic_params_3d.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing 3D candidate parquet: {path}")

    columns = list(dict.fromkeys(list(KEY_COLUMNS) + ["pdbs", "pdb_source", "pdb_type"]))
    available_columns = set(tabular_columns(path))
    frame = read_table(path, columns=[column for column in columns if column in available_columns])
    _require_columns(frame, ["sequence", "pdbs", "pdb_source", "pdb_type"], path)

    lookup: dict[str, list[dict]] = {}
    frame_columns = list(frame.columns)
    row_iter = (dict(zip(frame_columns, values)) for values in frame.itertuples(index=False, name=None))
    for row in tqdm(row_iter, total=len(frame), desc=f"Index {path.name}", leave=False):
        key = _row_key(row)
        seq_key = _sequence_key(row)
        if key in needed_keys:
            lookup.setdefault(key, []).append(row)
        if seq_key in needed_sequence_keys:
            seq_bucket = lookup.setdefault(seq_key, [])
            if len(seq_bucket) < 25:
                seq_bucket.append(row)
    return lookup


def _collect_needed_keys(jobs: list[dict], limit_rows: int | None) -> dict[str, dict[str, set[str]]]:
    needed: dict[str, dict[str, set[str]]] = {}
    for job in jobs:
        bucket = needed.setdefault(job["value_type"], {"keys": set(), "sequence_keys": set()})
        for split_key in ("train_path", "val_path", "test_path"):
            frame = read_table(job[split_key])
            rows = frame.to_dict("records")
            if limit_rows:
                rows = rows[:limit_rows]
            for row in rows:
                bucket["keys"].add(_row_key(row))
                bucket["sequence_keys"].add(_sequence_key(row))
    return needed


def _candidate_rows(row: dict, lookup: dict) -> list[dict]:
    exact = lookup.get(_row_key(row))
    if exact:
        return exact
    return lookup.get(_sequence_key(row), [])


def _needed_experimental_pdbs(rows: list[dict], lookup: dict, experimental_index: dict[str, str]) -> set[str]:
    needed = set()
    for row in rows:
        for candidate in _candidate_rows(row, lookup):
            source = _norm_text(candidate.get("pdb_source")).lower()
            kind = _norm_text(candidate.get("pdb_type")).lower()
            if source == "pdbe" or kind == "experimental":
                for token in _extract_candidates(candidate.get("pdbs")):
                    pdb_id = token.upper()
                    if pdb_id in experimental_index:
                        needed.add(pdb_id)
    return needed


def _prefill_sequence_cache(
    pdb_ids: set[str],
    experimental_index: dict[str, str],
    cache_path: Path,
    workers: int,
) -> dict:
    sequence_cache = _load_json_default(cache_path, {})
    missing = [(pdb_id, experimental_index[pdb_id]) for pdb_id in sorted(pdb_ids) if pdb_id not in sequence_cache]
    if missing:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
            iterator = executor.map(_parse_pdb_worker, missing, chunksize=max(1, len(missing) // max(1, workers * 8)))
            for pdb_id, entries, error in tqdm(iterator, total=len(missing), desc="Parse PDB chains", unit="pdb"):
                sequence_cache[pdb_id] = entries
                if error:
                    sequence_cache[f"{pdb_id}__error"] = error
        write_json(cache_path, sequence_cache)
    return sequence_cache


def _resolve_structure(
    row: dict,
    candidates: list[dict],
    experimental_index: dict[str, str],
    alphafold_index: dict[str, str],
    esm_index: dict[str, str],
    sequence_cache: dict,
    aligner: PairwiseAligner,
    identity_threshold: float,
) -> dict:
    sequence = _norm_seq(row.get("sequence", ""))
    best = {"pdbs": "", "chain_id": "", "identity": -1.0, "path": ""}
    alpha_accessions, esm_keys = [], []

    for candidate in candidates:
        pdbs = _norm_text(candidate.get("pdbs"))
        source = _norm_text(candidate.get("pdb_source")).lower()
        kind = _norm_text(candidate.get("pdb_type")).lower()
        if source == "pdbe" or kind == "experimental":
            for token in _extract_candidates(pdbs):
                pdb_id = token.upper()
                if pdb_id not in experimental_index:
                    continue
                for entry in sequence_cache.get(pdb_id, []):
                    identity = _identity_pct(aligner, sequence, _norm_seq(entry.get("sequence", "")))
                    if identity > best["identity"]:
                        best = {
                            "pdbs": pdb_id,
                            "chain_id": str(entry.get("chain_id", "")) or "A",
                            "identity": identity,
                            "path": experimental_index[pdb_id],
                        }
        elif source == "alphafold" or kind == "predicted":
            accession = _base_accession(pdbs)
            if accession in alphafold_index:
                alpha_accessions.append(accession)
        else:
            for key in (pdbs, _base_accession(pdbs), pdbs.upper()):
                if key in esm_index:
                    esm_keys.append(key)

    if best["path"] and best["identity"] >= identity_threshold:
        structure_id = f"PDBe:{best['pdbs']}:{best['chain_id']}"
        return {
            "pdbs": best["pdbs"],
            "pdb_source": "PDBe",
            "pdb_type": "experimental",
            "structure_path": best["path"],
            "chain_id": best["chain_id"],
            "catpred_structure_id": structure_id,
            "resolved_structure_status": "selected_experimental",
            "resolved_structure_identity": round(float(best["identity"]), 4),
            "resolved_structure_chain_id": best["chain_id"],
            "resolved_structure_reason": f"best_pdbe_identity>={identity_threshold:g}",
        }

    if alpha_accessions:
        accession = sorted(set(alpha_accessions))[0]
        return {
            "pdbs": accession,
            "pdb_source": "AlphaFold",
            "pdb_type": "predicted",
            "structure_path": alphafold_index[accession],
            "chain_id": "A",
            "catpred_structure_id": f"AlphaFold:{accession}:A",
            "resolved_structure_status": "selected_alphafold",
            "resolved_structure_identity": pd.NA if best["identity"] < 0 else round(float(best["identity"]), 4),
            "resolved_structure_chain_id": "A",
            "resolved_structure_reason": "no_qualifying_experimental_match",
        }

    if esm_keys:
        key = sorted(set(esm_keys))[0]
        return {
            "pdbs": key,
            "pdb_source": "ESM",
            "pdb_type": "predicted",
            "structure_path": esm_index[key],
            "chain_id": "A",
            "catpred_structure_id": f"ESM:{key}:A",
            "resolved_structure_status": "selected_esm",
            "resolved_structure_identity": pd.NA if best["identity"] < 0 else round(float(best["identity"]), 4),
            "resolved_structure_chain_id": "A",
            "resolved_structure_reason": "no_qualifying_experimental_or_alphafold_match",
        }

    return {
        "pdbs": pd.NA,
        "pdb_source": pd.NA,
        "pdb_type": pd.NA,
        "structure_path": pd.NA,
        "chain_id": pd.NA,
        "catpred_structure_id": pd.NA,
        "resolved_structure_status": "missing",
        "resolved_structure_identity": pd.NA if best["identity"] < 0 else round(float(best["identity"]), 4),
        "resolved_structure_chain_id": pd.NA,
        "resolved_structure_reason": "no_structure_candidate_found",
    }


def _backup_path(path: Path, backup_root: Path) -> Path:
    resolved = path.resolve()
    rel = Path(*resolved.parts[1:]) if resolved.is_absolute() else resolved
    return backup_root / rel


def _update_split_file(path: Path, frame: pd.DataFrame, dry_run: bool, overwrite: bool, backup_root: Path) -> None:
    if dry_run:
        return

    backup = _backup_path(path, backup_root)
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
    write_table_atomic(frame, path)


def _align_one_file(
    path: Path,
    lookup: dict,
    experimental_index: dict[str, str],
    alphafold_index: dict[str, str],
    esm_index: dict[str, str],
    sequence_cache: dict,
    selection_cache: dict,
    identity_threshold: float,
    limit_rows: int | None,
) -> tuple[pd.DataFrame, dict]:
    frame = read_table(path)
    _require_columns(frame, ["smiles", "sequence", "log10_value"], path)
    rows = frame.to_dict("records")
    if limit_rows:
        rows_to_resolve = rows[: int(limit_rows)]
    else:
        rows_to_resolve = rows

    aligner = _make_aligner()
    resolved = []
    cache_hits = 0
    for row in tqdm(rows_to_resolve, desc=f"Align {path.name}", leave=False):
        candidates = _candidate_rows(row, lookup)
        selection_key = stable_hash(
            {
                "sequence": _norm_seq(row.get("sequence", "")),
                "candidates": [
                    [_norm_text(item.get("pdbs")), _norm_text(item.get("pdb_source")), _norm_text(item.get("pdb_type"))]
                    for item in candidates
                ],
                "identity_threshold": float(identity_threshold),
            },
            length=24,
        )
        if selection_key in selection_cache:
            result = dict(selection_cache[selection_key])
            cache_hits += 1
        else:
            result = _resolve_structure(
                row,
                candidates,
                experimental_index,
                alphafold_index,
                esm_index,
                sequence_cache,
                aligner,
                identity_threshold,
            )
            selection_cache[selection_key] = {
                key: (None if pd.isna(value) else value)
                for key, value in result.items()
            }
        resolved.append(result)

    for column in STRUCTURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    for index, result in enumerate(resolved):
        for column, value in result.items():
            frame.at[index, column] = value

    status_counts = frame["resolved_structure_status"].value_counts(dropna=False).to_dict()
    return frame, {
        "path": str(path),
        "rows": int(len(frame)),
        "resolved_rows": int(len(resolved)),
        "selection_cache_hits": int(cache_hits),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Align one best structure per CatPred split row and update split parquet/CSV files.")
    parser.add_argument("--base_dir", default=str(default_base_dir()), type=str)
    parser.add_argument("--value_types", nargs="+", default=list(DEFAULT_VALUE_TYPES))
    parser.add_argument("--split_groups", nargs="+", default=list(DEFAULT_SPLIT_GROUPS))
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--experimental_pdb_dir", default=str(DEFAULT_EXPERIMENTAL_PDB_DIR), type=str)
    parser.add_argument("--alphafold_pdb_dir", default=str(DEFAULT_ALPHAFOLD_PDB_DIR), type=str)
    parser.add_argument("--esm_pdb_dir", default=str(DEFAULT_ESM_PDB_DIR), type=str)
    parser.add_argument("--identity_threshold", default=90.0, type=float)
    parser.add_argument("--workers", default=max(1, (os.cpu_count() or 8) // 2), type=int)
    parser.add_argument("--limit_rows", default=None, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    backup_root = base_dir / "_aligned_pdb_backups"
    cache_root = base_dir / "_structure_alignment_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    jobs = discover_split_jobs(
        base_dir=base_dir,
        value_types=args.value_types,
        split_groups=args.split_groups,
        thresholds=args.thresholds,
    )
    if not jobs:
        raise RuntimeError(f"No split jobs discovered under {base_dir}")

    experimental_index = _build_experimental_index(Path(args.experimental_pdb_dir))
    alphafold_index = _build_alphafold_index(Path(args.alphafold_pdb_dir))
    esm_index = _build_esm_index(Path(args.esm_pdb_dir))
    print(
        "[align] structure indexes "
        f"PDBe={len(experimental_index)} AlphaFold={len(alphafold_index)} ESM={len(esm_index)}",
        flush=True,
    )

    manifests = []
    needed_by_value = _collect_needed_keys(jobs, args.limit_rows)
    lookups: dict[str, dict] = {}
    for value_type in args.value_types:
        value_root = base_dir / value_type
        if value_root.exists():
            needed = needed_by_value.get(value_type, {"keys": set(), "sequence_keys": set()})
            lookups[value_type] = _candidate_lookup(
                value_root,
                needed_keys=needed["keys"],
                needed_sequence_keys=needed["sequence_keys"],
            )

    all_needed = set()
    for job in jobs:
        lookup = lookups[job["value_type"]]
        for split_key in ("train_path", "val_path", "test_path"):
            frame = read_table(job[split_key])
            rows = frame.to_dict("records")
            if args.limit_rows:
                rows = rows[: args.limit_rows]
            all_needed.update(_needed_experimental_pdbs(rows, lookup, experimental_index))

    sequence_cache = _prefill_sequence_cache(
        all_needed,
        experimental_index,
        cache_root / "pdb_sequence_cache.json",
        workers=args.workers,
    )
    selection_cache_path = cache_root / "selection_cache.json"
    selection_cache = _load_json_default(selection_cache_path, {})

    for job in tqdm(jobs, desc="Update split files", unit="job"):
        lookup = lookups[job["value_type"]]
        for split_key in ("train_path", "val_path", "test_path"):
            path = Path(job[split_key])
            existing_cols = set(tabular_columns(path))
            if not args.overwrite and all(column in existing_cols for column in STRUCTURE_COLUMNS):
                manifests.append({"path": str(path), "skipped": True, "reason": "structure_columns_exist"})
                continue
            frame, summary = _align_one_file(
                path,
                lookup,
                experimental_index,
                alphafold_index,
                esm_index,
                sequence_cache,
                selection_cache,
                args.identity_threshold,
                args.limit_rows,
            )
            summary.update(
                {
                    "value_type": job["value_type"],
                    "split_group": job["split_group"],
                    "threshold": job["threshold"],
                    "dry_run": bool(args.dry_run),
                }
            )
            manifests.append(summary)
            _update_split_file(path, frame, args.dry_run, args.overwrite, backup_root)
            if not args.dry_run and len(selection_cache) % 1000 < 20:
                write_json(selection_cache_path, selection_cache)

    manifest = {
        "base_dir": str(base_dir),
        "jobs": len(jobs),
        "identity_threshold": float(args.identity_threshold),
        "dry_run": bool(args.dry_run),
        "limit_rows": args.limit_rows,
        "files": manifests,
    }
    write_json(cache_root / "alignment_manifest.json", manifest)
    if not args.dry_run:
        write_json(selection_cache_path, selection_cache)
    print(f"[align] wrote manifest: {cache_root / 'alignment_manifest.json'}", flush=True)
    if not args.dry_run:
        print(f"[align] backups root: {backup_root}", flush=True)


if __name__ == "__main__":
    main()
