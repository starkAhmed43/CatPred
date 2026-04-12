import csv
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from bench_feature_cache import get_or_compute_esm
from common import default_cache_dir, materialize_tabular_as_csv


INLINE_PROTEIN_SENTINEL = "__bench_inline_protein__"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_protein_id_column(requested_col: str, fieldnames: List[str]) -> str:
    if requested_col in fieldnames:
        return requested_col

    fallback_candidates = ["pdbs", "pdb_id", "protein_id", "uniprot_id", "pdbpath"]
    for candidate in fallback_candidates:
        if candidate in fieldnames:
            return candidate

    available = ", ".join(fieldnames)
    raise ValueError(
        "Data file did not contain a valid protein id column. "
        f"Requested '{requested_col}'. Checked fallback columns {fallback_candidates}. "
        f"Available columns: {available}."
    )


def _cache_root() -> Path:
    root = os.getenv("CATPRED_BENCH_CACHE_DIR") or os.getenv("CATPRED_CACHE_PATH")
    if root is None:
        root_path = Path(default_cache_dir())
    else:
        root_path = Path(root).expanduser().resolve()
    path = root_path / "dataset_inline"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _file_signature(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    p = Path(path)
    try:
        st = p.stat()
        return f"{p.name}:{st.st_size}:{st.st_mtime_ns}"
    except Exception:
        return str(p)


def _dataset_cache_path(path: str, spec: dict) -> Path:
    key = hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    return _cache_root() / f"{Path(path).stem}.{key}.pkl"


def _dataset_signature(path: str, spec: dict) -> str:
    key = hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    return f"{Path(path).stem}.{key}"


def _load_dataset_cache(cache_path: Optional[Path], debug) -> Optional[object]:
    if cache_path is None or not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        try:
            cache_path.unlink()
        except Exception:
            pass
        debug(f"[cache] removed corrupt dataset cache: {cache_path}")
        return None


def _save_dataset_cache(cache_path: Optional[Path], data, debug) -> None:
    if cache_path is None:
        return
    try:
        tmp_path = cache_path.with_suffix(".tmp.pkl")
        with open(tmp_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)
    except Exception as exc:
        debug(f"[cache] dataset cache write skipped: {exc}")


def install_bench_patches(sequence_col: str = "sequence", uniprot_id_col: str = "uniprot_id") -> None:
    os.environ["CATPRED_BENCH_SEQUENCE_COL"] = sequence_col
    os.environ["CATPRED_BENCH_UNIPROT_ID_COL"] = uniprot_id_col

    import catpred.data as data_pkg
    import catpred.data.utils as data_utils

    original_get_data = data_utils.get_data
    if getattr(data_utils, "_bench_inline_patched", False):
        return

    def patched_get_data(
        path: str,
        protein_records_path: str = None,
        vocabulary_path: str = None,
        smiles_columns=None,
        target_columns: List[str] = None,
        ignore_columns: List[str] = None,
        skip_invalid_smiles: bool = True,
        args=None,
        data_weights_path: str = None,
        features_path: List[str] = None,
        features_generator: List[str] = None,
        phase_features_path: str = None,
        atom_descriptors_path: str = None,
        bond_descriptors_path: str = None,
        constraints_path: str = None,
        max_data_size: int = None,
        store_row: bool = True,
        logger=None,
        loss_function: str = None,
        skip_none_targets: bool = False,
    ):
        if protein_records_path != INLINE_PROTEIN_SENTINEL:
            return original_get_data(
                path=path,
                protein_records_path=protein_records_path,
                vocabulary_path=vocabulary_path,
                smiles_columns=smiles_columns,
                target_columns=target_columns,
                ignore_columns=ignore_columns,
                skip_invalid_smiles=skip_invalid_smiles,
                args=args,
                data_weights_path=data_weights_path,
                features_path=features_path,
                features_generator=features_generator,
                phase_features_path=phase_features_path,
                atom_descriptors_path=atom_descriptors_path,
                bond_descriptors_path=bond_descriptors_path,
                constraints_path=constraints_path,
                max_data_size=max_data_size,
                store_row=store_row,
                logger=logger,
                loss_function=loss_function,
                skip_none_targets=skip_none_targets,
            )

        return _get_data_inline(
            path=path,
            smiles_columns=smiles_columns,
            target_columns=target_columns,
            ignore_columns=ignore_columns,
            skip_invalid_smiles=skip_invalid_smiles,
            args=args,
            data_weights_path=data_weights_path,
            features_path=features_path,
            features_generator=features_generator,
            phase_features_path=phase_features_path,
            atom_descriptors_path=atom_descriptors_path,
            bond_descriptors_path=bond_descriptors_path,
            constraints_path=constraints_path,
            max_data_size=max_data_size,
            store_row=store_row,
            logger=logger,
            loss_function=loss_function,
            skip_none_targets=skip_none_targets,
            sequence_col=os.environ.get("CATPRED_BENCH_SEQUENCE_COL", "sequence"),
            uniprot_id_col=os.environ.get("CATPRED_BENCH_UNIPROT_ID_COL", "uniprot_id"),
        )

    data_utils.get_data = patched_get_data
    data_pkg.get_data = patched_get_data
    data_utils._bench_inline_patched = True


def _get_data_inline(
    path: str,
    smiles_columns=None,
    target_columns: List[str] = None,
    ignore_columns: List[str] = None,
    skip_invalid_smiles: bool = True,
    args=None,
    data_weights_path: str = None,
    features_path: List[str] = None,
    features_generator: List[str] = None,
    phase_features_path: str = None,
    atom_descriptors_path: str = None,
    bond_descriptors_path: str = None,
    constraints_path: str = None,
    max_data_size: int = None,
    store_row: bool = True,
    logger=None,
    loss_function: str = None,
    skip_none_targets: bool = False,
    sequence_col: str = "sequence",
    uniprot_id_col: str = "uniprot_id",
):
    import catpred.data.utils as u
    from catpred.data import MoleculeDatapoint, MoleculeDataset
    from catpred.rdkit import make_mol

    debug = logger.debug if logger is not None else print
    path = materialize_tabular_as_csv(path, cache_dir=os.environ.get("CATPRED_BENCH_TABULAR_CACHE_DIR"))

    if args is not None:
        smiles_columns = smiles_columns if smiles_columns is not None else args.smiles_columns
        target_columns = target_columns if target_columns is not None else args.target_columns
        ignore_columns = ignore_columns if ignore_columns is not None else args.ignore_columns
        features_path = features_path if features_path is not None else args.features_path
        features_generator = features_generator if features_generator is not None else args.features_generator
        phase_features_path = phase_features_path if phase_features_path is not None else args.phase_features_path
        atom_descriptors_path = atom_descriptors_path if atom_descriptors_path is not None else args.atom_descriptors_path
        bond_descriptors_path = bond_descriptors_path if bond_descriptors_path is not None else args.bond_descriptors_path
        constraints_path = constraints_path if constraints_path is not None else args.constraints_path
        max_data_size = max_data_size if max_data_size is not None else args.max_data_size
        loss_function = loss_function if loss_function is not None else args.loss_function

    if isinstance(smiles_columns, str) or smiles_columns is None:
        smiles_columns = u.preprocess_smiles_columns(path=path, smiles_columns=smiles_columns)

    max_data_size = max_data_size or float("inf")

    if features_path is not None:
        features_data = []
        for feat_path in features_path:
            features_data.append(u.load_features(feat_path))
        features_data = np.concatenate(features_data, axis=1)
    else:
        features_data = None

    if phase_features_path is not None:
        phase_features = u.load_features(phase_features_path)
        for d_phase in phase_features:
            if not (d_phase.sum() == 1 and np.count_nonzero(d_phase) == 1):
                raise ValueError("Phase features must be one-hot encoded.")
        if features_data is not None:
            features_data = np.concatenate((features_data, phase_features), axis=1)
        else:
            features_data = np.array(phase_features)
    else:
        phase_features = None

    if constraints_path is not None:
        constraints_data, raw_constraints_data = u.get_constraints(
            path=constraints_path,
            target_columns=args.target_columns,
            save_raw_data=args.save_smiles_splits,
        )
    else:
        constraints_data = None
        raw_constraints_data = None

    if data_weights_path is not None:
        data_weights = u.get_data_weights(data_weights_path)
    else:
        data_weights = None

    if target_columns is None:
        target_columns = u.get_task_names(
            path=path,
            smiles_columns=smiles_columns,
            target_columns=target_columns,
            ignore_columns=ignore_columns,
        )

    if loss_function == "bounded_mse":
        gt_targets, lt_targets = u.get_inequality_targets(path=path, target_columns=target_columns)
    else:
        gt_targets, lt_targets = None, None

    protein_enabled = args is not None and not getattr(args, "skip_protein", False)
    add_esm_feats = bool(getattr(args, "add_esm_feats", False)) if protein_enabled else False
    lazy_esm = _env_flag("CATPRED_BENCH_LAZY_ESM", default=True)
    sequence_feat_getter = None
    protein_record_cache = {}
    if protein_enabled:
        if add_esm_feats and not lazy_esm:
            sequence_feat_getter = u.get_protein_embedder("esm")["fn"]

    dataset_spec = {
        "schema_version": 2,
        "data": _file_signature(path),
        "features": [_file_signature(p) for p in (features_path or [])],
        "phase_features": _file_signature(phase_features_path),
        "constraints": _file_signature(constraints_path),
        "data_weights": _file_signature(data_weights_path),
        "smiles_columns": list(smiles_columns),
        "target_columns": list(target_columns),
        "ignore_columns": list(ignore_columns or []),
        "sequence_col": sequence_col,
        "uniprot_id_col": uniprot_id_col,
        "max_data_size": str(max_data_size),
        "skip_invalid_smiles": bool(skip_invalid_smiles),
        "skip_none_targets": bool(skip_none_targets),
        "store_row": bool(store_row),
        "loss_function": loss_function,
        "protein_enabled": bool(protein_enabled),
        "add_esm_feats": bool(add_esm_feats),
        "lazy_esm": bool(lazy_esm),
        "sequence_max_length": int(getattr(args, "sequence_max_length", 0) or 0),
    }
    dataset_sig = _dataset_signature(path, dataset_spec)

    dataset_cache_path = None
    if _env_flag("CATPRED_BENCH_DATASET_CACHE", default=True):
        dataset_cache_path = _dataset_cache_path(path, dataset_spec)
        cached_data = _load_dataset_cache(dataset_cache_path, debug)
        if cached_data is not None:
            setattr(cached_data, "_bench_dataset_signature", dataset_sig)
            return cached_data

    with open(path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        show_data_progress = _env_flag("CATPRED_BENCH_VERBOSE_DATA_PROGRESS", default=False)
        if any(c not in fieldnames for c in smiles_columns):
            raise ValueError(f"Data file did not contain all provided smiles columns: {smiles_columns}.")
        if any(c not in fieldnames for c in target_columns):
            raise ValueError(f"Data file did not contain all provided target columns: {target_columns}.")
        protein_id_col = uniprot_id_col
        if protein_enabled:
            if sequence_col not in fieldnames:
                raise ValueError(f"Data file did not contain sequence column: {sequence_col}.")
            protein_id_col = _resolve_protein_id_column(uniprot_id_col, fieldnames)
            if protein_id_col != uniprot_id_col:
                debug(
                    f"Protein id column '{uniprot_id_col}' not found. "
                    f"Using '{protein_id_col}' instead."
                )

        all_smiles, all_targets, all_atom_targets, all_bond_targets = [], [], [], []
        all_rows, all_features, all_phase_features = [], [], []
        all_constraints_data, all_raw_constraints_data, all_weights = [], [], []
        all_gt, all_lt, all_protein_records = [], [], []

        for i, row in enumerate(tqdm(reader, disable=not show_data_progress, leave=False)):
            smiles = [row[c] for c in smiles_columns]

            protein_record = None
            if protein_enabled:
                protein_name = str(row[protein_id_col]).strip()
                protein_seq = str(row[sequence_col]).strip()
                max_seq_len = int(getattr(args, "sequence_max_length", 0) or 0)
                if max_seq_len > 0 and len(protein_seq) > max_seq_len:
                    protein_seq = protein_seq[:max_seq_len]
                cache_key = (protein_name, protein_seq)
                if cache_key not in protein_record_cache:
                    protein_record = {
                        "name": protein_name,
                        "seq": protein_seq,
                        "token_ids": np.fromiter(
                            (
                                {
                                    "A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6, "G": 7, "H": 8,
                                    "I": 9, "L": 10, "K": 11, "M": 12, "F": 13, "P": 14, "S": 15, "T": 16,
                                    "W": 17, "Y": 18, "V": 19,
                                }.get(residue, 20)
                                for residue in protein_seq
                            ),
                            dtype=np.uint8,
                        ),
                    }
                    if sequence_feat_getter is not None:
                        protein_record["esm2_feats"] = get_or_compute_esm(
                            protein_seq,
                            protein_name,
                            sequence_feat_getter,
                            cache_dir=os.environ.get("CATPRED_BENCH_CACHE_DIR"),
                        )
                    elif add_esm_feats:
                        protein_record["_bench_cache_dir"] = os.environ.get("CATPRED_BENCH_CACHE_DIR")
                    protein_record_cache[cache_key] = protein_record
                protein_record = protein_record_cache[cache_key]

            targets, atom_targets, bond_targets = [], [], []
            for column in target_columns:
                value = row[column]
                if value in ["", "nan"]:
                    targets.append(None)
                elif ">" in value or "<" in value:
                    if loss_function == "bounded_mse":
                        targets.append(float(value.strip("<>")))
                    else:
                        raise ValueError("Inequality target requires bounded_mse.")
                elif "[" in value or "]" in value:
                    value = value.replace("None", "null")
                    target = np.array(json.loads(value))
                    if len(target.shape) == 1 and column in args.atom_targets:
                        atom_targets.append(target)
                        targets.append(target)
                    elif len(target.shape) == 1 and column in args.bond_targets:
                        bond_targets.append(target)
                        targets.append(target)
                    elif len(target.shape) == 2:
                        bond_target_arranged = []
                        mol = make_mol(smiles[0], args.explicit_h, args.adding_h, args.keeping_atom_map)
                        for bond in mol.GetBonds():
                            bond_target_arranged.append(target[bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
                        bond_targets.append(np.array(bond_target_arranged))
                        targets.append(np.array(bond_target_arranged))
                    else:
                        raise ValueError(f"Unrecognized targets of column {column} in {path}.")
                else:
                    targets.append(float(value))

            if skip_none_targets and all(x is None for x in targets):
                continue

            all_protein_records.append(protein_record)
            all_smiles.append(smiles)
            all_targets.append(targets)
            all_atom_targets.append(atom_targets)
            all_bond_targets.append(bond_targets)

            if features_data is not None:
                all_features.append(features_data[i])
            if phase_features is not None:
                all_phase_features.append(phase_features[i])
            if constraints_data is not None:
                all_constraints_data.append(constraints_data[i])
            if raw_constraints_data is not None:
                all_raw_constraints_data.append(raw_constraints_data[i])
            if data_weights is not None:
                all_weights.append(data_weights[i])
            if gt_targets is not None:
                all_gt.append(gt_targets[i])
            if lt_targets is not None:
                all_lt.append(lt_targets[i])
            if store_row:
                all_rows.append(row)
            if len(all_smiles) >= max_data_size:
                break

        atom_features = None
        atom_descriptors = None
        if args is not None and args.atom_descriptors is not None:
            descriptors = u.load_valid_atom_or_bond_features(atom_descriptors_path, [x[0] for x in all_smiles])
            if args.atom_descriptors == "feature":
                atom_features = descriptors
            elif args.atom_descriptors == "descriptor":
                atom_descriptors = descriptors

        bond_features = None
        bond_descriptors = None
        if args is not None and args.bond_descriptors is not None:
            descriptors = u.load_valid_atom_or_bond_features(bond_descriptors_path, [x[0] for x in all_smiles])
            if args.bond_descriptors == "feature":
                bond_features = descriptors
            elif args.bond_descriptors == "descriptor":
                bond_descriptors = descriptors

        data = MoleculeDataset([
            MoleculeDatapoint(
                smiles=smiles,
                protein_record=all_protein_records[i],
                vocabulary=None,
                targets=targets,
                atom_targets=all_atom_targets[i] if all_atom_targets[i] else None,
                bond_targets=all_bond_targets[i] if all_bond_targets[i] else None,
                row=all_rows[i] if store_row else None,
                data_weight=all_weights[i] if data_weights is not None else None,
                gt_targets=all_gt[i] if gt_targets is not None else None,
                lt_targets=all_lt[i] if lt_targets is not None else None,
                features_generator=features_generator,
                features=all_features[i] if features_data is not None else None,
                phase_features=all_phase_features[i] if phase_features is not None else None,
                atom_features=atom_features[i] if atom_features is not None else None,
                atom_descriptors=atom_descriptors[i] if atom_descriptors is not None else None,
                bond_features=bond_features[i] if bond_features is not None else None,
                bond_descriptors=bond_descriptors[i] if bond_descriptors is not None else None,
                constraints=all_constraints_data[i] if constraints_data is not None else None,
                raw_constraints=all_raw_constraints_data[i] if raw_constraints_data is not None else None,
                overwrite_default_atom_features=args.overwrite_default_atom_features if args is not None else False,
                overwrite_default_bond_features=args.overwrite_default_bond_features if args is not None else False,
            )
            for i, (smiles, targets) in tqdm(
                enumerate(zip(all_smiles, all_targets)),
                total=len(all_smiles),
                disable=not show_data_progress,
                leave=False,
            )
        ])

    if skip_invalid_smiles:
        original_data_len = len(data)
        data = u.filter_invalid_smiles(data)
        if len(data) < original_data_len:
            debug(f"Warning: {original_data_len - len(data)} SMILES are invalid.")

    setattr(data, "_bench_dataset_signature", dataset_sig)

    _save_dataset_cache(dataset_cache_path, data, debug)

    return data
