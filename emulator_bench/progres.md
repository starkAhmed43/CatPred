# PROGRES for CatPred Emulator Bench

This note documents the recommended way to generate protein structure embeddings with PROGRES and use them in emulator bench.

## TL;DR

- Do not use the legacy pdbpath + protein-records JSON flow for your bench runs.
- Use your table columns directly: smiles, sequence, pdbs.
- Build PROGRES embeddings keyed by pdbs values.
- Convert PROGRES output to a PyTorch dict: {pdb_id: embedding_tensor}.
- Pass that dict to bench with:
  - --add_pretrained_egnn_feats
  - --pretrained_egnn_feats_path /path/to/progres_egnn_by_pdbid.pt

## 1) Expected PDB directory layout

No special nesting is required.

Recommended layout:

- One file per protein in a single directory.
- File name stem must match the value in the pdbs column.

Example:

- /home/ubuntu/adhil/pdbs/1mzy.pdb
- /home/ubuntu/adhil/pdbs/2abc.pdb

Notes:

- Keep identifiers consistent (case and whitespace).
- PROGRES reads only the first chain from each PDB file.

## 2) Install and verify PROGRES

From your CatPred environment:

python3 -m pip install -e /home/ubuntu/adhil/progres

Check CLI:

progres -h

If the command is not on PATH, use:

python3 /home/ubuntu/adhil/progres/bin/progres -h

## 3) Build a PROGRES structure list

PROGRES embed expects a text file with one item per line:

<path_to_structure> <domain_id> <optional_note>

For this workflow, set domain_id == pdb_id.

Example line:

/home/ubuntu/adhil/pdbs/1mzy.pdb 1mzy sample

## 4) Generate structure list from parquet pdbs column

Create the list from your parquet (train/val/test or combined):

python3 - <<'PY'
import pandas as pd
from pathlib import Path

parquet_path = "/home/ubuntu/adhil/EMULaToR/data/processed/baselines/catpred/enzyme_sequence_splits/threshold_0.09/train.parquet"
pdb_dir = Path("/home/ubuntu/adhil/pdbs")
out_list = Path("/home/ubuntu/adhil/catpred/emulator_bench/progres_structurelist.txt")

# Read IDs and normalize
ids = pd.read_parquet(parquet_path)["pdbs"].astype(str).str.strip()
ids = sorted(set(ids))

missing = []
with out_list.open("w", encoding="utf-8") as f:
    for pid in ids:
        fp = pdb_dir / f"{pid}.pdb"
        if fp.exists():
            f.write(f"{fp} {pid} -\n")
        else:
            missing.append(pid)

print(f"wrote: {out_list}")
print(f"total IDs: {len(ids)}")
print(f"missing PDB files: {len(missing)}")
if missing:
    print("first missing:", missing[:20])
PY

If you want full coverage, build the ID set from all splits (train/val/test).

## 5) Run PROGRES embedding

Use CPU or GPU. Example with GPU:

progres embed \
  -l /home/ubuntu/adhil/catpred/emulator_bench/progres_structurelist.txt \
  -o /home/ubuntu/adhil/catpred/emulator_bench/progres_searchdb.pt \
  -f pdb \
  -d cuda

Important:

- Do not use -c for Chainsaw here.
- With Chainsaw on, IDs become suffixes like _D1, _D2 and will not match your pdbs keys.

## 6) Convert PROGRES output to CatPred EGNN dict

PROGRES output is a search DB format with keys ids, embeddings, nres, notes.
Convert to a dict keyed by pdb_id:

python3 - <<'PY'
import torch

src = "/home/ubuntu/adhil/catpred/emulator_bench/progres_searchdb.pt"
dst = "/home/ubuntu/adhil/catpred/emulator_bench/progres_egnn_by_pdbid.pt"

d = torch.load(src, map_location="cpu")
ids = d["ids"]
embs = d["embeddings"]

out = {}
for pid, emb in zip(ids, embs):
    key = str(pid).strip()
    out[key] = emb.detach().to(dtype=torch.float32, device="cpu")

torch.save(out, dst)
print("saved", dst, "entries", len(out))
PY

## 7) Validate key coverage against your parquet

python3 - <<'PY'
import pandas as pd
import torch

parquet_path = "/home/ubuntu/adhil/EMULaToR/data/processed/baselines/catpred/enzyme_sequence_splits/threshold_0.09/train.parquet"
eg_path = "/home/ubuntu/adhil/catpred/emulator_bench/progres_egnn_by_pdbid.pt"

need = set(pd.read_parquet(parquet_path)["pdbs"].astype(str).str.strip())
have = set(torch.load(eg_path, map_location="cpu").keys())
missing = sorted(need - have)

print("need", len(need), "have", len(have), "missing", len(missing))
if missing:
    print("first missing", missing[:20])
PY

## 8) Use the embeddings in emulator bench

Append these flags to train/tune/retrain commands:

--uniprot_id_col pdbs \
--add_pretrained_egnn_feats \
--pretrained_egnn_feats_path /home/ubuntu/adhil/catpred/emulator_bench/progres_egnn_by_pdbid.pt

Notes:

- Bench inline loader now auto-detects protein ID columns (including pdbs), but setting --uniprot_id_col pdbs is explicit and recommended.
- Sequence and ESM still come from sequence column.
- EGNN embedding lookup uses protein name key from the selected protein ID column.

## 9) Practical recommendation

- First run one small split to verify key coverage and model startup.
- Then run Optuna or retrain batch using the same EGNN dict and cache settings.
- Keep one canonical embedding file path for reproducibility.
