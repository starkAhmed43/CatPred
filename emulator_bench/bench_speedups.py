import os
from typing import List

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from bench_feature_cache import get_or_compute_esm, load_cached_esm


_PRETRAINED_EGNN_CACHE: dict[str, tuple[dict[str, torch.Tensor], torch.Tensor]] = {}
_LETTER_TO_NUM = {
    "A": 0,
    "R": 1,
    "N": 2,
    "D": 3,
    "C": 4,
    "Q": 5,
    "E": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "L": 10,
    "K": 11,
    "M": 12,
    "F": 13,
    "P": 14,
    "S": 15,
    "T": 16,
    "W": 17,
    "Y": 18,
    "V": 19,
}
_PAD_TOKEN = 20
_ESM_GETTER = None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _tokenize_sequence(sequence: str) -> torch.Tensor:
    return torch.as_tensor([_LETTER_TO_NUM.get(residue, _PAD_TOKEN) for residue in sequence], dtype=torch.long)


def _esm_getter():
    global _ESM_GETTER
    if _ESM_GETTER is None:
        import catpred.data.utils as data_utils

        _ESM_GETTER = data_utils.get_protein_embedder("esm")["fn"]
    return _ESM_GETTER


def _resolve_esm_feats(protein: dict) -> torch.Tensor:
    existing = protein.get("esm2_feats")
    if existing is not None:
        return existing if torch.is_tensor(existing) else torch.as_tensor(existing)

    seq = protein.get("seq", "")
    name = protein.get("name", "")
    cache_dir = protein.get("_bench_cache_dir") or os.getenv("CATPRED_BENCH_CACHE_DIR")

    cached = load_cached_esm(seq, cache_dir=cache_dir)
    if cached is None:
        cached = get_or_compute_esm(seq, name, _esm_getter(), cache_dir=cache_dir)

    if _env_flag("CATPRED_BENCH_PIN_ESM_IN_RECORD", default=False):
        protein["esm2_feats"] = cached

    return cached


def _load_pretrained_egnn_features(path: str) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    cached = _PRETRAINED_EGNN_CACHE.get(path)
    if cached is not None:
        return cached

    raw = torch.load(path, map_location="cpu")
    tensors = {}
    values = []
    for name, value in raw.items():
        tensor = value.detach().to(dtype=torch.float32, device="cpu") if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32)
        tensors[name] = tensor
        values.append(tensor)

    if not values:
        raise ValueError(f"No EGNN features found in {path}.")

    avg = torch.stack(values).mean(dim=0)
    cached = (tensors, avg)
    _PRETRAINED_EGNN_CACHE[path] = cached
    return cached


def install_model_speed_patches() -> None:
    import catpred.models.model as model_module
    from rotary_embedding_torch import RotaryEmbedding

    if getattr(model_module, "_bench_speed_patched", False):
        return

    molecule_model_cls = model_module.MoleculeModel
    AttentivePooling = model_module.AttentivePooling

    def create_protein_model(self, args) -> None:
        self.seq_embedder = nn.Embedding(21, args.seq_embed_dim, padding_idx=_PAD_TOKEN)

        if self.args.add_pretrained_egnn_feats:
            feats_dict, feats_avg = _load_pretrained_egnn_features(self.args.pretrained_egnn_feats_path)
            self.pretrained_egnn_feats_dict = feats_dict
            self.pretrained_egnn_feats_avg = feats_avg

        # Keep rotary dimension even to match core CatPred and avoid cache shape mismatches.
        rotary_dim = max(2, args.seq_embed_dim // 4)
        if rotary_dim % 2 != 0:
            rotary_dim -= 1
        self.rotary_embedder = RotaryEmbedding(dim=rotary_dim)
        self.multihead_attn = nn.MultiheadAttention(
            args.seq_embed_dim,
            args.seq_self_attn_nheads,
            batch_first=True,
        )

        seq_attn_pooling_dim = args.seq_embed_dim
        if args.add_esm_feats:
            seq_attn_pooling_dim += 1280

        self.attentive_pooler = AttentivePooling(seq_attn_pooling_dim, seq_attn_pooling_dim)
        self.max_pooler = lambda x: torch.max(x, dim=1, keepdim=False, out=None)

    def forward(
        self,
        batch,
        features_batch=None,
        atom_descriptors_batch=None,
        atom_features_batch=None,
        bond_descriptors_batch=None,
        bond_features_batch=None,
        constraints_batch: List[torch.Tensor] = None,
        bond_types_batch: List[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_atom_bond_targets:
            encodings = self.encoder(
                batch,
                features_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_descriptors_batch,
                bond_features_batch,
            )
            output = self.readout(encodings, constraints_batch, bond_types_batch)
        else:
            encodings = self.encoder(
                batch,
                features_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_descriptors_batch,
                bond_features_batch,
            )

            if not self.args.skip_protein and self.args.protein_records_path is not None:
                protein_records = batch[-1].protein_record_list
                pretrained_egnn_arr = []
                if self.args.add_pretrained_egnn_feats:
                    for protein in protein_records:
                        pretrained_egnn_arr.append(
                            self.pretrained_egnn_feats_dict.get(protein["name"], self.pretrained_egnn_feats_avg)
                        )
                    pretrained_egnn_arr = torch.stack(pretrained_egnn_arr).to(self.device, non_blocking=True)

                seq_arr = []
                for protein in protein_records:
                    token_ids = protein.get("token_ids")
                    if token_ids is None:
                        token_ids = _tokenize_sequence(protein["seq"])
                        protein["token_ids"] = token_ids
                    elif not torch.is_tensor(token_ids):
                        token_ids = torch.as_tensor(token_ids, dtype=torch.long)
                        protein["token_ids"] = token_ids
                    seq_arr.append(token_ids)

                seq_arr = pad_sequence(seq_arr, batch_first=True, padding_value=_PAD_TOKEN).to(self.device, non_blocking=True)

                if self.args.add_esm_feats:
                    esm_feature_arr = []
                    for protein in protein_records:
                        esm_feats = _resolve_esm_feats(protein)
                        if not torch.is_tensor(esm_feats):
                            esm_feats = torch.as_tensor(esm_feats)
                        esm_feature_arr.append(esm_feats)
                    esm_feature_arr = pad_sequence(esm_feature_arr, batch_first=True).to(self.device, non_blocking=True)
                    if seq_arr.shape[1] != esm_feature_arr.shape[1]:
                        seq_arr = seq_arr[:, : esm_feature_arr.shape[1]]

                seq_outs = self.seq_embedder(seq_arr)
                q = self.rotary_embedder.rotate_queries_or_keys(seq_outs, seq_dim=1)
                k = self.rotary_embedder.rotate_queries_or_keys(seq_outs, seq_dim=1)
                seq_outs, _ = self.multihead_attn(q, k, seq_outs)

                if self.args.add_esm_feats:
                    seq_outs = torch.cat([esm_feature_arr, seq_outs], dim=-1)

                if not self.args.skip_attentive_pooling:
                    seq_pooled_outs, _ = self.attentive_pooler(seq_outs)
                else:
                    seq_pooled_outs = seq_outs.mean(dim=1)

                if self.args.add_pretrained_egnn_feats:
                    seq_pooled_outs = torch.cat([seq_pooled_outs, pretrained_egnn_arr], dim=-1)

                if not self.args.skip_substrate:
                    total_outs = torch.cat([seq_pooled_outs, encodings], dim=-1)
                else:
                    total_outs = seq_pooled_outs

                output = self.readout(total_outs)
            else:
                output = self.readout(encodings)

        if self.classification and not (self.training and self.no_training_normalization) and self.loss_function != "dirichlet":
            if self.is_atom_bond_targets:
                output = [self.sigmoid(x) for x in output]
            else:
                output = self.sigmoid(output)

        if self.multiclass:
            output = output.reshape((output.shape[0], -1, self.num_classes))
            if not (self.training and self.no_training_normalization) and self.loss_function != "dirichlet":
                output = self.multiclass_softmax(output)

        if self.loss_function == "mve":
            if self.is_atom_bond_targets:
                outputs = []
                for x in output:
                    means, variances = torch.split(x, x.shape[1] // 2, dim=1)
                    variances = self.softplus(variances)
                    outputs.append(torch.cat([means, variances], axis=1))
                return outputs
            means, variances = torch.split(output, output.shape[1] // 2, dim=1)
            variances = self.softplus(variances)
            output = torch.cat([means, variances], axis=1)

        if self.loss_function == "evidential":
            if self.is_atom_bond_targets:
                outputs = []
                for x in output:
                    means, lambdas, alphas, betas = torch.split(x, x.shape[1] // 4, dim=1)
                    lambdas = self.softplus(lambdas)
                    alphas = self.softplus(alphas) + 1
                    betas = self.softplus(betas)
                    outputs.append(torch.cat([means, lambdas, alphas, betas], dim=1))
                return outputs
            means, lambdas, alphas, betas = torch.split(output, output.shape[1] // 4, dim=1)
            lambdas = self.softplus(lambdas)
            alphas = self.softplus(alphas) + 1
            betas = self.softplus(betas)
            output = torch.cat([means, lambdas, alphas, betas], dim=1)

        if self.loss_function == "dirichlet":
            if self.is_atom_bond_targets:
                outputs = []
                for x in output:
                    outputs.append(nn.functional.softplus(x) + 1)
                return outputs
            output = nn.functional.softplus(output) + 1

        return output

    molecule_model_cls.create_protein_model = create_protein_model
    molecule_model_cls.forward = forward
    model_module._bench_speed_patched = True
