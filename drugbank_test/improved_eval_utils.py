from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BASELINES_ROOT = os.path.join(PROJECT_ROOT, "baselines")
if BASELINES_ROOT not in sys.path:
    sys.path.insert(0, BASELINES_ROOT)

from common import (  # noqa: E402
    NegativeSampler,
    build_candidate_drugs,
    build_relation_vocab,
    compute_metrics,
    load_all_positive_set,
    load_smiles,
    load_triples,
    write_metrics,
)
from ecfp_mlp.train_eval_fold0 import (  # noqa: E402
    ECFPMLP,
    FingerprintStore,
    score_triples as ecfp_score_triples,
)
from data_preprocessing import DDIDataLoader, DrugDataset  # noqa: E402
from models import DGN_DDI  # noqa: E402
from transductive_test import _merge_eval_config_from_checkpoint, get_test_config  # noqa: E402


@dataclass
class LabeledTriples:
    triples: list[tuple[str, str, int]]
    labels: np.ndarray


class ExplicitTripleDataset(DrugDataset):
    """Score a fixed triple list without generating extra negatives in collate."""

    def __init__(
        self,
        source_csv: str,
        data_loader: DDIDataLoader,
        triples: Sequence[tuple[str, str, int]],
        negative_sampling_scope: str = "global",
    ):
        super().__init__(
            source_csv,
            data_loader,
            neg_ent=0,
            shuffle=False,
            negative_sampling_scope=negative_sampling_scope,
            freeze_negatives=True,
        )
        self.tri_list = list(triples)


def build_labeled_examples(split_csv: str, neg_ent: int = 3, seed: int = 44) -> LabeledTriples:
    positives = load_triples(split_csv)
    all_pos = list(load_all_positive_set())
    candidate_drugs = build_candidate_drugs(all_pos)
    sampler = NegativeSampler(all_pos, candidate_drugs, seed=seed)
    triples: list[tuple[str, str, int]] = []
    labels: list[int] = []
    for h, t, r in positives:
        triples.append((h, t, r))
        labels.append(1)
        for neg in sampler.sample(h, t, r, neg_ent):
            triples.append(neg)
            labels.append(0)
    return LabeledTriples(triples=triples, labels=np.asarray(labels, dtype=np.int64))


def select_best_threshold(probs: np.ndarray, labels: np.ndarray, min_recall: float = 0.0) -> float:
    best_t, best_f1, best_acc = 0.5, -1.0, -1.0
    for t in np.linspace(0.0, 1.0, 101):
        pred = (probs > t).astype(int)
        rec = float(metrics.recall_score(labels, pred, zero_division=0))
        if rec < min_recall:
            continue
        f1v = float(metrics.f1_score(labels, pred, zero_division=0))
        accv = float(metrics.accuracy_score(labels, pred))
        if (
            f1v > best_f1
            or (abs(f1v - best_f1) <= 1e-9 and accv > best_acc)
            or (abs(f1v - best_f1) <= 1e-9 and abs(accv - best_acc) <= 1e-9 and abs(t - 0.5) < abs(best_t - 0.5))
        ):
            best_t, best_f1, best_acc = float(t), f1v, accv
    return best_t


def evaluate_probabilities(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    out = compute_metrics(probs, labels)
    pred = (probs > float(threshold)).astype(int)
    out["accuracy"] = float(metrics.accuracy_score(labels, pred))
    out["f1_score"] = float(metrics.f1_score(labels, pred, zero_division=0))
    out["precision"] = float(metrics.precision_score(labels, pred, zero_division=0))
    out["recall"] = float(metrics.recall_score(labels, pred, zero_division=0))
    out["threshold"] = float(threshold)
    return out


def save_experiment_metrics(out_dir: str, metrics_out: dict, extra: dict | None = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    payload = dict(metrics_out)
    if extra:
        payload.update(extra)
    out_json = os.path.join(out_dir, "test_metrics.json")
    write_metrics(out_json, payload)
    return out_json


def load_ecfp_components(ckpt_path: str, device: torch.device):
    rel_to_id = build_relation_vocab()
    model = ECFPMLP(n_rel=len(rel_to_id)).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    fp_store = FingerprintStore(load_smiles())
    return model, fp_store, rel_to_id


@torch.no_grad()
def score_ecfp_probabilities(
    model: ECFPMLP,
    fp_store: FingerprintStore,
    rel_to_id: dict[int, int],
    triples: Sequence[tuple[str, str, int]],
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    probs: list[np.ndarray] = []
    for start in range(0, len(triples), batch_size):
        batch = list(triples[start:start + batch_size])
        scores = ecfp_score_triples(model, batch, fp_store, rel_to_id, device)
        probs.append(torch.sigmoid(scores).cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)


def load_dgn_components(model_path: str, device: torch.device):
    config = get_test_config(profile="full")
    config["model_path"] = model_path
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = _merge_eval_config_from_checkpoint(config, checkpoint)
    ddi_loader = DDIDataLoader()
    model = DGN_DDI(
        hidden_dim=config["hidden_dim"],
        kge_dim=config["kge_dim"],
        rel_total=max(1, int(getattr(ddi_loader, "rel_total", 0) or 0)),
        heads_out_feat_params=config["heads_out_feat_params"],
        blocks_params=config["blocks_params"],
        drug_graph_dict=ddi_loader.drug_graph_dict,
        int_to_drug_id=ddi_loader.int_to_drug_id,
        device=device,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, ddi_loader, config, checkpoint


def make_explicit_dgn_loader(
    ddi_loader: DDIDataLoader,
    source_csv: str,
    triples: Sequence[tuple[str, str, int]],
    batch_size: int,
    negative_sampling_scope: str = "global",
):
    dataset = ExplicitTripleDataset(
        source_csv=source_csv,
        data_loader=ddi_loader,
        triples=triples,
        negative_sampling_scope=negative_sampling_scope,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_positives_only)


@torch.no_grad()
def score_dgn(
    model: DGN_DDI,
    loader: Iterable,
    device: torch.device,
    return_pair_features: bool = False,
):
    all_probs: list[np.ndarray] = []
    all_h_repr: list[np.ndarray] = []
    all_t_repr: list[np.ndarray] = []
    for batch in loader:
        if batch is None:
            continue
        h_batch, t_batch, rels, b_graph = batch
        h_batch = h_batch.to(device)
        t_batch = t_batch.to(device)
        rels = rels.to(device)
        b_graph = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b_graph.items()}
        if return_pair_features:
            scores, h_repr, t_repr = model.forward_with_pair_features(h_batch, t_batch, rels, b_graph)
            all_h_repr.append(h_repr.cpu().numpy())
            all_t_repr.append(t_repr.cpu().numpy())
        else:
            scores = model(h_batch, t_batch, rels, b_graph)
        all_probs.append(torch.sigmoid(scores).cpu().numpy())
    probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0,), dtype=np.float32)
    if not return_pair_features:
        return probs
    h_repr_np = np.concatenate(all_h_repr, axis=0) if all_h_repr else np.zeros((0, 0), dtype=np.float32)
    t_repr_np = np.concatenate(all_t_repr, axis=0) if all_t_repr else np.zeros((0, 0), dtype=np.float32)
    return probs, h_repr_np, t_repr_np
