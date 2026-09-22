"""Shared helpers for fold0 DrugBank baselines.

All local baselines use the same protocol as the DGN-DDI main table:
fold0 train/test, val_noleak for checkpoint selection, and neg_ent=3 for
the final test metrics.
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import Dataset


BASELINES_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASELINES_ROOT)
DRUGBANK_DIR = os.path.join(PROJECT_ROOT, "drugbank_test", "drugbank")
FOLD0_DIR = os.path.join(DRUGBANK_DIR, "fold0")
TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, "drugbank_test", "test_results")

Triple = Tuple[str, str, int]


def ensure_noleak_val() -> str:
    """Return val_noleak.csv; fail if it has not been generated."""
    noleak = os.path.join(FOLD0_DIR, "val_noleak.csv")
    if os.path.exists(noleak):
        return noleak
    raise FileNotFoundError(
        f"{noleak} not found. Run `python drugbank_test/make_noleak_val.py` "
        "before training comparable baselines."
    )


def load_triples(path: str) -> List[Triple]:
    df = pd.read_csv(path, usecols=["d1", "d2", "type"])
    return [(str(h), str(t), int(r)) for h, t, r in zip(df["d1"], df["d2"], df["type"])]


def load_smiles() -> dict:
    df = pd.read_csv(os.path.join(DRUGBANK_DIR, "drug_smiles.csv"))
    return {str(row.drug_id): str(row.smiles) for row in df.itertuples(index=False)}


def build_relation_vocab() -> dict:
    all_triples = load_triples(os.path.join(DRUGBANK_DIR, "ddis.csv"))
    rels = sorted({r for _, _, r in all_triples})
    return {r: i for i, r in enumerate(rels)}


def load_protocol_splits() -> tuple[List[Triple], List[Triple], List[Triple], dict]:
    rel_to_id = build_relation_vocab()
    train = load_triples(os.path.join(FOLD0_DIR, "train.csv"))
    val = load_triples(ensure_noleak_val())
    test = load_triples(os.path.join(FOLD0_DIR, "test.csv"))
    return train, val, test, rel_to_id


def limit_triples(triples: Sequence[Triple], max_samples: Optional[int]) -> List[Triple]:
    """Return a deterministic prefix for smoke runs."""
    if max_samples is None or max_samples <= 0:
        return list(triples)
    return list(triples[:max_samples])


def load_all_positive_set() -> set:
    return set(load_triples(os.path.join(DRUGBANK_DIR, "ddis.csv")))


def build_candidate_drugs(triples: Iterable[Triple]) -> list[str]:
    drugs = set()
    for h, t, _ in triples:
        drugs.add(h)
        drugs.add(t)
    return sorted(drugs)


class NegativeSampler:
    """Corrupt heads or tails while avoiding known positives in ddis.csv."""

    def __init__(self, all_positive_triples: Sequence[Triple], candidate_drugs: Sequence[str], seed: int = 42):
        self.all_true = set(all_positive_triples)
        self.drugs = list(candidate_drugs)
        self.rng = random.Random(seed)
        self.true_heads = defaultdict(set)
        self.true_tails = defaultdict(set)
        for h, t, r in all_positive_triples:
            self.true_heads[(t, r)].add(h)
            self.true_tails[(h, r)].add(t)

    def sample(self, h: str, t: str, r: int, neg_ent: int) -> list[Triple]:
        out: list[Triple] = []
        for _ in range(neg_ent):
            corrupt_head = self.rng.random() < 0.5
            if corrupt_head:
                out.append((self._sample_head(t, r), t, r))
            else:
                out.append((h, self._sample_tail(h, r), r))
        return out

    def _sample_head(self, t: str, r: int) -> str:
        blocked = self.true_heads.get((t, r), set())
        for _ in range(100):
            h = self.rng.choice(self.drugs)
            if h not in blocked:
                return h
        return self.rng.choice(self.drugs)

    def _sample_tail(self, h: str, r: int) -> str:
        blocked = self.true_tails.get((h, r), set())
        for _ in range(100):
            t = self.rng.choice(self.drugs)
            if t not in blocked:
                return t
        return self.rng.choice(self.drugs)


class TripleDataset(Dataset):
    def __init__(
        self,
        triples: Sequence[Triple],
        sampler: NegativeSampler,
        neg_ent: int = 1,
        freeze_negatives: bool = False,
    ):
        self.triples = list(triples)
        self.sampler = sampler
        self.neg_ent = int(neg_ent)
        self.frozen_negatives = None
        if freeze_negatives:
            self.frozen_negatives = [self.sampler.sample(h, t, r, self.neg_ent) for h, t, r in self.triples]

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int):
        h, t, r = self.triples[idx]
        if self.frozen_negatives is not None:
            neg = self.frozen_negatives[idx]
        else:
            neg = self.sampler.sample(h, t, r, self.neg_ent)
        return (h, t, r), neg


def collate_triples(batch):
    pos: list[Triple] = []
    neg: list[Triple] = []
    for p, ns in batch:
        pos.append(p)
        neg.extend(ns)
    return pos, neg


def sigmoid_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    pos_scores = torch.nan_to_num(pos_scores, nan=0.0, posinf=50.0, neginf=-50.0)
    neg_scores = torch.nan_to_num(neg_scores, nan=0.0, posinf=50.0, neginf=-50.0)
    pos_scores = torch.clamp(pos_scores, min=-50.0, max=50.0)
    neg_scores = torch.clamp(neg_scores, min=-50.0, max=50.0)
    return 0.5 * (-F.logsigmoid(pos_scores).mean() - F.logsigmoid(-neg_scores).mean())


def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    pred = (scores >= 0.5).astype(int)
    return {
        "accuracy": float(metrics.accuracy_score(labels, pred)),
        "auc": float(metrics.roc_auc_score(labels, scores)),
        "aupr": float(metrics.average_precision_score(labels, scores)),
        "f1_score": float(metrics.f1_score(labels, pred)),
        "precision": float(metrics.precision_score(labels, pred, zero_division=0)),
        "recall": float(metrics.recall_score(labels, pred, zero_division=0)),
        "aupr_definition": "average_precision_score",
    }


def write_metrics(path: str, metrics_out: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)
