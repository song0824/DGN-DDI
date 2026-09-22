"""Train/evaluate an ECFP4-MLP baseline on local DrugBank fold0.

This is a local fingerprint baseline, not the official DeepDDI setup.
Final comparable metrics should be produced with --eval-only --neg-ent 3.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from torch import nn
from torch.utils.data import DataLoader

RDLogger.DisableLog("rdApp.*")

BASELINES_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASELINES_ROOT not in sys.path:
    sys.path.insert(0, BASELINES_ROOT)

from common import (  # noqa: E402
    TEST_RESULTS_DIR,
    Triple,
    TripleDataset,
    build_candidate_drugs,
    collate_triples,
    compute_metrics,
    load_all_positive_set,
    load_protocol_splits,
    load_smiles,
    limit_triples,
    sigmoid_loss,
    write_metrics,
    NegativeSampler,
)


class FingerprintStore:
    def __init__(self, smiles_by_id: dict[str, str], n_bits: int = 2048):
        self.n_bits = n_bits
        self.fp = {}
        for drug_id, smiles in smiles_by_id.items():
            arr = np.zeros((n_bits,), dtype=np.float32)
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
                DataStructs.ConvertToNumpyArray(bitvect, arr)
            self.fp[drug_id] = torch.from_numpy(arr)

    def tensor_for(self, drug_ids: list[str], device: torch.device) -> torch.Tensor:
        return torch.stack([self.fp[d] for d in drug_ids], dim=0).to(device=device)


class ECFPMLP(nn.Module):
    def __init__(self, n_rel: int, n_bits: int = 2048, rel_dim: int = 128, hidden_dim: int = 1024):
        super().__init__()
        self.rel_emb = nn.Embedding(n_rel, rel_dim)
        in_dim = 4 * n_bits + rel_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h_fp: torch.Tensor, t_fp: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_fp, t_fp, h_fp * t_fp, torch.abs(h_fp - t_fp), self.rel_emb(rel)], dim=-1)
        return self.net(x).squeeze(-1)


def _encode_triples(
    triples: list[Triple],
    fp_store: FingerprintStore,
    rel_to_id: dict[int, int],
    device: torch.device,
):
    h_ids = [h for h, _, _ in triples]
    t_ids = [t for _, t, _ in triples]
    rels = torch.tensor([rel_to_id[int(r)] for _, _, r in triples], dtype=torch.long, device=device)
    return fp_store.tensor_for(h_ids, device), fp_store.tensor_for(t_ids, device), rels


def score_triples(model, triples, fp_store, rel_to_id, device):
    h_fp, t_fp, rels = _encode_triples(triples, fp_store, rel_to_id, device)
    return model(h_fp, t_fp, rels)


@torch.no_grad()
def evaluate(model, loader, fp_store, rel_to_id, device):
    model.eval()
    all_scores, all_labels = [], []
    total_loss = 0.0
    n_pos = 0
    for pos, neg in loader:
        p_score = score_triples(model, pos, fp_store, rel_to_id, device)
        n_score = score_triples(model, neg, fp_store, rel_to_id, device)
        loss = sigmoid_loss(p_score, n_score)
        total_loss += float(loss.item()) * len(pos)
        n_pos += len(pos)
        scores = torch.sigmoid(torch.cat([p_score, n_score])).detach().cpu().numpy()
        labels = np.concatenate([np.ones(len(p_score)), np.zeros(len(n_score))])
        all_scores.append(scores)
        all_labels.append(labels)
    out = compute_metrics(np.concatenate(all_scores), np.concatenate(all_labels))
    out["avg_loss"] = total_loss / max(n_pos, 1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--neg-ent", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--metrics-name", type=str, default="test_metrics.json")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Smoke mode: limit train positives")
    parser.add_argument("--max-val-samples", type=int, default=0, help="Smoke mode: limit val positives")
    parser.add_argument("--max-test-samples", type=int, default=0, help="Smoke mode: limit test positives")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(TEST_RESULTS_DIR, "ecfp_mlp_fold0_noleak")
    os.makedirs(out_dir, exist_ok=True)

    train, val, test, rel_to_id = load_protocol_splits()
    train = limit_triples(train, args.max_train_samples)
    val = limit_triples(val, args.max_val_samples)
    test = limit_triples(test, args.max_test_samples)
    all_pos = load_all_positive_set()
    drugs = build_candidate_drugs(all_pos)
    train_sampler = NegativeSampler(list(all_pos), drugs, seed=42)
    val_sampler = NegativeSampler(list(all_pos), drugs, seed=43)
    test_sampler = NegativeSampler(list(all_pos), drugs, seed=44)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp_store = FingerprintStore(load_smiles())

    train_loader = DataLoader(
        TripleDataset(train, train_sampler, neg_ent=args.neg_ent),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_triples,
    )
    val_loader = DataLoader(
        TripleDataset(val, val_sampler, neg_ent=args.neg_ent, freeze_negatives=True),
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate_triples,
    )
    test_loader = DataLoader(
        TripleDataset(test, test_sampler, neg_ent=args.neg_ent, freeze_negatives=True),
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate_triples,
    )

    model = ECFPMLP(n_rel=len(rel_to_id)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best_path = args.ckpt or os.path.join(out_dir, "best_model.pth")

    def write_test(best_epoch=-1, best_val_auc=-1.0):
        metrics_out = evaluate(model, test_loader, fp_store, rel_to_id, device)
        metrics_out.update(
            {
                "model": "ECFP4-MLP",
                "split": "drugbank_test/drugbank/fold0",
                "best_epoch": int(best_epoch),
                "best_val_auc": float(best_val_auc),
                "n_epochs": int(args.n_epochs),
                "neg_ent": int(args.neg_ent),
                "aupr_definition": "average_precision_score",
                "note": "Morgan radius=2 2048-bit local baseline; not official DeepDDI.",
                "max_train_samples": int(args.max_train_samples),
                "max_val_samples": int(args.max_val_samples),
                "max_test_samples": int(args.max_test_samples),
            }
        )
        out_json = os.path.join(out_dir, args.metrics_name)
        write_metrics(out_json, metrics_out)
        print("TEST", metrics_out, flush=True)

    if args.eval_only:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        write_test(ckpt.get("epoch", -1), ckpt.get("best_val_auc", -1.0))
        return

    best_val_auc = -1.0
    history = []
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        seen = 0
        for pos, neg in train_loader:
            p_score = score_triples(model, pos, fp_store, rel_to_id, device)
            n_score = score_triples(model, neg, fp_store, rel_to_id, device)
            loss = sigmoid_loss(p_score, n_score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(pos)
            seen += len(pos)
        val_metrics = evaluate(model, val_loader, fp_store, rel_to_id, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "val_auc": val_metrics["auc"],
            "val_aupr": val_metrics["aupr"],
            "seconds": round(time.time() - t0, 2),
        }
        history.append(row)
        print(
            f"{datetime.now()} epoch {epoch}/{args.n_epochs} "
            f"train_loss={row['train_loss']:.4f} val_auc={row['val_auc']:.4f} "
            f"val_aupr={row['val_aupr']:.4f} ({row['seconds']}s)",
            flush=True,
        )
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "best_val_auc": best_val_auc},
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    write_test(ckpt.get("epoch", -1), ckpt.get("best_val_auc", best_val_auc))
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "train_history.csv"), index=False)


if __name__ == "__main__":
    main()
