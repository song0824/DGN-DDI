"""Train/evaluate an atom-only GAT-DDI baseline on local DrugBank fold0.

This baseline uses only atom graphs plus a RESCAL-style relation scorer. It is
trained from scratch and should not be confused with DGN inference ablations.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, global_mean_pool
from torch.utils.data import DataLoader

BASELINES_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASELINES_ROOT not in sys.path:
    sys.path.insert(0, BASELINES_ROOT)

from common import (  # noqa: E402
    DRUGBANK_DIR,
    TEST_RESULTS_DIR,
    Triple,
    TripleDataset,
    build_candidate_drugs,
    collate_triples,
    compute_metrics,
    load_all_positive_set,
    load_protocol_splits,
    limit_triples,
    sigmoid_loss,
    write_metrics,
    NegativeSampler,
)


class AtomGraphStore:
    def __init__(self):
        path = os.path.join(DRUGBANK_DIR, "drug_data_dict.pkl")
        with open(path, "rb") as f:
            graph_dict = pickle.load(f)
        self.atom_graphs = {}
        for drug_id, data in graph_dict.items():
            if drug_id == "__metadata__":
                continue
            if isinstance(data, dict) and "atom" in data:
                self.atom_graphs[str(drug_id)] = data["atom"]

    def batch(self, drug_ids: list[str], device: torch.device) -> Batch:
        graphs = []
        for drug_id in drug_ids:
            g = self.atom_graphs[drug_id].clone()
            g.x = torch.nan_to_num(g.x.float(), nan=0.0, posinf=1.0, neginf=-1.0)
            graphs.append(g)
        return Batch.from_data_list(graphs).to(device)


class GATDDI(nn.Module):
    def __init__(self, n_rel: int, in_dim: int = 66, hidden_dim: int = 256, kge_dim: int = 256):
        super().__init__()
        self.conv1 = GATConv(in_dim, 64, heads=4, concat=True, dropout=0.1)
        self.conv2 = GATConv(256, 64, heads=4, concat=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)
        self.proj = nn.Linear(hidden_dim, kge_dim)
        self.rel = nn.Embedding(n_rel, kge_dim * kge_dim)
        nn.init.xavier_uniform_(self.rel.weight)
        self.kge_dim = kge_dim
        self.score_scale = kge_dim ** -0.5

    def encode(self, batch: Batch) -> torch.Tensor:
        x = self.conv1(batch.x, batch.edge_index)
        x = torch.relu(self.norm1(x))
        x = self.conv2(x, batch.edge_index)
        x = torch.relu(self.norm2(x))
        return self.proj(global_mean_pool(x, batch.batch))

    def forward(self, h_batch: Batch, t_batch: Batch, rel: torch.Tensor) -> torch.Tensor:
        h = self.encode(h_batch)
        t = self.encode(t_batch)
        rel_mat = self.rel(rel).view(-1, self.kge_dim, self.kge_dim)
        return torch.bmm(torch.bmm(h.unsqueeze(1), rel_mat), t.unsqueeze(-1)).view(-1) * self.score_scale


def _encode_batch(triples: list[Triple], store: AtomGraphStore, rel_to_id: dict[int, int], device):
    h_ids = [h for h, _, _ in triples]
    t_ids = [t for _, t, _ in triples]
    rels = torch.tensor([rel_to_id[int(r)] for _, _, r in triples], dtype=torch.long, device=device)
    return store.batch(h_ids, device), store.batch(t_ids, device), rels


def score_triples(model, triples, store, rel_to_id, device):
    h_batch, t_batch, rels = _encode_batch(triples, store, rel_to_id, device)
    return model(h_batch, t_batch, rels)


@torch.no_grad()
def evaluate(model, loader, store, rel_to_id, device):
    model.eval()
    all_scores, all_labels = [], []
    total_loss = 0.0
    n_pos = 0
    for pos, neg in loader:
        p_score = score_triples(model, pos, store, rel_to_id, device)
        n_score = score_triples(model, neg, store, rel_to_id, device)
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
    parser.add_argument("--batch-size", type=int, default=128)
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

    out_dir = args.out_dir or os.path.join(TEST_RESULTS_DIR, "gat_ddi_fold0_noleak")
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
    store = AtomGraphStore()

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

    model = GATDDI(n_rel=len(rel_to_id)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best_path = args.ckpt or os.path.join(out_dir, "best_model.pth")

    def write_test(best_epoch=-1, best_val_auc=-1.0):
        metrics_out = evaluate(model, test_loader, store, rel_to_id, device)
        metrics_out.update(
            {
                "model": "GAT-DDI (atom-only, RESCAL scorer)",
                "split": "drugbank_test/drugbank/fold0",
                "best_epoch": int(best_epoch),
                "best_val_auc": float(best_val_auc),
                "n_epochs": int(args.n_epochs),
                "neg_ent": int(args.neg_ent),
                "aupr_definition": "average_precision_score",
                "note": "Atom graph GAT baseline trained from scratch; not a DGN inference ablation.",
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
            p_score = score_triples(model, pos, store, rel_to_id, device)
            n_score = score_triples(model, neg, store, rel_to_id, device)
            loss = sigmoid_loss(p_score, n_score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(pos)
            seen += len(pos)
        val_metrics = evaluate(model, val_loader, store, rel_to_id, device)
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
