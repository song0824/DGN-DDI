"""在本仓库 fold0 上训练 / 重评 SSI-DDI-like，写出与 DGN-DDI 同口径的 Test 指标。

Readout 已替换官方 SAGPooling，对外请称 SSI-DDI-like。
不改 DGN-DDI 的 paper_lock 权重。请在 baselines/ssi_ddi 目录运行。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch import optim

import custom_loss
import models
from data_preprocessing import DrugDataLoader, DrugDataset, TOTAL_ATOM_FEATS


def _load_triples(path):
    df = pd.read_csv(path)
    return [(h, t, int(r)) for h, t, r in zip(df["d1"], df["d2"], df["type"])]


def do_compute(model, batch, device):
    pos_tri, neg_tri = batch
    pos_tri = [tensor.to(device=device) for tensor in pos_tri]
    p_score = model(pos_tri)
    neg_tri = [tensor.to(device=device) for tensor in neg_tri]
    n_score = model(neg_tri)
    probas = np.concatenate(
        [
            torch.sigmoid(p_score.detach()).cpu().numpy(),
            torch.sigmoid(n_score.detach()).cpu().numpy(),
        ]
    )
    labels = np.concatenate([np.ones(len(p_score)), np.zeros(len(n_score))])
    return p_score, n_score, probas, labels


def compute_metrics(probas, target):
    pred = (probas >= 0.5).astype(int)
    return {
        "accuracy": float(metrics.accuracy_score(target, pred)),
        "auc": float(metrics.roc_auc_score(target, probas)),
        "aupr": float(metrics.average_precision_score(target, probas)),
        "f1_score": float(metrics.f1_score(target, pred)),
        "precision": float(metrics.precision_score(target, pred, zero_division=0)),
        "recall": float(metrics.recall_score(target, pred, zero_division=0)),
        "aupr_definition": "average_precision_score",
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_p, all_y = [], []
    total_loss = 0.0
    loss_fn = custom_loss.SigmoidLoss()
    n = 0
    for batch in loader:
        p_score, n_score, probas, labels = do_compute(model, batch, device)
        loss, _, _ = loss_fn(p_score, n_score)
        total_loss += float(loss.item()) * len(p_score)
        n += len(p_score)
        all_p.append(probas)
        all_y.append(labels)
    metrics_out = compute_metrics(np.concatenate(all_p), np.concatenate(all_y))
    metrics_out["avg_loss"] = total_loss / max(n, 1)
    return metrics_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--neg-ent", type=int, default=1, help="negatives per positive; use 3 to match DGN test")
    parser.add_argument("--eval-only", action="store_true", help="load existing best_model.pth and evaluate only")
    parser.add_argument("--ckpt", type=str, default="", help="checkpoint path for --eval-only")
    parser.add_argument("--metrics-name", type=str, default="test_metrics.json")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(here)),
        "drugbank_test",
        "test_results",
        "ssi_ddi_fold0",
    )
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tup = _load_triples("data/ddi_training.csv")
    val_tup = _load_triples("data/ddi_validation.csv")
    test_tup = _load_triples("data/ddi_test.csv")

    train_data = DrugDataset(train_tup, neg_ent=args.neg_ent)
    val_data = DrugDataset(val_tup, neg_ent=args.neg_ent, disjoint_split=False)
    test_data = DrugDataset(test_tup, neg_ent=args.neg_ent, disjoint_split=False)
    train_loader = DrugDataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DrugDataLoader(val_data, batch_size=args.batch_size * 3)
    test_loader = DrugDataLoader(test_data, batch_size=args.batch_size * 3)

    model = models.SSI_DDI(
        TOTAL_ATOM_FEATS,
        64,
        64,
        86,
        heads_out_feat_params=[32, 32, 32, 32],
        blocks_params=[2, 2, 2, 2],
    ).to(device)
    loss_fn = custom_loss.SigmoidLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** epoch)

    print(
        f"SSI-DDI-like fold0: train={len(train_data)} val={len(val_data)} test={len(test_data)} "
        f"neg_ent={args.neg_ent} device={device}",
        flush=True,
    )
    best_val_auc = -1.0
    best_path = args.ckpt or os.path.join(out_dir, "best_model.pth")
    history = []

    def _write_test(model, tag_best_epoch=-1, tag_best_val=-1.0):
        test_metrics = evaluate(model, test_loader, device)
        test_metrics["model"] = "SSI-DDI-like (softmax-gated add-pool)"
        test_metrics["split"] = "drugbank_test/drugbank/fold0"
        test_metrics["best_epoch"] = int(tag_best_epoch)
        test_metrics["best_val_auc"] = float(tag_best_val)
        test_metrics["n_epochs"] = args.n_epochs
        test_metrics["neg_ent"] = int(args.neg_ent)
        test_metrics["aupr_definition"] = "average_precision_score"
        test_metrics["readout"] = "softmax-gated add-pool (not official SAGPooling)"
        test_metrics["val_leak_note"] = "original fold0 val ∩ test = 3152; do not claim official SSI-DDI fold"
        out_json = os.path.join(out_dir, args.metrics_name)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)
        print("TEST", json.dumps(test_metrics, ensure_ascii=False), flush=True)
        print("Wrote", out_json, flush=True)
        return test_metrics

    if args.eval_only:
        if not os.path.exists(best_path):
            raise FileNotFoundError(best_path)
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        _write_test(model, ckpt.get("epoch", -1), ckpt.get("val", {}).get("auc", -1.0) if isinstance(ckpt.get("val"), dict) else ckpt.get("best_val_auc", -1.0))
        return

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        seen = 0
        for batch in train_loader:
            p_score, n_score, _, _ = do_compute(model, batch, device)
            loss, _, _ = loss_fn(p_score, n_score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(p_score)
            seen += len(p_score)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
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
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val": val_metrics}, best_path)
            print(f"  saved best val_auc={best_val_auc:.4f} -> {best_path}", flush=True)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _write_test(model, ckpt.get("epoch", -1), best_val_auc)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "train_history.csv"), index=False)


if __name__ == "__main__":
    main()
