from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from improved_eval_utils import (
    SCRIPT_DIR,
    build_labeled_examples,
    build_relation_vocab,
    evaluate_probabilities,
    load_dgn_components,
    load_ecfp_components,
    make_explicit_dgn_loader,
    save_experiment_metrics,
    score_dgn,
    score_ecfp_probabilities,
    select_best_threshold,
)


class HybridFeatureDataset(Dataset):
    def __init__(self, dgn_features: np.ndarray, aux_scores: np.ndarray, rel_ids: np.ndarray, labels: np.ndarray):
        self.dgn_features = torch.tensor(dgn_features, dtype=torch.float32)
        self.aux_scores = torch.tensor(aux_scores, dtype=torch.float32)
        self.rel_ids = torch.tensor(rel_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return int(self.labels.size(0))

    def __getitem__(self, idx: int):
        return self.dgn_features[idx], self.aux_scores[idx], self.rel_ids[idx], self.labels[idx]


class HybridHead(nn.Module):
    def __init__(self, dgn_dim: int, n_rel: int, rel_dim: int = 64, hidden_dim: int = 512):
        super().__init__()
        self.rel_emb = nn.Embedding(n_rel, rel_dim)
        self.net = nn.Sequential(
            nn.Linear(dgn_dim + 2 + rel_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, dgn_features: torch.Tensor, aux_scores: torch.Tensor, rel_ids: torch.Tensor) -> torch.Tensor:
        rel = self.rel_emb(rel_ids)
        x = torch.cat([dgn_features, aux_scores, rel], dim=-1)
        return self.net(x).squeeze(-1)


def build_rel_ids(triples: list[tuple[str, str, int]]) -> np.ndarray:
    rel_to_id = build_relation_vocab()
    return np.asarray([rel_to_id[int(r)] for _, _, r in triples], dtype=np.int64)


def limit_examples(data, max_samples: int):
    if max_samples and max_samples > 0:
        return type(data)(triples=data.triples[:max_samples], labels=data.labels[:max_samples])
    return data


def extract_dgn_features(model, loader, device):
    probs, h_repr, t_repr = score_dgn(model, loader, device, return_pair_features=True)
    pair = np.concatenate([h_repr, t_repr, h_repr * t_repr, np.abs(h_repr - t_repr)], axis=1)
    return probs, pair


@torch.no_grad()
def predict_model(model: HybridHead, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for dgn_features, aux_scores, rel_ids, _ in loader:
        logits = model(dgn_features.to(device), aux_scores.to(device), rel_ids.to(device))
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0,), dtype=np.float32)


def train_epoch(model, loader, optimizer, device, teacher_weight: float = 0.0):
    model.train()
    bce = nn.BCEWithLogitsLoss()
    total = 0.0
    total_n = 0
    for dgn_features, aux_scores, rel_ids, labels in loader:
        dgn_features = dgn_features.to(device)
        aux_scores = aux_scores.to(device)
        rel_ids = rel_ids.to(device)
        labels = labels.to(device)
        logits = model(dgn_features, aux_scores, rel_ids)
        cls_loss = bce(logits, labels)
        if teacher_weight > 0:
            teacher_prob = aux_scores[:, 1]
            distill_loss = nn.functional.mse_loss(torch.sigmoid(logits), teacher_prob)
            loss = cls_loss + teacher_weight * distill_loss
        else:
            loss = cls_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * int(labels.size(0))
        total_n += int(labels.size(0))
    return total / max(total_n, 1)


def main():
    parser = argparse.ArgumentParser(description="Train a lightweight DGN hybrid / distillation head")
    parser.add_argument("--mode", choices=["hybrid", "distill"], default="hybrid")
    parser.add_argument(
        "--dgn-model",
        default=os.path.join(SCRIPT_DIR, "checkpoints", "paper_lock", "best_model.pth"),
    )
    parser.add_argument(
        "--ecfp-model",
        default=os.path.join(SCRIPT_DIR, "test_results", "ecfp_mlp_fold0_noleak", "best_model.pth"),
    )
    parser.add_argument(
        "--train-csv",
        default=os.path.join(SCRIPT_DIR, "drugbank", "fold0", "train.csv"),
    )
    parser.add_argument(
        "--val-csv",
        default=os.path.join(SCRIPT_DIR, "drugbank", "fold0", "val_noleak.csv"),
    )
    parser.add_argument(
        "--test-csv",
        default=os.path.join(SCRIPT_DIR, "drugbank", "fold0", "test.csv"),
    )
    parser.add_argument("--neg-ent", type=int, default=3)
    parser.add_argument("--seed-train", type=int, default=42)
    parser.add_argument("--seed-val", type=int, default=43)
    parser.add_argument("--seed-test", type=int, default=44)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dgn-batch-size", type=int, default=48)
    parser.add_argument("--ecfp-batch-size", type=int, default=4096)
    parser.add_argument("--teacher-weight", type=float, default=0.3)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        default=os.path.join(SCRIPT_DIR, "test_results", "dgn_hybrid_head"),
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dgn_model, ddi_loader, dgn_cfg, _ = load_dgn_components(args.dgn_model, device)
    ecfp_model, fp_store, rel_to_id = load_ecfp_components(args.ecfp_model, device)
    neg_scope = dgn_cfg.get("negative_sampling_scope", "global")

    train_data = limit_examples(build_labeled_examples(args.train_csv, args.neg_ent, args.seed_train), args.max_train_samples)
    val_data = limit_examples(build_labeled_examples(args.val_csv, args.neg_ent, args.seed_val), args.max_val_samples)
    test_data = limit_examples(build_labeled_examples(args.test_csv, args.neg_ent, args.seed_test), args.max_test_samples)

    def _make_loader(csv_path, triples):
        return make_explicit_dgn_loader(ddi_loader, csv_path, triples, args.dgn_batch_size, negative_sampling_scope=neg_scope)

    train_loader_dgn = _make_loader(args.train_csv, train_data.triples)
    val_loader_dgn = _make_loader(args.val_csv, val_data.triples)
    test_loader_dgn = _make_loader(args.test_csv, test_data.triples)

    dgn_train_prob, dgn_train_features = extract_dgn_features(dgn_model, train_loader_dgn, device)
    dgn_val_prob, dgn_val_features = extract_dgn_features(dgn_model, val_loader_dgn, device)
    dgn_test_prob, dgn_test_features = extract_dgn_features(dgn_model, test_loader_dgn, device)

    ecfp_train_prob = score_ecfp_probabilities(ecfp_model, fp_store, rel_to_id, train_data.triples, device, args.ecfp_batch_size)
    ecfp_val_prob = score_ecfp_probabilities(ecfp_model, fp_store, rel_to_id, val_data.triples, device, args.ecfp_batch_size)
    ecfp_test_prob = score_ecfp_probabilities(ecfp_model, fp_store, rel_to_id, test_data.triples, device, args.ecfp_batch_size)

    train_aux = np.column_stack([dgn_train_prob, ecfp_train_prob]).astype(np.float32)
    val_aux = np.column_stack([dgn_val_prob, ecfp_val_prob]).astype(np.float32)
    test_aux = np.column_stack([dgn_test_prob, ecfp_test_prob]).astype(np.float32)

    train_rel = build_rel_ids(train_data.triples)
    val_rel = build_rel_ids(val_data.triples)
    test_rel = build_rel_ids(test_data.triples)

    train_ds = HybridFeatureDataset(dgn_train_features, train_aux, train_rel, train_data.labels)
    val_ds = HybridFeatureDataset(dgn_val_features, val_aux, val_rel, val_data.labels)
    test_ds = HybridFeatureDataset(dgn_test_features, test_aux, test_rel, test_data.labels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = HybridHead(dgn_dim=dgn_train_features.shape[1], n_rel=len(rel_to_id)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_state = None
    best_f1 = -1.0
    best_t = 0.5
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            teacher_weight=args.teacher_weight if args.mode == "distill" else 0.0,
        )
        val_prob = predict_model(model, val_loader, device)
        best_t_epoch = select_best_threshold(val_prob, val_data.labels)
        val_metrics = evaluate_probabilities(val_prob, val_data.labels, best_t_epoch)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_f1": val_metrics["f1_score"], "val_auc": val_metrics["auc"]})
        if val_metrics["f1_score"] > best_f1:
            best_f1 = float(val_metrics["f1_score"])
            best_t = float(best_t_epoch)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No best state captured during training")
    model.load_state_dict(best_state)
    test_prob = predict_model(model, test_loader, device)
    test_metrics = evaluate_probabilities(test_prob, test_data.labels, best_t)
    test_metrics["model"] = f"DGN hybrid head ({args.mode})"
    test_metrics["neg_ent"] = int(args.neg_ent)
    test_metrics["aupr_definition"] = "average_precision_score"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mode": args.mode,
            "best_threshold": best_t,
            "history": history,
            "dgn_model": args.dgn_model,
            "ecfp_model": args.ecfp_model,
        },
        os.path.join(args.out_dir, "best_model.pth"),
    )
    extra = {
        "mode": args.mode,
        "best_threshold": best_t,
        "history": history,
        "dgn_model": args.dgn_model,
        "ecfp_model": args.ecfp_model,
        "negative_sampling_scope": neg_scope,
        "teacher_weight": float(args.teacher_weight if args.mode == "distill" else 0.0),
    }
    metrics_path = save_experiment_metrics(args.out_dir, test_metrics, extra=extra)
    with open(os.path.join(args.out_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(json.dumps({"metrics_path": metrics_path, "metrics": test_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
