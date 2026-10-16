from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from improved_eval_utils import (
    SCRIPT_DIR,
    build_labeled_examples,
    evaluate_probabilities,
    load_dgn_components,
    load_ecfp_components,
    make_explicit_dgn_loader,
    save_experiment_metrics,
    score_dgn,
    score_ecfp_probabilities,
    select_best_threshold,
)


def main():
    parser = argparse.ArgumentParser(description="Late fusion / stacking for DGN-DDI + ECFP4-MLP")
    parser.add_argument(
        "--dgn-model",
        default=os.path.join(SCRIPT_DIR, "checkpoints", "paper_lock", "best_model.pth"),
    )
    parser.add_argument(
        "--ecfp-model",
        default=os.path.join(SCRIPT_DIR, "test_results", "ecfp_mlp_fold0_noleak", "best_model.pth"),
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
    parser.add_argument("--seed-val", type=int, default=43)
    parser.add_argument("--seed-test", type=int, default=44)
    parser.add_argument("--dgn-batch-size", type=int, default=48)
    parser.add_argument("--ecfp-batch-size", type=int, default=4096)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        default=os.path.join(SCRIPT_DIR, "test_results", "dgn_ecfp_fusion"),
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dgn_model, ddi_loader, dgn_cfg, _ = load_dgn_components(args.dgn_model, device)
    ecfp_model, fp_store, rel_to_id = load_ecfp_components(args.ecfp_model, device)
    neg_scope = dgn_cfg.get("negative_sampling_scope", "global")

    val_data = build_labeled_examples(args.val_csv, neg_ent=args.neg_ent, seed=args.seed_val)
    test_data = build_labeled_examples(args.test_csv, neg_ent=args.neg_ent, seed=args.seed_test)
    if args.max_val_samples and args.max_val_samples > 0:
        val_data = type(val_data)(triples=val_data.triples[:args.max_val_samples], labels=val_data.labels[:args.max_val_samples])
    if args.max_test_samples and args.max_test_samples > 0:
        test_data = type(test_data)(triples=test_data.triples[:args.max_test_samples], labels=test_data.labels[:args.max_test_samples])

    val_loader = make_explicit_dgn_loader(
        ddi_loader, args.val_csv, val_data.triples, args.dgn_batch_size, negative_sampling_scope=neg_scope
    )
    test_loader = make_explicit_dgn_loader(
        ddi_loader, args.test_csv, test_data.triples, args.dgn_batch_size, negative_sampling_scope=neg_scope
    )

    dgn_val = score_dgn(dgn_model, val_loader, device, return_pair_features=False)
    dgn_test = score_dgn(dgn_model, test_loader, device, return_pair_features=False)
    ecfp_val = score_ecfp_probabilities(
        ecfp_model, fp_store, rel_to_id, val_data.triples, device, batch_size=args.ecfp_batch_size
    )
    ecfp_test = score_ecfp_probabilities(
        ecfp_model, fp_store, rel_to_id, test_data.triples, device, batch_size=args.ecfp_batch_size
    )

    x_val = np.column_stack([dgn_val, ecfp_val])
    x_test = np.column_stack([dgn_test, ecfp_test])
    y_val = val_data.labels
    y_test = test_data.labels

    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(x_val, y_val)
    val_prob = clf.predict_proba(x_val)[:, 1]
    test_prob = clf.predict_proba(x_test)[:, 1]
    best_t = select_best_threshold(val_prob, y_val)
    metrics_out = evaluate_probabilities(test_prob, y_test, threshold=best_t)
    metrics_out["model"] = "DGN-DDI + ECFP4-MLP (late fusion)"
    metrics_out["neg_ent"] = int(args.neg_ent)
    metrics_out["aupr_definition"] = "average_precision_score"

    extra = {
        "dgn_model": args.dgn_model,
        "ecfp_model": args.ecfp_model,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
        "negative_sampling_scope": neg_scope,
        "weights": clf.coef_[0].tolist(),
        "bias": float(clf.intercept_[0]),
        "val_threshold": float(best_t),
        "val_metrics": evaluate_probabilities(val_prob, y_val, threshold=best_t),
    }
    metrics_path = save_experiment_metrics(args.out_dir, metrics_out, extra=extra)

    pd.DataFrame(
        {
            "label": y_test,
            "dgn_prob": dgn_test,
            "ecfp_prob": ecfp_test,
            "fusion_prob": test_prob,
        }
    ).to_csv(os.path.join(args.out_dir, "fusion_predictions.csv"), index=False)
    with open(os.path.join(args.out_dir, "fusion_model.json"), "w", encoding="utf-8") as f:
        json.dump(extra, f, ensure_ascii=False, indent=2)
    print(json.dumps({"metrics_path": metrics_path, "metrics": metrics_out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
