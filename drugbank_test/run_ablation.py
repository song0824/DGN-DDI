"""推理消融：Full / w/o Fusion(=atom_only) / w/o Inter / substruct_only。

不改权重、不重训。结果写入 test_results/ablation_<mode>/ ，并汇总 comparison.csv。
atom_only 与 no_fusion 同一路径，默认只跑 no_fusion。
"""
import argparse
import json
import logging
import os

import pandas as pd

from transductive_test import DDITester, get_test_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODES = ("full", "no_fusion", "no_inter", "substruct_only")


def main():
    parser = argparse.ArgumentParser(description="Inference-time ablation for DGN-DDI")
    parser.add_argument(
        "--model-path",
        default=os.path.join(_SCRIPT_DIR, "checkpoints", "paper_lock", "best_model.pth"),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(_SCRIPT_DIR, "test_results"),
    )
    parser.add_argument("--subset-size", type=int, default=None, help="Optional smoke subset")
    parser.add_argument("--batch-size", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        args.model_path = os.path.join(_SCRIPT_DIR, "checkpoints", "best_model.pth")
        logger.warning("paper_lock missing, use %s", args.model_path)

    config = get_test_config(profile="full")
    config["model_path"] = args.model_path
    if args.subset_size:
        config["subset_size"] = args.subset_size
    if args.batch_size and args.batch_size > 0:
        config["batch_size"] = args.batch_size
    config["ablation_mode"] = "full"
    config["results_dir"] = os.path.join(args.out_dir, "ablation_full")
    os.makedirs(config["results_dir"], exist_ok=True)

    tester = DDITester(config)
    rows = []
    for mode in MODES:
        tester.model.set_ablation_mode(mode)
        tester.config["ablation_mode"] = mode
        tester.config["results_dir"] = os.path.join(args.out_dir, f"ablation_{mode}")
        os.makedirs(tester.config["results_dir"], exist_ok=True)
        logger.info("===== Ablation mode: %s =====", mode)
        metrics = tester.test()
        tester.save_results()
        row = {
            "mode": mode,
            "auc": metrics.get("auc"),
            "aupr": metrics.get("aupr"),
            "accuracy": metrics.get("accuracy"),
            "f1_score": metrics.get("f1_score"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
        }
        rows.append(row)
        with open(os.path.join(tester.config["results_dir"], "ablation_mode.json"), "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(args.out_dir, "ablation_comparison.csv")
    summary.to_csv(summary_path, index=False)
    logger.info("Ablation comparison saved: %s", summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
