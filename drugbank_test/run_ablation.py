"""推理消融：默认 full / no_fusion / substruct_only。

不改权重、不重训。默认权重为无泄漏正式 checkpoint，
结果写入 test_results/ablation_noleak_<mode>/。
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
ALL_MODES = ("full", "no_fusion", "no_inter", "substruct_only")
# 中期主消融只回答融合模块贡献；no_inter 需显式 --modes 才跑。
DEFAULT_MODES = ("full", "no_fusion", "substruct_only")
NOLEAK_CKPT = os.path.join(
    _SCRIPT_DIR, "checkpoints", "dgn_noleak_sigmoid_full", "best_model.pth"
)


def main():
    parser = argparse.ArgumentParser(description="Inference-time ablation for DGN-DDI")
    parser.add_argument("--model-path", default=NOLEAK_CKPT)
    parser.add_argument(
        "--out-dir",
        default=os.path.join(_SCRIPT_DIR, "test_results"),
    )
    parser.add_argument(
        "--tag",
        default="noleak",
        help="结果子目录前缀：ablation_<tag>_<mode>，避免覆盖旧 ablation_*",
    )
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help="逗号分隔，可选 full,no_fusion,no_inter,substruct_only",
    )
    parser.add_argument("--subset-size", type=int, default=None, help="Optional smoke subset")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="已有 test_metrics.json 时也重测",
    )
    args = parser.parse_args()

    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    unknown = [m for m in modes if m not in ALL_MODES]
    if unknown or not modes:
        raise SystemExit(f"未知消融模式 {unknown or modes}，允许 {ALL_MODES}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"checkpoint 不存在: {args.model_path}")

    config = get_test_config(profile="full")
    config["model_path"] = args.model_path
    if args.subset_size:
        config["subset_size"] = args.subset_size
    if args.batch_size and args.batch_size > 0:
        config["batch_size"] = args.batch_size
    config["ablation_mode"] = "full"
    config["results_dir"] = os.path.join(args.out_dir, f"ablation_{args.tag}_full")
    os.makedirs(config["results_dir"], exist_ok=True)

    tester = DDITester(config)
    rows = []
    for mode in modes:
        tester.model.set_ablation_mode(mode)
        tester.config["ablation_mode"] = mode
        tester.config["results_dir"] = os.path.join(args.out_dir, f"ablation_{args.tag}_{mode}")
        os.makedirs(tester.config["results_dir"], exist_ok=True)
        done_metrics = os.path.join(tester.config["results_dir"], "test_metrics.json")
        if os.path.exists(done_metrics) and not args.force:
            logger.info("===== Skip finished mode: %s (%s) =====", mode, done_metrics)
            with open(done_metrics, encoding="utf-8") as f:
                metrics = json.load(f)
        else:
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
    summary_path = os.path.join(args.out_dir, f"ablation_{args.tag}_comparison.csv")
    summary.to_csv(summary_path, index=False)
    logger.info("Ablation comparison saved: %s", summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
