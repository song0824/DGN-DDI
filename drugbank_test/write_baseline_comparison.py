"""汇总 DGN-DDI 主结果 / 消融 / SSI-DDI-like fold0 对照表。

主表只把「同一 fold0 + 同一 neg_ent + 同一 AUPR 定义」标成可并排。
未完成同协议重评前，SSI-DDI-like 的 AUPR/Acc/F1 标为不可比。
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "test_results", "baseline_comparison.csv")
OUT_MAIN = os.path.join(ROOT, "test_results", "baseline_comparison_main.csv")

HOW_TO_CITE = (
    "同一 DrugBank 86 类语料、本仓库 fold0 划分上复现 SSI-DDI-like"
    "（softmax-gated add-pool，非官方 SAGPooling）；"
    "不要写“与 SSI-DDI 论文完全同一 fold”。"
)
AUPR_AP = "average_precision_score"
AUPR_TRAP = "sklearn.auc(recall, precision) trapezoid"


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _row(name, metrics, *, neg_ent, aupr_definition, comparable, note=""):
    if not metrics:
        return {
            "model": name,
            "auc": None,
            "aupr": None,
            "accuracy": None,
            "f1_score": None,
            "precision": None,
            "recall": None,
            "neg_ent": neg_ent,
            "aupr_definition": aupr_definition,
            "comparable_to_dgn_main": False,
            "note": note or "not ready",
        }
    acc = metrics.get("accuracy")
    if acc is None:
        acc = metrics.get("acc")
    return {
        "model": name,
        "auc": metrics.get("auc"),
        "aupr": metrics.get("aupr"),
        "accuracy": acc,
        "f1_score": metrics.get("f1_score"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "neg_ent": metrics.get("neg_ent", neg_ent),
        "aupr_definition": metrics.get("aupr_definition", aupr_definition),
        "comparable_to_dgn_main": comparable,
        "note": note,
    }


def main():
    paper = _load_json(os.path.join(ROOT, "test_results", "paper_lock_full", "test_metrics.json"))
    paper_neg1 = _load_json(os.path.join(ROOT, "test_results", "paper_lock_neg1", "test_metrics.json"))
    ablate = os.path.join(ROOT, "test_results", "ablation_fullset")
    ssi_neg3 = _load_json(os.path.join(ROOT, "test_results", "ssi_ddi_fold0", "test_metrics_neg3.json"))
    ssi_orig = _load_json(os.path.join(ROOT, "test_results", "ssi_ddi_fold0", "test_metrics.json"))
    sub_fair = _load_json(os.path.join(ablate, "ablation_substruct_only_fair", "test_metrics.json"))
    sub_old = _load_json(os.path.join(ablate, "ablation_substruct_only", "test_metrics.json"))

    rows = [
        _row(
            "DGN-DDI (full, paper_lock)",
            paper,
            neg_ent=3,
            aupr_definition=AUPR_AP,
            comparable=True,
            note="fold0 test; main protocol; " + HOW_TO_CITE,
        ),
        _row(
            "DGN-DDI w/o Inter",
            _load_json(os.path.join(ablate, "ablation_no_inter", "test_metrics.json")),
            neg_ent=3,
            aupr_definition=AUPR_AP,
            comparable=True,
            note="inference ablation; same protocol as DGN main",
        ),
        _row(
            "DGN-DDI w/o Fusion (= atom_only)",
            _load_json(os.path.join(ablate, "ablation_no_fusion", "test_metrics.json")),
            neg_ent=3,
            aupr_definition=AUPR_AP,
            comparable=True,
            note="inference ablation; atom_only 与 no_fusion 同一张量路径，只报一行",
        ),
        _row(
            "DGN-DDI substruct_only (fair residual)",
            sub_fair,
            neg_ent=3,
            aupr_definition=AUPR_AP,
            comparable=bool(sub_fair),
            note=(
                "inference ablation; keeps atom residual, fusion increment from substructure only"
                if sub_fair
                else "not ready — do not cite residual-free 0.56 AUC"
            ),
        ),
        _row(
            "SSI-DDI-like (softmax-gated add-pool, neg_ent=3)",
            ssi_neg3,
            neg_ent=3,
            aupr_definition=AUPR_AP,
            comparable=bool(ssi_neg3),
            note=(
                "same fold0 + same neg_ent=3 + average_precision_score; "
                "not official SAGPooling; val∩test=3152 leaked into original checkpoint selection"
                if ssi_neg3
                else "pending re-eval of existing weights under DGN test protocol"
            ),
        ),
        _row(
            "SSI-DDI-like (original run, neg_ent=1, not comparable)",
            ssi_orig,
            neg_ent=1,
            aupr_definition=ssi_orig.get("aupr_definition", AUPR_TRAP) if ssi_orig else AUPR_TRAP,
            comparable=False,
            note=(
                "do not put AUPR/Acc/F1 beside DGN main; different class ratio; "
                "SAGPooling replaced; val∩test=3152 used for best checkpoint"
            ),
        ),
        _row(
            "DGN-DDI (appendix, neg_ent=1)",
            paper_neg1,
            neg_ent=1,
            aupr_definition=AUPR_AP,
            comparable=False,
            note="appendix only; same weights as paper_lock, 1 neg/pos to mirror SSI-DDI-like original run",
        ),
    ]
    if sub_old and not sub_fair:
        rows.append(
            _row(
                "DGN-DDI substruct_only (UNFAIR, residual dropped — do not cite)",
                sub_old,
                neg_ent=3,
                aupr_definition=AUPR_AP,
                comparable=False,
                note="legacy path dropped atom residual; 0.56 AUC is an implementation artifact",
            )
        )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    main_df = df[df["comparable_to_dgn_main"] == True]  # noqa: E712
    main_df.to_csv(OUT_MAIN, index=False)
    print(df.to_string(index=False))
    print("Wrote", OUT)
    print("Wrote main-protocol table", OUT_MAIN)
    print("how_to_cite:", HOW_TO_CITE)


if __name__ == "__main__":
    main()
