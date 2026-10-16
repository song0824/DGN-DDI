"""汇总 DGN-DDI 主结果 / 消融 / fold0 baseline 对照表。

主表只把「同一 fold0 + 同一 neg_ent + 同一 AUPR 定义」标成可并排。
未完成同协议重评前，baseline 的 AUPR/Acc/F1 标为不可比。
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "test_results", "baseline_comparison.csv")
OUT_MAIN = os.path.join(ROOT, "test_results", "baseline_comparison_main.csv")

HOW_TO_CITE = (
    "同一 DrugBank 86 类语料、本仓库 fold0 划分；"
    "val_noleak 选模、neg_ent=3 测试、AUPR=average_precision_score；"
    "不要抄文献表格或声称官方同一 fold。"
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
    dgn_noleak = _load_json(os.path.join(ROOT, "test_results", "dgn_noleak_sigmoid_full", "test_metrics.json"))
    dgn_fusion = _load_json(os.path.join(ROOT, "test_results", "dgn_ecfp_fusion", "test_metrics.json"))
    dgn_hybrid = _load_json(os.path.join(ROOT, "test_results", "dgn_hybrid_head", "test_metrics.json"))
    dgn_distill = _load_json(os.path.join(ROOT, "test_results", "dgn_distill_head", "test_metrics.json"))
    ablate = os.path.join(ROOT, "test_results", "ablation_fullset")
    ecfp_neg3 = _load_json(os.path.join(ROOT, "test_results", "ecfp_mlp_fold0_noleak", "test_metrics_neg3.json"))
    gat_neg3 = _load_json(os.path.join(ROOT, "test_results", "gat_ddi_fold0_noleak", "test_metrics_neg3.json"))
    dsn_neg3 = _load_json(os.path.join(ROOT, "test_results", "dsn_ddi_fold0_noleak", "test_metrics_neg3.json"))
    ssi_noleak_neg3 = _load_json(os.path.join(ROOT, "test_results", "ssi_ddi_fold0_noleak", "test_metrics_neg3.json"))
    ssi_noleak_neg1 = _load_json(os.path.join(ROOT, "test_results", "ssi_ddi_fold0_noleak", "test_metrics.json"))
    ssi_leaked_neg3 = _load_json(os.path.join(ROOT, "test_results", "ssi_ddi_fold0", "test_metrics_neg3.json"))
    ssi_leaked_neg1 = _load_json(os.path.join(ROOT, "test_results", "ssi_ddi_fold0", "test_metrics.json"))
    sub_fair = _load_json(os.path.join(ablate, "ablation_substruct_only_fair", "test_metrics.json"))
    sub_old = _load_json(os.path.join(ablate, "ablation_substruct_only", "test_metrics.json"))

    rows = [
        _row(
            "ECFP4-MLP (noleak val, neg_ent=3)",
            ecfp_neg3,
            neg_ent=3,
            aupr_definition=AUPR_AP,
            comparable=bool(ecfp_neg3),
            note=(
                "Morgan radius=2 2048-bit local baseline; not official DeepDDI; " + HOW_TO_CITE
                if ecfp_neg3
                else "pending: python baselines/ecfp_mlp/train_eval_fold0.py, then eval-only --neg-ent 3"
            ),
        ),
        _row(
            "GAT-DDI (atom-only RESCAL, noleak val, neg_ent=3)",
            gat_neg3,
            neg_ent=3,
            aupr_definition=AUPR_AP,
            comparable=bool(gat_neg3),
            note=(
                "atom-only GAT encoder + RESCAL-style scorer; trained from scratch; " + HOW_TO_CITE
                if gat_neg3
                else "pending: python baselines/gat_ddi/train_eval_fold0.py, then eval-only --neg-ent 3"
            ),
        ),
        _row(
            "DGN-DDI (full, paper_lock)",
            paper,
            neg_ent=3,
            aupr_definition=AUPR_AP,
            comparable=True,
            note="fold0 test; main protocol; " + HOW_TO_CITE,
        ),
        _row(
            "DGN-DDI (noleak val, improved protocol)",
            dgn_noleak,
            neg_ent=3,
            aupr_definition=AUPR_AP,
            comparable=bool(dgn_noleak),
            note=(
                "noleak val + filtered global negatives + frozen eval negatives; " + HOW_TO_CITE
                if dgn_noleak
                else "pending: transductive_train.py --use-val-noleak --negative-sampling-scope global"
            ),
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
            "SSI-DDI-like (noleak val, neg_ent=3)",
            ssi_noleak_neg3,
            neg_ent=3,
            aupr_definition=AUPR_AP,
            comparable=bool(ssi_noleak_neg3),
            note=(
                "softmax-gated add-pool; not official SAGPooling or official SSI-DDI fold; " + HOW_TO_CITE
                if ssi_noleak_neg3
                else "pending noleak re-eval; keep leaked SSI rows in appendix only"
            ),
        ),
        _row(
            "DGN + ECFP late fusion",
            dgn_fusion,
            neg_ent=3,
            aupr_definition=AUPR_AP,
            comparable=bool(dgn_fusion),
            note=(
                "late fusion on val_noleak; preserves original paper_lock checkpoint"
                if dgn_fusion
                else "pending: python drugbank_test/dgn_ecfp_fusion.py"
            ),
        ),
        _row(
            "DGN hybrid head",
            dgn_hybrid,
            neg_ent=3,
            aupr_definition=AUPR_AP,
            comparable=bool(dgn_hybrid),
            note=(
                "frozen DGN pair features + lightweight relation-aware head"
                if dgn_hybrid
                else "pending: python drugbank_test/train_dgn_hybrid_head.py --mode hybrid"
            ),
        ),
        _row(
            "DGN distill head",
            dgn_distill,
            neg_ent=3,
            aupr_definition=AUPR_AP,
            comparable=bool(dgn_distill),
            note=(
                "hybrid head with ECFP teacher distillation signal"
                if dgn_distill
                else "pending: python drugbank_test/train_dgn_hybrid_head.py --mode distill"
            ),
        ),
        _row(
            "DSN-DDI-like (noleak val, neg_ent=3)",
            dsn_neg3,
            neg_ent=3,
            aupr_definition=AUPR_AP,
            comparable=bool(dsn_neg3),
            note=(
                "external DSN-DDI adapter; cite as -like if upstream code changed; " + HOW_TO_CITE
                if dsn_neg3
                else "pending: baselines/dsn_ddi/train_eval_fold0.py requires external upstream checkout"
            ),
        ),
        _row(
            "SSI-DDI-like (leaked val, neg_ent=3, appendix)",
            ssi_leaked_neg3,
            neg_ent=3,
            aupr_definition=AUPR_AP,
            comparable=False,
            note=(
                "appendix only; same test protocol but checkpoint selected on old val ∩ test = 3152"
                if ssi_leaked_neg3
                else "not ready"
            ),
        ),
        _row(
            "SSI-DDI-like (noleak val, neg_ent=1, not comparable)",
            ssi_noleak_neg1,
            neg_ent=1,
            aupr_definition=ssi_noleak_neg1.get("aupr_definition", AUPR_AP) if ssi_noleak_neg1 else AUPR_AP,
            comparable=False,
            note=(
                "appendix only; noleak checkpoint, 1 neg/pos — do not put AUPR/F1 beside DGN main"
                if ssi_noleak_neg1
                else "not ready"
            ),
        ),
        _row(
            "SSI-DDI-like (leaked val, neg_ent=1, not comparable)",
            ssi_leaked_neg1,
            neg_ent=1,
            aupr_definition=ssi_leaked_neg1.get("aupr_definition", AUPR_TRAP) if ssi_leaked_neg1 else AUPR_TRAP,
            comparable=False,
            note=(
                "appendix only; different class ratio; val∩test=3152 used for best checkpoint"
                if ssi_leaked_neg1
                else "not ready"
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
