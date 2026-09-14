"""核对 fold0 与 SSI-DDI / 公开 DrugBank 86 类语料是否同一套。"""
import json
import os

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BANK = os.path.join(_SCRIPT_DIR, "drugbank")


def _triples(path):
    df = pd.read_csv(path, usecols=["d1", "d2", "type"])
    return set(
        zip(df["d1"].astype(str), df["d2"].astype(str), df["type"].astype(str))
    )


def main():
    ddis = _triples(os.path.join(_BANK, "ddis.csv"))
    train = _triples(os.path.join(_BANK, "fold0", "train.csv"))
    val = _triples(os.path.join(_BANK, "fold0", "val.csv"))
    test = _triples(os.path.join(_BANK, "fold0", "test.csv"))
    smiles = pd.read_csv(os.path.join(_BANK, "drug_smiles.csv"))
    ddis_df = pd.read_csv(os.path.join(_BANK, "ddis.csv"), usecols=["d1", "d2", "type"])
    drugs = set(ddis_df["d1"].astype(str)) | set(ddis_df["d2"].astype(str))

    lit = {
        "source": "SSI-DDI / DSN-DDI / TDC DrugBank multi-typed DDI",
        "n_drugs": 1706,
        "n_types": 86,
        "n_pos_ddi": 191808,
        "note": "SSI-DDI 官方仓库不附带划分文件，只约定 data/ddi_{training,validation,test}.csv + drug_smiles.csv + ddis.csv",
    }
    local = {
        "n_drugs": len(drugs),
        "n_smiles": int(len(smiles)),
        "n_types": int(ddis_df["type"].nunique()),
        "n_pos_ddi": len(ddis),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "train_plus_test_equals_ddis": (train | test) == ddis,
        "train_test_overlap": len(train & test),
        "val_in_train": len(val & train),
        "val_in_test": len(val & test),
        "val_only": len(val - train - test),
        "train_ratio": round(len(train) / len(ddis), 4),
        "test_ratio": round(len(test) / len(ddis), 4),
        "type_id_range": [int(ddis_df["type"].min()), int(ddis_df["type"].max())],
    }
    same_corpus = (
        local["n_drugs"] == lit["n_drugs"]
        and local["n_types"] == lit["n_types"]
        and local["n_pos_ddi"] == lit["n_pos_ddi"]
    )
    report = {
        "literature_standard": lit,
        "local_fold0": local,
        "same_corpus_as_ssiddi_drugbank": same_corpus,
        "same_official_ssiddi_fold": False,
        "conclusion": (
            "语料与 SSI-DDI/DSN-DDI/TDC 公开 DrugBank（1706 药 / 86 类 / 191808 正样本）一致；"
            "fold0 为 train:test=80:20（train∪test=全集、无重叠），val 从全集另抽，与 train/test 都有交集。"
            "SSI-DDI 仓库不含官方 fold 文件，因此不能声称“同一官方划分”，"
            "但可以在本 fold0 上复现 SSI-DDI，与 DGN-DDI 做同划分对照。"
        ),
        "how_to_cite": (
            "对照表请写：同一 DrugBank 86 类语料、本仓库 fold0 划分上复现 SSI-DDI-like"
            "（softmax-gated add-pool，非官方 SAGPooling）；"
            "不要写“与 SSI-DDI 论文完全同一 fold”。"
            "旧 val ∩ test = 3152，主表勿把泄漏 val 选模的数字写成官方复现。"
        ),
    }
    out = os.path.join(_SCRIPT_DIR, "test_results", "split_comparison.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
