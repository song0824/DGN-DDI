"""从 fold0 train 再切一份 val，保证 val ∩ test = ∅。

不重训。当前 SSI-DDI-like 的 0.914 仍来自泄漏 val 选模，只能当内部 sanity。
若要报“公平复现”，必须用本脚本产出的 val_noleak.csv 重新选 checkpoint。
"""
import json
import os

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BANK = os.path.join(_SCRIPT_DIR, "drugbank")
_FOLD = os.path.join(_BANK, "fold0")


def _triples(df):
    return set(zip(df["d1"].astype(str), df["d2"].astype(str), df["type"].astype(str)))


def main():
    train_df = pd.read_csv(os.path.join(_FOLD, "train.csv"))
    test_df = pd.read_csv(os.path.join(_FOLD, "test.csv"))
    old_val_df = pd.read_csv(os.path.join(_FOLD, "val.csv"))

    train = _triples(train_df)
    test = _triples(test_df)
    old_val = _triples(old_val_df)

    n_val = len(old_val_df)
    sampled = train_df.sample(n=min(n_val, len(train_df)), random_state=42)
    sampled_set = _triples(sampled)
    assert sampled_set.isdisjoint(test), "noleak val still overlaps test"

    out_csv = os.path.join(_FOLD, "val_noleak.csv")
    sampled.to_csv(out_csv, index=False)

    decision = {
        "val_noleak_path": out_csv,
        "n_val_noleak": int(len(sampled)),
        "val_noleak_in_train": int(len(sampled_set & train)),
        "val_noleak_in_test": int(len(sampled_set & test)),
        "old_val_in_test": int(len(old_val & test)),
        "old_val_in_train": int(len(old_val & train)),
        "retrain_now": False,
        "reason": (
            "现有 SSI-DDI-like 80 epoch / best_epoch=79 已用泄漏 val 选模；"
            "本脚本只准备无泄漏 val，不自动重训。"
            "论文主表先用已有权重做 neg_ent=3 重评；正文必须写 val∩test=3152。"
            "若要报公平复现，再用 val_noleak.csv 重选 checkpoint 或固定最后 epoch。"
        ),
        "how_to_retrain": (
            "python baselines/prepare_ssiddi_data.py --val-csv drugbank_test/drugbank/fold0/val_noleak.csv; "
            "然后在 baselines/ssi_ddi 跑 train_eval_fold0.py"
        ),
    }
    out_json = os.path.join(_SCRIPT_DIR, "test_results", "noleak_val_decision.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(decision, f, ensure_ascii=False, indent=2)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print("Wrote", out_csv)
    print("Wrote", out_json)


if __name__ == "__main__":
    main()
