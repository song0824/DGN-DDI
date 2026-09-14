"""给已有 interpret_results 补 split / 双向 logit，不重画热力图。"""
import json
import os

import pandas as pd

from interpretability import _pair_split_membership

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(_SCRIPT_DIR, "interpret_results")


def _csvs():
    bank = os.path.join(_SCRIPT_DIR, "drugbank")
    return [
        os.path.join(bank, "fold0", "train.csv"),
        os.path.join(bank, "fold0", "val.csv"),
        os.path.join(bank, "fold0", "test.csv"),
    ]


def _score_lookup():
    path = os.path.join(_OUT, "positive_case_candidates.csv")
    lookup = {}
    if not os.path.exists(path):
        return lookup
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        lookup[(str(r["head"]), str(r["tail"]))] = r
    return lookup


def main():
    csvs = _csvs()
    scores = _score_lookup()
    updated = 0
    for root, _dirs, files in os.walk(_OUT):
        if "case_summary.json" not in files:
            continue
        path = os.path.join(root, "case_summary.json")
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        h, t = rec.get("head"), rec.get("tail")
        if not h or not t:
            continue
        rec["split"] = _pair_split_membership(csvs, h, t)
        hit = scores.get((h, t))
        if hit is not None:
            if pd.notna(hit.get("reverse_logit")):
                rec["reverse_logit"] = float(hit["reverse_logit"])
            rec["logit"] = float(hit["logit"]) if pd.notna(hit.get("logit")) else rec.get("logit")
            rec["prediction_positive"] = bool(rec.get("logit", 0) > 0)
        elif rec.get("reverse_logit") is None:
            rec["reverse_logit"] = None
        rec.setdefault("note", "")
        extra = "必须同时看两个方向的 logit；alignment_recall 不是预测准确率；请标明 train/val/test。"
        if extra not in str(rec.get("note", "")):
            rec["note"] = (str(rec.get("note") or "") + extra).strip()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        updated += 1

    align_path = os.path.join(_OUT, "mechanism_alignment.csv")
    if os.path.exists(align_path):
        df = pd.read_csv(align_path)
        splits, revs, in_tests = [], [], []
        for _, r in df.iterrows():
            info = _pair_split_membership(csvs, str(r["head"]), str(r["tail"]))
            splits.append("|".join(info.get("splits") or []))
            hit = scores.get((str(r["head"]), str(r["tail"])))
            revs.append(float(hit["reverse_logit"]) if hit is not None and pd.notna(hit.get("reverse_logit")) else r.get("reverse_logit"))
            in_tests.append(info.get("in_test"))
        df["splits"] = splits
        df["reverse_logit"] = revs
        df["in_test"] = in_tests
        df.to_csv(align_path, index=False, encoding="utf-8-sig")

    index_path = os.path.join(_OUT, "index.json")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        for rec in index.get("cases") or []:
            h, t = rec.get("head"), rec.get("tail")
            if not h or not t:
                continue
            rec["split"] = _pair_split_membership(csvs, h, t)
            hit = scores.get((h, t))
            if hit is not None and pd.notna(hit.get("reverse_logit")):
                rec["reverse_logit"] = float(hit["reverse_logit"])
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"Updated {updated} case_summary.json files")


if __name__ == "__main__":
    main()
