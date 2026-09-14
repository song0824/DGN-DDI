"""筛有已知机制的药物对，并同时报两个方向的 logit / 划分。

只读 paper_lock 权重。禁止只按 logit>0 挤选出例。
"""
import csv
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch

from data_preprocessing import DDIDataLoader, DrugDataset
from interpretability import (
    KNOWN_MECHANISMS,
    _build_model,
    _find_pair_relation,
    _pair_split_membership,
    _score_logit,
)
from transductive_test import get_test_config

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CANDIDATE_PAIRS = [
    ("DB00945", "DB00682", "aspirin_warfarin"),
    ("DB00682", "DB00945", "warfarin_aspirin"),
    ("DB00316", "DB00682", "acetaminophen_warfarin"),
    ("DB00758", "DB00682", "clopidogrel_warfarin"),
    ("DB01076", "DB00682", "atorvastatin_warfarin"),
    ("DB00213", "DB00682", "pantoprazole_warfarin"),
    ("DB01118", "DB00682", "amiodarone_warfarin"),
    ("DB00472", "DB00682", "fluoxetine_warfarin"),
    ("DB00501", "DB00682", "cimetidine_warfarin"),
    ("DB00338", "DB00682", "omeprazole_warfarin"),
    ("DB00641", "DB00682", "simvastatin_warfarin"),
    ("DB00176", "DB00682", "fluvoxamine_warfarin"),
    ("DB00537", "DB00682", "ciprofloxacin_warfarin"),
    ("DB01026", "DB00682", "ketoconazole_warfarin"),
    ("DB00641", "DB01118", "simvastatin_amiodarone"),
    ("DB00381", "DB00641", "amlodipine_simvastatin"),
    ("DB00313", "DB00555", "valproate_lamotrigine"),
    ("DB00563", "DB00945", "methotrexate_aspirin"),
    ("DB00682", "DB01118", "warfarin_amiodarone"),
    ("DB00682", "DB00472", "warfarin_fluoxetine"),
    ("DB00682", "DB00338", "warfarin_omeprazole"),
    ("DB00682", "DB00537", "warfarin_ciprofloxacin"),
    ("DB00338", "DB00682", "omeprazole_warfarin"),
    ("DB00641", "DB00381", "simvastatin_amlodipine"),
    ("DB01118", "DB00641", "amiodarone_simvastatin"),
]


def main():
    out_csv = os.path.join(_SCRIPT_DIR, "interpret_results", "positive_case_candidates.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    config = get_test_config(profile="full")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddi_loader = DDIDataLoader()
    dataset = DrugDataset(config["test_csv"], ddi_loader, neg_ent=0, shuffle=False)
    model, config, _ckpt = _build_model(config, ddi_loader, device)
    model.eval()

    bank = os.path.join(_SCRIPT_DIR, "drugbank")
    csvs = [
        os.path.join(bank, "fold0", "train.csv"),
        os.path.join(bank, "fold0", "val.csv"),
        os.path.join(bank, "fold0", "test.csv"),
    ]

    rows = []
    seen = set()
    for h, t, name in CANDIDATE_PAIRS:
        key = (h, t)
        if key in seen:
            continue
        seen.add(key)
        if h not in ddi_loader.drug_graph_dict or t not in ddi_loader.drug_graph_dict:
            continue
        rel = _find_pair_relation(csvs, h, t)
        if rel is None:
            continue
        logit = _score_logit(dataset, model, device, h, t, rel)
        if logit is None:
            continue
        reverse_logit = _score_logit(dataset, model, device, t, h, rel)
        split_info = _pair_split_membership(csvs, h, t)
        mech = KNOWN_MECHANISMS.get(frozenset({h, t}))
        prob = float(1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50))))
        rows.append({
            "case_name": name,
            "head": h,
            "tail": t,
            "relation": str(rel),
            "logit": logit,
            "reverse_logit": reverse_logit,
            "probability": prob,
            "prediction_positive": bool(logit > 0),
            "reverse_prediction_positive": bool(reverse_logit is not None and reverse_logit > 0),
            "has_known_mechanism": bool(mech),
            "pair_name": (mech or {}).get("pair_name", ""),
            "in_test": bool(split_info.get("in_test")),
            "in_train": bool(split_info.get("in_train")),
            "in_val": bool(split_info.get("in_val")),
            "forward_splits": "|".join(split_info.get("forward_splits") or []),
            "reverse_splits": "|".join(split_info.get("reverse_splits") or []),
        })
        print(
            f"{name}: logit={logit:.4f} rev={reverse_logit} test={split_info.get('in_test')} mech={bool(mech)}",
            flush=True,
        )

    rows.sort(key=lambda r: (not r["has_known_mechanism"], not r["in_test"], -abs(r["logit"])))
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    # 选例：已知机制 + 尽量在 test；两个方向都保留，不按 logit>0 过滤。
    grouped = {}
    for r in rows:
        key = frozenset({r["head"], r["tail"]})
        grouped.setdefault(key, []).append(r)
    selected_pairs = []
    for key, pair_rows in grouped.items():
        if not any(r["has_known_mechanism"] for r in pair_rows):
            continue
        selected_pairs.append({
            "pair_name": pair_rows[0]["pair_name"],
            "in_test": any(r["in_test"] for r in pair_rows),
            "directions": pair_rows,
        })
    selected_pairs.sort(key=lambda p: (not p["in_test"], p["pair_name"]))
    # 开题案例必须保留两个方向，不能被 in_test 排序挤掉。
    must_keep = {"阿司匹林–华法林", "aspirin_warfarin"}
    head = [p for p in selected_pairs if p["pair_name"] in must_keep or any(d.get("case_name") in ("aspirin_warfarin", "warfarin_aspirin") for d in p["directions"])]
    rest = [p for p in selected_pairs if p not in head]
    selected_pairs = head + rest
    pick_path = os.path.join(_SCRIPT_DIR, "interpret_results", "positive_cases_selected.json")
    with open(pick_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_candidates": len(rows),
                "selection_rule": "known mechanism; keep both directions; do not filter logit>0",
                "paper_figure_rule": "only use prediction_positive=True AND known mechanism; always show both directions for aspirin-warfarin",
                "selected": selected_pairs[:6],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Wrote {out_csv}", flush=True)
    print(f"Selected {min(6, len(selected_pairs))} unordered pairs -> {pick_path}", flush=True)
    for p in selected_pairs[:6]:
        print(f"  {p['pair_name']} in_test={p['in_test']} dirs={len(p['directions'])}", flush=True)


if __name__ == "__main__":
    main()
