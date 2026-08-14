"""
离线预计算 (d1, d2) 无向药物对的二部图，写入 drugbank/bipartite_cache/，
与训练时 data_preprocessing 中的 BipartiteGraphCache 共用。

用法:
  python precompute_bipartite_cache.py
  python precompute_bipartite_cache.py --csvs drugbank/fold0/train.csv drugbank/fold0/val.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# 预计算脚本默认启用磁盘缓存读写。
os.environ.setdefault("DDI_USE_BIPARTITE_DISK_CACHE", "1")
os.environ.setdefault("DDI_WRITE_BIPARTITE_DISK_CACHE", "1")

from data_preprocessing import (  # noqa: E402
    DDIDataLoader,
    DrugDataset,
    _canonical_drug_pair_key,
    get_bipartite_cache_signature,
)


def _collect_pairs_from_csvs(csv_paths) -> set:
    pairs: set = set()
    for path in csv_paths:
        if not path or not os.path.isfile(path):
            continue
        df = pd.read_csv(path, usecols=["d1", "d2"])
        for a, b in zip(df["d1"], df["d2"]):
            x, y, _ = _canonical_drug_pair_key(a, b)
            pairs.add((x, y))
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Precompute bipartite graph cache for drug pairs")
    ap.add_argument(
        "--csvs",
        nargs="*",
        default=None,
        help="DDI CSVs with columns d1,d2 (default: fold0 train/val/test)",
    )
    args = ap.parse_args()

    bank = os.path.join(_SCRIPT_DIR, "drugbank")
    fold0 = os.path.join(bank, "fold0")
    if args.csvs:
        csvs = args.csvs
    else:
        csvs = [os.path.join(fold0, f) for f in ("train.csv", "val.csv", "test.csv")]

    print("Loading DDIDataLoader...")
    ddi = DDIDataLoader()
    drug_keys = [k for k in ddi.drug_graph_dict.keys() if k != "__metadata__"]
    ds = DrugDataset.__new__(DrugDataset)
    ds.data_loader = ddi
    ds.drug_ids = np.array(drug_keys)
    ds.neg_ent = 1
    ds.return_pos_triples = False
    ds.hard_neg_ratio = 0.7
    ds.tri_list = []  # unused for _get_drug_data

    pairs = _collect_pairs_from_csvs(csvs)
    print(
        f"Unique undirected pairs: {len(pairs)}; "
        f"cache signature={get_bipartite_cache_signature()}; "
        f"cache dir={getattr(ddi.bipartite_cache, 'cache_dir', '')}"
    )

    n_ok, n_skip = 0, 0
    for h, t in tqdm(sorted(pairs), desc="bipartite_cache"):
        hd = DrugDataset._get_drug_data(ds, h)
        td = DrugDataset._get_drug_data(ds, t)
        if hd is None or td is None:
            n_skip += 1
            continue
        out = DrugDataset._create_dual_granularity_b_graph(ds, hd, td, h_id=h, t_id=t)
        if out is not None:
            n_ok += 1
        else:
            n_skip += 1

    print(f"Done. computed={n_ok} skipped={n_skip}")


if __name__ == "__main__":
    main()
