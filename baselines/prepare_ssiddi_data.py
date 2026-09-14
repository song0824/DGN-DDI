"""把本仓库 fold0 写成 SSI-DDI 官方脚本期望的 data/*.csv。"""
import argparse
import os
import shutil

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(ROOT), "drugbank_test", "drugbank")
DST = os.path.join(ROOT, "ssi_ddi", "data")


def _pos_csv(src, dst):
    df = pd.read_csv(src, usecols=["d1", "d2", "type"])
    df.to_csv(dst, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-csv",
        default="",
        help="Optional leak-free val CSV (e.g. drugbank_test/drugbank/fold0/val_noleak.csv)",
    )
    args = parser.parse_args()

    os.makedirs(DST, exist_ok=True)
    _pos_csv(os.path.join(SRC, "ddis.csv"), os.path.join(DST, "ddis.csv"))
    _pos_csv(os.path.join(SRC, "fold0", "train.csv"), os.path.join(DST, "ddi_training.csv"))
    val_src = args.val_csv or os.path.join(SRC, "fold0", "val.csv")
    _pos_csv(val_src, os.path.join(DST, "ddi_validation.csv"))
    _pos_csv(os.path.join(SRC, "fold0", "test.csv"), os.path.join(DST, "ddi_test.csv"))
    shutil.copy2(
        os.path.join(SRC, "drug_smiles.csv"),
        os.path.join(DST, "drug_smiles.csv"),
    )
    print("Prepared SSI-DDI data in", DST)
    print("val source:", val_src)


if __name__ == "__main__":
    main()
