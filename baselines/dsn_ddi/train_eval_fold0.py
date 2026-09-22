"""Adapter stub for running DSN-DDI on local DrugBank fold0.

The upstream DSN-DDI repository is not vendored here. This script documents the
expected protocol and fails loudly until an upstream checkout is supplied and a
thin conversion layer is implemented.
"""
from __future__ import annotations

import argparse
import os
import sys

BASELINES_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASELINES_ROOT not in sys.path:
    sys.path.insert(0, BASELINES_ROOT)

from common import FOLD0_DIR, TEST_RESULTS_DIR, ensure_noleak_val  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", default="", help="Path to an external DSN-DDI checkout.")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(TEST_RESULTS_DIR, "dsn_ddi_fold0_noleak"),
        help="Where DSN-DDI metrics should be written.",
    )
    parser.add_argument("--neg-ent", type=int, default=3)
    args = parser.parse_args()

    train_csv = os.path.join(FOLD0_DIR, "train.csv")
    val_csv = ensure_noleak_val()
    test_csv = os.path.join(FOLD0_DIR, "test.csv")

    msg = f"""
DSN-DDI adapter is intentionally not executed yet.

Expected protocol:
  upstream-dir: {args.upstream_dir or '<required external checkout>'}
  train csv:    {train_csv}
  val csv:      {val_csv}
  test csv:     {test_csv}
  final neg_ent:{args.neg_ent}
  output dir:   {args.out_dir}

Do not cite a DSN-DDI row until this script writes:
  {os.path.join(args.out_dir, 'test_metrics_neg3.json')}

If upstream code must be changed to run on this fold, cite the row as
DSN-DDI-like and describe the adapter in the thesis.
"""
    raise SystemExit(msg)


if __name__ == "__main__":
    main()
