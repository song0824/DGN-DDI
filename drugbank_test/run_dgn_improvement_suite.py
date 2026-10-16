from __future__ import annotations

import argparse
import os
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def build_commands():
    val_noleak = os.path.join(SCRIPT_DIR, "drugbank", "fold0", "val_noleak.csv")
    cmds = [
        [
            PY, "transductive_train.py",
            "--profile", "full",
            "--fresh",
            "--use-val-noleak",
            "--experiment-name", "dgn_noleak_sigmoid_full",
            "--negative-sampling-scope", "global",
            "--freeze-eval-negatives",
        ],
        [
            PY, "transductive_train.py",
            "--profile", "full",
            "--fresh",
            "--val-csv", val_noleak,
            "--experiment-name", "dgn_noleak_hinge_full",
            "--loss-fn", "hinge",
            "--negative-sampling-scope", "global",
            "--freeze-eval-negatives",
        ],
        [
            PY, "transductive_train.py",
            "--profile", "full",
            "--fresh",
            "--val-csv", val_noleak,
            "--experiment-name", "dgn_noleak_hardneg16",
            "--num-candidates-per-pos", "16",
            "--hard-neg-ratio", "0.5",
            "--negative-sampling-scope", "global",
            "--freeze-eval-negatives",
        ],
        [
            PY, "precompute_bipartite_cache.py",
            "--csvs",
            os.path.join(SCRIPT_DIR, "drugbank", "fold0", "train.csv"),
            val_noleak,
            os.path.join(SCRIPT_DIR, "drugbank", "fold0", "test.csv"),
        ],
        [
            PY, "dgn_ecfp_fusion.py",
            "--out-dir", os.path.join(SCRIPT_DIR, "test_results", "dgn_ecfp_fusion"),
        ],
        [
            PY, "train_dgn_hybrid_head.py",
            "--mode", "hybrid",
            "--out-dir", os.path.join(SCRIPT_DIR, "test_results", "dgn_hybrid_head"),
        ],
        [
            PY, "train_dgn_hybrid_head.py",
            "--mode", "distill",
            "--out-dir", os.path.join(SCRIPT_DIR, "test_results", "dgn_distill_head"),
        ],
    ]
    return cmds


def main():
    parser = argparse.ArgumentParser(description="Run the staged DGN improvement experiment suite")
    parser.add_argument("--execute", action="store_true", help="Actually run the commands instead of only printing them")
    parser.add_argument("--start", type=int, default=1, help="1-based index of the first command to run")
    parser.add_argument("--end", type=int, default=0, help="1-based index of the last command to run (0=all)")
    args = parser.parse_args()

    commands = build_commands()
    start = max(1, int(args.start))
    end = len(commands) if args.end <= 0 else min(len(commands), int(args.end))
    selected = list(enumerate(commands, start=1))[start - 1:end]
    for idx, cmd in selected:
        rendered = " ".join(f'"{x}"' if " " in x else x for x in cmd)
        print(f"[{idx}] {rendered}")
        if args.execute:
            subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)


if __name__ == "__main__":
    main()
