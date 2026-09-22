# Baselines

只提交本仓库补丁，不要把上游克隆整目录推进 git。

- 可跟踪：`prepare_ssiddi_data.py`、`ssi_ddi/train_eval_fold0.py`、`ssi_ddi/models.py`（readout 注释/替换）。
- 不要提交：`ssi_ddi/.git/`、`ssi_ddi/data/`、权重、`__pycache__`。
- 不要执行 `git add baselines/` 或仓库根目录的 `git add .`。
- 对外名称：`SSI-DDI-like (softmax-gated add-pool)`，不要写“与 SSI-DDI 论文完全同一 fold”。
- 无泄漏 val：先跑 `drugbank_test/make_noleak_val.py`，再 `python prepare_ssiddi_data.py --val-csv .../val_noleak.csv`。当前 80 epoch 结果仍用了泄漏 val，默认不重训。

## Midterm baseline protocol

中期主表只放同一协议的行：

- DrugBank 86 类语料，本仓库 `drugbank_test/drugbank/fold0`。
- 用 `val_noleak.csv` 选模；缺失时 baseline 脚本会直接报错，不回退旧泄漏 val。
- 最终测试用 `neg_ent=3`，AUPR 使用 `average_precision_score`。
- 验证集/测试集负样本固定；训练集负样本按 epoch 动态采样。
- 不能把 DGN 推理消融当外部 baseline；不能抄论文数字。

推荐运行顺序：

```powershell
cd D:\DGN-DDI
python drugbank_test\make_noleak_val.py
python baselines\prepare_ssiddi_data.py --val-csv drugbank_test\drugbank\fold0\val_noleak.csv

# Fast local fingerprint baseline.
python baselines\ecfp_mlp\train_eval_fold0.py --n-epochs 40 --neg-ent 1
python baselines\ecfp_mlp\train_eval_fold0.py --eval-only --neg-ent 3 `
  --ckpt drugbank_test\test_results\ecfp_mlp_fold0_noleak\best_model.pth `
  --metrics-name test_metrics_neg3.json

# Atom-only graph baseline.
python baselines\gat_ddi\train_eval_fold0.py --n-epochs 40 --neg-ent 1
python baselines\gat_ddi\train_eval_fold0.py --eval-only --neg-ent 3 `
  --ckpt drugbank_test\test_results\gat_ddi_fold0_noleak\best_model.pth `
  --metrics-name test_metrics_neg3.json

# Existing SSI-DDI-like row.
cd D:\DGN-DDI\baselines\ssi_ddi
python train_eval_fold0.py --n-epochs 80 --neg-ent 1 `
  --out-dir D:\DGN-DDI\drugbank_test\test_results\ssi_ddi_fold0_noleak
python train_eval_fold0.py --eval-only --neg-ent 3 `
  --ckpt D:\DGN-DDI\drugbank_test\test_results\ssi_ddi_fold0_noleak\best_model.pth `
  --out-dir D:\DGN-DDI\drugbank_test\test_results\ssi_ddi_fold0_noleak `
  --metrics-name test_metrics_neg3.json

cd D:\DGN-DDI
python drugbank_test\write_baseline_comparison.py
```

快速 smoke 可先加：

```powershell
--max-train-samples 512 --max-val-samples 256 --max-test-samples 256 --n-epochs 1
```

`baselines/dsn_ddi/` 目前只提供适配入口，不提交上游代码。只有生成
`drugbank_test/test_results/dsn_ddi_fold0_noleak/test_metrics_neg3.json`
之后，才能在论文/中期表中引用 DSN-DDI-like 行。
