# Baselines

只提交本仓库补丁，不要把上游克隆整目录推进 git。

- 可跟踪：`prepare_ssiddi_data.py`、`ssi_ddi/train_eval_fold0.py`、`ssi_ddi/models.py`（readout 注释/替换）。
- 不要提交：`ssi_ddi/.git/`、`ssi_ddi/data/`、权重、`__pycache__`。
- 不要执行 `git add baselines/` 或仓库根目录的 `git add .`。
- 对外名称：`SSI-DDI-like (softmax-gated add-pool)`，不要写“与 SSI-DDI 论文完全同一 fold”。
- 无泄漏 val：先跑 `drugbank_test/make_noleak_val.py`，再 `python prepare_ssiddi_data.py --val-csv .../val_noleak.csv`。当前 80 epoch 结果仍用了泄漏 val，默认不重训。
