# DSN-DDI adapter

This directory is reserved for a local DSN-DDI fold0 adapter.

Rules:

- Do not commit the upstream DSN-DDI repository into this project.
- Put or clone the upstream code outside this repo, then pass its path to the
  adapter script.
- Use the same local protocol as the other baselines:
  `fold0`, `val_noleak.csv` for checkpoint selection, `neg_ent=3` for final
  test metrics, and `average_precision_score` as AUPR.
- If the upstream code is modified, cite the row as `DSN-DDI-like`.

Planned output:

```text
drugbank_test/test_results/dsn_ddi_fold0_noleak/test_metrics_neg3.json
```
