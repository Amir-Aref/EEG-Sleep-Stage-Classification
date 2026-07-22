# Phase 2 Baseline Validation

This directory preserves the original Phase 2 environment and data
characteristics before the project rebuild.

## Validation policy

The SHA-256 value stored in `phase2_data_baseline.json` is an exact
byte-level fingerprint. It is useful for archival purposes, but it is
not the final criterion for numerical regression testing.

Floating-point outputs may differ at machine precision because of:

- CSV serialization and parsing
- NumPy floating-point operations
- scikit-learn transformations
- dependency and platform differences

Therefore, numerical pipeline outputs are considered equivalent when:

- shapes are identical
- column names and order are identical
- categorical values are identical
- class distributions are identical
- numerical values satisfy:

  - relative tolerance: 1e-10
  - absolute tolerance: 1e-12

The Phase 2 refactor comparison passed these semantic checks. The only
observed differences were approximately 1.78e-15 in
`log_signal_energy`, which is below the defined tolerance.
