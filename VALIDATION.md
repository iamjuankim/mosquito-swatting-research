# Historical verification and current public checks

The following are facts reported by the existing runs, not newly executed package validations:

| Execution | Existing evidence | Recorded result |
|---|---|---|
| Stage8 | `validation_results.json` | 48 conditions, 9,600 draw pairs; checkpoint/array reconciliation, legacy hashes, finite/infinite time handling and paired summaries checked. Scientific H2 verdict deferred. |
| Stage9G run_04 | `gate_summary.json`, `cfi_gate_results.json`, `sentinel_exact_comparison.json` | CFI0–4 PASS. Four baseline sentinel schemas and all arrays, including float64 bits, matched Stage7. This is a four-sentinel check, not proof over every condition. |
| Stage9H P3 | `reuse_and_baseline_gate.json`, `p3_verification.json`, `p3_resource_gate.json` | Recorded PASS; 1,200 draw rows / 2,400 condition rows / six Rule G rows; 2,364 new and 36 reused L checkpoints. Existing B Delta_7 draw values, intervals and classifications matched Stage8. |

Stage9H recorded maximum uniform recovery error 2.498001805406602e-16, maximum tail residual fraction of existing tolerance 0.06385992556912397, and zero maximum independent-drive error/survival increase. The full posterior was not scanned, historical full RNG arrays were unavailable for direct comparison, and overall H2 was not rejudged.

P3 necessity was not fixed before P2 results were read. P3 must be described as exploratory robustness analysis. L is a declared boundary scenario, not an observed biological pre-start stimulus. All six local decrease classifications remain for fixed draws/MC a; +2 MC SE changes the original/calibrated n=1.2 blocks to exclusion. This diagnostic is not a new posterior or MC confidence interval.

Current public-package checks are restricted to hash/size/allowlist checks, static syntax and recorded source identity comparisons, plus recalculation of summary tables from included CSVs. `provenance/public_checks.json` records these separately. It does not confer PASS on production simulations or historical Gates.
