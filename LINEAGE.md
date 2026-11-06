# Execution lineage

All paths below are relative source labels; personal absolute locations are removed.

1. Existing `mq10`: current detection/flow/escape/simulation source, `phases/p3b/run.py`, original judge and H2 results. Original selection metadata records commit `03db936286bdec6e2378d6ae8b7ef14d50e6c97e` with `run_start_dirty=true`; this commit alone cannot reconstruct that execution.
2. Stage7: final folder's `executor.py` + `common.py` + `stable_threshold.py`, frozen `source/run_original.py` and `source/swat`. Production `conditions.json`, `input_manifest.json`, `code_manifest.json` are kept. Earlier failed/preflight copies are excluded. Production signature is retained in the manifest.
3. Stage8: paired analysis of existing H2 legacy outputs versus Stage7 stable checkpoints. Its own analysis code and output tables are retained; no simulator rerun occurred in this step.
4. Stage9G: only `run_04` is the verified P1/P2 execution. Uses its own `baseline.py`, `variant_L.py`, strict sampler, `extracted.py` and dependency copies. Earlier upper-folder/run_02/run_03 attempts must not be used as its code. Four baseline sentinels were newly calculated; 36 L batches include four P1 L batches reused in P2.
5. Stage9H: `run_p3.py` + `p3_core.py` use the run_04 numerical code, its own dependency copies and local judge. Reuses 36 run_04 L batches and 2,400 Stage7 B batches; adds 2,364 L batches. `analyze_p3.py`/`finalize_p3.py` analyze and verify stored results. `p3_core_changes.diff` records original execution changes; release location redactions are separately listed in source_lineage.json.

`provenance/source_lineage.json` binds every selected source file to its release copy with both SHA256 values. Public wrappers do not replace any historical driver. Historical code manifests are checked against original source hashes, rather than claiming that redacted bytes generated the original results.
