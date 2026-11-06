# Reproducibility scope

## A. Calculations executable from this release

`python tools/reproduce_tables.py` recomputes Stage8 effect intervals/classifications from `h2_effects_by_draw.csv` and Stage9H B/L local Delta_7 intervals, empirical 95% intervals, and ±2 MC SE classification diagnostics from `p3_draw_results.csv`. It calls the actual archived Stage9H `local_judge.py` functions (AST-selected from the original judge). It compares the calculations against the published summary cells with explicit numerical tolerances and writes new CSVs in `generated/`. This is summary reproducibility conditional on the published draw-level estimates. It cannot recover missing trajectory-level covariance or independently validate their MC SE.

`python tools/verify_package.py` checks the explicit release allowlist, byte sizes/SHA256, parses every provided Python file without importing production drivers, and compares copied code hashes to the Stage7, run_04 and P3 historical code manifests. It distinguishes original-source hashes from hashes after privacy redaction. A current-source mismatch with a historical manifest must be reported, never silently accepted as an exact executable match.

## B. Public external inputs required

Raw trajectory and CFD analysis/preparation requires separately obtaining the Dryad 2022/2024 datasets. MATLAB comparison code is linked through Zenodo and is not redistributed. FlyWire/connectome acquisition also requires the original version/export/filter specification and relevant access terms; this release does not bundle those exports. See DATA_SOURCES.md. No end-to-end production result is classified as reproducible solely by downloading public raw data: the exact derived artifacts below are also required.

## C. Full computations not guaranteed reproducible here

Existing H2 simulation, Stage7 production, Stage8 trajectory-level validation, Stage9G P1/P2, and Stage9H P3 need omitted derived initial states (841 rows), flow model/grid, escape bootstrap, exact fitted posterior Zarr and stored baseline/L trajectory checkpoints. `provenance/input_inventory.json` and the Stage7 historical input manifest identify expected sizes/hashes. No archive/DOI for these derived artifacts was established in the inspected records. Do not substitute newly fitted inputs and label the results exact reproduction.

The `archive/mq10` snapshot contains the current existing model source and original H2 result tables, with an explicitly dirty original draw-selection run recorded in `inputs/draws_P1_meta.json`. Exact historical H2 source identity is not certified. Stage7's frozen `source/run_original.py` and `source/swat/` preserve the later baseline's actual source dependencies. Stage8 reads checkpoints and includes manuscript comparison/reporting logic requiring an omitted manuscript; that PDF is not a scientific simulation input. Stage9 drivers additionally depend on historical planning records, manifests, checkpoint sidecars, fixed filesystem layout, Windows audit guards and past deadlines. Some administrative dependencies are deliberately omitted. No full-driver execution command is advertised.

## Parameters and execution identity

- Conditions: Stage7 `conditions.json` (48 model×speed×n combinations). Speeds are 0.5, 1, 2, 3, 4, 6, 8, 12 m/s; n is 0.8, 1, 1.2; original and calibrated branches are separate.
- Fixed 200 coordinates/order: `inputs/draws_P1.csv`, metadata selection seed 20261101. Existing selection is PSIS probability-proportional without replacement, summarized with equal empirical weights; no reselection/reweighting occurs in this release. Exact selection implementation is in archived `phases/p3b/run.py`.
- Simulation RNG: MC seed `[20261105,j,0]` for set a, followed by pick, uniform(N,6), z_a and z; k uses 20261103/j and escape uses 20261104/j. N=4000. Stage9G uses j=0,73,199; P3 uses j=0..199.
- Transformation: preserve original `params_from_draw`, sigma = raw sigma + 1e-3; a_sp/b_sp shape (1,); original session order. Session eta uses raw sigma_sess×z_sess; newly generated MC eta uses raw sigma_sess×MC z. Selected parameter provenance is included separately for run_04 and P3; it does not supply all simulation inputs.
- Rule G: interval is the hull of empirical quantiles and mean ± z×draw SD, not SD/√200. Adjacent effects use alpha=0.05/35, delta=0.02; E uses alpha=0.01. Six local P3 blocks do not change the multiplicity denominator to six.
- L changes the pre-start lag boundary only. It retains the size floor max(theta,1e-6) and strict conditional sampler; it does not force total drive/F0=0 or Z=1.

## Changes made for release

Only location strings were redacted in archived files, plus UTF-8 serialization when required by redaction. `/RESEARCH_ROOT` and `/PRIVATE_INPUT/manuscript.pdf` are provenance placeholders, not runnable paths. No numerical algorithm was ported or modified. `provenance/source_lineage.json` lists each source/release hash and transformation. Historical manifests retain historical hashes, so redacted files intentionally no longer match those bytes. The new public wrapper is separate, and calculates only summaries from published CSVs.

Historical Stage9G baseline sentinel bit equality and Stage9H Gate PASS refer to earlier runs, as recorded in VALIDATION.md. No sentinel, simulation Gate, full audit, posterior scan, or research rerun was executed while packaging this release. Passing public table comparisons cannot establish those stronger claims. Installation on a fresh machine has not been tested; local public checks use the existing research interpreter.
