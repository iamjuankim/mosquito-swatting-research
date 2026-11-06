# Mosquito swatting study: code and reproducibility release

This repository documents the MQ10 detection and capture simulations: how approach speed affects predicted mosquito capture, and how a declared pre-start lag boundary changes those predictions. It contains the actual numerical code, selected parameter coordinates, analysis tables and historical verification records. It does not contain the full posterior or trajectory checkpoints.

The five computations are separate snapshots under `archive/`:

| Computation | Directory | Scope |
|---|---|---|
| Existing H2 model | `mq10/` | Detection, flow, escape and simulation sources; original Rule G and H2 tables. Current source snapshot is not asserted to be the exact historical H2 executable. |
| Stage7 baseline | `mq10_audit_stage7_h2_stable_20261105/` | Frozen production executor, source, conditions, strict threshold sampler and input manifest. |
| Stage8 verification | `mq10_audit_stage8_paired_20261105/` | Paired legacy/baseline analysis and Rule G, 48 conditions × 200 draws × 4,000 trajectories. |
| Stage9G P1/P2 | `mq10_audit_stage9g_p1p2_20261106/run_04/` | Verified run only: four baseline sentinels and 36 L batches, three selected draws. |
| Stage9H P3 | `mq10_audit_stage9h_p3_20261106/` | Exploratory L sensitivity, speeds 8/12 m/s, six model×n blocks, fixed 200 draws. Reuses 36 L batches and adds 2,364. |

Use Python 3.12. For the small public result calculation, install:

```sh
python -m pip install -r requirements-public.txt
python tools/reproduce_tables.py
python tools/verify_package.py
```

These commands recompute Stage8 effect summaries and Stage9H local Rule G from published draw tables, and check package hashes, syntax and recorded source lineage. They do not simulate trajectories, refit a posterior, verify sentinel bits or repeat historical Gates. Outputs go to ignored `generated/`. The public calculation uses NumPy/SciPy/pandas; the original broader environment is recorded in `ENVIRONMENT.md`.

Raw data are obtained separately from the two Dryad records listed in [DATA_SOURCES.md](DATA_SOURCES.md). No download is required for the commands above. Downloading those records alone does not restore the fitted models, initialization table, flow grid, exact posterior and checkpoints required by the archived production drivers. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) before using any archived driver; those files retain expired one-shot guards and historical path contracts and are not public execution entry points.

Main result files (under the corresponding archive directory):

- Existing H2: `results/p3b_judge/h2_rule_g.csv` and `h2_speed_table.csv`.
- Stage8: `paired_capture_by_condition_draw.csv` (9,600 pairs), `posterior_by_speed_n_model.csv`, `h2_effects_by_draw.csv` and `h2_effects_rule_G.csv`. `legacy` and `stable` distinguish algorithms.
- Stage9G run_04: `paired_screening_results.csv` (36 rows) and `paired_speed_contrasts.csv` (18 rows). These three-draw screening results do not establish a posterior-wide conclusion.
- Stage9H: `p3_condition_draw_results.csv` (2,400 rows), `p3_draw_results.csv` (1,200 rows), `p3_contrast_summary.csv` and `p3_rule_g_summary.csv` (six blocks). B is Stage7 baseline; L extends pre-start lag features by zero. Rates/differences are fractions; multiply by 100 for percentage points. Read the CSV headers and `p3_science_report.md` for MC SE and censoring definitions.

Historical verification is summarized in [VALIDATION.md](VALIDATION.md). Six L 8→12 m/s decrease classifications were retained for the fixed draws and MC set; positive 2 MC SE shifts change classification in the two n=1.2 blocks. P3 is exploratory, and does not rejudge overall H2. B/L condition on different start-survival populations; their paired differences are not identified individual causal effects. Equal-weight summaries of the fixed selected draws do not guarantee representation of the full reweighted posterior.

Exact file sizes are in [PUBLIC_FILES.csv](PUBLIC_FILES.csv); provenance and publication limitations are in [LINEAGE.md](LINEAGE.md) and [RIGHTS.md](RIGHTS.md).

FlyWire FAFB v783 informed the aggregate-based prior choices in `detection.py` (rho_vs centers 1.9/2.6 and rho_av center 0.15); no raw connectivity or annotation export is bundled. Credit: Dorkenwald et al. / FlyWire Consortium ([2024 DOI](https://doi.org/10.1038/s41586-024-07558-y)) and Schlegel et al. ([2024 DOI](https://doi.org/10.1038/s41586-024-07686-5)). The upstream data are [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) under [FlyWire's official public-release guidelines](https://home.flywire.ai/guidelines). MQ10 aggregated and rounded those inputs into priors, then fitted models and calculated results. This is noncommercial scholarly sharing; applicable upstream attribution and noncommercial conditions remain in force for protected upstream material. Full credits, transformations and scope are in [RIGHTS.md](RIGHTS.md). No license is assigned to independently authored MQ10 code or results; neither MIT nor CC0 is applied to third-party material.
