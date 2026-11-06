# Input origin and acquisition

Source links below were recorded in the research bibliography. Live DOI pages could not be retrieved during packaging; current availability and redistribution terms were not independently verified. No third-party raw files, PDFs, MATLAB packages or connectome exports are redistributed.

| Input | Original source / acquisition | Role and limitation |
|---|---|---|
| 2022 looming trajectories | [Dryad dataset](https://doi.org/10.5061/dryad.ttdz08m09); obtain files from its download page | Original raw-data location `mq10/data/raw/2022_looming/` |
| 2024 bow-wave trajectories and CFD | [Dryad dataset](https://doi.org/10.5061/dryad.nvx0k6dvz); obtain experiment 1, experiment 2 and CFD files | Original locations `mq10/data/raw/2024_bowwave/exp1/`, `exp2/`, `cfd/`; extraction/preparation is required |
| External comparison software | [2022 Methods S1](https://doi.org/10.5281/zenodo.5784222), [2024 Code S1–S2](https://doi.org/10.5281/zenodo.6896748) | Download from source subject to its terms; omitted from release |
| Connectome prior information | FlyWire Codex exports, cited [connectome study](https://doi.org/10.1038/s41586-024-07558-y) | Exact export/version/filter provenance is not completely resolved in this package; no claim of exact acquisition reproducibility |
| Fitted posterior | Internal MQ10 P2d-R artifact, exact expected Zarr chunks in Stage7 input manifest | No public artifact URL established; omitted |
| Initial states / flow model / flow grid / escape bootstrap | MQ10 derived artifacts from data preparation, fitting and simulation preparation | Exact required paths/sizes/hashes in input_inventory.json; omitted rather than distributing unreviewed source-derived records |
| Selected draws / parameter mapping | MQ10 `draws_P1.csv` and Stage9G/H selected-parameter provenance | Included numerical model outputs; scientific chain/draw/session indices are not personal identifiers |
| Capture and analysis CSVs | MQ10 calculated results at the recorded execution stages | Included aggregate/per-draw numerical outputs; no raw recordings |

The original preparation pipeline used `phases/p1_repro/`, `phases/p2b_detection/`, `phases/p2d_r/`, `phases/p2e/`, then `phases/p3b/run.py` preparation. Only the latter and the calculation core are included in this minimal release. Missing upstream preparation and fitted artifacts prevent a claimed full reproduction from raw download alone. The input inventory is a manifest, not a download service or an assertion that every input is publicly hosted.
