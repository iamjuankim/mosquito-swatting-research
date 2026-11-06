# Rights and third-party material review

This package contains selected MQ10 research Python sources, configurations, derived numerical results and verification records, prepared at the research owner's request for noncommercial scholarly sharing. No license is assigned to the independently authored MQ10 code or results. Public availability does not grant a general reuse license for that material. This reservation does not restrict rights already granted in upstream CC-licensed material, public-domain elements, or applicable exceptions. No repository-wide MIT, CC0 or CC BY-NC license is asserted.

Third-party raw datasets, MATLAB code/packages, article PDFs, images, videos, connectome exports and fitted data objects are omitted. Original data/software DOI links are provided instead. No bundled third-party package implementation or copied connectome table was identified in the selected payload. Importing a dependency does not redistribute that dependency here. Licenses for omitted software are not relied upon to authorize copying it into this repository.

## FlyWire v783 attribution and material scope

FlyWire Consortium and the authors credited below provide FAFB v783 public-release data under [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). The [official public-release guidelines](https://home.flywire.ai/guidelines) state that all data available in Codex for snapshot 783 are publicly released, including newer annotations. Source: [FlyWire Codex](https://codex.flywire.ai/). The original MQ10 acquisition record labels its inputs FAFB v783, downloaded 2026-10-28; the post-July-2025 synapse-prediction label is recorded user-supplied provenance, not a newly audited export.

Credit for the upstream resources used in the original aggregation:

- Dorkenwald et al. and the FlyWire Consortium (2024), *Neuronal wiring diagram of an adult brain*, [doi:10.1038/s41586-024-07558-y](https://doi.org/10.1038/s41586-024-07558-y): reconstruction/connectivity.
- Schlegel et al. (2024), *Whole-brain annotation and multi-connectome cell typing of Drosophila*, [doi:10.1038/s41586-024-07686-5](https://doi.org/10.1038/s41586-024-07686-5): cell types/classification. These first two works are co-cited as requested by the [official citation guide](https://docs.google.com/spreadsheets/d/1eOPxOYoalArVDIhzVis3CsKjETzkdFkRtXSHHY3XdSU/edit).
- Zheng et al. (2018), *A complete electron microscopy volume of the brain of adult Drosophila melanogaster*, [doi:10.1016/j.cell.2018.06.019](https://doi.org/10.1016/j.cell.2018.06.019): FAFB volume/reconstruction source credit.
- Yu et al. (2025), synapse/connectivity prediction credit for the recorded post-July-2025 inputs, [doi:10.1101/2025.07.11.664377](https://doi.org/10.1101/2025.07.11.664377).
- Eckstein, Bates et al. (2024), *Neurotransmitter classification from electron microscopy images at synaptic sites in Drosophila melanogaster*, [doi:10.1016/j.cell.2024.03.016](https://doi.org/10.1016/j.cell.2024.03.016): neurotransmitter predictions.

The license permits noncommercial sharing and adaptation with appropriate credit, a license link and an indication of modifications. Commercial use of licensed material is outside that permission; no additional legal/technical restriction may remove its licensed freedoms. These are explicit upstream conditions, not a prohibition on publishing this noncommercial research repository. No endorsement by FlyWire or the cited authors is implied. Upstream material is provided without warranties; see the [license legal code](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en).

## What detection.py actually contains

The four copies are `archive/mq10/src/swat/detection.py`, Stage7 `source/swat/detection.py`, Stage9G `run_04/source/swat/detection.py`, and Stage9H `source/swat/detection.py`. All retain the same relevant `PRIORS` entries. The paths use the stage directory names in PUBLIC_FILES.csv.

| Value in PRIORS | Original interpretation and transformation |
|---|---|
| P1 rho_vs log-center ln(1.9) | Rounded model choice combining the Ache functional LC4/LPLC2 ratio 1.5/1.1 (about 1.36) with the rounded connectome ratio 2.6, using their geometric mean. |
| P2 rho_vs log-center ln(2.6) | Rounded prior center informed by aggregate left/right GF direct LC4/LPLC2 synapse ratios about 2.83 and 2.46 after pair-sum threshold >=5. It is not a copied connection row. |
| P1/P2 rho_av log-center ln(0.15) | Model choice informed by unsigned JO-C+E effective strength about 12–18% of LC4, summing normalized path products of lengths 1–3. Positive sign is a modeling assumption informed by mosquito bow-wave evidence, not an identified connectome sign. |
| Log SDs 0.5/0.15 and 1.5/0.25 | Research-selected prior widths, not exported FlyWire measurements. P0 is a broad reference prior. |

Original evidence, read without rerunning: `mq10/phases/p2a_connectome/run.py` and `connectome.py`, `results/p2a/ratios.csv`, `logs/phase_reports/p2a.md`, and `logs/decisions.md` P2b entry dated 2026-10-28. Those upstream research files remain outside this minimal public package. Aggregation summed neuropil rows by neuron pair, applied the declared synapse threshold, normalized by each target's total input, and summed group/path contributions. The code then used the rounded aggregate-informed choices above. No neuron-level connectivity rows, cell-annotation exports, or full aggregation tables are included. Other detection constants (e.g. C1, C2, theta peak and size width) come from the cited Ache model, not FlyWire; see [Ache et al. 2019, doi:10.1016/j.cub.2019.01.079](https://doi.org/10.1016/j.cub.2019.01.079).

Modification notice: MQ10 transformed the upstream data through aggregation and prior selection, then fitting, simulation and paired summaries. Included H2/Stage8/Stage9G/Stage9H CSVs and Stage9G/H `selected_parameter_provenance.json` contain model parameters or calculated estimates, not copied raw connectome records. Copyright/database-right applicability to a few numerical facts and remote downstream statistics is distinct from direct data redistribution; this review does not assume that every independent output automatically inherits a data license. To the extent any protected FlyWire-derived material is carried into those constants or results, the upstream attribution, noncommercial condition and license remain applicable to that material. This notice does not relicense the independently authored Python implementation or independent numerical results.

## Publication decision

For the stated noncommercial scholarly release, no mandatory unresolved redistribution restriction was identified in the reviewed payload after adding this attribution and modification notice. The earlier lack of a mapped source license is resolved by the official v783 public-release guidance. The remaining uncertainty about the legal protection of individual numerical aggregates is not evidence of a publication ban and does not require an invented permission step. Commercial reuse of protected upstream material would need separate permission or another applicable basis. This is a scoped publication review, not an ownership warranty for every possible future use.

Personal filesystem locations and manuscript filename references in selected files were replaced by neutral placeholders. Scientific session/chain/draw/trajectory indices remain because they are required to interpret calculations. Secrets and personal location patterns are scanned over the exact allowlist and staged content before the first commit. The Git commit uses a neutral local author identity, avoiding a personal email in this new public history.
