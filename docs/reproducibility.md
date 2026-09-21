# Reproducing the software

This repository provides source code and inference interfaces. Learned
inference requires locally supplied checkpoints; training datasets and study
data are separate from the software distribution.

## Component map

| Component | Source | Entry/config | Main software tests |
| --- | --- | --- | --- |
| iMut-CDR-JM | `models/fr_cdr.py`, `generation.py` | `generate --model jm`; `examples/configs/jm.json` | `test_generation_jm.py`, `test_generation_models.py` |
| iMut-CDR-JM-Epi | `models/jm_epi.py`, `models/antigen.py`, `generation.py` | `generate --model jm-epi`; `examples/configs/jm_epi.json` | `test_generation_models.py`, `test_generation_cli_integration.py`, `test_antigen_graph.py` |
| iMut-CDR-GC | `pipeline.py`, `workflow.py`, `selection.py` | `run`; `examples/demo_pipeline.json` | `test_pipeline.py`, `test_workflow_contract.py`, `test_stage_input_identity.py` |
| DeepCDR-3D | `scoring/native_structure.py`, `scoring/_vendor/deepcdr3d_model.py` | `score --models 3d`; `examples/configs/deepcdr3d.json` | `test_scoring_contracts.py`, `test_scoring_source_identity.py`, `test_scoring_workflow_identity.py` |

Source paths in this table are relative to `imut_cdr_gc/`; tests are under
`tests/`. Generation requires the `generation` extra; structural scoring
requires `structure`. Numbering/folding and naturalness additionally require
their separately documented native tools and authorized assets.

## Start with software checks

```bash
python -m pip install '.[test]'
python examples/make_demo.py
python -m imut_cdr_gc run --config examples/demo_pipeline.json
python examples/check_demo.py
python -B -m pytest tests/test_pipeline.py tests/test_workflow_contract.py -p no:cacheprovider
```

The demo is fully synthetic and loads no model. Optional neural tests use small
random-state CPU fixtures. Install optional dependencies to run the complete
suite. Native-model checks and their runtime scope are documented in
[VALIDATION.md](../VALIDATION.md).

## Reproduce a learned inference

Record the package revision, Python/library versions, source/asset SHA256 values,
input identities, model configuration, seed, device and precision. Supply the
corresponding checkpoint and tokenizer locally; an unrelated checkpoint is not
an equivalent replacement. Keep stage outputs and their provenance together.
Changing any numerical setting requires a separately identified run.

JM and JM-Epi have separate architectures and checkpoints. Numerical
validation is specific to the tested model, weights and runtime.

To implement V3, apply DeepCDR screening after each `propose` call and use the
accepted candidates as the next parent bank. Configure reference P5 calibration
and verify final complexes separately. The synthetic workflow runs without
model downloads or study data.
