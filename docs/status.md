# Scope and validation

This alpha package provides model definitions and explicit inference, scoring,
selection and export interfaces. It does not bundle trained weights, a training
dataset, generated antibody libraries, benchmark outcomes or experimental data.

The intended use is focused search around an existing candidate binder. A
smaller permitted search space is not evidence of better binding, lower
experimental cost or superiority over de novo design.

## What has been checked

Software tests cover input/output identity, sequence reconstruction, deterministic
selection, model interfaces and failure handling. Most use synthetic data or
explicit mocks. One separately identified JM-Epi checkpoint matched its original
implementation on a fixed CPU input; the scope and environment are described in
[VALIDATION.md](../VALIDATION.md).

That check does not establish end-to-end generation, real-antigen preprocessing,
GPU equivalence, independent predictive accuracy or physical structure quality.
Folding and reconstruction have distinct validation boundaries documented in
their respective guides.

## Interpreting results

DeepCDR outputs are computational scores, not measured affinity or specificity.
AbNatiV2 is a human-repertoire compatibility measure, not universal nativeness.
Sequence/structure correspondence does not establish stability or binding.

The repository contains no molecular-dynamics results, no MD-selected library
and no prospective experimental validation. Supplying source code does not
certify a complete scientific dataset or its claims. Asset permissions and
third-party terms remain separate from software verification.
