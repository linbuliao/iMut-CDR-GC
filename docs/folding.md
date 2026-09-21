# True antibody folds with explicit native assets

`imut_cdr_gc.folding.fold_antibodies` creates **antibody-only H/L folds** for
structure measurement. It does not construct a full antigen complex, dock an
antibody, mutate a founder structure, run refinement, or perform screening.
Those products must retain separate provenance.

```python
from imut_cdr_gc.folding import fold_antibodies

folded_rows, receipt = fold_antibodies(
    candidates,
    project_root=project_root,
    output_dir="runs/example/antibody_folds",  # must not already exist
    assets=fold_assets,
    device="cpu",  # only cpu or an explicit cuda:N; no implicit GPU selection
)
```

## Inputs and fixed numerical profile

Each record needs a unique `id`, canonical full `heavy`/`light` **variable-domain**
sequences, and a nonempty `antigen_id`. The input sequences must already have
been reconstructed from FR277 proposals. The entire provided sequence must be
covered by exactly one appropriately typed ANARCI domain: no dummy residues,
constant-domain clipping, terminal trimming or automatic alternate numbering.
H/L sequences longer than 70 residues are required by the native admission
rule. Candidate metadata, founder identity and existing selection evidence are
preserved; existing fold evidence is rejected rather than replaced.

Only `native_ab2_123_chothia_unrefined_v1` is implemented:

- Native ABodyBuilder2 models **1, 2, 3**, in that order; embedding dimensions
  128/256/256, relative-position dimension 64, float32, evaluation mode.
- Native ANARCI IMGT recognition followed by **Chothia** numbering, human/mouse
  species, H versus L/K role checks. Unlike the original wrapper, complete
  sequence coverage and unambiguous domain identity are enforced explicitly.
- Native ensemble CA-dispersion ranking. Only `ranking[0]` is written by
  `Antibody.save_single_unrefined`; the saved model ID is recorded per candidate.
- No model 4, automatic downloads, upstream refinement, AMP conversion,
  automatic GPU/CPU fallback or changed geometry convention.

The antigen ID is an input identity link only. Antigen coordinates are **not**
used by this antibody-only predictor.

## Asset declaration

All paths are explicit **project-relative** existing files, with lowercase
SHA256 values. The following structure is mandatory; names are exact keys:

```text
profile: native_ab2_123_chothia_unrefined_v1
sources:
  constants.py:    {path: ..., sha256: ...}
  rigids.py:       {path: ..., sha256: ...}
  models.py:       {path: ..., sha256: ...}
  util.py:         {path: ..., sha256: ...}
  ABodyBuilder2.py:{path: ..., sha256: ...}
weights:
  antibody_model_1:{path: ..., sha256: ...}
  antibody_model_2:{path: ..., sha256: ...}
  antibody_model_3:{path: ..., sha256: ...}
numbering:
  sources:
    __init__.py: {path: ..., sha256: ...}
    anarci.py:   {path: ..., sha256: ...}
    germlines.py:{path: ..., sha256: ...}
    schemes.py: {path: ..., sha256: ...}
  hmms:
    ALL.hmm:    {path: ..., sha256: ...}
    ALL.hmm.h3f:{path: ..., sha256: ...}
    ALL.hmm.h3i:{path: ..., sha256: ...}
    ALL.hmm.h3m:{path: ..., sha256: ...}
    ALL.hmm.h3p:{path: ..., sha256: ...}
  hmmscan:      {path: ..., sha256: ...}
```

Accepted native digests are published as `SOURCE_SHA256`, `WEIGHT_SHA256`,
`NUMBERING_SOURCE_SHA256`, `HMM_SHA256`, and `HMMSCAN_SHA256` in `folding.py`.
They identify the inspected production sources/assets, not arbitrary files
renamed to match. Missing, substituted, extra or changed assets fail closed.
Assets are checked **before learned-model imports** and rechecked before a
completion receipt. Model states use `torch.load(weights_only=True)` and strict
state loading, never unsafe pickle fallback.

The four ANARCI source files must describe the installed package resolved by
the current interpreter. HMM files must be its `dat/HMMs/ALL.hmm*`; `hmmscan`
resolved on PATH must be the declared executable. Install/configure that
runtime and executable inside the explicit project beforehand. The API does
not change `sys.path`, create module aliases, install dependencies or rewrite
the native runtime. Runtime, library and asset identities are recorded in
the output metadata.

Compatible PyTorch, NumPy, einops, BioPython, ANARCI and HMMER are prerequisites.
The adapter reads the five pinned ImmuneBuilder source assets into an isolated
namespace; it never imports upstream top-level `ImmuneBuilder`, whose import
also loads refinement. It includes only the native numerical classes/utilities
and the two necessary `Antibody` methods. No vendor source or learned weights
are bundled by this module; acquire authorized local copies and retain their
upstream licenses separately. OpenMM/PDBFixer are unnecessary for these
unrefined antibody folds, but are required by the separate founder-sidechain
complex constructor.

## Files, failure behavior and provenance

Numbering for all records runs before learned inference, in an isolated child
whose temporary files remain inside this attempt and are cleaned afterward.
Neither the parent environment nor ANARCI globals are patched. The child uses
the same installed interpreter/package; install the package before calling
this API rather than relying on a temporary parent-only `sys.path` edit.

On total success, the new output directory contains:

- `antibody_000001.pdb`, ...: actual, unrefined H/L atom coordinates.
- `fold_manifest.jsonl`: all original candidate metadata plus `pdb_path`,
  `pdb_sha256`, `numbering`, and `fold_provenance`. This matches the existing
  DeepCDR-3D `verify_mutant_fold` interface directly.
- `receipt.json`: completed antibody-fold count, source/weight/runtime/profile
  identities, input-record fingerprint, all PDB hashes and manifest hash.

Every output requires exact full H/L sequences, insertion-aware Chothia
residue correspondence, finite coordinates and all required backbone and
residue-specific sidechain heavy atoms. These checks establish identity and
completeness, **not physical structure quality, affinity or specificity**.

Any failed numbering, model loading, inference, malformed PDB, asset drift,
resource exhaustion or interruption aborts the attempt. `partial.jsonl` and
`failure.json` retain incomplete evidence and already-created PDBs; there is no
completion receipt. Nothing is silently retried or replaced. Inspect the
failure and use an explicitly new output directory for a new attempt; do not
treat partial structures as a completed library.

## Verification boundary

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  python -B -m unittest discover -s tests -p 'test_folding_contract.py' -v
```

The tests use synthetic CPU records, fake predictor outputs, explicit mocked
runtime loading and AST-isolation fixtures. They test the file/API/ranking and
failure contract, **not real checkpoint predictions or learned-weight parity**.
No GPU, model download, real weights, real ANARCI/HMMER run or new production
structures were used to validate this component. A separate authorized native
asset smoke test remains necessary before scientific inference is claimed.
