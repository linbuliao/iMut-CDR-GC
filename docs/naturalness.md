# Naturalness: measurement and campaign eligibility

`imut_cdr_gc.naturalness` exposes two different analyses. AbNatiV-2 paired
scores evaluate human-repertoire compatibility for **all comparison methods**.
They are not probabilities of binding, measured specificity, developability or
species-native naturalness for murine antibodies. Campaign likelihood is the
generator-related SAbDab-298 masked pseudo-loglikelihood and is restricted to
records explicitly labelled `model: "iMut-CDR-JM-Epi"`. Other methods do not
receive likelihood/P5/GC-selected libraries through this API.

Both analyses require locally supplied model assets. See
[VALIDATION.md](../VALIDATION.md) for tested runtimes and numerical checks.

## Inputs and local assets

CSV/JSONL records require unique `id`, actual mutated `heavy` and `light`
variable chains and, for campaign likelihood, the exact `model` label above.
`heavy_mut_variable`/`light_mut_variable` are accepted aliases. Conflicting
aliases fail. `heavy_full`/`light_full` are **not** founder fallbacks. Equal
sequences with distinct IDs are preserved in their original order.
The campaign output starts with each full input record, retaining its actual
chains, founder, mutation annotations, design/version, lineage, DeepCDR scores
and other scientific metadata. It then adds the new likelihood measurement.
Inputs containing `likelihood` or `average_log_likelihood` are rejected even
when those values are null: use a fresh candidate file, never silently replace
historical measurements. The retained metadata lets `scores.jsonl` feed the
selection workflow directly once a matching policy and measurements exist.

Supply an asset manifest; paths resolve within the explicit project root:

```json
{
  "schema": "campaign-likelihood-assets.v1",
  "scorer_source": {"path": "assets/likelihood/cdr_likelihood.py", "sha256": "<actual SHA256>"},
  "model_source": {"path": "assets/likelihood/imut_cdr_jm_model.py", "sha256": "<actual SHA256>"},
  "weights": {"path": "assets/likelihood/model.pt", "sha256": "<actual SHA256>"},
  "local_model_dir": "assets/esm2_config",
  "config_tokenizer_sha256": {
    "config.json": "<actual SHA256>",
    "vocab.txt": "<actual SHA256>"
  }
}
```

Replace the placeholders; they are not usable identities. Pin **every present**
config/tokenizer file, including `tokenizer_config.json`,
`special_tokens_map.json`, `tokenizer.json` or `added_tokens.json` if present.
Configuration and weights are read only on an explicit scoring invocation.
The private loader accepts only the reviewed native scorer/model source
revisions listed in `_likelihood_native.SOURCE_SHA256`; a new source revision
requires review rather than silently running as the same metric.

Native sources are not copied into this package: provide them as separate,
authorized local assets. The sibling import is resolved in a module-private
builtins import hook, without a `sys.path` change or a global module alias.
Native tokenizer/config calls already use `local_files_only=True`; custom
`auto_map` hooks and undersized/non-ESM configurations are rejected. The
legacy unsafe pickle fallback is disabled. This changes deserialization error
handling, not model mathematics. Architecture overrides are rejected.

Install the original compatible CPU ANARCI/IMGT and local hmmscan/HMM assets.
The API records hashes of the actual numberer source, executable, HMMs and
numbering defaults. It calls heavy-chain numbering then light-chain numbering,
requires one complete matching domain, and constructs the light-then-heavy
298-position representation. Slots are right-padded with X and separated by
X. Unknown residues, ambiguous domains, overlength slots, missing chains and
internal padding are errors, never truncated or replaced.

## Campaign API and command

```python
from imut_cdr_gc.naturalness import read_pairs, score_campaign_likelihood_pairs

rows, metadata = score_campaign_likelihood_pairs(
    read_pairs("candidates.jsonl"), "analysis/likelihood",
    project_root=".", assets=asset_manifest,
    device="cpu", batch_size=64, calibration=None)
```

```bash
python -m imut_cdr_gc.naturalness campaign \
  --project-root . --input candidates.jsonl \
  --output-dir analysis/likelihood --assets likelihood_assets.json \
  --device cpu --batch-size 64
```

Use a new, project-relative output directory. The default device is CPU;
request CUDA explicitly with `cuda:N`. Reference calibration must match the
execution runtime.

The adapter calls the original `score_many` once on valid native inputs in
original order, with `strict=True`, `return_per_position=True`, native default
CDR mask positions, temperature 1 and the requested per-antibody mask batch
size (default 64). The native scorer keeps each antibody's forward passes
separate and uses AA20 log-softmax followed by the original mean over its CDR
sites. No cross-antibody mask mixing, score caching, precision change,
reduction rewrite or custom mask list is introduced. It preserves native
per-position evidence and verifies identity, coverage and aggregate values.

Each completed `scores.jsonl` row contains:

```text
id, ordinal, model, group
average_log_likelihood: finite number | null
status: ok | conversion_error
error: per-chain conversion error list | null
model_result: full original native result | null
likelihood: {id, heavy_sha256, light_sha256, native298_sha256,
             score, status, metric_identity, source_identity}
p5_eligibility: pass | below_reference_p5 | missing | null
gc_selected: null
```

`input_conversion.jsonl` retains native298 inputs, zero-based positions and
both-chain conversion errors. `metadata.json` binds input, source, model,
tokenizer/config, numberer, runtime and score-contract identities, output hash
and separate input/scored/conversion-error/P5 counts. This API does not choose
Top-k or certify final complexes. `scientific_clearance` remains false.
Nested likelihood values bind the actual candidate ID and both complete
mutated chain strings by SHA256, not just the model/runtime. Selection must
reject values copied from another candidate or a changed sequence.

## P5 and metric identity

With `calibration=None`, no cutoff is applied. Optional `--calibration` must
contain the fixed independent-reference identity, actual reference SHA256,
`reference_not_candidates: true`, `lower_quantile: 0.05`, inclusive numeric
`lower_cutoff`, **`upper_quantile: null` and `upper_cutoff: null`**, and exactly
matching `metric_identity` **and** `source_identity`. For a successful score,
pass means `score >= reference P5`. It does not mean exactly 5% of new
candidates fail. There is no P97.5 ceiling.

The source identity includes numeric runtime/device characteristics and
adapter/scorer settings. A reference from another source, checkpoint,
tokenizer, mask contract, batch size or runtime must not be silently reused.
Use a validated reference calibration with matching identity when changing
runtimes.

The generic `calibrate` / `apply_calibration` functions retain explicit
two-bound analysis for other research questions, using standard linear
quantiles. Such general analysis is **not** campaign policy. Call
`validate_campaign_calibration` before workflow eligibility and compare both
identities, not just the numeric floor. CLI `calibrate` accepts `--reference`,
`--column`, `--metric-identity`, `--source-identity`, `--lower-quantile`,
`--upper-quantile`, `--independent-reference`, and `--output`. The
independent-reference flag is a documented caller assertion, not an automated
training-leakage or population-independence certification.

## Errors and resuming

Known per-record input/domain/slot errors retain a null score. They are
counted separately from finite measurements below P5. An ANARCI execution
failure, OOM, unknown scorer exception, malformed/short/misordered native
result, interruption or stale asset **fails the entire measurement attempt**;
there is no automatic per-sequence retry, and no error is a low score. Failed
attempts carry an explicit status/stage/error in metadata and do not commit a
completed score file. Atomic score commits prevent partial JSONL acceptance.

After fixing a failure, use a fresh purpose-named output directory. Inference
does not resume automatically or switch scoring methods after an exception.

## AbNatiV evaluation

```bash
python -m imut_cdr_gc.naturalness paired \
  --project-root . --input all_methods.jsonl \
  --output-dir analysis/abnativ --models-dir assets/abnativ \
  --checkpoint-sha256 ACTUAL_PAIRED_CHECKPOINT_SHA256 --cpus 6
```

Provide the official installed AbNatiV CLI and authorized local
`vpaired2_model.ckpt`; there is no automatic weight download. The official
`paired_score` command's raw CSV and process log are retained, joined by exact
ID and actual heavy/light sequences. Failure or absent values are not zero.
This route emits `abnativ_paired_score`/status only, with **no campaign
likelihood, P5 or GC selection** for external methods. The AbNatiV execution
environment is caller-provided, not a validated campaign calibration backend.

## Tests and boundaries

Run from this package root:

```bash
python -B -m unittest discover -s tests -p 'test_naturalness*.py' -v
```

Tests preserve the original seven analysis checks and add CPU-only fixtures
for explicit constructor wiring, source/config/weight identity, private
imports, no pickle fallback, malformed numbering/results, duplicates/order,
P5 boundaries, runtime mismatch, OOM/interruption, atomic output and external
evaluation separation. One source-pinned pure-function oracle compares 298
layout bytes with the original native builder; it imports no model. These
checks do not read real checkpoint bytes, use GPUs, or certify numeric parity.
The local original-source oracle is explicitly skipped in standalone clones
without that separately authorized source; the remaining fixtures are
self-contained. No missing source is downloaded or replaced with fake evidence.

The metadata-preserving successor adds a CPU integration check that runs actual
conversion/output code with an explicitly simulated numberer and scorer, then
passes the persisted score JSONL directly through the real `workflow.select_run`
and `select_records`. It verifies P5 rejection, sequence deduplication and final
selection without reconstructing dropped metadata between stages. Current
software validation and its limitations are summarized in
[VALIDATION.md](../VALIDATION.md); internal execution records are not part of
this source distribution.
