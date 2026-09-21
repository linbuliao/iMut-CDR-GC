# JM and JM-Epi generation

Use `imut-cdr-gc generate --model jm` for joint CDR proposals without antigen
conditioning, or `--model jm-epi` for antigen-conditioned proposals. The latter
is the default. [Configuration templates](../examples/configs/README.md) show
the required local assets and runnable command syntax after those assets are
supplied.

The Python factory accepts `load_local_generator(model_kind="jm", ...)` or
`model_kind="jm-epi"`. JM uses `ProteinMLMContrastModel` and accepts no antigen
graph (`propose(records, graphs=None)`). JM-Epi uses
`EpitopeConditionedFRCDRModel` and requires one graph per record. JM's optional
`model_config` contains flat `BackboneConfig` fields; JM-Epi uses its documented
`EpiConfig`. Neither interface adds screening. Likelihood/P5/GC selection remains
restricted to JM-Epi; a JM model cannot be relabelled as an Epi candidate.

The asset and graph examples below describe JM-Epi unless stated otherwise.

The model definitions come from the FR-v1 and JM-Epi sources used by the native
V3 protocols. Their inference imports are side-effect-free;
`models/source_identity.json` records the original source identities and hashes.

The reference checkpoint passed the single-input CPU numerical check in
[VALIDATION.md](../VALIDATION.md). Validation of training, GPU inference and
the full portable pipeline remains outstanding.

## Assets and construction

```python
from imut_cdr_gc.generation import load_local_generator, load_antigen_graph

generator = load_local_generator(
    model_dir="assets/esm2_config_and_tokenizer",
    checkpoint="assets/jm_epi/approved_checkpoint.pt",
    expected_checkpoint_sha256=checkpoint_sha256,
    asset_sha256=config_and_tokenizer_hashes,
    device="cpu", precision="float32",
    temperature=2.5, top_k=6, parental_residue_logit_penalty=3.0,
    seed=123,
)
graph = load_antigen_graph("inputs/native_antigen_graph.npz", graph_sha256)
```

The example paths are placeholders for explicitly supplied, authorized assets;
no assets are bundled, downloaded or inferred from a private server path.
`asset_sha256` maps every present config/tokenizer filename to its SHA-256 and
must include `config.json` and a vocabulary. Local-only Transformers APIs build
ESM from its configuration. The complete fine-tuned tensor state is then loaded
with `weights_only=True`; missing/extra keys, shapes, dtypes, nonfinite tensors and
contradictory tied-weight aliases are rejected. There is no permissive partial
checkpoint fallback. Checkpoint and tokenizer/config identities are rechecked
after loading. Compatibility is specific to checkpoint and library version;
synthetic tests alone do not certify it.

The exact reference checkpoint
`a19bd33544bbfcb8a18dfb88bb007439cd57e702c4d89963e143df79991c144e`
uses a serialized ESM layout with a persistent deterministic position-ID buffer
and no contact-prediction regressor. For this artifact only, the loader verifies
the complete architecture, four config/tokenizer identities and Transformers
4.46.3, restores that saved layout, then strictly loads all 699 keys. No saved
tensor is dropped, renamed, cast or replaced. Ordinary sequence generation
does not use the absent contact predictor; calling it fails explicitly rather
than using random parameters. Other checkpoints receive no such exception.
The `generation` extra pins `transformers==4.46.3` to match this contract;
the caller must still provide the exact authorized weights and tokenizer.

The default `EpiConfig` explicitly specifies the executed ESM2 1280-dimensional
backbone, p1/c1/s3 recurrent blocks, 128-dimensional projection, 30-dimensional
antigen features and the native pocket/cross-attention dimensions. Constructors
also permit explicit configurations and encoder/tokenizer injection for testing.
Different dimensions require compatible weights and separate validation.
Callers select precision and device explicitly; CPU is the default. Mixed
precision has not been evaluated by the CPU numerical check.

### Antigen graph contract

`load_antigen_graph(path, expected_sha256, node_dim=30)` loads a **materialized
native graph**, not an atom-level pocket file. Safe NPZ must have exactly:

| Array | Type | Shape |
|---|---|---|
| `x` | float32 | N × node_dim |
| `pos` | float32 | N × 3 |
| `edge_index` | int64 | 2 × E |
| `edge_attr` | float32 | E × 2: distance, sequence-edge flag |

It preserves all nodes, edges and values; checks finite values, indices, flags
and hashes; refuses empty graphs and pickle/object arrays. Extra schema fields,
including the historical atom-level `res_name/res_id/chain_id/atom_name` format,
are rejected rather than silently converted. Producing these materialized
arrays from raw pockets is a separate preprocessing step requiring the actual
native residue grouping/contact rules and feature-map provenance. This loader
does not certify an unproven graph's scientific origin merely because its hash
matches an input supplied by a caller. Do not reuse a DeepCDR-3D graph here:
JM-Epi uses a different native antigen graph representation.

## One proposal, then verified full-chain reconstruction

```python
rows = generator.propose([{
    "id": "candidate_1", "design": "design3", "version": "v3",
    "antigen_id": "pinned_antigen",
    "founder_fr_cdr_seq": founder_fr277,
    "fr_cdr_seq": parent_fr277,
    "proposal_positions": eligible_positions,  # 1–10 unique FR277 indices
    "sampling_seed": 471,
}], graphs=[graph])
```

Both FR sequences must contain exactly 277 positions; padding/framework changes,
padding requests, duplicate/invalid positions and >10 requests are rejected.
No sequence or mask list is silently truncated. Tokenization must preserve one
token per position without extra special tokens. This is the fixed generation
representation; it is **not** the scorer's 95-slot input and **not** a full-chain
sequence obtained by dropping X padding.

The output is deliberately typed as
`fr_cdr_proposal_requires_full_chain_reconstruction`. It contains `fr_cdr_seq`,
`parent_fr_cdr_seq`, `founder_fr_cdr_seq`, `requested_sites`, `changed_sites`,
`actual_mutation_positions`, all corresponding counts and `iteration_trace`.
Indices are zero-based **FR277** positions. Reconstruct the mutant H/L sequences
using the verified founder CDR→full-chain map, retaining the original framework
residues. Score the reconstructed candidate; parent scores do not describe the
new sequence.

Sampling keeps AA20 in the native order, moves logits to float32 CPU, divides
by temperature, subtracts the parental-residue penalty, applies top-k and draws
with a per-record CPU generator. The parent residue is **not forbidden**, so a
request need not produce a substitution. The recorded native schedule uses
temperature/top-k `(3, 10)` before round 4 and `(2.5, 6)` subsequently, with
penalty 3; those values belong to the enclosing explicit protocol, not a hidden
automatic schedule in this module. Default API sampling values are not labelled
as a replay of that recorded schedule. Temperature zero is an explicit greedy
option. Provided `sampling_seed` is retained; otherwise the supplied base seed
and record ID deterministically derive one without seeding a global RNG.

For example, requesting 10 sites can leave all 10 unchanged. Changing 10 sites,
then reverting 9 of them, produces 19 committed stepwise changes but **one**
final mutation relative to the founder. The implementation recalculates final
burden from sequences rather than summing requested or changed counts.

`iterate(record, graph=..., position_rounds=...)` is an **unscreened accounting
helper**. It is not a V3 controller. Production V3 must use `propose`, apply its
measured DeepCDR gate, and allow only accepted children to become parents.
`iteration_trace` therefore must not be treated as a scored `stepwise_trace`.
This module adds no likelihood, selection or filtering to Epi/external controls.

## What the CPU tests establish

Run from the prepared package root:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  python -B -m unittest discover -s tests -p 'test_generation*.py' -v
```

The independent oracle evaluates only the original, hash-bound model class and
helper AST nodes, injecting a small synthetic encoder/tokenizer/configuration.
It never imports or executes a trainer. The successor and oracle have identical
state keys and exactly equal CPU float32 logits, anchor and positive vectors.
The test activates FiLM and recurrent-depth queries, uses non-identical antigen
graphs and padded sequences, and includes a ten-site query. This avoids a
trivial comparison caused solely by identity-initialized FiLM. Sampling parity
is separately checked against the recorded native operation order.

Further tests cover strict state loading, no import-time asset/device/RNG
actions, the actual extracted model→proposal path, zero/mixed changes,
reversions, invalid mask/sequence/graph inputs and safe graph/asset hashes.
The CLI integration fixtures mock only the learned-model loader: the real
safe-NPZ reader, proposal adapter, batching and full-chain reconstruction run on
CPU. They check ordered IDs, preserved founder frameworks, stale-score removal,
explicit target/version consistency and refusal to commit when input files
change during inference. Their prepared/unscreened outputs are not new selected
libraries or structures.
Temporary fixture files are inside this package and removed after testing.
See [validation boundaries](../VALIDATION.md) for the current test scope.

These are synthetic CPU implementation checks, not pretrained ESM output
parity, biological validation, GPU/AMP equivalence, successful real-asset
generation or a completed pipeline/structure delivery. Those boundaries remain
explicit even when every CPU test passes.
