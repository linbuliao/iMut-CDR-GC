# Native DeepCDR scoring

The scoring package measures a candidate; it does **not** decide whether to
select it. It uses explicit local assets and returns the full input, checkpoint,
source and runtime identity alongside each measurement. There are no cluster
paths, automatic downloads, GPU allocation or historical score-cache lookups.
Importing `imut_cdr_gc.scoring` does not import Torch, PyG or ANARCI.

## API and assets

```python
from imut_cdr_gc.scoring import score_esm2, score_3d

esm_result = score_esm2(record, project_root=project_root,
                        assets=esm_assets, antigen=antigen, threads=1)
structure_result = score_3d(record, project_root=project_root,
                           assets=structure_assets, antigen=antigen, threads=1)
```

`project_root` must be an existing working-project directory. Every declared
asset and PDB must resolve to a file **inside** it, including through symlinks.
The small native feature JSONs are included in `imut_cdr_gc/scoring/_vendor/`;
if the installed package is outside the working project, explicitly copy the
needed feature JSON into that project's `assets/` before declaring its path.
Learned weights are not included, copied or fetched by these APIs.

Each model's `assets` mapping is:

```json
{
  "model": "DeepCDR-ESM2",
  "weights": {"path": "assets/esm2_state.pt", "sha256": "EXACT_FILE_SHA256"},
  "features": {"path": "assets/amino_acid_vectors_20dim.json", "sha256": "EXACT_FILE_SHA256"}
}
```

Use `model: "DeepCDR-3D"` and the 30-dimensional feature JSON for 3D. The
reference ESM2 state-dictionary SHA256 is
`28250f0b24e713e4b73f13fdcfb1a440db4a5473ea3804c5dd4e29dcb77c2387`;
the reference 3D SHA256 is
`35dd2b41dc4765ec08fd2cb09006811c6078c389845205b7772a76030c262672`.
No public download URL or redistribution permission is asserted. Private
project asset access and public licensing must be handled by their owners.

A different checkpoint is refused unless the caller explicitly supplies
`allow_nonreference_checkpoint: true`. This is for clearly identified
user-trained models or fixtures, never an automatic fallback. Results then
record `reference_checkpoint: false`. Both paths still require exact asset
hashes, safe `torch.load(weights_only=True)`, strict state-dictionary loading,
the same architecture and the original numeric feature tables. Reformatting a
feature JSON is permitted with its new explicit hash; changing its float32
vectors is not. Unsafe feature pickle and full-object model loading are absent.

## Candidate and antigen identity

A candidate requires `id` (or `mutant_id`), actual mutant `heavy` and `light`
full-chain sequences, and `antigen_id`. `heavy_full`/`light_full` are supported
aliases. Supplying both forms with different sequences is an error; supplying
conflicting identifier aliases is also an error. Sequences must be nonempty,
ungapped, uppercase canonical amino acids.

The explicit antigen descriptor is:

```json
{
  "id": "target_name",
  "pocket_pdb": {"path": "inputs/antigen_pocket.pdb", "sha256": "EXACT_FILE_SHA256"},
  "chains": {"A": "ACTUAL_POCKET_CHAIN_SEQUENCE"}
}
```

Its name must match the candidate. PDB hash and every chain's actual atomic
sequence must match this declaration. If a candidate additionally supplies
`antigen_pocket_sha256`, that must match too. The strict parser rejects ambiguous
multiple models, duplicate CA residues, noncanonical CA residues, unsupported
alternate locations and nonfinite coordinates; it does not repair the input.
An antigen name alone is never sufficient. This validates a declared pocket's
identity, not its biological appropriateness or an experimentally bound pose.

## DeepCDR-ESM2: derive native95 from actual mutant H/L

The API never reads the generator's `cdr_seq` for scoring. ANARCI runs on the
actual heavy and light chains (`scheme="imgt"`, one CPU); a unique heavy domain
and a unique K/L light domain are required. Numbered nongap residues must map
exactly back to each actual input chain.

The retained author contract takes **zero-based ANARCI list indices**, inclusive
27–38, 56–65 and 105–117. These are not literal IMGT residue labels. Each CDR
is truncated/padded to widths 10, 10 and 25, respectively. Order is heavy CDR1,
CDR2, CDR3, then light CDR1, CDR2, CDR3, with one `X` separator between blocks:
95 slots in total. Returned provenance includes each mapped residue, chain
index, ANARCI label, padding and every truncated residue. Missing domains or
empty required windows are errors, not all-zero substitute sequences.

The 95×20 tensor uses the author's static 20-dimensional amino-acid vectors;
this retained scorer does not compute new large-model ESM embeddings at runtime.
The antigen graph preserves the native `train_like`, single-direction CA
contact edges at strictly less than 5 Å. Native set-derived node ordering and
historical residue-label/insertion-code conventions are retained, not silently
"corrected". Such changes would define a different model input protocol.

ANARCI/HMMER and their local data must already be installed. Numbering scratch
files are confined to the working project's `.cache/tmp/` and cleaned on exit.
A `numberer=` argument exists for deterministic fixtures/integration; results
mark it as caller-supplied with scientific equivalence **not** asserted.

## DeepCDR-3D: actual fold, correct branch order

3D records additionally require `pdb_path`, `pdb_sha256`, `numbering: "chothia"`
and this explicit folding provenance:

```json
{
  "kind": "antibody_fold",
  "method": "THE_ACTUAL_FOLDING_METHOD",
  "numbering": "chothia",
  "heavy_sha256": "SHA256_OF_ACTUAL_HEAVY_STRING",
  "light_sha256": "SHA256_OF_ACTUAL_LIGHT_STRING",
  "pdb_sha256": "EXACT_FOLD_FILE_SHA256"
}
```

Place that object in `record["fold_provenance"]`. The PDB must contain exactly
the declared mutant H and L atomic sequences; neither swapped chains nor an
unchanged founder fold can stand in for a different mutant. Founder-based
side-chain reconstructions are explicitly refused as actual folds. The caller
must supply a truly folded, Chothia-numbered structure with honest provenance;
the scorer does not run a folding model or independently certify that provenance.

The 30-feature dual-GCN is called with named arguments: the **antigen pocket**
enters the first GCN branch and the **actual antibody CDR-context graph** enters
the second. Contacts are strictly <5 Å with the original directed-edge rule.
The historical Chothia CDR-context windows are H26–35/H50–65/H95–102 and
L24–34/L50–56/L89–97. No inter-partner docking pose enters this dual-graph
network. It is not interchangeable with a full-complex pose model.

## Outputs and selection

Only `status: "ok"` has finite `score` in [0,1] and a finite `logit`. Other
statuses (`invalid_input`, `asset_error`, `dependency_error`, `inference_error`,
`resource_error`)
retain a descriptive error and **null** numerical measurements. They must not
be coerced to zero, counted as ordinary cutoff failures or ranked.
`selection_decision` remains null even for a valid low score. Binding/structure
alias fields are included for explicit workflow integration; they refer to the
same computational measurement, not experimental affinity or specificity.
`resource_error` denotes memory/resource exhaustion and must abort the current
batch rather than repeatedly attempting the same failure for every candidate.

For the orchestrator, store the entire object under `scores.esm2` or `scores.3d`;
only the separate selection stage may apply route, common DeepCDR cutoff and
JM-Epi-only P5 eligibility. Neither scorer computes likelihood/P5 or assigns
GC-selected status. Assets and PDB hashes are rechecked after inference.

## Dependencies, provenance and limits

Runtime inference needs NumPy, Torch supporting `weights_only=True` and PyG;
ESM2 sequence preparation additionally needs ANARCI plus its local HMMER/data.
CPU `threads` is explicit and positive; calls configure Torch's process-wide
intra-op thread count. Use serial calls per worker process rather than sharing
the model cache across caller-managed concurrent threads. No dependency is
installed by the API and no CUDA device is selected.

`scoring/source_identity.json` binds every vendored file to its audited source.
The 3D model, graph and JSON are exact copies. ESM2 model and input functions
have exact source ASTs with import-safe surrounding code. The 20-feature pickle
was converted by inert symbolic parsing and every float32 value was checked;
runtime uses JSON only. Upstream ESM2 code source:
<https://github.com/haiping1010/DeepCDR_esm2>.

CPU tests cover aliases, wrong antigen/fold identities, missing/changed assets,
native95 slots, strict features, typed failures and both architectures' actual
forward passes using **synthetic random state dictionaries**. These are not
learned-weight predictive validation, a four-model benchmark, corrected-library
acceptance or evidence of biological superiority. Public release additionally
requires the unresolved project and third-party rights decisions; no learned
weights are redistributed here.
