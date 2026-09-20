# Real-asset templates

These are configuration **templates**, not runnable scientific examples.
Replace every `REPLACE_WITH_...` value and supply authorized local assets.
No weights, targets, actual antibody inputs or download credentials are included.
Use [the synthetic example](../README.md) for a runnable no-model introduction.

| Template | Component | Extra input |
| --- | --- | --- |
| `jm.json` | iMut-CDR-JM | Prepared founder and explicit proposal JSONL |
| `jm_epi.json` | iMut-CDR-JM-Epi | The same plus the target's native antigen graph |
| `deepcdr3d.json` | DeepCDR-3D | Actual mutant H/L folds with provenance and an antigen pocket |

After completing the templates and preparing the corresponding inputs:

```bash
imut-cdr-gc generate --model jm --version v1 \
  --founder inputs/founder.json --input inputs/proposals.jsonl \
  --config examples/configs/jm.json --output-dir outputs/jm_candidates

imut-cdr-gc generate --model jm-epi --version v1 \
  --founder inputs/founder.json --input inputs/proposals.jsonl \
  --config examples/configs/jm_epi.json --output-dir outputs/jm_epi_candidates

imut-cdr-gc score --models 3d --input inputs/fold_manifest.jsonl \
  --config examples/configs/deepcdr3d.json --output-dir outputs/deepcdr3d_scores
```

All paths resolve from the working project. Pin every config/tokenizer file
actually present; do not keep missing files or omit additional recognized
tokenizer files. The checkpoint must match the selected architecture. JM has
no antigen graph; supplying one is an error. JM-Epi requires its own native
graph format, not a DeepCDR graph.

The explicit sampling values are API examples, not a reproduction of a research
campaign's schedule. A candidate may keep requested residues unchanged. These
commands generate proposals or scores, not a selected V3 library.

The 30-dimensional DeepCDR feature table is supplied under
`imut_cdr_gc/scoring/_vendor/`; place it in the working project's declared asset
location and hash that actual file. Antigen chain declarations must match the
prepared pocket's atomic sequences exactly. A founder-sidechain reconstruction
is not an antibody fold and cannot substitute for the required fold input.

See [generation](../../docs/generation.md), [scoring](../../docs/scoring.md),
[pipeline configuration](../../docs/pipeline.md), and
[third-party terms](../../THIRD_PARTY.md).
