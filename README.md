# iMut-CDR-GC

Focused antibody design from an existing candidate binder: propose CDR changes,
measure computational properties, and select an explicitly defined library.
Starting binders may have experimental or computational support; validated
binding is not assumed.

## Four components, one command-line interface

| Component | Purpose | Entry point |
| --- | --- | --- |
| **iMut-CDR-JM** | Joint CDR proposals without antigen conditioning | `imut-cdr-gc generate --model jm` |
| **iMut-CDR-JM-Epi** | Joint CDR proposals conditioned on an antigen-pocket graph | `imut-cdr-gc generate --model jm-epi` |
| **iMut-CDR-GC** | Explicit preparation, scoring, selection and export stages | `imut-cdr-gc run --config CONFIG.json` |
| **DeepCDR-3D** | Computational interaction scoring from an antibody fold and antigen pocket | `imut-cdr-gc score --models 3d` |

**DeepCDR-ESM2** is an additional antibody–antigen interaction scoring model
used in the GC screening workflow. See the
[original DeepCDR-ESM2 repository](https://github.com/haiping1010/DeepCDR_esm2)
for its implementation.

The package also provides likelihood analysis, antibody folding and founder-pose
side-chain reconstruction. Learned weights and research datasets are
**not included or downloaded automatically**.

## Quickstart: no models or GPU required

From a checkout of this repository, using Python 3.11 or later:

```bash
python -m pip install .
python examples/make_demo.py
imut-cdr-gc run --config examples/demo_pipeline.json
python examples/check_demo.py
```

This runs the real preparation and selection code on **eight synthetic test
records with invented scores**. It uses no trained model, real antibody, PDB or
scientific result. Two records are selected; every exclusion is recorded.

- Prepared input: `outputs/demo_prepared/founder.json`
- Selected test records: `outputs/demo_selected/selected.jsonl`
- Counts, decisions and cutoffs: `outputs/demo_selected/selection_summary.json`
- Pipeline execution record: `outputs/demo_run/`

Existing output directories are never overwritten. See
[the example guide](examples/README.md) to inspect the expected results, or
[the pipeline guide](docs/pipeline.md) to configure your own stages.

## Run with your own assets

Install the optional dependencies you need, then supply local model assets,
input files and their SHA256 hashes:

```bash
python -m pip install '.[generation,structure]'
imut-cdr-gc generate --help
imut-cdr-gc score --help
imut-cdr-gc run --help
```

All configured paths are relative to the working project; `--project-root .`
is the default. [Configuration templates](examples/configs/README.md) contain
deliberately invalid placeholders until you supply authorized assets.

Guides: [generation](docs/generation.md), [antigen graphs](docs/antigen_graph.md),
[DeepCDR scoring](docs/scoring.md), [naturalness](docs/naturalness.md),
[folding](docs/folding.md), [complex reconstruction](docs/rebuilding.md), and
[reproducibility](docs/reproducibility.md).

## Interpretation and limits

- A proposal requests at most ten CDR sites; requested residues may remain
  unchanged. Final substitutions are counted against the founder.
- V1 permits 3–10 final substitutions; V2/V3 permit 11–20. V3 requires a measured
  DeepCDR gate after each joint proposal. The configurable runner executes
  declared stages; it does not invent a production V3 iteration schedule.
- Likelihood/P5 and GC selection apply only to **JM-Epi**. Epi and external
  comparison methods are not screened by these stages. P5 is an inclusive
  lower bound from a separate reference, with no upper bound—not selection of
  the lowest-scoring 5%.
- DeepCDR scores are predictions, not measured affinity or specificity.
  AbNatiV2 measures human-repertoire compatibility. A fold and a fixed-pose
  side-chain reconstruction are distinct outputs; neither validates binding.

This is an alpha research implementation, not a complete trained-model or
scientific data release. See [scope](docs/status.md) and
[validation boundaries](VALIDATION.md). No molecular-dynamics results or
prospective experimental validation are supplied.

## Tests

```bash
python -m pip install '.[test,generation,structure]'
python -B -m pytest tests -p no:cacheprovider
```

This installs the optional model-test dependencies as well. Synthetic tests
check software behavior; they do not establish predictive accuracy. The
quickstart above remains a no-model check.

## Attribution and licensing

Code sources: [iMut-CDR](https://github.com/linbuliao/iMut-CDR) and
[DeepCDR-ESM2](https://github.com/haiping1010/DeepCDR_esm2).
DeepCDR-3D is integrated here; its dual-GCN architecture is not claimed as a
newly invented graph architecture.

The rights holder has not yet selected this project's software license.
See [LICENSE_REVIEW.md](LICENSE_REVIEW.md) and [THIRD_PARTY.md](THIRD_PARTY.md)
before reuse or redistribution. This repository does not grant rights to
third-party weights, datasets or generated outputs.
