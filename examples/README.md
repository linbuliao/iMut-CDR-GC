# Synthetic quickstart

Run from the repository root after `python -m pip install .`:

```bash
python examples/make_demo.py
imut-cdr-gc run --config examples/demo_pipeline.json
python examples/check_demo.py
```

No model, GPU, network request or real antibody data is used. The short sequences
are format-testing strings, not antibody domains. All scores are invented. The
demo executes real preparation/selection code, **not generation or prediction**.
Synthetic markers remain in every selected record.

## Expected output

| Sequential decision | Records |
| --- | ---: |
| Input | 8 |
| Outside the allowed substitution count | 1 |
| Below reference P5 | 1 |
| Below both active DeepCDR floors | 1 |
| Missing a required score | 1 |
| Eligible before deduplication | 4 |
| Duplicate sequence pair removed | 1 |
| Eligible but outside Top-2 | 1 |
| Selected | 2 |

`outputs/demo_selected/selected.jsonl` contains `keep_high` and
`keep_p5_boundary`. `selection_summary.json` holds every record's decision,
counts, policy and observed ranking boundary. No PDB is created or implied.

The separate synthetic reference is `[-3, -2, -1, 0, 1]`. Linear interpolation
gives P5 = **-2.8**. Equality passes and there is no upper cutoff. This number is
only for the demo; never copy it into a scientific selection policy. Both mock
DeepCDR models use 0.5, with OR eligibility. Terminal ranking requires both
measurements; a missing score remains missing, even if the other score is high.
The last selected record has primary ranking score 0.8, which is distinct from
the configured eligibility floor.

## Inspect or rerun

```bash
python -m json.tool outputs/demo_selected/selection_summary.json
imut-cdr-gc prepare --help
imut-cdr-gc select --help
```

Before the first run, `imut-cdr-gc run --config examples/demo_pipeline.json
--dry-run` checks the plan without executing it. It also requires fresh outputs.

The maker and stages refuse existing output directories. Keep previous results;
use a fresh working project or deliberately choose new output directories in a
copy of the configuration. Generated `outputs/` files are not source assets and
must not be committed as scientific data.

[Real-asset configuration templates](configs/README.md) are separate from this
fully runnable synthetic example.
