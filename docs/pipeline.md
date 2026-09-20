# Running a pipeline

`imut-cdr-gc run` executes an ordered list of existing commands from a JSON file.
It uses relative paths, not shell snippets, cluster settings or implicit model
downloads. Start with the [synthetic example](../examples/README.md).

```bash
imut-cdr-gc run --config examples/demo_pipeline.json --dry-run
imut-cdr-gc run --config examples/demo_pipeline.json
```

## Configuration

```json
{
  "schema": "imut-cdr-gc-pipeline.v1",
  "output_dir": "outputs/pipeline_run",
  "steps": [
    {
      "name": "prepare",
      "command": "prepare",
      "arguments": {
        "founder": "inputs/founder.json",
        "output_dir": "outputs/prepared"
      }
    }
  ]
}
```

`output_dir` at the top level stores pipeline execution information. Each step
also needs its own new output directory. Argument names match the corresponding
CLI options, with underscores instead of hyphens; list-valued options use JSON
arrays. Use `imut-cdr-gc COMMAND --help` for the command's required arguments.
Paths resolve from the working project, not from the configuration's directory.
The default project root is `.`; an explicit root goes before the command:

```bash
imut-cdr-gc --project-root . run --config examples/demo_pipeline.json
```

## Inputs and outputs

| Stage | Reads | Writes |
| --- | --- | --- |
| `prepare` | Full founder H/L chains and CDR mapping | `founder.json` |
| `prepare-antigen` | Prepared pocket atom arrays and native feature map | `antigen_graph.npz` |
| `generate --model jm` | Prepared founder, proposal requests and local JM assets | `candidates.jsonl` |
| `generate --model jm-epi` | The same input types plus an explicit antigen graph | `candidates.jsonl` |
| `fold` | Actual candidate H/L chains and local folding assets | `fold_manifest.jsonl` and antibody-only PDBs |
| `score` | Candidate records, target identity and model assets | `scores.jsonl` |
| `likelihood` | JM-Epi records and native likelihood assets | `scores.jsonl` with retained metadata |
| `select` | Scored JM-Epi records and a reference-bound policy | `selected.jsonl`, `selection_summary.json` |
| `rebuild-complexes` | Candidate records and a verified founder complex | `reconstructed.jsonl` and fixed-pose complexes |
| `export` | Records with verified complex PDBs | `library.jsonl`, FASTA and PDB manifest |

Connect stages by declaring the earlier output file as the next input. Nothing
is inferred from a filename: folds need sequence/structure identity, scores
retain their model/input identity, and export needs actual matching PDB files.
See [scoring](scoring.md) and [reconstruction](rebuilding.md) for these formats.

JM generation is unscreened. Only JM-Epi candidates may enter the supplied
likelihood/P5/GC selection stages. External comparison methods may receive
post-hoc property scores, not JM-specific selection labels.

## Failures and scientific scope

The runner stops at an error; it does not retry, overwrite earlier outputs or
replace missing measurements with zero. Keep partial output and the recorded
failure, inspect the cause, and use a new output directory for a new attempt.
`--dry-run` validates the plan without executing stages or loading models; it
does not certify that model assets are compatible or that scientific results
will be complete.

This is file-based orchestration, not an automatic V3 generation controller.
A genuine V3 candidate requires its measured gate and lineage at every joint
proposal. A single endpoint score, a syntactically valid config or a completed
synthetic demo does not establish that lineage.
