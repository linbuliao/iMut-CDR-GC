# Founder-pose full-complex side-chain reconstruction

`rebuilding.rebuild_complexes` makes an explicitly labelled approximate mutant
complex from a verified founder antibody–antigen pose. It is **not antibody
folding, backbone prediction, docking, binding validation, or final campaign
selection**. `folding` and `complexes.export_library` are separate operations.
Exporting a previously generated PDB does not generate a new structure.

## API

```python
from imut_cdr_gc.rebuilding import inspect_rebuilding_runtime, rebuild_complexes

# Optional metadata inspection: no reconstruction or GPU context.
runtime = inspect_rebuilding_runtime(project_root=project_root, cpu_threads=1)
# Review and save the runtime identity before an explicitly authorized attempt.
rows, receipt = rebuild_complexes(
    records,
    project_root=project_root,
    output_dir="analysis/sidechain_canary",
    prepared_founder=prepared_founder,
    runtime_identity=runtime,
)
```

Both calls use a fresh subprocess of the current Python executable. Therefore
that interpreter must already have compatible PDBFixer/OpenMM installed. The
module never installs dependencies, downloads structures, chooses a GPU, or
changes a running campaign. Importing the module does not import those optional
dependencies. Project-local temporary files are removed after success or failure.

`runtime_identity` is a strict object with:

- `schema: "founder-sidechain-cpu-runtime.v1"`, `platform: "CPU"`, and explicit
  integer `cpu_threads` from 1 through 8;
- `python_version`, `openmm_version`;
- `implementation_sha256` for `pdbfixer.pdbfixer`, `openmm.app.pdbfile`,
  `openmm.app.modeller`, and `openmm._openmm`;
- `physical_asset_files_sha256`, a package-relative map covering PDBFixer
  `soft.xml`/templates and OpenMM application data;
- `physical_assets_sha256`, the SHA256 of the sorted compact JSON encoding of
  that map.

The actual runtime and assets must equal this identity before and after physical
work. Identity inspection is not scientific approval or proof of numerical
equivalence. System/shared libraries beyond the recorded Python extension and
package assets are not a completely hermetic environment.

## Input and structural contract

`prepared_founder` must pass `complexes.verify_founder_complex`: explicit
project-relative PDB path and SHA256, distinct H/L/antigen chain roles, full
antigen sequences, full founder H/L sequences, and a validated FR277 CDR mapping.
Declared constant-region `heavy_structure_suffix` and `light_structure_suffix`
are preserved exactly, never inferred or trimmed.

Every candidate requires unique `id`, `design`, `antigen_id`, `founder`, `heavy`,
and `light`. Full H/L sequences are checked by `preparation.reconstruct` against
the prepared founder. If FR277 is absent, the representation is derived only
from those full chains at the verified CDR mapping, then reconstructed again;
any framework/unmapped difference fails. Supplied FR277, mutation count/list,
and indexing convention must agree. No unknown residues, insertion/deletion,
chain truncation, or hidden sequence repair is accepted. All original metadata
and row order are preserved. Different IDs with the same H/L pair remain
different reconstruction records; this builder performs no selection or dedup.
The downstream final-library exporter separately rejects duplicate pairs.

Existing `complex` or delivered-PDB fields, including `null`, are refused rather
than overwritten. Output directories must be new, project-relative, and
date-free. Candidate errors, physical-runtime exceptions, malformed results,
source races, and missing atoms stop the entire attempt. They are not biological
screen failures, zero scores, or silently skipped candidates. There is no
automatic resume over a prior output directory.

## Physical operation and validation

The native reference is `mutate_and_minimize.py` at SHA256
`a7a18c92ca662739651908a27f7a2c7f8ecbae05f13eed3495245df1f8af4d47`.
This implementation preserves its fast-path operation order:

1. Read ATOM records and renumber each chain sequentially, keeping an explicit
   original residue-number/insertion-code → sequential-index mapping. Insertion
   residues remain distinct; their original labels are recorded, not collapsed.
   HETATM waters/ligands are excluded, with their count reported. This is a
   protein ATOM complex, not a promise to preserve every founder file record.
2. Diff the actual full H/L sequences; apply the heavy mutations, then light
   mutations using `PDBFixer.applyMutations`.
3. Call `findMissingResidues`, set `missingResidues={}`, then
   `findMissingAtoms`, `addMissingAtoms`, and `addMissingHydrogens(7.0)`.
4. Write with `PDBFile.writeFile(..., keepIds=True)`.

The successor explicitly constructs PDBFixer with
`openmm.Platform.getPlatformByName("CPU")`. In the separately configured child,
`OPENMM_CPU_THREADS` is bounded and `OPENMM_DEFAULT_PLATFORM=CPU`; custom
`OPENMM_FORCEFIELD_DIR` is removed because it is not in the bound package asset
manifest. The parent's environment is untouched. This differs from the native
fast-path constructor's unspecified platform. Do not describe it as already
numerically equivalent to that historical runtime.

There is no **explicit final** minimization. PDBFixer atom/hydrogen placement
can itself perform energy optimization; “no energy operation” would be wrong.
There is no clash-free or physically relaxed structure guarantee.

Afterward, actual PDB residues must exactly match mutant H/L plus declared
constant suffixes and all fixed antigen sequences. Every standard residue must
contain its required backbone and side-chain heavy atoms. All original antigen
heavy atoms and antibody backbone heavy atoms (including original OXT where
present) must remain within **0.001 Å per Cartesian coordinate**, one PDB decimal
grid step. These checks are performed independently in worker and parent.
Hydrogen positions are not claimed to be preserved; newly added hydrogens are
not heavy-atom completeness evidence. Geometry, stereochemistry, affinity, and
biological activity remain unvalidated.

## Outputs and interpretation

The function returns `(rows, receipt)` and writes:

- `reconstructed.jsonl`: unchanged scientific metadata plus explicit `complex`
  path/hash/roles and `founder_sidechain_reconstruction` provenance;
- `complexes/sequence_<index>_<id-hash>.pdb`: one verified approximate complex
  for each input ID, with new sequential residue numbering;
- `founder_residue_mapping.json`: original PDB numbering/insertion codes;
- `pdb_manifest.jsonl`: IDs, PDB hashes, H/L hashes, verified mutations, and
  measured fixed-coordinate deviations;
- `rebuild.json`: input, source, physical-runtime and output hashes, counts,
  and explicit unvalidated scientific-status flags.

On failure, `failure.json` marks the attempt incomplete; any partial files are
not a delivery. Do not merge them into a final library by file existence alone.
Successful rows can be passed directly to `complexes.export_library` for a
separate one-to-one library export. Neither command marks corrected V3 selection
complete or promotes any production checkpoint.

## Validation status and native checks

The implementation is initially tested with an explicitly **mock physical
engine** and synthetic complete-atom PDBs. The CPU fixtures execute the real
request/worker, full-chain, mapping, coordinate, failure, output, and export
logic; they do not demonstrate actual OpenMM/PDBFixer performance or accuracy.
They never execute a GPU or native molecular calculation.

A separate real native check remains necessary. Use an authorized, explicitly
identified founder complex and actual candidate full H/L sequences. Record
their source hashes, the inspected PDBFixer/OpenMM runtime, the exact operation
profile, atom/residue identities, measured fixed-coordinate deviations and
export validation in a new project-relative directory. Preserve failures and
the original structures; do not substitute a different candidate after failure.

Synthetic checks do not certify native-runtime parity, throughput, physical
accuracy, production integration or newly selected complex counts. This source
package does not ship a private campaign record or prescribe access to a
particular server. It does not authorize molecular dynamics or change the
current study's paused MD status.
