# Antigen-pocket graph preparation

The generator consumes a **materialized residue graph**, not arbitrary PDB atoms.
`imut_cdr_gc.antigen_graph.prepare_antigen_graph` converts an explicitly prepared
pocket into the four safe numeric arrays expected by `load_antigen_graph`.

Supply a project-relative atom NPZ and its SHA256, feature JSON and SHA256,
explicit antigen ID, cutoff and a new output directory. Atom columns are
`pos` (N×3 floating coordinates), `res_name`, `res_id` (integers), `chain_id` and
`atom_name`. Strings must be Unicode arrays, not object arrays requiring pickle.
An optional insertion-code column must contain only blanks. This utility does
not select a pocket, download structures, renumber a PDB or infer chain roles.
If numbering must change, retain an explicit original-to-prepared residue map.

Use the **actual JM-Epi training feature dictionary**; a DeepCDR feature table is
not automatically interchangeable merely because it also has 30 columns. Every
used residue needs 30 finite numbers. Unlike the native permissive loader, invalid
features are rejected rather than resized or replaced with unknown/zero vectors.
Unknown residues, ambiguous residue IDs, duplicate atoms and unresolved alternate
conformers are likewise rejected. Standard protein residues are supported here.

For valid inputs the executed trainer's arithmetic is preserved: node order is
first atom occurrence; coordinates use CA, or the all-atom mean when CA is absent;
sequence edges join adjacent **observed sorted residue IDs**, including numeric
gaps; directed contact edges use distance ≤ cutoff (default 8 Å). A pair may have
both a sequence edge and a contact edge. These duplicates are intentional and
must not be deduplicated. Features/coordinates/edge attributes are float32 and
edge indices int64.

The output records both input hashes, output hash, cutoff and node/edge counts.
Use its numeric `antigen_graph.npz` and recorded hash in the generator config.
This preparatory transform does not establish equivalence of another pocket,
feature dictionary, learned checkpoint, or a new scientific result.
