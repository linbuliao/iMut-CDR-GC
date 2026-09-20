"""One-to-one mutant/full-complex export with actual atom and sequence checks.

Identity verification is not structure-quality or binding validation. Founder
side-chain reconstruction is labelled separately from an antibody fold.
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import shutil

from .records import new_output, relative_file, sequence, sha256, write_json, write_jsonl
from .preparation import prepare_founder

RESIDUES = dict(zip(
    "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
    "ACDEFGHIKLMNPQRSTVWY"))
SIDECHAINS = dict(zip("ACDEFGHIKLMNPQRSTVWY", (
    "CB", "CB SG", "CB CG OD1 OD2", "CB CG CD OE1 OE2", "CB CG CD1 CD2 CE1 CE2 CZ", "",
    "CB CG ND1 CD2 CE1 NE2", "CB CG1 CG2 CD1", "CB CG CD CE NZ", "CB CG CD1 CD2",
    "CB CG SD CE", "CB CG OD1 ND2", "CB CG CD", "CB CG CD OE1 NE2", "CB CG CD NE CZ NH1 NH2",
    "CB OG", "CB OG1 CG2", "CB CG1 CG2", "CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2",
    "CB CG CD1 CD2 CE1 CE2 CZ OH")))


def inspect_pdb(path, *, require_complete_atoms=True):
    chains, model_count, last, closed = OrderedDict(), 0, None, set()
    with Path(path).open(encoding="ascii") as stream:
        for line in stream:
            if line.startswith("MODEL "):
                model_count += 1
                if model_count > 1:
                    raise ValueError("Exactly one coordinate model is required")
            if not line.startswith("ATOM  "):
                continue
            if len(line.rstrip("\r\n")) < 54:
                raise ValueError("Truncated PDB ATOM record")
            atom, alternate, resname = line[12:16].strip(), line[16], line[17:20].strip()
            chain = line[21]
            if alternate not in (" ", "A"):
                raise ValueError("Resolve alternate conformers explicitly before delivery")
            if resname not in RESIDUES or chain.isspace():
                raise ValueError("Explicit protein chain IDs and standard residues are required")
            number, insertion = int(line[22:26]), line[26]
            coords = tuple(float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54)))
            if not all(math.isfinite(v) for v in coords):
                raise ValueError("Nonfinite PDB coordinate")
            key = (chain, number, insertion)
            if last != key:
                if key in closed:
                    raise ValueError("Discontinuous or duplicated residue identifiers")
                if last is not None:
                    closed.add(last)
                last = key
            residues = chains.setdefault(chain, OrderedDict())
            residue = residues.setdefault((number, insertion), {"aa": RESIDUES[resname], "atoms": set()})
            if residue["aa"] != RESIDUES[resname] or atom in residue["atoms"]:
                raise ValueError("Conflicting residue identity or duplicate atom")
            residue["atoms"].add(atom)
    if not chains:
        raise ValueError("PDB contains no protein atoms")
    for residues in chains.values():
        for residue in residues.values():
            required = {"N", "CA", "C", "O"} | set(SIDECHAINS[residue["aa"]].split())
            if "CA" not in residue["atoms"] or (require_complete_atoms and not required <= residue["atoms"]):
                raise ValueError("Missing backbone or residue-specific side-chain heavy atoms")
    return {chain: "".join(residue["aa"] for residue in residues.values())
            for chain, residues in chains.items()}


def verify_founder_complex(founder, root):
    founder = prepare_founder(founder)
    spec = founder.get("complex_reference", {})
    if spec.get("antigen_id") != founder["antigen_id"]:
        raise ValueError("Founder requires a fixed full-complex antigen reference")
    source = relative_file(root, spec["path"], spec["sha256"])
    expected = dict(spec.get("antigen_chains", {}))
    if not expected:
        raise ValueError("Founder reference must declare full antigen chain sequences")
    for role in ("heavy", "light"):
        chain = spec.get(role + "_chain")
        if not isinstance(chain, str) or len(chain) != 1 or chain in expected:
            raise ValueError("Founder H/L and antigen roles must be distinct")
        expected[chain] = founder[role] + founder.get(role + "_structure_suffix", "")
    if inspect_pdb(source) != expected:
        raise ValueError("Physical founder complex differs from its fixed full-chain reference")
    return founder


def verify_complex(row, root, founder):
    if any(row.get("founder", {}).get(key) != founder[key] for key in ("id", "heavy", "light")):
        raise ValueError("Candidate belongs to a different founder")
    if row.get("antigen_id") != founder["antigen_id"]:
        raise ValueError("Candidate target differs from the fixed founder antigen")
    complex_spec = row.get("complex", {})
    reference = founder["complex_reference"]
    if (complex_spec.get("antigen_id") != founder["antigen_id"]
            or complex_spec.get("antigen_chains") != reference["antigen_chains"]
            or complex_spec.get("heavy_chain") != reference["heavy_chain"]
            or complex_spec.get("light_chain") != reference["light_chain"]):
        raise ValueError("Complex roles/antigen differ from the fixed full founder reference")
    path = relative_file(root, complex_spec["path"], complex_spec["sha256"])
    if complex_spec.get("provenance", {}).get("kind") not in (
            "antibody_fold_aligned_to_antigen", "founder_sidechain_reconstruction"):
        raise ValueError("Explicit real complex construction provenance is required")
    chains = inspect_pdb(path)
    heavy_chain, light_chain = complex_spec.get("heavy_chain"), complex_spec.get("light_chain")
    if not heavy_chain or not light_chain or heavy_chain == light_chain:
        raise ValueError("Distinct antibody H/L chain roles are required")
    expected_antigens = complex_spec.get("antigen_chains")
    if not isinstance(expected_antigens, dict) or not expected_antigens:
        raise ValueError("Expected antigen chain sequences are required, not just chain presence")
    if {heavy_chain, light_chain} & set(expected_antigens):
        raise ValueError("Antibody and antigen chain roles overlap")
    if set(chains) != {heavy_chain, light_chain} | set(expected_antigens):
        raise ValueError("PDB chain inventory differs from the explicit complex")
    for role, chain in (("heavy", heavy_chain), ("light", light_chain)):
        mutant = sequence(row.get(role), role)
        suffix = founder.get(role + "_structure_suffix", "")
        if suffix:
            sequence(suffix, role + " constant suffix")
        if chains.get(chain) != mutant + suffix:
            raise ValueError("PDB does not contain the actual full mutant chain plus declared founder suffix")
    for chain, expected in expected_antigens.items():
        if chains.get(chain) != sequence(expected, "antigen"):
            raise ValueError("Antigen sequence differs from the prepared complex")
    return path, {"id": row["id"], "source_path": complex_spec["path"],
                  "source_sha256": complex_spec["sha256"], "chain_sequences": chains,
                  "construction": complex_spec["provenance"], "all_required_heavy_atoms_present": True,
                  "sequence_identity_verified": True, "binding_or_structural_quality_validated": False}


def export_library(rows, *, project_root, output_dir, prepared_founder):
    if not rows:
        raise ValueError("Cannot export an empty library")
    if len({row.get("id") for row in rows}) != len(rows):
        raise ValueError("Duplicate library IDs")
    if len({(row.get("heavy"), row.get("light")) for row in rows}) != len(rows):
        raise ValueError("Duplicate full H/L sequence pairs")
    # Refuse the entire export before creating outputs if any source is invalid.
    founder = verify_founder_complex(prepared_founder, project_root)
    verified = [verify_complex(row, project_root, founder) for row in rows]
    output = new_output(project_root, output_dir)
    (output / "complexes").mkdir()
    exported, manifest = [], []
    try:
        for index, (row, (source, identity)) in enumerate(zip(rows, verified), 1):
            stem = hashlib.sha256(row["id"].encode()).hexdigest()[:12]
            destination = output / "complexes" / f"sequence_{index:05d}_{stem}.pdb"
            if sha256(source) != identity["source_sha256"]:
                raise ValueError("Source changed after physical verification")
            shutil.copyfile(source, destination)
            if sha256(destination) != identity["source_sha256"]:
                raise ValueError("PDB copy differs from verified source")
            relative = destination.relative_to(Path(project_root).resolve()).as_posix()
            manifest.append({**identity, "pdb_path": relative, "pdb_sha256": sha256(destination)})
            exported.append({**row, "delivered_pdb_path": relative,
                             "delivered_pdb_sha256": sha256(destination)})
        write_jsonl(output / "library.jsonl", exported)
        write_jsonl(output / "pdb_manifest.jsonl", manifest)
        with (output / "sequences.fasta").open("x") as stream:
            for row in exported:
                for chain in ("heavy", "light"):
                    identifier = row["id"].replace("\n", " ").replace("\r", " ")
                    stream.write(f">{identifier}|{chain}\n{row[chain]}\n")
        receipt = {"status": "sequence_and_atom_identity_verified", "count": len(rows),
                   "recorded_utc": datetime.now(timezone.utc).isoformat(),
                   "library_sha256": sha256(output / "library.jsonl"),
                   "pdb_manifest_sha256": sha256(output / "pdb_manifest.jsonl"),
                   "founder_reference": founder["complex_reference"],
                   "paths_relative_to_project_root": True,
                   "scientific_protocol_acceptance": False,
                   "new_structures_generated_by_this_command": 0}
        write_json(output / "export.json", receipt)
        return receipt
    except BaseException as error:
        write_json(output / "failure.json", {"status": "failed", "error": type(error).__name__ + ": " + str(error),
                                             "partial_output_not_a_delivery": True})
        raise
