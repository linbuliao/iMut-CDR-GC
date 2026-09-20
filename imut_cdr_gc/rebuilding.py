"""Explicit CPU-only founder-pose side-chain reconstruction, not folding/docking.

PDBFixer/OpenMM are optional and imported only in an isolated child process.
The child binds the installed implementation and physical assets before use.
CPU fixtures can exercise the complete I/O/validation contract without either
dependency; those fixtures are not evidence of physical reconstruction quality.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile

from .complexes import RESIDUES, inspect_pdb, verify_complex, verify_founder_complex
from .preparation import prepare_founder, reconstruct
from .records import new_output, relative_file, sequence, sha256, write_json, write_jsonl


NATIVE_REFERENCE_SHA256 = "a7a18c92ca662739651908a27f7a2c7f8ecbae05f13eed3495245df1f8af4d47"
RUNTIME_SCHEMA = "founder-sidechain-cpu-runtime.v1"
IMPLEMENTATION_MODULES = ("pdbfixer.pdbfixer", "openmm.app.pdbfile",
                          "openmm.app.modeller", "openmm._openmm")
COORDINATE_TOLERANCE_ANGSTROM = 0.001
_BACKBONE = frozenset(("N", "CA", "C", "O", "OXT"))
_THREE = {v: k for k, v in RESIDUES.items()}


class RebuildingError(RuntimeError):
    """A reconstruction attempt failed; it is not a screening rejection."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _threads(value):
    if type(value) is not int or not 1 <= value <= 8:
        raise ValueError("cpu_threads must be an explicit integer from 1 through 8")
    return value


def _valid_sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _runtime_spec(value):
    result = deepcopy(value)
    if not isinstance(result, dict) or result.get("schema") != RUNTIME_SCHEMA:
        raise ValueError("An explicit founder-sidechain CPU runtime identity is required")
    if result.get("platform") != "CPU":
        raise ValueError("Only the explicit OpenMM CPU platform is allowed")
    _threads(result.get("cpu_threads"))
    for key in ("python_version", "openmm_version"):
        if not isinstance(result.get(key), str) or not result[key]:
            raise ValueError(f"Runtime identity requires {key}")
    implementations = result.get("implementation_sha256", {})
    if set(implementations) != set(IMPLEMENTATION_MODULES) or not all(
            _valid_sha(v) for v in implementations.values()):
        raise ValueError("Runtime requires exact implementation module SHA256 identities")
    assets = result.get("physical_asset_files_sha256")
    if not isinstance(assets, dict) or not assets or any(
            not isinstance(k, str) or not k.startswith(("pdbfixer/", "openmm/app/"))
            or ".." in Path(k).parts or not _valid_sha(v) for k, v in assets.items()):
        raise ValueError("Runtime requires explicit package-relative physical asset SHA256 identities")
    if result.get("physical_assets_sha256") != _digest(assets):
        raise ValueError("Physical asset manifest SHA256 differs")
    return result


def _source_identity():
    return {name: sha256(Path(__file__).with_name(name + ".py"))
            for name in ("rebuilding", "complexes", "preparation", "records")}


def _physical_runtime(cpu_threads):
    """Run only in the explicitly CPU-configured child; no GPU context is made."""
    if os.environ.get("OPENMM_CPU_THREADS") != str(_threads(cpu_threads)):
        raise RebuildingError("CPU thread environment was not explicitly bounded before imports")
    modules = {name: importlib.import_module(name) for name in IMPLEMENTATION_MODULES}
    openmm = importlib.import_module("openmm")
    pdbfixer = importlib.import_module("pdbfixer")
    cpu = openmm.Platform.getPlatformByName("CPU")
    if cpu.getName() != "CPU":
        raise RebuildingError("Requested CPU platform was not obtained")
    fix_root = Path(pdbfixer.__file__).resolve().parent
    app_root = Path(modules["openmm.app.pdbfile"].__file__).resolve().parent
    assets = {}
    for prefix, base, paths in (
            ("pdbfixer", fix_root, [fix_root / "soft.xml", *sorted((fix_root / "templates").rglob("*"))]),
            ("openmm/app", app_root, sorted((app_root / "data").rglob("*")))):
        for path in paths:
            if path.is_file():
                assets[prefix + "/" + path.relative_to(base).as_posix()] = sha256(path)
    if "pdbfixer/soft.xml" not in assets or not any(k.startswith("pdbfixer/templates/") for k in assets):
        raise RebuildingError("PDBFixer physical template assets are missing")
    if not any(k.startswith("openmm/app/data/") for k in assets):
        raise RebuildingError("OpenMM physical data assets are missing")
    identity = {"schema": RUNTIME_SCHEMA, "platform": "CPU", "cpu_threads": cpu_threads,
                "python_version": platform.python_version(), "openmm_version": openmm.__version__,
                "implementation_sha256": {name: sha256(module.__file__) for name, module in modules.items()},
                "physical_asset_files_sha256": assets, "physical_assets_sha256": _digest(assets)}
    return _runtime_spec(identity), modules["pdbfixer.pdbfixer"].PDBFixer, modules["openmm.app.pdbfile"].PDBFile, cpu


def _atoms(path):
    """Retain original number/insertion identity and ordered ATOM coordinates."""
    inspect_pdb(path)
    chains = OrderedDict()
    for line in Path(path).read_text(encoding="ascii").splitlines():
        if not line.startswith("ATOM  "):
            continue
        residues = chains.setdefault(line[21], OrderedDict())
        residue = residues.setdefault((int(line[22:26]), line[26]),
                                      {"aa": RESIDUES[line[17:20].strip()], "atoms": {}})
        name = line[12:16].strip()
        element = line[76:78].strip() if len(line) >= 78 else ""
        is_hydrogen = element in ("H", "D") or (not element and name.lstrip("0123456789").startswith(("H", "D")))
        residue["atoms"][name] = {"coordinates": tuple(float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))),
                                   "hydrogen": is_hydrogen}
    return chains


def _renumber_template(source, destination):
    atoms = _atoms(source)
    mapping = {chain: [{"chain_index": i, "sequential_residue_number": i + 1,
                        "original_residue_number": key[0], "original_insertion_code": key[1], "amino_acid": res["aa"]}
                       for i, (key, res) in enumerate(residues.items())]
               for chain, residues in atoms.items()}
    lookup = {(chain, r["original_residue_number"], r["original_insertion_code"]): r["sequential_residue_number"]
              for chain, residues in mapping.items() for r in residues}
    counts = {"source_atom_records": 0, "excluded_hetatm_records": 0}
    with Path(destination).open("x", encoding="ascii") as output:
        for line in Path(source).read_text(encoding="ascii").splitlines(keepends=True):
            if line.startswith("ATOM  "):
                number = lookup[(line[21], int(line[22:26]), line[26])]
                if number > 9999:
                    raise ValueError("PDB sequential residue numbering exceeds four columns")
                output.write(line[:22] + f"{number:4d} " + line[27:])
                counts["source_atom_records"] += 1
            elif line[:6] in ("TER   ", "END   ", "ENDMDL", "MODEL ") or line.strip() == "END":
                output.write(line)
            elif line.startswith("HETATM"):
                counts["excluded_hetatm_records"] += 1
    return mapping, counts


def _candidate(row, founder):
    if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
        raise ValueError("Every candidate requires a unique nonempty string id")
    if any(key in row for key in ("complex", "delivered_pdb_path", "delivered_pdb_sha256")):
        raise ValueError("Existing complex evidence must not be overwritten; use fresh candidate records")
    supplied = prepare_founder(row.get("founder", {}))
    for key in ("id", "design", "antigen_id", "heavy", "light", "fr_cdr_seq", "cdr_position_map"):
        if supplied[key] != founder[key]:
            raise ValueError("Candidate belongs to a different founder or CDR mapping")
    for key in ("complex_reference", "heavy_structure_suffix", "light_structure_suffix"):
        if key in supplied and supplied[key] != founder.get(key):
            raise ValueError("Candidate declares a conflicting founder complex reference/suffix")
    if row.get("design") != founder["design"] or row.get("antigen_id") != founder["antigen_id"]:
        raise ValueError("Candidate design/antigen differs from the fixed founder")
    chains = {role: sequence(row.get(role), role) for role in ("heavy", "light")}
    if any(len(chains[role]) != len(founder[role]) for role in chains):
        raise ValueError("Full H/L chain lengths must match; only substitutions are supported")
    fr = list(founder["fr_cdr_seq"])
    for position, entry in founder["cdr_position_map"].items():
        fr[int(position)] = chains[entry["chain"]][entry["index"]]
    fr = "".join(fr)
    rebuilt = reconstruct(founder, fr)
    if any(rebuilt[role] != chains[role] for role in chains):
        raise ValueError("Actual full chains contain a framework/unmapped change")
    for key in ("fr_cdr_seq", "mutant_fr_cdr_seq"):
        if key in row and row[key] != fr:
            raise ValueError("Candidate FR277 differs from its actual full H/L chains")
    if "founder_fr_cdr_seq" in row and row["founder_fr_cdr_seq"] != founder["fr_cdr_seq"]:
        raise ValueError("Candidate founder FR277 differs")
    for key in ("mutations", "mutation_count", "position_index_base"):
        if key in row and key != "mutations" and type(row[key]) is not int:
            raise ValueError("Candidate mutation count/index must be an actual integer")
        if key in row and row[key] != rebuilt[key]:
            raise ValueError("Candidate mutation metadata differs from actual full chains")
    return rebuilt


def _coordinate_check(founder_path, mutant_path, roles):
    if any(line.startswith("HETATM") for line in Path(mutant_path).read_text(encoding="ascii").splitlines()):
        raise ValueError("Unexpected HETATM output in a protein-ATOM reconstruction")
    original, mutant = _atoms(founder_path), _atoms(mutant_path)
    if set(original) != set(mutant):
        raise ValueError("Reconstructed chain inventory differs")
    checked, maximum = {"antigen_heavy_atoms": 0, "antibody_backbone_heavy_atoms": 0}, 0.0
    for chain, residues in original.items():
        new = mutant[chain]
        if len(new) != len(residues) or list(new) != [(i + 1, " ") for i in range(len(residues))]:
            raise ValueError("Reconstructed residue count/sequential numbering differs")
        is_antigen = chain in roles["antigen_chains"]
        for old_res, new_res in zip(residues.values(), new.values()):
            for name, atom in old_res["atoms"].items():
                if atom["hydrogen"] or (not is_antigen and name not in _BACKBONE):
                    continue
                if name not in new_res["atoms"]:
                    raise ValueError("A fixed antigen/backbone atom is missing")
                delta = max(abs(a - b) for a, b in zip(atom["coordinates"], new_res["atoms"][name]["coordinates"]))
                if delta > COORDINATE_TOLERANCE_ANGSTROM + 1e-9:
                    raise ValueError("Antigen/backbone coordinates moved beyond PDB precision")
                maximum = max(maximum, delta)
                checked["antigen_heavy_atoms" if is_antigen else "antibody_backbone_heavy_atoms"] += 1
    return {**checked, "maximum_coordinate_component_difference_angstrom": maximum,
            "tolerance_per_coordinate_angstrom": COORDINATE_TOLERANCE_ANGSTROM,
            "hydrogen_coordinates_not_claimed_preserved": True}


def _graft(template, row, founder, output, fixer_class, pdb_class, cpu):
    fixer = fixer_class(filename=str(template), platform=cpu)
    for role in ("heavy", "light"):
        chain = founder["complex_reference"][role + "_chain"]
        suffix = founder.get(role + "_structure_suffix", "")
        before, after = founder[role] + suffix, row[role] + suffix
        mutations = [f"{_THREE[a]}-{i}-{_THREE[b]}" for i, (a, b) in enumerate(zip(before, after), 1) if a != b]
        if mutations:
            fixer.applyMutations(mutations, chain)
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    with Path(output).open("x", encoding="ascii") as stream:
        pdb_class.writeFile(fixer.topology, fixer.positions, stream, keepIds=True)


def _invoke_child(request_path, response_path, threads):
    # Package code is selected explicitly, including when run from an unpacked
    # source tree. The parent interpreter's sys.path/environment are untouched.
    command = [sys.executable, "-I", "-B", "-c",
               "import sys; sys.path.insert(0, sys.argv[1]); from imut_cdr_gc.rebuilding import _worker_cli; _worker_cli(sys.argv[2:])",
               str(Path(__file__).resolve().parent.parent), str(request_path), str(response_path), sha256(request_path)]
    environment = dict(os.environ)
    environment.update(OPENMM_CPU_THREADS=str(_threads(threads)), OPENMM_DEFAULT_PLATFORM="CPU",
                       TMPDIR=str(request_path.parent), TEMP=str(request_path.parent), TMP=str(request_path.parent))
    # Custom force-field search roots would invalidate the recorded package assets.
    environment.pop("OPENMM_FORCEFIELD_DIR", None)
    with subprocess.Popen(command, cwd=request_path.parent, env=environment,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        try:
            stdout, stderr = process.communicate()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise
    if not response_path.is_file():
        raise RebuildingError(f"CPU worker did not produce a response (exit {process.returncode}): {stderr[-2000:]}")
    response = json.loads(response_path.read_text())
    if process.returncode != 0 or response.get("status") != "complete":
        raise RebuildingError(f"CPU worker failed: {response.get('error', stderr[-2000:])}")
    return response


def _scratch(root):
    parent = root / ".cache" / "tmp"
    if not parent.resolve().is_relative_to(root):
        raise ValueError("Project temporary directory escapes project root")
    parent.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="rebuilding_", dir=parent)


def inspect_rebuilding_runtime(*, project_root, cpu_threads=1):
    """Capture installed CPU runtime/assets only; no reconstruction or GPU work.

    This is metadata inspection, not approval or an assertion of native parity.
    Save/review its return value and pass it explicitly to rebuild_complexes.
    """
    root = Path(project_root).resolve(strict=True)
    with _scratch(root) as folder:
        temporary = Path(folder)
        request = {"schema": "sidechain-worker-request.v1", "operation": "inspect_runtime",
                   "project_root": str(root), "cpu_threads": _threads(cpu_threads), "source_identity": _source_identity()}
        write_json(temporary / "request.json", request)
        response = _invoke_child(temporary / "request.json", temporary / "response.json", cpu_threads)
    return _runtime_spec(response["runtime_identity"])


def _worker_cli(arguments):
    request_path, response_path, expected = arguments
    request_path, response_path = Path(request_path), Path(response_path)
    current_id = None
    try:
        if sha256(request_path) != expected:
            raise ValueError("Worker request SHA256 differs")
        request = json.loads(request_path.read_text())
        root = Path(request["project_root"]).resolve(strict=True)
        if not request_path.resolve().is_relative_to(root) or response_path.parent != request_path.parent:
            raise ValueError("Worker paths must remain in its project temporary directory")
        if request.get("schema") != "sidechain-worker-request.v1" or request["source_identity"] != _source_identity():
            raise ValueError("Worker source identity changed")
        actual, fixer_class, pdb_class, cpu = _physical_runtime(request["cpu_threads"])
        if request["operation"] == "inspect_runtime":
            write_json(response_path, {"status": "complete", "runtime_identity": actual})
            return
        if request["operation"] != "rebuild" or _runtime_spec(request["runtime_identity"]) != actual:
            raise ValueError("Installed runtime/physical assets differ from explicit identity")
        founder = verify_founder_complex(request["founder"], root)
        snapshot = request["founder_snapshot"]
        if snapshot["sha256"] != founder["complex_reference"]["sha256"]:
            raise ValueError("Founder snapshot identity differs from the fixed reference")
        source = relative_file(root, snapshot["path"], snapshot["sha256"])
        snapshot_founder = dict(founder, complex_reference={**founder["complex_reference"], "path": snapshot["path"]})
        verify_founder_complex(snapshot_founder, root)
        template = request_path.parent / "template.pdb"
        mapping, template_counts = _renumber_template(source, template)
        products = []
        for index, row in enumerate(request["records"], 1):
            current_id = row["id"]
            _candidate(row, founder)
            name = f"sequence_{index:05d}_{hashlib.sha256(current_id.encode()).hexdigest()[:12]}.pdb"
            destination = request_path.parent / name
            _graft(template, row, founder, destination, fixer_class, pdb_class, cpu)
            coordinates = _coordinate_check(source, destination, founder["complex_reference"])
            products.append({"id": current_id, "file": name, "sha256": sha256(destination), "coordinate_verification": coordinates})
        relative_file(root, founder["complex_reference"]["path"], founder["complex_reference"]["sha256"])
        relative_file(root, snapshot["path"], snapshot["sha256"])
        if _physical_runtime(request["cpu_threads"])[0] != actual or request["source_identity"] != _source_identity():
            raise ValueError("Runtime/source changed during reconstruction")
        write_json(response_path, {"status": "complete", "runtime_identity": actual, "products": products,
                                   "mapping": mapping, "template_record_counts": template_counts})
    except BaseException as error:
        if not response_path.exists():
            write_json(response_path, {"status": "failed", "id": current_id,
                                       "error": type(error).__name__ + ": " + str(error),
                                       "scientific_protocol_acceptance": False})
        raise


def rebuild_complexes(records, *, project_root, output_dir, prepared_founder, runtime_identity):
    """Return ``(rows, receipt)`` for complete, identity-checked CPU reconstructions.

    Full candidate metadata and order are preserved. Duplicate IDs, stale
    inputs/runtime, non-CDR changes and existing complex evidence are refused.
    Any error aborts the attempt, writes failure.json and never becomes a
    failed biological screen. No automatic resume/overwrite is supported.
    """
    root = Path(project_root).resolve(strict=True)
    runtime = _runtime_spec(runtime_identity)
    founder = verify_founder_complex(prepared_founder, root)
    rows = deepcopy(list(records))
    if not rows:
        raise ValueError("Cannot rebuild an empty library")
    verified_mutations = [_candidate(row, founder) for row in rows]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate candidate IDs are not accepted")
    record_identity, source_identity = _digest(rows), _source_identity()
    output = new_output(root, output_dir)
    output_rows, manifest = [], []
    try:
        with _scratch(root) as folder:
            temporary = Path(folder)
            reference = founder["complex_reference"]
            source = relative_file(root, reference["path"], reference["sha256"])
            # Bind the bytes actually used by all physical and coordinate reads,
            # not only a hash of a mutable external source path.
            snapshot = temporary / "founder_input.pdb"
            with source.open("rb") as original, snapshot.open("xb") as copy:
                shutil.copyfileobj(original, copy)
            snapshot_spec = {"path": snapshot.relative_to(root).as_posix(), "sha256": reference["sha256"]}
            relative_file(root, snapshot_spec["path"], snapshot_spec["sha256"])
            request = {"schema": "sidechain-worker-request.v1", "operation": "rebuild", "project_root": str(root),
                       "records": rows, "founder": founder, "cpu_threads": runtime["cpu_threads"],
                       "runtime_identity": runtime, "source_identity": source_identity, "founder_snapshot": snapshot_spec}
            write_json(temporary / "request.json", request)
            response = _invoke_child(temporary / "request.json", temporary / "response.json", runtime["cpu_threads"])
            if response.get("runtime_identity") != runtime or len(response.get("products", [])) != len(rows):
                raise RebuildingError("Worker runtime or output count differs")
            if source_identity != _source_identity():
                raise RebuildingError("Rebuilding implementation changed during execution")
            relative_file(root, reference["path"], reference["sha256"])
            source = relative_file(root, snapshot_spec["path"], snapshot_spec["sha256"])
            for row, product in zip(rows, response["products"]):
                name = product["file"]
                if product["id"] != row["id"] or Path(name).name != name or not name.endswith(".pdb"):
                    raise RebuildingError("Worker output identity/order/path differs")
                staged = relative_file(root, (temporary / name).relative_to(root), product["sha256"])
                provenance = {"kind": "founder_sidechain_reconstruction", "runtime_identity": runtime,
                              "native_reference_source_sha256": NATIVE_REFERENCE_SHA256,
                              "explicit_cpu_successor_not_native_runtime_equivalence": True,
                              "founder_complex_sha256": reference["sha256"], "implementation_sha256": source_identity,
                              "new_backbone_prediction": False, "new_docking": False,
                              "explicit_final_minimization": False, "internal_atom_or_hydrogen_energy_placement_possible": True,
                              "scientific_protocol_acceptance": False}
                augmented = dict(row, complex={**reference, "path": staged.relative_to(root).as_posix(),
                                               "sha256": product["sha256"], "provenance": provenance})
                verify_complex(augmented, root, founder)
                actual_coordinates = _coordinate_check(source, staged, reference)
                if product.get("coordinate_verification") != actual_coordinates:
                    raise RebuildingError("Worker coordinate verification differs")
                output_rows.append(augmented)
                manifest.append({"id": row["id"], "sha256": product["sha256"], "coordinate_verification": actual_coordinates})
            # Check the independently reproduced insertion-code mapping before committing.
            mapping_probe = temporary / "mapping_check.pdb"
            actual_mapping, actual_counts = _renumber_template(source, mapping_probe)
            if response.get("mapping") != actual_mapping or response.get("template_record_counts") != actual_counts:
                raise RebuildingError("Founder numbering/ATOM mapping differs")
            (output / "complexes").mkdir()
            for augmented, item, mutations in zip(output_rows, manifest, verified_mutations):
                staged = relative_file(root, augmented["complex"]["path"], item["sha256"])
                destination = output / "complexes" / staged.name
                with staged.open("rb") as original, destination.open("xb") as target:
                    shutil.copyfileobj(original, target)
                if sha256(destination) != item["sha256"]:
                    raise RebuildingError("Committed PDB differs from verified reconstruction")
                augmented["complex"]["path"] = destination.relative_to(root).as_posix()
                item.update(path=augmented["complex"]["path"], verified_mutations=mutations["mutations"],
                            heavy_sha256=hashlib.sha256(augmented["heavy"].encode()).hexdigest(),
                            light_sha256=hashlib.sha256(augmented["light"].encode()).hexdigest())
            relative_file(root, reference["path"], reference["sha256"])
        write_json(output / "founder_residue_mapping.json", actual_mapping)
        write_jsonl(output / "reconstructed.jsonl", output_rows)
        write_jsonl(output / "pdb_manifest.jsonl", manifest)
        receipt = {"schema": "founder-sidechain-rebuild.v1", "status": "sequence_atom_and_fixed_coordinate_identity_verified",
                   "recorded_utc": datetime.now(timezone.utc).isoformat(), "input_count": len(rows), "completed_count": len(output_rows),
                   "input_records_sha256": record_identity, "source_identity": source_identity, "runtime_identity": runtime,
                   "founder_reference": founder["complex_reference"], "template_record_counts": actual_counts,
                   "reconstructed_jsonl_sha256": sha256(output / "reconstructed.jsonl"),
                   "pdb_manifest_sha256": sha256(output / "pdb_manifest.jsonl"),
                   "founder_residue_mapping_sha256": sha256(output / "founder_residue_mapping.json"),
                   "construction": "founder_sidechain_reconstruction", "new_backbone_predictions": 0, "new_docking_runs": 0,
                   "scientific_protocol_acceptance": False, "binding_or_structural_quality_validated": False,
                   "native_runtime_numerical_parity_validated": False, "paths_relative_to_project_root": True}
        write_json(output / "rebuild.json", receipt)
        return output_rows, receipt
    except BaseException as error:
        write_json(output / "failure.json", {"status": "failed", "error": type(error).__name__ + ": " + str(error),
                                             "input_count": len(rows), "input_records_sha256": record_identity,
                                             "partial_output_not_a_delivery": True, "scientific_protocol_acceptance": False})
        raise
