"""CPU-only simulated physical engine: no OpenMM/PDBFixer execution or quality claim."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from imut_cdr_gc import rebuilding as rebuild
from imut_cdr_gc.complexes import RESIDUES, SIDECHAINS, export_library, inspect_pdb
from imut_cdr_gc.preparation import prepare_founder, reconstruct
from imut_cdr_gc.records import read_jsonl, sha256
from test_workflow_contract import RootFixture, founder, pdb_text, row


def runtime():
    assets = {"pdbfixer/soft.xml": "a" * 64, "pdbfixer/templates/GLY.pdb": "b" * 64,
              "openmm/app/data/hydrogens.xml": "c" * 64}
    return {"schema": rebuild.RUNTIME_SCHEMA, "platform": "CPU", "cpu_threads": 1,
            "python_version": "synthetic_cpu_fixture", "openmm_version": "not_a_real_engine",
            "implementation_sha256": {name: "d" * 64 for name in rebuild.IMPLEMENTATION_MODULES},
            "physical_asset_files_sha256": assets, "physical_assets_sha256": rebuild._digest(assets)}


class FakeEngine:
    """Retain original coordinates, substitute atom names: explicitly not physics."""
    def __init__(self):
        self.calls = []
        self.mutate_output = lambda text: text
        owner = self

        class Fixer:
            def __init__(self, *, filename, platform):
                owner.calls.append(("construct", platform))
                self.chains = inspect_pdb(filename)
                self.original = rebuild._atoms(filename)
                self.missingResidues = None
                self.topology, self.positions = self, "mock_positions_not_physical"

            def applyMutations(self, mutations, chain):
                owner.calls.append(("applyMutations", chain, mutations))
                sequence = list(self.chains[chain])
                for mutation in mutations:
                    old, index, new = mutation.split("-")
                    index = int(index) - 1
                    if sequence[index] != RESIDUES[old]:
                        raise ValueError("Synthetic engine received incorrect native residue")
                    sequence[index] = RESIDUES[new]
                self.chains[chain] = "".join(sequence)

            def findMissingResidues(self):
                owner.calls.append(("findMissingResidues",))
                self.missingResidues = {"must_be_cleared": True}

            def findMissingAtoms(self):
                if self.missingResidues != {}:
                    raise ValueError("Missing residues were not explicitly disabled")
                owner.calls.append(("findMissingAtoms",))

            def addMissingAtoms(self):
                owner.calls.append(("addMissingAtoms",))

            def addMissingHydrogens(self, ph):
                owner.calls.append(("addMissingHydrogens", ph))

        class PDB:
            @staticmethod
            def writeFile(topology, positions, stream, *, keepIds):
                owner.calls.append(("writeFile", keepIds))
                lines, serial = [], 0
                inverse = {v: k for k, v in RESIDUES.items()}
                for chain, sequence in topology.chains.items():
                    old = list(topology.original[chain].values())
                    for index, aa in enumerate(sequence, 1):
                        for atom in ["N", "CA", "C", "O", *SIDECHAINS[aa].split(), "H"]:
                            serial += 1
                            coords = old[index - 1]["atoms"].get(atom, {"coordinates": (float(index), 0., 0.)})["coordinates"]
                            lines.append(f"ATOM  {serial:5d} {atom:^4s} {inverse[aa]:3s} {chain}{index:4d}    "
                                         f"{coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}{1.:6.2f}{20.:6.2f}          {atom[0]:>2s}\n")
                stream.write(owner.mutate_output("".join(lines) + "END\n"))

        self.fixer, self.pdb, self.cpu = Fixer, PDB, SimpleNamespace(getName=lambda: "CPU")

    def physical_runtime(self, threads):
        result = runtime(); result["cpu_threads"] = threads
        return result, self.fixer, self.pdb, self.cpu


class RebuildingTests(RootFixture):
    def setUp(self):
        super().setUp()
        self.engine = FakeEngine()
        self.prepared = prepare_founder(founder())
        source = self.root / "founder.pdb"
        source.write_text(pdb_text({"H": self.prepared["heavy"], "L": self.prepared["light"], "C": "GG"}))
        self.prepared["complex_reference"] = {"path": "founder.pdb", "sha256": sha256(source),
             "heavy_chain": "H", "light_chain": "L", "antigen_id": self.prepared["antigen_id"], "antigen_chains": {"C": "GG"}}
        self.candidate = row()
        self.candidate["founder"] = deepcopy(self.prepared)
        self.identity = runtime()

    def invoke_fixture(self, request, response, threads):
        with patch.object(rebuild, "_physical_runtime", self.engine.physical_runtime):
            rebuild._worker_cli([str(request), str(response), sha256(request)])
        return json.loads(response.read_text())

    def run_rebuild(self, records=None, output="rebuilt", runtime_identity=None):
        with patch.object(rebuild, "_invoke_child", self.invoke_fixture):
            return rebuild.rebuild_complexes(records if records is not None else [self.candidate],
                project_root=self.root, output_dir=output, prepared_founder=self.prepared,
                runtime_identity=runtime_identity if runtime_identity is not None else self.identity)

    def test_complete_native_call_order_and_cpu_constructor(self):
        mutant = list(self.prepared["fr_cdr_seq"]); mutant[26] = "G"; mutant[165] = "A"
        candidate = {**self.candidate, **reconstruct(self.prepared, "".join(mutant)), "fr_cdr_seq": "".join(mutant)}
        records, receipt = self.run_rebuild([candidate])
        names = [call[0] for call in self.engine.calls]
        self.assertEqual(names, ["construct", "applyMutations", "applyMutations", "findMissingResidues",
                                 "findMissingAtoms", "addMissingAtoms", "addMissingHydrogens", "writeFile"])
        self.assertIs(self.engine.calls[0][1], self.engine.cpu)
        self.assertEqual([call[1] for call in self.engine.calls if call[0] == "applyMutations"], ["H", "L"])
        self.assertIn(("addMissingHydrogens", 7.0), self.engine.calls)
        self.assertIn(("writeFile", True), self.engine.calls)
        self.assertEqual(receipt["completed_count"], 1)
        self.assertFalse(receipt["scientific_protocol_acceptance"])
        self.assertFalse(receipt["native_runtime_numerical_parity_validated"])
        self.assertTrue(records[0]["complex"]["provenance"]["internal_atom_or_hydrogen_energy_placement_possible"])
        self.assertEqual(list((self.root / ".cache/tmp").iterdir()), [])

    def test_full_metadata_preserved_and_export_accepts_actual_output(self):
        self.candidate["custom_science_metadata"] = {"purpose": "explicit CPU fixture", "number": 7}
        before = deepcopy(self.candidate)
        records, receipt = self.run_rebuild()
        self.assertEqual(self.candidate, before)
        self.assertEqual({k: v for k, v in records[0].items() if k != "complex"}, before)
        self.assertEqual(read_jsonl(self.root / "rebuilt/reconstructed.jsonl"), records)
        exported = export_library(records, project_root=self.root, output_dir="exported", prepared_founder=self.prepared)
        self.assertEqual(exported["count"], 1)
        self.assertEqual(exported["new_structures_generated_by_this_command"], 0)

    def test_ids_order_and_same_sequence_distinct_ids_preserved(self):
        second = deepcopy(self.candidate); second["id"] = "second_id_same_sequence"
        records, _ = self.run_rebuild([second, self.candidate])
        self.assertEqual([r["id"] for r in records], [second["id"], self.candidate["id"]])
        self.assertNotEqual(records[0]["complex"]["path"], records[1]["complex"]["path"])

    def test_duplicate_ids_fail_before_outputs(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_rebuild([self.candidate, deepcopy(self.candidate)])
        self.assertFalse((self.root / "rebuilt").exists())

    def test_empty_input_and_bad_id_fail_before_outputs(self):
        for records in ([], [{**self.candidate, "id": ""}]):
            with self.assertRaises(ValueError): self.run_rebuild(records)
        self.assertFalse((self.root / "rebuilt").exists())

    def test_no_existing_complex_evidence_even_null(self):
        for key in ("complex", "delivered_pdb_path", "delivered_pdb_sha256"):
            with self.assertRaisesRegex(ValueError, "Existing complex"):
                self.run_rebuild([{**self.candidate, key: None}])

    def test_founder_antigen_and_roles_strict(self):
        for change in (lambda r: r.update(antigen_id="other"),
                       lambda r: r.update(design="other"),
                       lambda r: r["founder"].update(id="other"),
                       lambda r: r["founder"]["complex_reference"].update(antigen_chains={"C": "AA"})):
            candidate = deepcopy(self.candidate); change(candidate)
            with self.assertRaises(ValueError): self.run_rebuild([candidate])

    def test_founder_missing_atoms_and_stale_hash_rejected(self):
        source = self.root / "founder.pdb"; source.write_text(source.read_text().replace("ATOM  ", "HETATM", 1))
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.run_rebuild()
        self.prepared["complex_reference"]["sha256"] = sha256(source)
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.run_rebuild()

    def test_framework_change_and_truncation_rejected(self):
        for heavy in ("A" + self.candidate["heavy"][1:], self.candidate["heavy"][:-1]):
            with self.assertRaisesRegex(ValueError, "framework|lengths"):
                self.run_rebuild([{**self.candidate, "heavy": heavy}])

    def test_inconsistent_fr_mutations_and_counts_rejected(self):
        for change in ({"fr_cdr_seq": self.prepared["fr_cdr_seq"]}, {"mutations": []}, {"mutation_count": 0},
                       {"position_index_base": 1}, {"position_index_base": False}, {"mutation_count": True},
                       {"founder_fr_cdr_seq": "X" * 277}):
            with self.assertRaises(ValueError): self.run_rebuild([{**self.candidate, **change}])

    def test_zero_mutations_keeps_actual_founder_chains(self):
        candidate = {**self.candidate, **reconstruct(self.prepared, self.prepared["fr_cdr_seq"])}
        records, _ = self.run_rebuild([candidate])
        self.assertNotIn("applyMutations", [call[0] for call in self.engine.calls])
        self.assertEqual(inspect_pdb(self.root / records[0]["complex"]["path"])["H"], self.prepared["heavy"])

    def test_declared_constant_suffix_preserved_without_hidden_changes(self):
        self.prepared.update(heavy_structure_suffix="AG", light_structure_suffix="GG")
        source = self.root / "founder.pdb"
        source.write_text(pdb_text({"H": self.prepared["heavy"] + "AG", "L": self.prepared["light"] + "GG", "C": "GG"}))
        self.prepared["complex_reference"]["sha256"] = sha256(source)
        self.candidate["founder"] = deepcopy(self.prepared)
        records, _ = self.run_rebuild()
        chains = inspect_pdb(self.root / records[0]["complex"]["path"])
        self.assertEqual(chains["H"], self.candidate["heavy"] + "AG")
        self.assertEqual(chains["L"], self.candidate["light"] + "GG")

    def test_original_insertion_codes_retained_in_mapping_not_collapsed(self):
        source = self.root / "founder.pdb"
        lines = []
        for line in source.read_text().splitlines(keepends=True):
            if line.startswith("ATOM  ") and line[21] == "H":
                number = int(line[22:26]); insertion = "A" if number == 2 else " "
                line = line[:22] + f"{number - 1 if number > 1 else 1:4d}" + insertion + line[27:]
            lines.append(line)
        source.write_text("".join(lines)); self.prepared["complex_reference"]["sha256"] = sha256(source)
        self.candidate["founder"] = deepcopy(self.prepared)
        records, _ = self.run_rebuild()
        mapping = json.loads((self.root / "rebuilt/founder_residue_mapping.json").read_text())
        self.assertEqual([(r["original_residue_number"], r["original_insertion_code"]) for r in mapping["H"][:3]],
                         [(1, " "), (1, "A"), (2, " ")])
        self.assertEqual(list(rebuild._atoms(self.root / records[0]["complex"]["path"])["H"])[:3],
                         [(1, " "), (2, " "), (3, " ")])

    def test_hetatm_exclusion_is_explicit_in_receipt(self):
        source = self.root / "founder.pdb"; source.write_text(source.read_text() + "HETATM excluded water fixture\n")
        self.prepared["complex_reference"]["sha256"] = sha256(source)
        self.candidate["founder"] = deepcopy(self.prepared)
        records, receipt = self.run_rebuild()
        self.assertEqual(receipt["template_record_counts"]["excluded_hetatm_records"], 1)
        self.assertNotIn("HETATM", (self.root / records[0]["complex"]["path"]).read_text())

    def test_antigen_and_backbone_move_rejected(self):
        for chain, atom in (("C", "CA"), ("H", "N")):
            def corrupt(text, chain=chain, atom=atom):
                lines = []
                for line in text.splitlines(keepends=True):
                    if line.startswith("ATOM  ") and line[21] == chain and line[12:16].strip() == atom:
                        line = line[:30] + f"{float(line[30:38]) + 0.002:8.3f}" + line[38:]
                    lines.append(line)
                return "".join(lines)
            self.engine.mutate_output = corrupt
            with self.assertRaisesRegex(ValueError, "coordinates moved"):
                self.run_rebuild(output="bad_" + chain)
            self.assertFalse((self.root / ("bad_" + chain) / "reconstructed.jsonl").exists())

    def test_one_coordinate_grid_step_allowed_and_measured(self):
        def shift(text):
            return "".join(line[:30] + f"{float(line[30:38]) + 0.001:8.3f}" + line[38:]
                           if line.startswith("ATOM  ") else line for line in text.splitlines(keepends=True))
        self.engine.mutate_output = shift
        _, receipt = self.run_rebuild()
        manifest = read_jsonl(self.root / "rebuilt/pdb_manifest.jsonl")
        self.assertAlmostEqual(manifest[0]["coordinate_verification"]["maximum_coordinate_component_difference_angstrom"], .001)

    def test_missing_mutant_sidechain_atom_rejected(self):
        self.engine.mutate_output = lambda text: "".join(line for line in text.splitlines(keepends=True) if line[12:16].strip() != "CB")
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.run_rebuild()

    def test_incorrect_output_sequence_rejected_even_complete_atoms(self):
        def change(text):
            return "".join(line[:17] + "ALA" + line[20:] if line.startswith("ATOM  ") and line[21] == "C" else line
                           for line in text.splitlines(keepends=True))
        self.engine.mutate_output = change
        with self.assertRaises(ValueError): self.run_rebuild()

    def test_unexpected_hetatm_output_rejected(self):
        self.engine.mutate_output = lambda text: text + "HETATM unexpected atom\n"
        with self.assertRaisesRegex(ValueError, "Unexpected HETATM"):
            self.run_rebuild()

    def test_runtime_hash_platform_thread_and_asset_errors(self):
        for change in ({"platform": "CUDA"}, {"cpu_threads": 0}, {"cpu_threads": 9}, {"cpu_threads": True},
                       {"implementation_sha256": {}}, {"physical_assets_sha256": "f" * 64}):
            with self.assertRaises(ValueError): self.run_rebuild(runtime_identity={**self.identity, **change})
        self.assertFalse((self.root / "rebuilt").exists())

    def test_installed_runtime_mismatch_fails_not_skips(self):
        changed = deepcopy(self.identity); changed["openmm_version"] = "different"
        with self.assertRaisesRegex(ValueError, "runtime/physical"):
            self.run_rebuild(runtime_identity=changed)
        self.assertFalse(self.engine.calls)
        self.assertTrue((self.root / "rebuilt/failure.json").is_file())

    def test_unknown_engine_failure_and_keyboard_interrupt_clean_temp(self):
        for exception, name in ((RuntimeError("synthetic engine failure"), "error"), (KeyboardInterrupt(), "interrupt")):
            with patch.object(self.engine.fixer, "addMissingAtoms", side_effect=exception):
                with self.assertRaises(type(exception)):
                    self.run_rebuild(output=name)
            self.assertTrue((self.root / name / "failure.json").is_file())
            self.assertFalse((self.root / name / "rebuild.json").exists())
        self.assertEqual(list((self.root / ".cache/tmp").iterdir()), [])

    def test_malformed_worker_missing_extra_wrong_order_or_hash(self):
        original = self.invoke_fixture
        cases = [lambda r: r.update(products=[]), lambda r: r["products"].append(r["products"][0]),
                 lambda r: r["products"][0].update(id="wrong"), lambda r: r["products"][0].update(sha256="f" * 64),
                 lambda r: r["products"][0].update(file="../escape.pdb"), lambda r: r.update(mapping={}),
                 lambda r: r["products"][0].update(coordinate_verification={})]
        for i, mutate in enumerate(cases):
            def wrong(request, response, threads):
                result = original(request, response, threads); mutate(result); return result
            with patch.object(rebuild, "_invoke_child", wrong), self.assertRaises((ValueError, rebuild.RebuildingError)):
                rebuild.rebuild_complexes([self.candidate], project_root=self.root, output_dir=f"malformed_{i}",
                                          prepared_founder=self.prepared, runtime_identity=self.identity)

    def test_founder_file_race_after_worker_is_refused(self):
        original = self.invoke_fixture
        def changed(request, response, threads):
            result = original(request, response, threads)
            source = self.root / "founder.pdb"; source.write_text(source.read_text() + "REMARK changed\n")
            return result
        with patch.object(rebuild, "_invoke_child", changed), self.assertRaisesRegex(ValueError, "SHA256"):
            rebuild.rebuild_complexes([self.candidate], project_root=self.root, output_dir="race",
                                      prepared_founder=self.prepared, runtime_identity=self.identity)

    def test_snapshot_mutation_is_refused_even_if_original_unchanged(self):
        original = self.invoke_fixture
        def changed(request, response, threads):
            result = original(request, response, threads)
            snapshot = request.parent / "founder_input.pdb"
            snapshot.write_text(snapshot.read_text() + "REMARK snapshot changed\n")
            return result
        with patch.object(rebuild, "_invoke_child", changed), self.assertRaisesRegex(ValueError, "SHA256"):
            rebuild.rebuild_complexes([self.candidate], project_root=self.root, output_dir="snapshot_race",
                                      prepared_founder=self.prepared, runtime_identity=self.identity)

    def test_output_no_overwrite_escape_or_date(self):
        self.run_rebuild()
        before = sha256(self.root / "rebuilt/rebuild.json")
        for output in ("rebuilt", "../outside", "/outside", "rebuilding_20260919"):
            with self.assertRaises((ValueError, FileExistsError)): self.run_rebuild(output=output)
        self.assertEqual(sha256(self.root / "rebuilt/rebuild.json"), before)

    def test_metadata_inspection_does_not_graft_or_leave_temporary_files(self):
        with patch.object(rebuild, "_invoke_child", self.invoke_fixture):
            actual = rebuild.inspect_rebuilding_runtime(project_root=self.root, cpu_threads=1)
        self.assertEqual(actual, runtime()); self.assertEqual(self.engine.calls, [])
        self.assertEqual(list((self.root / ".cache/tmp").iterdir()), [])

    def test_child_environment_not_parent_mutation_and_no_implicit_gpu(self):
        request, response = self.root / "request.json", self.root / "response.json"
        request.write_text("{}")
        response.write_text(json.dumps({"status": "complete"}))
        child = SimpleNamespace(communicate=lambda: ("", ""), returncode=0)
        class Process:
            def __enter__(self): return child
            def __exit__(self, *args): pass
        with patch.dict(os.environ, {"OPENMM_CPU_THREADS": "17", "OPENMM_DEFAULT_PLATFORM": "CUDA",
                                    "OPENMM_FORCEFIELD_DIR": "unrecorded_custom_asset"}), \
                patch.object(rebuild.subprocess, "Popen", return_value=Process()) as launch:
            rebuild._invoke_child(request, response, 2)
            environment = launch.call_args.kwargs["env"]
            self.assertEqual(environment["OPENMM_CPU_THREADS"], "2")
            self.assertEqual(environment["OPENMM_DEFAULT_PLATFORM"], "CPU")
            self.assertNotIn("OPENMM_FORCEFIELD_DIR", environment)
            self.assertEqual(environment["TMPDIR"], str(self.root))
            self.assertEqual(os.environ["OPENMM_CPU_THREADS"], "17")
            self.assertEqual(os.environ["OPENMM_DEFAULT_PLATFORM"], "CUDA")
            self.assertIn("-I", launch.call_args.args[0])

    def test_public_import_is_lazy(self):
        self.assertNotIn("openmm", rebuild.__dict__)
        self.assertNotIn("pdbfixer", rebuild.__dict__)


if __name__ == "__main__":
    unittest.main()
