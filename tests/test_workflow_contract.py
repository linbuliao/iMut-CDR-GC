"""Synthetic CPU contracts; no native weights or scientific results."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from imut_cdr_gc.cli import main, parser
from imut_cdr_gc.complexes import SIDECHAINS, RESIDUES, export_library, inspect_pdb
from imut_cdr_gc.preparation import prepare_founder, reconstruct
from imut_cdr_gc.records import new_output, read_jsonl, sha256, write_json, write_jsonl
from imut_cdr_gc.selection import select_records
from imut_cdr_gc.workflow import check_stepwise_trace, prepare_run, select_run

PACKAGE = Path(__file__).resolve().parents[1]


def founder():
    fr = ["X"] * 277
    for start, motif in ((26, "ACD"), (55, "EFG"), (102, "HIK"),
                         (165, "LMN"), (194, "PQR"), (241, "STV")):
        fr[start:start + 3] = motif
    return {"id": "synthetic_founder", "design": "synthetic", "antigen_id": "synthetic_antigen",
            "heavy": "WLMNYPQRWSTVY", "light": "WACDYEFGWHIKY", "fr_cdr_seq": "".join(fr)}


def policy(version="v1", route="esm2_OR_native3d", target=1):
    return {"version": version, "route": route, "target": target, "common_deepcdr_floor": 0.5,
            "likelihood_calibration": {"lower_quantile": 0.05, "lower_cutoff": -1.9,
                "upper_quantile": None, "upper_cutoff": None, "metric_identity": "synthetic_not_native",
                "source_identity": {"fixture": "not_a_native_model"},
                "reference_sha256": "a" * 64, "reference_not_candidates": True}}


def row(identifier="candidate", k=3, esm=0.8, structure=0.7, likelihood=-1.0, version="v1"):
    f = prepare_founder(founder())
    mutant = list(f["fr_cdr_seq"])
    positions = [int(p) for p in f["cdr_position_map"]]
    for p in positions[:k]:
        mutant[p] = "A" if mutant[p] != "A" else "C"
    rebuilt = reconstruct(f, "".join(mutant))
    return {**rebuilt, "id": identifier, "model": "iMut-CDR-JM-Epi", "version": version,
            "design": "synthetic", "antigen_id": f["antigen_id"], "founder": f,
            "scores": {name: measured(identifier, name, value, rebuilt)
                       for name, value in (("esm2", esm), ("3d", structure))},
            "likelihood": {"status": "ok", "score": likelihood, "metric_identity": "synthetic_not_native",
                           "source_identity": {"fixture": "not_a_native_model"}, "id": identifier,
                           **{role + "_sha256": hashlib.sha256(rebuilt[role].encode()).hexdigest() for role in ("heavy", "light")}}}


def measured(identifier, name, value, chains):
    protocol = {"esm2": "deepcdr_esm2_hfirst_anarci_list_index95_v1", "3d": "deepcdr_3d_antigen_first_v1"}[name]
    return {"id": identifier, "model": {"esm2": "DeepCDR-ESM2", "3d": "DeepCDR-3D"}[name],
            "protocol": protocol, "identity": {"protocol": protocol},
            "status": "ok", "score": value, "inputs": {
                "heavy_sha256": hashlib.sha256(chains["heavy"].encode()).hexdigest(),
                "light_sha256": hashlib.sha256(chains["light"].encode()).hexdigest(),
                "antigen": {"id": "synthetic_antigen"}}}


def pdb_text(chains):
    """Coordinate placeholders for parsing/identity tests, not physical structures."""
    inverse = {value: key for key, value in RESIDUES.items()}
    lines, serial = [], 0
    for chain, seq in chains.items():
        for index, aa in enumerate(seq, 1):
            for atom in ["N", "CA", "C", "O", *SIDECHAINS[aa].split()]:
                serial += 1
                lines.append(f"ATOM  {serial:5d} {atom:^4s} {inverse[aa]:3s} {chain}{index:4d}    "
                             f"{float(index):8.3f}{float(serial % 7):8.3f}{0.:8.3f}{1.:6.2f}{20.:6.2f}          {atom[0]:>2s}\n")
    return "".join(lines) + "END\n"


class RootFixture(unittest.TestCase):
    def setUp(self):
        scratch = PACKAGE / ".cache" / "tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="workflow_", dir=scratch)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()


class PreparationTests(RootFixture):
    def test_lossless_chain_reconstruction(self):
        f = prepare_founder(founder())
        mutant = list(f["fr_cdr_seq"]); mutant[26] = "G"
        result = reconstruct(f, "".join(mutant))
        self.assertEqual(result["heavy"], f["heavy"])
        self.assertEqual(result["light"], "WGCDYEFGWHIKY")
        self.assertEqual(result["mutation_count"], 1)

    def test_unchanged_proposal_is_not_a_mutation(self):
        f = founder()
        self.assertEqual(reconstruct(f, f["fr_cdr_seq"])["mutation_count"], 0)

    def test_padding_and_framework_rejected(self):
        f = founder(); mutant = list(f["fr_cdr_seq"]); mutant[0] = "A"
        with self.assertRaisesRegex(ValueError, "framework"):
            reconstruct(f, "".join(mutant))

    def test_repeated_cdr_requires_explicit_map(self):
        f = founder(); explicit = prepare_founder(f)["cdr_position_map"]
        f["light"] += "ACD"
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            prepare_founder(f)
        f["cdr_position_map"] = explicit
        self.assertEqual(prepare_founder(f)["light"], f["light"])

    def test_wrong_map_and_extra_map_rejected(self):
        for mutation in ({"26": {"chain": "heavy", "index": 1}}, {"0": {"chain": "light", "index": 0}}):
            f = prepare_founder(founder()); f["cdr_position_map"].update(mutation)
            with self.assertRaises(ValueError):
                prepare_founder(f)

    def test_prepare_file_and_no_overwrite(self):
        write_json(self.root / "input.json", founder())
        prepare_run(self.root, "input.json", "prepared")
        self.assertTrue((self.root / "prepared/founder.json").is_file())
        with self.assertRaises(FileExistsError):
            prepare_run(self.root, "input.json", "prepared")


class SelectionTests(RootFixture):
    def test_same_common_floor_or_and_recorded_boundary(self):
        candidate = row(esm=0.49, structure=0.8)
        selected, result = select_records([candidate], policy())
        self.assertEqual(len(selected), 1)
        self.assertEqual(result["actual_boundary"]["active_model_scores"], {"esm2": 0.49, "3d": 0.8})
        self.assertTrue(result["or_boundary_is_not_two_independent_model_cutoffs"])

    def test_p5_inclusive_and_no_upper_bound(self):
        self.assertEqual(select_records([row(likelihood=-1.9)], policy())[1]["counts"]["selected"], 1)
        self.assertEqual(select_records([row(likelihood=0.0)], policy())[1]["counts"]["selected"], 1)
        self.assertEqual(select_records([row(likelihood=-1.901)], policy())[1]["counts"]["below_reference_p5"], 1)

    def test_upper_bound_and_candidate_quantile_refused(self):
        for change in ({"upper_cutoff": -0.1}, {"upper_quantile": 0.975}, {"reference_not_candidates": False}):
            cfg = policy(); cfg["likelihood_calibration"].update(change)
            with self.assertRaises(ValueError):
                select_records([row()], cfg)

    def test_external_control_never_screened(self):
        candidate = row(); candidate["model"] = "iMut-CDR-Epi"
        with self.assertRaisesRegex(ValueError, "restricted"):
            select_records([candidate], policy())

    def test_missing_and_infinite_scores_are_not_zero_or_threshold_failures(self):
        for value in (None, float("nan"), float("inf"), True):
            candidate = row(esm=value)
            _, result = select_records([candidate], policy())
            self.assertEqual(result["counts"]["unmeasured_or_error"], 1)
            self.assertNotIn("below_common_deepcdr_floor", result["counts"])

    def test_missing_likelihood_not_below_p5(self):
        candidate = row(likelihood=None)
        _, result = select_records([candidate], policy())
        self.assertEqual(result["counts"]["likelihood_unmeasured_error_or_identity_mismatch"], 1)
        self.assertNotIn("below_reference_p5", result["counts"])

    def test_p5_identity_required(self):
        candidate = row(); candidate["likelihood"]["metric_identity"] = "wrong_model"
        _, result = select_records([candidate], policy())
        self.assertEqual(result["counts"]["selected"], 0)

    def test_p5_runtime_source_identity_required(self):
        candidate = row(); candidate["likelihood"]["source_identity"] = {"fixture": "other_runtime"}
        _, result = select_records([candidate], policy())
        self.assertEqual(result["counts"]["selected"], 0)

    def test_stale_sequence_and_wrong_antigen_scores_are_not_accepted(self):
        for key, value in (("heavy_sha256", "f" * 64), ("antigen", {"id": "other_target"})):
            candidate = row(); candidate["scores"]["esm2"]["inputs"][key] = value
            _, result = select_records([candidate], policy())
            self.assertEqual(result["counts"]["unmeasured_or_error"], 1)

    def test_single_active_model_no_other_score_required(self):
        candidate = row(); candidate["scores"].pop("esm2")
        selected, _ = select_records([candidate], policy(route="native3d_only"))
        self.assertEqual(len(selected), 1)

    def test_dedup_keeps_best_rank_and_reports_loss(self):
        worse, best = row("a", esm=0.7), row("z", esm=0.9)
        selected, result = select_records([worse, best], policy(target=2))
        self.assertEqual(selected[0]["id"], "z")
        self.assertEqual(result["counts"]["duplicate_eligible_sequence_pair"], 1)
        self.assertEqual(result["status"], "insufficient_eligible")
        self.assertFalse(result["actual_boundary"]["is_requested_top_k_boundary"])

    def test_deterministic_tie_and_original_decision_order(self):
        a, b = row("z"), row("a")
        chosen, result = select_records([a, b], policy())
        self.assertEqual(chosen[0]["id"], "a")
        self.assertEqual([d["id"] for d in result["decisions"]], ["z", "a"])

    def test_mutation_count_independently_checked(self):
        candidate = row(); candidate["mutation_count"] = 5
        with self.assertRaisesRegex(ValueError, "actual full-chain"):
            select_records([candidate], policy())

    def test_disjoint_sequential_loss_accounting(self):
        rows = [row("burden", k=1), row("ll", likelihood=-5), row("score", esm=.1, structure=.1), row("pass")]
        _, result = select_records(rows, policy())
        self.assertEqual([d["state"] for d in result["decisions"]], [
            "outside_mutation_burden", "below_reference_p5", "below_common_deepcdr_floor", "selected"])
        self.assertEqual(len(result["decisions"]), result["counts"]["input"])

    def test_wrong_version_and_cross_design_refused(self):
        for field, value in (("version", "v2"), ("design", "other")):
            a, b = row("a"), row("b"); b[field] = value
            with self.assertRaises(ValueError):
                select_records([a, b], policy())

    def test_existing_pair_exclusion(self):
        candidate = row()
        _, result = select_records([candidate], policy(), excluded_pairs=[(candidate["heavy"], candidate["light"])])
        self.assertEqual(result["counts"]["excluded_existing_sequence_pair"], 1)


class TraceTests(RootFixture):
    def valid(self):
        r = row(k=11, version="v3")
        parent = {key: r["founder"][key] for key in ("heavy", "light")}
        mutations = r["mutations"]
        r["stepwise_trace"] = []
        for round_id, group in enumerate((mutations[:10], mutations[10:]), 1):
            child = dict(parent)
            for change in group:
                seq = list(child[change["chain"]]); seq[change["chain_index"]] = change["to"]
                child[change["chain"]] = "".join(seq)
            identifier = "proposal_" + str(round_id)
            r["stepwise_trace"].append({"round": round_id, "proposal_id": identifier, "parent": dict(parent), "child": child,
                "requested_sites": [{"chain": m["chain"], "index": m["chain_index"]} for m in group],
                "scores": {"esm2": measured(identifier, "esm2", 0.8, child)}})
            parent = child
        return r

    def test_per_proposal_gate_and_chain_lineage(self):
        check_stepwise_trace(self.valid(), policy(version="v3"))

    def test_missing_trace_cannot_claim_v3(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            check_stepwise_trace(row(version="v3", k=11), policy(version="v3"))

    def test_gate_after_each_joint_proposal(self):
        r = self.valid(); r["stepwise_trace"][0]["scores"] = {}
        with self.assertRaisesRegex(ValueError, "every|Every"):
            check_stepwise_trace(r, policy(version="v3"))

    def test_wrong_parent_rejected(self):
        r = self.valid(); r["stepwise_trace"][1]["parent"] = r["stepwise_trace"][0]["parent"]
        with self.assertRaisesRegex(ValueError, "contiguous"):
            check_stepwise_trace(r, policy(version="v3"))

    def test_more_than_ten_positions_rejected(self):
        r = self.valid(); r["stepwise_trace"][0]["requested_sites"].append({"chain": "heavy", "index": 0})
        with self.assertRaisesRegex(ValueError, "ten"):
            check_stepwise_trace(r, policy(version="v3"))

    def test_changed_unrequested_site_rejected(self):
        r = self.valid(); r["stepwise_trace"][0]["requested_sites"].pop()
        with self.assertRaisesRegex(ValueError, "outside"):
            check_stepwise_trace(r, policy(version="v3"))


class ExportTests(RootFixture):
    def export(self, records, output_dir="delivery"):
        return export_library(records, project_root=self.root, output_dir=output_dir, prepared_founder=self.prepared_founder)

    def candidate(self):
        r = row()
        pdb = self.root / "synthetic.pdb"
        pdb.write_text(pdb_text({"H": r["heavy"], "L": r["light"], "A": "GG"}))
        original = self.root / "founder.pdb"
        original.write_text(pdb_text({"H": r["founder"]["heavy"], "L": r["founder"]["light"], "A": "GG"}))
        self.prepared_founder = deepcopy(r["founder"])
        self.prepared_founder["complex_reference"] = {"path": "founder.pdb", "sha256": sha256(original),
            "antigen_id": "synthetic_antigen", "heavy_chain": "H", "light_chain": "L", "antigen_chains": {"A": "GG"}}
        r["complex"] = {"path": "synthetic.pdb", "sha256": sha256(pdb), "heavy_chain": "H", "light_chain": "L",
            "antigen_id": "synthetic_antigen",
            "antigen_chains": {"A": "GG"}, "provenance": {"kind": "founder_sidechain_reconstruction",
                                                        "synthetic_fixture_not_physical_model": True}}
        return r

    def test_real_files_relative_paths_metadata_retained(self):
        r = self.candidate()
        receipt = self.export([r])
        delivered = read_jsonl(self.root / "delivery/library.jsonl")[0]
        self.assertTrue((self.root / delivered["delivered_pdb_path"]).is_file())
        self.assertEqual(delivered["mutations"], r["mutations"])
        self.assertEqual(receipt["new_structures_generated_by_this_command"], 0)

    def test_missing_pdb_or_wrong_sequence_refused(self):
        r = self.candidate(); r["light"] = r["founder"]["light"]
        with self.assertRaisesRegex(ValueError, "actual full mutant"):
            self.export([r])
        self.assertFalse((self.root / "delivery").exists())

    def test_missing_sidechain_atoms_refused(self):
        r = self.candidate(); path = self.root / "synthetic.pdb"
        path.write_text("".join(line for line in path.read_text().splitlines(keepends=True)
                                if line[12:16].strip() != "CB"))
        r["complex"]["sha256"] = sha256(path)
        with self.assertRaisesRegex(ValueError, "side-chain"):
            self.export([r])

    def test_wrong_antigen_refused(self):
        r = self.candidate(); r["complex"]["antigen_chains"] = {"A": "GA"}
        with self.assertRaisesRegex(ValueError, "antigen"):
            self.export([r])

    def test_stale_pdb_hash_refused(self):
        r = self.candidate(); r["complex"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.export([r])

    def test_duplicate_sequences_refused(self):
        r = self.candidate(); copy = deepcopy(r); copy["id"] = "another"
        with self.assertRaisesRegex(ValueError, "Duplicate full"):
            self.export([r, copy])

    def test_output_escape_dates_and_overwrite_refused(self):
        for output in ("../bad", "/outside", "delivery_20260919"):
            with self.assertRaises(ValueError):
                new_output(self.root, output)
        new_output(self.root, "okay")
        with self.assertRaises(FileExistsError):
            new_output(self.root, "okay")

    def test_synthetic_prepare_select_export_file_chain(self):
        r = self.candidate()
        write_json(self.root / "founder_input.json", founder())
        prepare_run(self.root, "founder_input.json", "prepared")
        write_jsonl(self.root / "scored.jsonl", [r])
        write_json(self.root / "policy.json", policy())
        summary = select_run(self.root, "scored.jsonl", "policy.json", "selection")
        self.assertEqual(summary["status"], "complete")
        self.export(read_jsonl(self.root / "selection/selected.jsonl"))
        self.assertTrue((self.root / "delivery/pdb_manifest.jsonl").is_file())


class CliTests(unittest.TestCase):
    def test_help_does_not_require_models(self):
        with self.assertRaises(SystemExit) as outcome:
            parser().parse_args(["--help"])
        self.assertEqual(outcome.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
