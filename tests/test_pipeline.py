"""Portable pipeline contracts with synthetic software-only fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from imut_cdr_gc.cli import parser, execute
from imut_cdr_gc.pipeline import load_plan, run_pipeline
from imut_cdr_gc.records import write_json

PACKAGE = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        scratch = PACKAGE / ".cache" / "tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="pipeline_", dir=scratch)
        self.root = Path(self.temporary.name)
        self.plan = {"schema": "imut-cdr-gc-pipeline.v1", "output_dir": "run",
            "steps": [{"name": "prepare", "command": "prepare", "arguments": {
                "founder": "founder.json", "output_dir": "prepared"}}]}

    def tearDown(self):
        self.temporary.cleanup()

    def write_plan(self):
        path = self.root / "pipeline.json"
        path.write_text(json.dumps(self.plan), encoding="utf-8")

    def test_default_project_is_current_directory(self):
        args = parser().parse_args(["run", "--config", "pipeline.json", "--dry-run"])
        self.assertEqual(args.project_root, Path("."))
        self.assertTrue(args.dry_run)

    def test_dry_run_does_not_import_or_execute_model(self):
        self.write_plan()
        with patch("imut_cdr_gc.cli.execute") as execute_stage:
            result = run_pipeline(self.root, "pipeline.json", dry_run=True)
        execute_stage.assert_not_called()
        self.assertEqual(result["status"], "plan_validated_not_executed")
        self.assertEqual(result["model_calls"], 0)
        self.assertFalse((self.root / "run").exists())

    def test_real_preparation_pipeline_and_receipts(self):
        fr = ["X"] * 277
        for index, motif in ((26, "ACD"), (55, "EFG"), (102, "HIK"),
                             (165, "LMN"), (194, "PQR"), (241, "STV")):
            fr[index:index + 3] = motif
        write_json(self.root / "founder.json", dict(id="synthetic", design="synthetic",
            antigen_id="synthetic", heavy="WLMNYPQRWSTVY", light="WACDYEFGWHIKY",
            fr_cdr_seq="".join(fr)))
        self.write_plan()
        result = run_pipeline(self.root, "pipeline.json")
        self.assertEqual(result["completed_steps"], 1)
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["scientific_validation"])
        self.assertTrue((self.root / "prepared/founder.json").is_file())
        self.assertTrue((self.root / "run/stage_01_prepare.json").is_file())
        self.assertTrue((self.root / "run/completion.json").is_file())
        with self.assertRaises(FileExistsError):
            run_pipeline(self.root, "pipeline.json")

    def test_unknown_commands_keys_and_duplicate_names_rejected_before_calls(self):
        variants = []
        plan = deepcopy(self.plan); plan["steps"][0]["command"] = "run"; variants.append(plan)
        plan = deepcopy(self.plan); plan["steps"][0]["arguments"]["shell"] = "anything"; variants.append(plan)
        plan = deepcopy(self.plan); plan["steps"].append(deepcopy(plan["steps"][0])); variants.append(plan)
        plan = deepcopy(self.plan); plan["steps"][0]["arguments"].pop("founder"); variants.append(plan)
        plan = deepcopy(self.plan); plan["unexpected"] = True; variants.append(plan)
        for candidate in variants:
            self.plan = candidate
            self.write_plan()
            with self.subTest(plan=candidate), self.assertRaises(ValueError):
                load_plan(self.root, "pipeline.json")
        self.assertFalse((self.root / "run").exists())

    def test_absolute_traversal_and_overlapping_outputs_refused(self):
        original = deepcopy(self.plan)
        for value in (str(self.root / "prepared"), "../escape", "run/child", "run", "."):
            self.plan = deepcopy(original)
            self.plan["steps"][0]["arguments"]["output_dir"] = value
            self.write_plan()
            with self.subTest(path=value), self.assertRaises((ValueError, FileExistsError)):
                load_plan(self.root, "pipeline.json")
        self.assertFalse((self.root / "run").exists())

    def test_symlink_output_escape_refused(self):
        (self.root / "outside").symlink_to(self.root.parent, target_is_directory=True)
        self.plan["steps"][0]["arguments"]["output_dir"] = "outside/should_not_be_created"
        self.write_plan()
        with self.assertRaisesRegex(ValueError, "inside"):
            load_plan(self.root, "pipeline.json")

    def test_list_model_argument_and_numeric_batch_parsed(self):
        self.plan["steps"] = [{"name": "score", "command": "score", "arguments": {
            "input": "input.jsonl", "config": "score.json", "models": ["esm2", "3d"],
            "output_dir": "scored"}}, {"name": "likelihood", "command": "likelihood", "arguments": {
            "input": "scored/scores.jsonl", "assets": "likelihood.json", "batch_size": 2,
            "output_dir": "likelihood"}}]
        self.write_plan()
        _, _, _, stages = load_plan(self.root, "pipeline.json")
        self.assertEqual(stages[0][1].models, ["esm2", "3d"])
        self.assertEqual(stages[1][1].batch_size, 2)

    def test_incomplete_selection_stops_without_export_or_completion(self):
        self.plan["steps"] = [{"name": "select", "command": "select", "arguments": {
            "input": "input.jsonl", "policy": "policy.json", "output_dir": "selected"}},
            {"name": "export", "command": "export", "arguments": {
                "input": "selected/selected.jsonl", "founder": "founder.json", "output_dir": "library"}}]
        self.write_plan()
        with patch("imut_cdr_gc.cli.execute", return_value={"status": "insufficient_eligible"}) as stage:
            with self.assertRaisesRegex(RuntimeError, "insufficient_eligible"):
                run_pipeline(self.root, "pipeline.json")
        self.assertEqual(stage.call_count, 1)
        self.assertFalse((self.root / "run/completion.json").exists())
        failure = json.loads((self.root / "run/failure.json").read_text())
        self.assertEqual(failure["failed_stage"], "select")
        self.assertFalse(failure["automatic_retry"])

    def test_changed_plan_during_stage_does_not_commit_success(self):
        self.write_plan()
        def mutate(_):
            (self.root / "pipeline.json").write_text("{}", encoding="utf-8")
            return {"status": "complete"}
        with self.assertRaisesRegex(ValueError, "changed"):
            run_pipeline(self.root, "pipeline.json", stage_executor=mutate)
        self.assertTrue((self.root / "run/failure.json").is_file())
        self.assertFalse((self.root / "run/completion.json").exists())


if __name__ == "__main__":
    unittest.main()
