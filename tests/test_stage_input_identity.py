"""Input changes must not be attributed to already-computed measurements."""
import json
from unittest.mock import patch

from imut_cdr_gc import workflow
from imut_cdr_gc.records import sha256, write_json, write_jsonl
from test_workflow_contract import RootFixture, founder, measured, policy, row


class InputIdentityTests(RootFixture):
    def test_prepare_change_prevents_output(self):
        source = self.root / "founder.json"
        write_json(source, founder())
        original = workflow.prepare_founder
        def changed(value):
            result = original(value)
            source.write_text(source.read_text() + "\n")
            return result
        with patch.object(workflow, "prepare_founder", changed):
            with self.assertRaisesRegex(ValueError, "input changed"):
                workflow.prepare_run(self.root, "founder.json", "prepared")
        self.assertFalse((self.root / "prepared").exists())

    def test_score_input_or_config_change_retains_failure_not_completion(self):
        for key in ("input", "config"):
            base = self.root / key
            base.mkdir()
            candidate = row()
            candidate["scores"] = {}
            source, config = base / "input.jsonl", base / "config.json"
            write_jsonl(source, [candidate])
            write_json(config, {"assets": {"esm2": {}}, "antigen": {}})
            before = {p: sha256(p) for p in (source, config)}
            def scorer(record, **kwargs):
                changed = source if key == "input" else config
                changed.write_text(changed.read_text() + "\n")
                return measured(record["id"], "esm2", .8, record)
            with patch("imut_cdr_gc.scoring.score_esm2", scorer):
                with self.assertRaisesRegex(ValueError, "input changed"):
                    workflow.score_run(base, "input.jsonl", "config.json", "scored", ["esm2"])
            receipt = json.loads((base / "scored/failure.json").read_text())
            self.assertEqual(receipt["input_sha256"], before[source])
            self.assertEqual(receipt["config_sha256"], before[config])
            self.assertTrue(receipt["partial_output_not_committed"])
            self.assertFalse((base / "scored/score.json").exists())
            self.assertFalse((base / "scored/scores.jsonl").exists())

    def test_selection_change_prevents_output(self):
        for key in ("input", "policy", "excluded"):
            base = self.root / key
            base.mkdir()
            write_jsonl(base / "input.jsonl", [row()])
            write_json(base / "policy.json", policy())
            write_jsonl(base / "excluded.jsonl", [{"id": "unrelated", "heavy": "A", "light": "C"}])
            original = workflow.select_records
            def changed(*args, **kwargs):
                result = original(*args, **kwargs)
                suffix = ".json" if key == "policy" else ".jsonl"
                source = base / (key + suffix)
                source.write_text(source.read_text() + "\n")
                return result
            with patch.object(workflow, "select_records", changed):
                with self.assertRaisesRegex(ValueError, "input changed"):
                    workflow.select_run(base, "input.jsonl", "policy.json", "selected", "excluded.jsonl")
            self.assertFalse((base / "selected").exists())

    def test_invalid_model_requests_fail_before_inputs_or_outputs(self):
        for models in ([], ["esm2", "esm2"], ["other"]):
            with self.assertRaisesRegex(ValueError, "exactly once"):
                workflow.score_run(self.root, "absent", "absent", "scored", models)
        self.assertFalse((self.root / "scored").exists())
