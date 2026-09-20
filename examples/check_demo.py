"""Check the persisted synthetic demo without importing a model or scoring data."""
import argparse
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError("Demo check failed: " + message)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.project_root.resolve(strict=True)
    inputs = root / "outputs" / "demo_inputs"
    result = root / "outputs" / "demo_selected"
    expected = json.loads((inputs / "expected.json").read_text())
    summary = json.loads((result / "selection_summary.json").read_text())
    rows = [json.loads(line) for line in (result / "selected.jsonl").read_text().splitlines()]
    prepared = json.loads((root / "outputs" / "demo_prepared" / "founder.json").read_text())
    require(prepared["synthetic"] is True and prepared["prepared"] is True, "prepared toy founder")
    require(summary["counts"] == expected["counts"], "all sequential counts")
    require(summary["status"] == "complete", "two synthetic records selected")
    require([row["id"] for row in rows] == expected["selected_ids"], "selected IDs/order")
    require(len(summary["decisions"]) == 8, "one decision for every synthetic record")
    require(all(row["synthetic"] is True and row["model_was_executed"] is False for row in rows),
            "synthetic labels retained")
    require(summary["full_complexes_verified"] is False, "no claimed PDB verification")
    require(summary["corrected_production_accepted"] is False, "no claimed scientific delivery")
    calibration = summary["policy"]["likelihood_calibration"]
    require(calibration["lower_cutoff"] == expected["lower_cutoff"], "independent reference P5")
    require(calibration["upper_cutoff"] is None and calibration["upper_quantile"] is None, "no upper gate")
    require(calibration["reference_sha256"] == hashlib.sha256((inputs / "reference.json").read_bytes()).hexdigest(),
            "actual synthetic reference identity")
    require(summary["input_sha256"] == hashlib.sha256((inputs / "scored.jsonl").read_bytes()).hexdigest(),
            "actual input identity")
    require(summary["actual_boundary"]["primary_score"] == expected["primary_score_at_selection_boundary"],
            "observed selection boundary")
    print(json.dumps({"status": "demo_checks_passed", "input_records": 8, "selected": 2,
                      "model_calls": 0, "structures_created": 0, "scientific_results": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
