"""Write small synthetic software-test inputs; no models or real antibodies."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


NOTICE = "SYNTHETIC SOFTWARE DEMO: invented sequences and scores; not scientific results."
METRIC = "synthetic-demo-likelihood-not-a-trained-model.v1"
SOURCE = {"synthetic": True, "producer": "examples/make_demo.py", "metric": METRIC}


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def linear_quantile(values, quantile):
    values = sorted(values)
    position = (len(values) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (position - low) * (values[high] - values[low])


def founder():
    # Short strings exercise the data format; these are not antibody domains.
    fr = ["X"] * 277
    mapping = {}
    for chain, offset, index, motif in (
        ("light", 26, 1, "ACD"), ("light", 55, 5, "EFG"),
        ("light", 102, 9, "HIK"), ("heavy", 165, 1, "LMN"),
        ("heavy", 194, 5, "PQR"), ("heavy", 241, 9, "STV"),
    ):
        fr[offset:offset + 3] = motif
        for delta in range(3):
            mapping[str(offset + delta)] = {"chain": chain, "index": index + delta}
    return {
        "id": "synthetic_founder", "design": "synthetic_design",
        "antigen_id": "synthetic_antigen", "heavy": "WLMNYPQRWSTVY",
        "light": "WACDYEFGWHIKY", "fr_cdr_seq": "".join(fr),
        "cdr_position_map": mapping, "synthetic": True, "notice": NOTICE,
    }


def candidate(identifier, positions, esm2, structure, likelihood):
    parent = founder()
    heavy = list(parent["heavy"])
    mutations = []
    for index in positions:
        old = heavy[index]
        heavy[index] = "A"
        mutations.append({"chain": "heavy", "chain_index": index, "from": old, "to": "A"})
    heavy = "".join(heavy)
    row = {
        "id": identifier, "design": parent["design"], "antigen_id": parent["antigen_id"],
        "model": "iMut-CDR-JM-Epi", "version": "v1", "founder": parent,
        "heavy": heavy, "light": parent["light"], "mutation_count": len(positions),
        "mutations": mutations, "synthetic": True, "notice": NOTICE,
        "model_was_executed": False, "position_index_base": 0,
    }
    hashes = {role + "_sha256": digest(row[role]) for role in ("heavy", "light")}
    row["likelihood"] = {
        "id": identifier, "status": "ok", "score": likelihood,
        "metric_identity": METRIC, "source_identity": deepcopy(SOURCE), **hashes,
    }
    row["scores"] = {}
    for name, value, label, protocol in (
        ("esm2", esm2, "DeepCDR-ESM2", "deepcdr_esm2_hfirst_anarci_list_index95_v1"),
        ("3d", structure, "DeepCDR-3D", "deepcdr_3d_antigen_first_v1"),
    ):
        # Native field names exercise the reader, not a native model execution.
        row["scores"][name] = {
            "id": identifier, "model": label, "protocol": protocol,
            "identity": {"protocol": protocol, "synthetic": True, "notice": NOTICE},
            "inputs": {**hashes, "antigen": {"id": parent["antigen_id"]}},
            "status": "ok" if value is not None else "dependency_error", "score": value,
            "synthetic": True, "model_was_executed": False,
        }
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.project_root.resolve(strict=True)
    output = root / "outputs" / "demo_inputs"
    if output.exists() or output.is_symlink():
        raise FileExistsError("outputs/demo_inputs already exists; keep it or use a fresh working project")
    if not output.resolve().is_relative_to(root):
        raise ValueError("Demo output must stay within the working project")
    output.mkdir(parents=True, exist_ok=False)
    reference = {
        "synthetic": True, "notice": NOTICE, "reference_not_candidates": True,
        "metric_identity": METRIC, "source_identity": deepcopy(SOURCE),
        "scores": [-3.0, -2.0, -1.0, 0.0, 1.0],
        "quantile_method": "linear interpolation at (n - 1) * q",
    }
    write_json(output / "reference.json", reference)
    cutoff = linear_quantile(reference["scores"], 0.05)
    policy = {
        "version": "v1", "route": "esm2_OR_native3d", "target": 2,
        "common_deepcdr_floor": 0.5, "synthetic": True, "notice": NOTICE,
        "likelihood_calibration": {
            "lower_quantile": 0.05, "lower_cutoff": cutoff,
            "upper_quantile": None, "upper_cutoff": None,
            "metric_identity": METRIC, "source_identity": deepcopy(SOURCE),
            "reference_sha256": hashlib.sha256((output / "reference.json").read_bytes()).hexdigest(),
            "reference_not_candidates": True,
        },
    }
    rows = [
        candidate("keep_high", [1, 2, 3], .95, .70, -1.0),
        candidate("keep_p5_boundary", [5, 6, 7], .60, .80, cutoff),
        candidate("eligible_outside_top", [9, 10, 11], .70, .60, -1.5),
        candidate("duplicate", [1, 2, 3], .90, .65, -1.0),
        candidate("outside_burden", [1], .95, .90, -1.0),
        candidate("below_p5", [1, 5, 9], .95, .90, cutoff - .1),
        candidate("below_floor", [2, 6, 10], .40, .49, -1.0),
        candidate("missing_structure_score", [3, 7, 11], .90, None, -1.0),
    ]
    expected = {
        "synthetic": True, "selected_ids": ["keep_high", "keep_p5_boundary"],
        "counts": {
            "input": 8, "outside_mutation_burden": 1, "below_reference_p5": 1,
            "below_common_deepcdr_floor": 1, "unmeasured_or_error": 1,
            "eligible_before_dedup": 4, "duplicate_eligible_sequence_pair": 1,
            "eligible_unique": 3, "selected": 2, "eligible_not_top_k": 1,
        },
        "lower_cutoff": cutoff, "primary_score_at_selection_boundary": .80,
    }
    write_json(output / "founder.json", founder())
    write_json(output / "policy.json", policy)
    write_json(output / "expected.json", expected)
    with (output / "scored.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps({"status": "synthetic_inputs_written", "input_records": 8,
                      "output_dir": "outputs/demo_inputs", "model_calls": 0,
                      "scientific_results": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
