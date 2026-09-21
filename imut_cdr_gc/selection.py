"""JM-Epi-only P5 eligibility and explicit score-ranked or random selection.

Missing/errored scores are never zero or threshold failures. This component
does not certify the lineage of externally supplied V3 candidates; use the
workflow trace checker before calling it on production data.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import re

from .records import finite, sequence

ROUTES = {"esm2_only": ("esm2",), "native3d_only": ("3d",),
          "esm2_OR_native3d": ("esm2", "3d")}
PROTOCOLS = {"esm2": "deepcdr_esm2_hfirst_anarci_list_index95_v1",
             "3d": "deepcdr_3d_antigen_first_v1"}
MODEL = "iMut-CDR-JM-Epi"


def validate_policy(policy):
    policy = deepcopy(policy)
    if policy.get("route") not in ROUTES:
        raise ValueError("Declare the actual DeepCDR route")
    if policy.get("version") not in ("v1", "v2", "v3"):
        raise ValueError("Declare v1, v2 or v3")
    if not finite(policy.get("common_deepcdr_floor"), probability=True):
        raise ValueError("Both active scorers must share one finite DeepCDR floor")
    if policy["common_deepcdr_floor"] < 0.5:
        raise ValueError("This workflow requires a common DeepCDR floor >= 0.5")
    calibration = policy.get("likelihood_calibration", {})
    if (calibration.get("lower_quantile") != 0.05
            or calibration.get("upper_quantile") is not None
            or calibration.get("upper_cutoff") is not None
            or not finite(calibration.get("lower_cutoff"))
            or not calibration.get("metric_identity")
            or not isinstance(calibration.get("source_identity"), dict)
            or not calibration["source_identity"]
            or not re.fullmatch("[0-9a-f]{64}", str(calibration.get("reference_sha256", "")))
            or calibration.get("reference_not_candidates") is not True):
        raise ValueError("Require a named, fixed reference P5 calibration with no upper bound")
    target = policy.get("target")
    if type(target) is not int or target < 1:
        raise ValueError("target must be a positive integer")
    method = policy.get("selection_method", "score_top_k")
    if method not in ("score_top_k", "random_without_replacement"):
        raise ValueError("Choose score_top_k or random_without_replacement")
    if method == "random_without_replacement":
        seed = policy.get("random_seed")
        if not isinstance(seed, str) or not seed.strip():
            raise ValueError("Random selection requires an explicit nonempty random_seed")
    elif "random_seed" in policy:
        raise ValueError("random_seed requires random_without_replacement selection")
    return policy


def random_priority(seed, cohort, heavy, light):
    """Frozen hash lottery: sequence identity, not score, ID or structure state."""
    if not all(isinstance(value, str) and value for value in (seed, cohort)):
        raise ValueError("Seed and cohort must be nonempty strings")
    sequence(heavy, "heavy")
    sequence(light, "light")
    payload = json.dumps([seed, cohort, heavy, light], sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _provenance_key(row):
    """Retain a duplicate's provenance without looking at its scores."""
    if "source_path" in row or "source_row_zero_based" in row:
        source, index = row.get("source_path"), row.get("source_row_zero_based")
        if not isinstance(source, str) or not source or type(index) is not int or index < 0:
            raise ValueError("Duplicate provenance requires source_path and nonnegative source_row_zero_based")
        return (0, source, index, row["id"])
    return (1, row["id"], -1, row["id"])


def measured_score(row, model):
    item = row.get("scores", {}).get(model, {})
    value = item.get("score")
    if item.get("status") != "ok" or not finite(value, probability=True):
        return None
    expected_model = {"esm2": "DeepCDR-ESM2", "3d": "DeepCDR-3D"}[model]
    inputs = item.get("inputs") or {}
    if item.get("model") != expected_model or item.get("id") != row.get("id"):
        return None
    if (item.get("protocol") != PROTOCOLS[model]
            or (item.get("identity") or {}).get("protocol") != PROTOCOLS[model]):
        return None
    if inputs.get("antigen", {}).get("id") != row.get("antigen_id") or not row.get("antigen_id"):
        return None
    for role in ("heavy", "light"):
        seq = row.get(role)
        if not isinstance(seq, str) or inputs.get(role + "_sha256") != hashlib.sha256(seq.encode()).hexdigest():
            return None
    return value


def interaction_gate(row, policy, *, require_all=True):
    """Terminal ranking needs both scores for a two-model OR route.

    A V3 intermediate OR may short-circuit on one measured passing score;
    missing all passing scores is unknown unless all active scores are known.
    """
    names = ROUTES[policy["route"]]
    scores = [measured_score(row, name) for name in names]
    if require_all and any(value is None for value in scores):
        return "unmeasured_or_error"
    if any(value is not None and value >= policy["common_deepcdr_floor"] for value in scores):
        return "pass"
    return "unmeasured_or_error" if any(value is None for value in scores) else "below_common_deepcdr_floor"


def select_records(records, policy, *, excluded_pairs=()):
    policy = validate_policy(policy)
    rows = [deepcopy(row) for row in records]
    if not rows:
        raise ValueError("No candidates supplied")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("Candidate IDs must be unique nonempty strings")
    if any(row.get("model") != MODEL for row in rows):
        raise ValueError("Likelihood/P5/GC selection is restricted to JM-Epi candidates")
    if any(row.get("version") != policy["version"] for row in rows):
        raise ValueError("Do not pool version-specific mutation burdens in selection")
    if len({row.get("design") for row in rows}) != 1 or not rows[0].get("design"):
        raise ValueError("Select one explicitly named design at a time")
    order = {identifier: index for index, identifier in enumerate(ids)}
    bounds = (3, 10) if policy["version"] == "v1" else (11, 20)
    excluded_pairs = set(excluded_pairs)
    counts, decisions, eligible = Counter(input=len(rows)), [], []
    calibration = policy["likelihood_calibration"]
    for row in rows:
        heavy, light = sequence(row.get("heavy"), "heavy"), sequence(row.get("light"), "light")
        founder = row.get("founder", {})
        fh, fl = sequence(founder.get("heavy"), "founder heavy"), sequence(founder.get("light"), "founder light")
        if len(heavy) != len(fh) or len(light) != len(fl):
            raise ValueError("This substitution workflow does not accept indels")
        actual_k = sum(a != b for a, b in zip(heavy + light, fh + fl))
        if row.get("mutation_count") != actual_k:
            raise ValueError("Declared mutation count differs from actual full-chain sequence")
        reason = None
        if not bounds[0] <= actual_k <= bounds[1]:
            reason = "outside_mutation_burden"
        elif (heavy, light) in excluded_pairs:
            reason = "excluded_existing_sequence_pair"
        else:
            likelihood = row.get("likelihood", {})
            if (likelihood.get("status") != "ok" or not finite(likelihood.get("score"))
                    or likelihood.get("metric_identity") != calibration["metric_identity"]
                    or likelihood.get("source_identity") != calibration["source_identity"]
                    or likelihood.get("id") != row["id"]
                    or any(likelihood.get(role + "_sha256") != hashlib.sha256(row[role].encode()).hexdigest()
                           for role in ("heavy", "light"))):
                reason = "likelihood_unmeasured_error_or_identity_mismatch"
            elif likelihood["score"] < calibration["lower_cutoff"]:
                reason = "below_reference_p5"
            else:
                gate = interaction_gate(row, policy)
                if gate != "pass":
                    reason = gate
        if reason:
            counts[reason] += 1
            decisions.append({"id": row["id"], "state": reason})
        else:
            eligible.append(row)
    names = ROUTES[policy["route"]]
    method = policy.get("selection_method", "score_top_k")
    random_selection = method == "random_without_replacement"
    cohort = rows[0]["design"] + "/" + policy["version"]

    def rank_key(row):
        scores = sorted((measured_score(row, name) for name in names), reverse=True)
        return (*(-v for v in scores), -row["likelihood"]["score"], row["id"])

    eligible.sort(key=_provenance_key if random_selection else rank_key)
    unique, seen = [], set()
    for row in eligible:
        pair = row["heavy"], row["light"]
        if pair in seen:
            counts["duplicate_eligible_sequence_pair"] += 1
            decisions.append({"id": row["id"], "state": "duplicate_eligible_sequence_pair"})
        else:
            seen.add(pair); unique.append(row)
    if random_selection:
        unique.sort(key=lambda r: (
            random_priority(policy["random_seed"], cohort, r["heavy"], r["light"]),
            hashlib.sha256((r["heavy"] + "\0" + r["light"]).encode("ascii")).hexdigest()))
    selected = unique[:policy["target"]]
    for rank, row in enumerate(selected, 1):
        row["common_deepcdr_floor"] = policy["common_deepcdr_floor"]
        if random_selection:
            row.pop("selection_rank", None)
            row.pop("selection_primary_score", None)
            row.update(selection_method=method, sampling_order=rank,
                       sampling_priority_sha256=random_priority(
                           policy["random_seed"], cohort, row["heavy"], row["light"]))
            decisions.append({"id": row["id"], "state": "selected", "sampling_order": rank})
        else:
            row.pop("sampling_order", None)
            row.pop("sampling_priority_sha256", None)
            row.update(selection_method=method, selection_rank=rank,
                       selection_primary_score=-rank_key(row)[0])
            decisions.append({"id": row["id"], "state": "selected", "rank": rank})
    outside = "eligible_not_sampled" if random_selection else "eligible_not_top_k"
    for row in unique[policy["target"]:]:
        decisions.append({"id": row["id"], "state": outside})
    counts.update(eligible_before_dedup=len(eligible), eligible_unique=len(unique),
                  selected=len(selected))
    counts[outside] = max(0, len(unique) - len(selected))
    boundary = None
    if selected and not random_selection:
        last = selected[-1]
        boundary = {"rank": len(selected), "id": last["id"],
                    "primary_score": -rank_key(last)[0],
                    "active_model_scores": {name: measured_score(last, name) for name in names},
                    "likelihood": last["likelihood"]["score"],
                    "minimum_selected_model_scores": {name: min(measured_score(r, name) for r in selected)
                                                       for name in names},
                    "is_requested_top_k_boundary": len(selected) == policy["target"]}
    summary = {"status": "complete" if len(selected) == policy["target"] else "insufficient_eligible",
        "counts": dict(counts), "policy": policy, "actual_boundary": boundary,
        "decisions": sorted(decisions, key=lambda d: order[d["id"]]),
        "ranking": None if random_selection else "active scores descending (max, min for OR), likelihood descending, id ascending",
        "or_boundary_is_not_two_independent_model_cutoffs": True,
        "full_complexes_verified": False, "corrected_production_accepted": False}
    if random_selection:
        summary.update(selection_method=method, score_ranking_cutoff=None,
            sampling={"seed": policy["random_seed"], "cohort": cohort,
                      "ordering": "Ascending SHA256(canonical JSON [seed,cohort,heavy,light])",
                      "without_replacement": True,
                      "duplicate_provenance": "source path/row then ID, or ID when no source pointer is supplied",
                      "score_or_pdb_based_sampling": False},
            selected_score_ranges={name: {
                "min": min(measured_score(r, name) for r in selected),
                "max": max(measured_score(r, name) for r in selected),
                "available": len(selected)} if selected else None for name in names})
    return selected, summary
