"""Thin explicit-file stage orchestration; no cluster or implicit GPU policy."""
from __future__ import annotations

from copy import deepcopy
import json

from .preparation import prepare_founder, reconstruct
from .records import new_output, read_jsonl, relative_file, sequence, sha256, write_json, write_jsonl
from .selection import MODEL, interaction_gate, select_records, validate_policy


def _require_unchanged(pinned):
    if any(sha256(path) != expected for path, expected in pinned.items()):
        raise ValueError("Stage input changed during execution; outputs are not committed")


def check_stepwise_trace(row, policy):
    """Verify full chain lineage; no old endpoint row can pretend to be V3."""
    if policy["version"] != "v3":
        return
    trace = row.get("stepwise_trace")
    if not isinstance(trace, list) or not 1 <= len(trace) <= 5:
        raise ValueError("V3 requires one to five explicit joint-proposal/gate records")
    parent = {role: row["founder"][role] for role in ("heavy", "light")}
    founder = prepare_founder(row["founder"])
    allowed = {(entry["chain"], entry["index"]) for entry in founder["cdr_position_map"].values()}
    seen_proposals = set()
    for index, step in enumerate(trace, 1):
        if step.get("round") != index or step.get("parent") != parent:
            raise ValueError("Stepwise parent/round does not form a contiguous founder lineage")
        requested = step.get("requested_sites")
        if not isinstance(requested, list) or not 1 <= len(requested) <= 10:
            raise ValueError("Each joint proposal must request one to ten sites")
        sites = set()
        for site in requested:
            if not isinstance(site, dict) or site.get("chain") not in parent or type(site.get("index")) is not int:
                raise ValueError("Each requested site must name its chain and zero-based index")
            key = site["chain"], site["index"]
            if not 0 <= key[1] < len(parent[key[0]]) or key in sites:
                raise ValueError("Invalid/duplicated requested site")
            if key not in allowed:
                raise ValueError("Requested site is outside the prepared founder CDR mapping")
            sites.add(key)
        child = step.get("child", {})
        changed = set()
        for role in parent:
            if not isinstance(child.get(role), str) or len(child[role]) != len(parent[role]):
                raise ValueError("V3 substitutions must preserve both full-chain lengths")
            sequence(child[role], "V3 child " + role)
            changed.update((role, p) for p, (a, b) in enumerate(zip(parent[role], child[role])) if a != b)
        if not changed <= sites:
            raise ValueError("A residue outside the requested sites changed")
        if not isinstance(step.get("proposal_id"), str) or not step["proposal_id"]:
            raise ValueError("Every V3 step must identify the scored proposal")
        if step["proposal_id"] in seen_proposals:
            raise ValueError("V3 proposal identities must be unique across steps")
        seen_proposals.add(step["proposal_id"])
        measured_child = {**step, **child, "id": step["proposal_id"], "antigen_id": row["antigen_id"]}
        if interaction_gate(measured_child, policy, require_all=False) != "pass":
            raise ValueError("Every accepted V3 joint proposal requires a measured DeepCDR pass")
        parent = {role: child[role] for role in parent}
    if any(parent[role] != row[role] for role in parent):
        raise ValueError("Stepwise final sequence differs from the candidate being selected")


def prepare_run(root, founder_path, output_dir):
    path = relative_file(root, founder_path)
    pinned = {path: sha256(path)}
    result = prepare_founder(json.loads(path.read_text()))
    _require_unchanged(pinned)
    output = new_output(root, output_dir)
    write_json(output / "founder.json", result)
    write_json(output / "prepare.json", {"status": "prepared_not_generated", "input_sha256": pinned[path],
                                        "output_sha256": sha256(output / "founder.json")})
    return result


def reconstruct_proposals(founder, proposals, *, version, model=MODEL):
    if model not in ("iMut-CDR-JM", MODEL):
        raise ValueError("Declare the actual supported joint generator model")
    founder = prepare_founder(founder)
    result = []
    for proposal in proposals:
        if proposal.get("model", model) != model:
            raise ValueError("Proposal model identity differs from the requested generator")
        if proposal.get("founder_fr_cdr_seq") != founder["fr_cdr_seq"]:
            raise ValueError("Generated proposal uses a different founder representation")
        fr = proposal.get("fr_cdr_seq") or proposal.get("mutant_fr_cdr_seq")
        rebuilt = reconstruct(founder, fr)
        result.append({**proposal, **rebuilt, "model": model, "version": version,
                       "design": founder["design"], "antigen_id": founder["antigen_id"],
                       "founder": deepcopy(founder),
                       "full_chain_reconstruction_required": False,
                       "full_chain_reconstruction_verified": True,
                       "representation": "actual_full_mutant_chains"})
    return result


def score_run(root, input_path, config_path, output_dir, models):
    from .scoring import score_esm2, score_3d
    if not models or len(set(models)) != len(models) or set(models) - {"esm2", "3d"}:
        raise ValueError("Choose each supported scoring model exactly once")
    source = relative_file(root, input_path)
    config_file = relative_file(root, config_path)
    pinned = {path: sha256(path) for path in (source, config_file)}
    config = json.loads(config_file.read_text())
    rows = read_jsonl(source)
    output = new_output(root, output_dir)
    result = []
    try:
        for source_row in rows:
            row = deepcopy(source_row)
            row.setdefault("scores", {})
            for model in models:
                if model in row["scores"]:
                    raise ValueError("Do not overwrite prior score evidence; use a fresh input cohort")
                function = {"esm2": score_esm2, "3d": score_3d}[model]
                row["scores"][model] = function(row, project_root=root,
                    assets=config["assets"][model], antigen=config["antigen"])
                if row["scores"][model]["status"] in ("resource_error", "asset_error", "dependency_error", "inference_error"):
                    result.append(row)
                    raise RuntimeError("Scoring attempt aborted: " + row["scores"][model]["status"])
            result.append(row)
        _require_unchanged(pinned)
        write_jsonl(output / "scores.jsonl", result)
        counts = {model: {state: sum(r["scores"][model]["status"] == state for r in result)
                          for state in sorted({r["scores"][model]["status"] for r in result})}
                  for model in models}
        receipt = {"status": "complete", "input_sha256": pinned[source],
                   "config_sha256": pinned[config_file], "score_states": counts,
                   "output_sha256": sha256(output / "scores.jsonl"), "selection_performed": False}
        write_json(output / "score.json", receipt)
        return receipt
    except BaseException as error:
        write_jsonl(output / "partial_scores.jsonl", result)
        write_json(output / "failure.json", {"status": "failed", "error": type(error).__name__ + ": " + str(error),
                   "input_sha256": pinned[source], "config_sha256": pinned[config_file],
                   "partial_output_not_committed": True})
        raise


def select_run(root, input_path, policy_path, output_dir, excluded_path=None):
    source, policy_file = relative_file(root, input_path), relative_file(root, policy_path)
    pinned = {path: sha256(path) for path in (source, policy_file)}
    policy = validate_policy(json.loads(policy_file.read_text()))
    rows = read_jsonl(source)
    for row in rows:
        check_stepwise_trace(row, policy)
    excluded = []
    if excluded_path:
        excluded_file = relative_file(root, excluded_path)
        pinned[excluded_file] = sha256(excluded_file)
        excluded = [(r["heavy"], r["light"]) for r in read_jsonl(excluded_file)]
    chosen, summary = select_records(rows, policy, excluded_pairs=excluded)
    _require_unchanged(pinned)
    output = new_output(root, output_dir)
    summary.update(input_sha256=pinned[source], policy_sha256=pinned[policy_file])
    if excluded_path:
        summary["excluded_input_sha256"] = pinned[excluded_file]
    write_jsonl(output / "selected.jsonl", chosen)
    write_json(output / "selection_summary.json", summary)
    return summary
