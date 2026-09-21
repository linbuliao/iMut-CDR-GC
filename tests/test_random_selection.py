"""Hash-sampling contracts using short synthetic sequences and invented scores."""
from copy import deepcopy
import hashlib
import json

import pytest

from imut_cdr_gc.records import read_jsonl, write_json, write_jsonl
from imut_cdr_gc.selection import random_priority, select_records, validate_policy
from imut_cdr_gc.workflow import select_run
from test_workflow_contract import RootFixture, policy, row


def random_policy(target=3, version="v1"):
    return {**policy(version=version, target=target),
            "selection_method": "random_without_replacement", "random_seed": "fixed-test-seed"}


def candidates():
    return [row(identifier=f"synthetic_{k}", k=k, esm=.51 + k / 100,
                structure=.65, likelihood=-1.5 + k / 100) for k in range(3, 11)]


def pairs(rows):
    return [(r["heavy"], r["light"]) for r in rows]


def test_priority_exact_canonical_bytes():
    expected = hashlib.sha256(b'["seed","design3/v1","ACD","EFG"]').hexdigest()
    assert random_priority("seed", "design3/v1", "ACD", "EFG") == expected


@pytest.mark.parametrize("seed", [None, "", "   ", 0, True, [], {}])
def test_random_seed_must_be_explicit_nonempty_string(seed):
    cfg = random_policy()
    cfg["random_seed"] = seed
    with pytest.raises(ValueError, match="random_seed"):
        validate_policy(cfg)


def test_unknown_method_and_ignored_seed_refused():
    cfg = random_policy()
    cfg["selection_method"] = "random_typo"
    with pytest.raises(ValueError, match="Choose"):
        validate_policy(cfg)
    cfg["selection_method"] = "score_top_k"
    with pytest.raises(ValueError, match="random_seed"):
        validate_policy(cfg)


def test_membership_ignores_scores_and_structure_availability_after_eligibility():
    original = candidates()
    before = deepcopy(original)
    selected, summary = select_records(original, random_policy())
    changed = deepcopy(list(reversed(original)))
    for r in changed:
        r["scores"]["esm2"]["score"] = 1 - r["scores"]["esm2"]["score"] / 2
        r["scores"]["3d"]["score"] = .99
        r["likelihood"]["score"] = 0.0
        r["complex"] = {"path": "does_not_exist.pdb", "verified": True}
    repeated, other = select_records(changed, random_policy())
    assert pairs(selected) == pairs(repeated)
    assert original == before
    assert summary["sampling"] == other["sampling"]
    assert summary["actual_boundary"] is None
    assert summary["score_ranking_cutoff"] is None
    assert summary["ranking"] is None
    assert summary["counts"]["eligible_not_sampled"] == 5
    assert "eligible_not_top_k" not in summary["counts"]
    assert not summary["full_complexes_verified"]
    assert [r["sampling_order"] for r in selected] == [1, 2, 3]
    assert all("selection_rank" not in r and "selection_primary_score" not in r for r in selected)


def test_identity_not_record_label_defines_priority():
    original = candidates()
    selected, _ = select_records(original, random_policy())
    renamed = [row(identifier=f"other_{k}", k=k) for k in range(3, 11)]
    repeated, _ = select_records(renamed, random_policy())
    assert pairs(selected) == pairs(repeated)


def test_duplicate_does_not_get_extra_chance_and_source_not_score_selects_provenance():
    original = candidates()
    original[0].update(source_path="z.jsonl", source_row_zero_based=0)
    duplicate = row(identifier="duplicate", k=3, esm=.5, structure=.5, likelihood=-1.9)
    duplicate.update(source_path="a.jsonl", source_row_zero_based=90)
    selected, _ = select_records(original, random_policy())
    repeated, summary = select_records(original + [duplicate], random_policy())
    assert pairs(selected) == pairs(repeated)
    assert summary["counts"]["duplicate_eligible_sequence_pair"] == 1
    all_rows, _ = select_records(original + [duplicate], random_policy(target=8))
    kept = next(r for r in all_rows if (r["heavy"], r["light"]) == pairs([duplicate])[0])
    assert kept["id"] == "duplicate"


def test_duplicate_fallback_uses_id_not_input_order_or_score():
    a = row(identifier="aaa", esm=.5, structure=.5, likelihood=-1.9)
    z = row(identifier="zzz", esm=.99, structure=.99)
    for group in ([a, z], [z, a]):
        chosen, _ = select_records(group, random_policy(target=1))
        assert chosen[0]["id"] == "aaa"


@pytest.mark.parametrize("bad", [{"source_path": "a"}, {"source_row_zero_based": 0},
    {"source_path": "a", "source_row_zero_based": True},
    {"source_path": "a", "source_row_zero_based": -1}])
def test_partial_or_invalid_source_identity_rejected(bad):
    r = row()
    r.update(bad)
    with pytest.raises(ValueError, match="provenance"):
        select_records([r], random_policy(target=1))


def test_exclusions_apply_before_sampling_and_do_not_modify_inputs():
    original = candidates()
    exclusions = pairs(original[:2])
    selected, summary = select_records(original, random_policy(), excluded_pairs=exclusions)
    assert not set(pairs(selected)).intersection(exclusions)
    assert summary["counts"]["excluded_existing_sequence_pair"] == 2
    assert summary["counts"]["eligible_unique"] == 6


def test_shortage_is_not_silently_called_complete():
    chosen, summary = select_records(candidates(), random_policy(target=2000))
    assert len(chosen) == 8
    assert summary["status"] == "insufficient_eligible"
    assert summary["counts"]["eligible_not_sampled"] == 0


def test_no_eligible_records_have_no_score_range_or_boundary():
    rejected = [row(likelihood=-3)]
    chosen, summary = select_records(rejected, random_policy())
    assert chosen == []
    assert summary["selected_score_ranges"] == {"esm2": None, "3d": None}
    assert summary["actual_boundary"] is None


def test_seed_changes_order_but_is_reproducible():
    cfg = random_policy(target=8)
    first, _ = select_records(candidates(), cfg)
    cfg["random_seed"] = "a-different-fixed-seed"
    second, _ = select_records(candidates(), cfg)
    repeat, _ = select_records(candidates(), cfg)
    assert pairs(first) != pairs(second)
    assert pairs(second) == pairs(repeat)
    assert set(pairs(first)) == set(pairs(second))


def test_original_score_top_k_remains_default():
    selected, summary = select_records(candidates(), policy(target=3))
    assert [r["id"] for r in selected] == ["synthetic_10", "synthetic_9", "synthetic_8"]
    assert summary["actual_boundary"]["is_requested_top_k_boundary"]
    assert "sampling" not in summary


def test_top_k_reselection_does_not_retain_random_sampling_labels():
    whole_pool, _ = select_records(candidates(), random_policy(target=8))
    chosen, summary = select_records(whole_pool, policy(target=3))
    assert summary["actual_boundary"] is not None
    assert all(r["selection_method"] == "score_top_k" for r in chosen)
    assert all("sampling_order" not in r and "sampling_priority_sha256" not in r for r in chosen)


def test_random_reselection_does_not_retain_top_k_labels():
    whole_pool, _ = select_records(candidates(), policy(target=8))
    chosen, summary = select_records(whole_pool, random_policy())
    assert summary["actual_boundary"] is None
    assert all(r["selection_method"] == "random_without_replacement" for r in chosen)
    assert all("selection_rank" not in r and "selection_primary_score" not in r for r in chosen)


class RandomStageTests(RootFixture):
    def test_file_stage_records_source_hash_and_policy_seed(self):
        original = candidates()
        write_jsonl(self.root / "scored.jsonl", original)
        write_json(self.root / "policy.json", random_policy())
        summary = select_run(self.root, "scored.jsonl", "policy.json", "chosen")
        persisted = json.loads((self.root / "chosen/selection_summary.json").read_text())
        assert persisted == summary
        assert persisted["policy"]["random_seed"] == "fixed-test-seed"
        assert persisted["input_sha256"] == hashlib.sha256((self.root / "scored.jsonl").read_bytes()).hexdigest()
        assert len(read_jsonl(self.root / "chosen/selected.jsonl")) == 3
        with pytest.raises(FileExistsError):
            select_run(self.root, "scored.jsonl", "policy.json", "chosen")

    def test_random_sampling_does_not_bypass_v3_lineage(self):
        write_jsonl(self.root / "scored.jsonl", [row(k=11, version="v3")])
        write_json(self.root / "policy.json", random_policy(target=1, version="v3"))
        with pytest.raises(ValueError, match="V3 requires"):
            select_run(self.root, "scored.jsonl", "policy.json", "chosen")
        assert not (self.root / "chosen").exists()
