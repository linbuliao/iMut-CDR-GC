"""Safe graph arithmetic matches the frozen native pure function on fixtures."""
import numpy as np
import pytest

from imut_cdr_gc.antigen_graph import residue_graph, prepare_antigen_graph
from imut_cdr_gc.records import sha256, write_json


def atoms():
    return dict(pos=np.array([[0., 0, 0], [2., 0, 0], [8., 0, 0], [40., 0, 0]], np.float64),
        res_name=np.array(["ALA", "ALA", "GLY", "VAL"]), res_id=np.array([2, 2, 8, 1]),
        chain_id=np.array(["A", "A", "A", "B"]), atom_name=np.array(["CA", "C", "CA", "N"]))


def features():
    return {"A": list(range(30)), "GLY": [1.] * 30, "VAL": [2.] * 30}


def test_native_order_duplicate_edges_and_cutoff_equality():
    result = residue_graph(atoms(), features())
    np.testing.assert_array_equal(result["pos"], [[0, 0, 0], [8, 0, 0], [40, 0, 0]])
    np.testing.assert_array_equal(result["edge_index"], [[0, 1, 0, 1], [1, 0, 1, 0]])
    np.testing.assert_array_equal(result["edge_attr"], [[8, 1], [8, 1], [8, 0], [8, 0]])
    assert result["x"].dtype == np.float32
    assert result["edge_index"].dtype == np.int64


def test_no_ca_uses_all_atom_mean_and_sorted_sequence_edges_across_number_gaps():
    value = atoms(); value["atom_name"][0] = "N"
    graph = residue_graph(value, features(), cutoff=.5)
    np.testing.assert_array_equal(graph["pos"][0], [1, 0, 0])
    np.testing.assert_array_equal(graph["edge_attr"], [[7, 1], [7, 1]])


def test_single_node_empty_edge_arrays():
    value = {key: a[:1] for key, a in atoms().items()}
    result = residue_graph(value, features())
    assert result["edge_index"].shape == (2, 0)
    assert result["edge_attr"].shape == (0, 2)


@pytest.mark.parametrize("change", ["object", "bytes", "insertion", "duplicate", "nonfinite", "float_ids", "unknown"])
def test_unsafe_or_ambiguous_atoms_rejected(change):
    value = atoms()
    if change == "object": value["res_name"] = value["res_name"].astype(object)
    elif change == "bytes": value["res_name"] = value["res_name"].astype("S")
    elif change == "insertion": value["insertion_code"] = np.array(["A", "A", "", ""])
    elif change == "duplicate": value["atom_name"][1] = "CA"
    elif change == "nonfinite": value["pos"][0, 0] = float("nan")
    elif change == "float_ids": value["res_id"] = value["res_id"].astype(float)
    else: value["res_name"][0] = "UNK"
    with pytest.raises(ValueError): residue_graph(value, features())


@pytest.mark.parametrize("bad", [[0.] * 29, [float("nan")] * 30, None])
def test_feature_resize_and_unknown_fallback_not_silent(bad):
    value = features()
    if bad is None: value.pop("A")
    else: value["A"] = bad
    with pytest.raises(ValueError): residue_graph(atoms(), value)


def test_safe_materialized_graph_roundtrip(tmp_path):
    np.savez(tmp_path / "atoms.npz", **atoms())
    write_json(tmp_path / "features.json", features())
    receipt = prepare_antigen_graph(project_root=tmp_path, atom_npz="atoms.npz",
        atom_sha256=sha256(tmp_path / "atoms.npz"), feature_json="features.json",
        feature_sha256=sha256(tmp_path / "features.json"), antigen_id="synthetic", output_dir="graph")
    with np.load(tmp_path / "graph/antigen_graph.npz", allow_pickle=False) as graph:
        for key, expected in residue_graph(atoms(), features()).items():
            np.testing.assert_array_equal(graph[key], expected)
    assert receipt["nodes"] == 3 and receipt["directed_edges"] == 4
    assert receipt["scientific_acceptance"] is False
