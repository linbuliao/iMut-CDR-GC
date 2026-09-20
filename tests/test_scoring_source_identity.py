"""Standalone source/feature identity checks; no research checkout is imported."""
import ast
import hashlib
import json
from pathlib import Path
import struct

import pytest

from imut_cdr_gc.assets import AssetError

SCORING = Path(__file__).resolve().parents[1] / 'imut_cdr_gc/scoring'


def test_vendored_source_hashes_and_exact_extracted_math():
    manifest = json.loads((SCORING/'source_identity.json').read_text())
    for entry in manifest['vendor_files']:
        path = SCORING / entry['target']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['target_sha256']
        if 'ast_sha256' in entry:
            nodes = {}
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    nodes[node.name] = node
                elif isinstance(node, ast.Assign):
                    for name in node.targets:
                        if isinstance(name, ast.Name): nodes[name.id] = node
            for name, expected in entry['ast_sha256'].items():
                actual = ast.dump(nodes[name], include_attributes=False)
                assert hashlib.sha256(actual.encode()).hexdigest() == expected
        if 'numeric_float32_sha256' in entry:
            vectors = json.loads(path.read_text())
            for aa, expected in entry['numeric_float32_sha256'].items():
                assert hashlib.sha256(struct.pack('<20f', *vectors[aa])).hexdigest() == expected
            assert vectors['X'] == [0.] * 20
    assert manifest['rights']['weights_bundled'] is False
    assert manifest['rights']['public_push_performed'] is False


@pytest.mark.parametrize('dimension', [20, 30])
def test_feature_dimensions_and_original_values_both_enforced(tmp_path, dimension):
    pytest.importorskip('numpy')
    from imut_cdr_gc.scoring.backend import feature_json
    source = SCORING / '_vendor' / f'amino_acid_vectors_{dimension}dim.json'
    assert feature_json(source, dimension)
    values = json.loads(source.read_text())
    aa = 'A' if dimension == 20 else 'ALA'
    values[aa][0] += 0.25
    changed = tmp_path / 'changed_features.json'
    changed.write_text(json.dumps(values))
    with pytest.raises(AssetError, match='native numeric vectors'):
        feature_json(changed, dimension)
    values[aa] = values[aa][:-1]
    changed.write_text(json.dumps(values))
    with pytest.raises(AssetError, match='dimensional'):
        feature_json(changed, dimension)


def test_cpu_thread_validation_and_no_unsafe_checkpoint_fallback(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('torch_geometric')
    from imut_cdr_gc.scoring.backend import configure_cpu, load_model
    for invalid in (True, 0, -1, 1.2, '1'):
        with pytest.raises(ValueError, match='positive integer'):
            configure_cpu(invalid)
    weights = tmp_path / 'not_a_state.pt'
    weights.write_bytes(b'not an executable pickle or state dictionary')
    with pytest.raises(AssetError, match='safe weights_only'):
        load_model('DeepCDR-3D', str(weights), hashlib.sha256(weights.read_bytes()).hexdigest(), 1)
    load_model.cache_clear()


def test_safe_loader_preserves_resource_failure_and_shape_error_is_asset(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('torch_geometric')
    from imut_cdr_gc.scoring.backend import load_model
    from imut_cdr_gc.scoring.common import is_resource_failure
    def exhausted(*args, **kwargs): raise MemoryError('Fixture no memory during safe loading')
    monkeypatch.setattr(torch, 'load', exhausted)
    with pytest.raises(MemoryError):
        load_model('DeepCDR-3D', str(tmp_path/'fixture.pt'), 'resource-fixture', 1)
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: {})
    with pytest.raises(AssetError, match='exact compatible'):
        load_model('DeepCDR-3D', str(tmp_path/'fixture.pt'), 'shape-fixture', 1)
    assert not is_resource_failure(RuntimeError('size mismatch for out.weight'))
    load_model.cache_clear()
