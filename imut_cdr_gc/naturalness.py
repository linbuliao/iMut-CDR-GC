"""Naturalness analysis with explicit local assets and JM-Epi-only P5 eligibility.

AbNatiV-2 is an evaluation metric, not campaign likelihood or measured binding.
The campaign API calls the original SAbDab-298 scorer. It does not download
models, silently truncate, substitute founder chains, or create GC libraries.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile

from .assets import AssetError, file_sha256 as digest, verified_asset
from ._likelihood_native import identity_hash, load_native_scorer
from .records import new_output

AA = frozenset('ACDEFGHIKLMNPQRSTVWY')
METRIC = 'AbNatiV Heavy-Light Score'
CAMPAIGN_MODEL = 'iMut-CDR-JM-Epi'
SEGMENTS = ['fwr1', 'cdr1', 'fwr2', 'cdr2', 'fwr3', 'cdr3', 'fwr4']
RANGES = [(1, 26), (27, 38), (39, 55), (56, 65), (66, 104), (105, 117), (118, 128)]
WIDTHS = {'light': [26, 12, 17, 10, 38, 25, 10], 'heavy': [25, 12, 18, 12, 39, 30, 11]}


class MeasurementError(RuntimeError):
    """Incomplete measurement attempt: never a P5 rejection or a zero score."""


def _dump(path, value):
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.metadata_', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_rows(path, rows):
    path = Path(path)
    text = ''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.rows_', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # atomically commit without replacing an existing file
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_records(records, *, campaign=False):
    if not records:
        raise ValueError('No antibody records supplied')
    if any(not isinstance(row, dict) for row in records):
        raise ValueError('Every record must be a mapping')
    ids = [row.get('id') for row in records]
    if any(not isinstance(key, str) or not key for key in ids) or len(ids) != len(set(ids)):
        raise ValueError('Missing or duplicated sequence ID; equal sequences with distinct IDs are allowed')
    if campaign and any(row.get('model') != CAMPAIGN_MODEL for row in records):
        raise ValueError('Campaign likelihood/P5 is only for explicit iMut-CDR-JM-Epi candidates')
    if campaign and any('likelihood' in row or 'average_log_likelihood' in row for row in records):
        raise ValueError('Existing likelihood measurement fields cannot be overwritten; supply fresh candidate inputs')


def read_pairs(path):
    """Read actual mutant heavy/light chains; never infer heavy_full/light_full."""
    path = Path(path)
    with path.open() as handle:
        rows = list(csv.DictReader(handle)) if path.suffix.lower() == '.csv' else [
            json.loads(line) for line in handle if line.strip()]
    output = []
    for row in rows:
        item = dict(row)
        item['id'] = str(row.get('id') or row.get('mutant_id') or '')
        for chain in ('heavy', 'light'):
            alias = chain + '_mut_variable'
            if chain in row and alias in row and row[chain] != row[alias]:
                raise ValueError('Conflicting actual mutant chain fields: ' + chain)
            item[chain] = row.get(chain, row.get(alias, ''))
        item['group'] = str(row.get('group') or row.get('model') or 'all')
        output.append(item)
    _validate_records(output)
    return output


def read_score_rows(path):
    path = Path(path)
    with path.open() as handle:
        if path.suffix.lower() == '.csv':
            return list(csv.DictReader(handle))
        if path.suffix.lower() == '.jsonl':
            return [json.loads(line) for line in handle if line.strip()]
    raise ValueError('Score input must be CSV or JSONL')


def _finite(value):
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def score_qc(values):
    values = list(values)
    finite = [float(x) for x in values if _finite(x)]
    return dict(input=len(values), finite=len(finite), missing_or_nonfinite=len(values)-len(finite),
        minimum=min(finite) if finite else None, maximum=max(finite) if finite else None,
        mean=statistics.fmean(finite) if finite else None,
        standard_deviation=statistics.pstdev(finite) if finite else None,
        degenerate=len(finite) < 2 or max(finite)-min(finite) <= 1e-12)


def _quantile(values, q):
    ordered = sorted(float(x) for x in values)
    index = (len(ordered)-1)*q
    low = math.floor(index)
    high = math.ceil(index)
    fraction = index-low
    # The standard linear reference quantile; never a candidate percentile score.
    if fraction >= 0.5:
        return ordered[high] - (ordered[high]-ordered[low])*(1-fraction)
    return ordered[low] + (ordered[high]-ordered[low])*fraction


def calibrate(values, metric_identity, lower_quantile, upper_quantile, reference_name,
              *, source_identity=None):
    """Generic analysis API. Campaign selection separately enforces P5/no upper."""
    values = list(values)
    qc = score_qc(values)
    if not metric_identity or not reference_name:
        raise ValueError('Name the metric identity and reference population')
    if qc['missing_or_nonfinite'] or qc['finite'] < 10 or qc['degenerate']:
        raise ValueError('Calibration requires >=10 finite, nonconstant reference scores')
    for q in (lower_quantile, upper_quantile):
        if q is not None and (not _finite(q) or not 0 <= q <= 1):
            raise ValueError('Quantiles must be in [0,1]')
    if lower_quantile is not None and upper_quantile is not None and lower_quantile > upper_quantile:
        raise ValueError('Reversed quantiles')
    result = dict(metric_identity=metric_identity, reference_name=reference_name,
        reference_count=len(values), lower_quantile=lower_quantile, upper_quantile=upper_quantile,
        lower_cutoff=_quantile(values, lower_quantile) if lower_quantile is not None else None,
        upper_cutoff=_quantile(values, upper_quantile) if upper_quantile is not None else None,
        quantile_method='linear', bounds='inclusive', qc=qc)
    if source_identity is not None:
        result['source_identity'] = source_identity
    return result


def apply_calibration(values, calibration, metric_identity, *, source_identity=None):
    if metric_identity != calibration.get('metric_identity'):
        raise ValueError('Scorer/checkpoint/normalization identity mismatch')
    if source_identity is not None or 'source_identity' in calibration:
        if source_identity != calibration.get('source_identity'):
            raise ValueError('Calibration source/runtime identity mismatch')
    low, high = calibration['lower_cutoff'], calibration['upper_cutoff']
    if any(x is not None and not _finite(x) for x in (low, high)) or (
            low is not None and high is not None and low > high):
        raise ValueError('Invalid calibration cutoffs')
    labels = ['missing' if not _finite(x) else 'below_lower' if low is not None and float(x) < low
              else 'above_upper' if high is not None and float(x) > high else 'pass' for x in values]
    return labels, {label: labels.count(label) for label in ('pass', 'below_lower', 'above_upper', 'missing')}


def validate_campaign_calibration(calibration, metric_identity, source_identity):
    import re
    if (not isinstance(calibration, dict) or calibration.get('lower_quantile') != 0.05
            or calibration.get('upper_quantile') is not None or calibration.get('upper_cutoff') is not None
            or not _finite(calibration.get('lower_cutoff'))
            or not calibration.get('reference_name')
            or not re.fullmatch('[0-9a-f]{64}', str(calibration.get('reference_sha256', '')))
            or calibration.get('reference_not_candidates') is not True
            or calibration.get('bounds') != 'inclusive'
            or not isinstance(source_identity, dict) or not source_identity
            or calibration.get('source_identity') != source_identity):
        raise ValueError('Campaign requires a fixed independent-reference P5, exact source identity and no upper bound')
    apply_calibration([], calibration, metric_identity, source_identity=source_identity)
    return calibration


def build_layout298(heavy_segments, light_segments):
    parts = []
    for chain, segments in (('light', light_segments), ('heavy', heavy_segments)):
        for name, width in zip(SEGMENTS, WIDTHS[chain]):
            segment = segments.get(name)
            if not isinstance(segment, str) or not segment or not set(segment) <= AA or len(segment) > width:
                raise ValueError('Invalid or overlength ' + chain + '/' + name)
            parts.append(segment.ljust(width, 'X'))
    sequence = 'X'.join(parts)
    native_positions(sequence)
    return sequence


def native_positions(sequence):
    if not isinstance(sequence, str) or len(sequence) != 298 or set(sequence) - (AA | {'X'}):
        raise ValueError('Expected native 298-position AA20/X sequence')
    cursor, positions = 0, []
    for index, (name, width) in enumerate([(name, width) for chain in ('light', 'heavy')
                                         for name, width in zip(SEGMENTS, WIDTHS[chain])]):
        segment = sequence[cursor:cursor+width].rstrip('X')
        if not segment or 'X' in segment:
            raise ValueError('Empty or internally padded native segment')
        if name.startswith('cdr'):
            positions.extend(range(cursor, cursor+len(segment)))
        cursor += width
        if index < 13:
            if sequence[cursor] != 'X':
                raise ValueError('Native separator is not X')
            cursor += 1
    return positions


def _numbering_backend():
    """Bind the actual CPU numbering code, executable, HMMs and defaults."""
    package = importlib.import_module('anarci')
    implementation = importlib.import_module('anarci.anarci')
    numberer = package.anarci
    binary = shutil.which('hmmscan')
    hmm_dir = Path(implementation.HMM_path)
    assets = sorted(p for p in hmm_dir.glob('*.hmm*') if p.is_file())
    if not binary or not assets:
        raise AssetError('ANARCI requires its explicit installed hmmscan and local HMM assets')
    defaults = {}
    for name, parameter in inspect.signature(numberer).parameters.items():
        if parameter.default is not inspect.Parameter.empty:
            value = parameter.default
            defaults[name] = sorted(value) if isinstance(value, set) else value
    identity = dict(package_source_sha256=digest(package.__file__),
        implementation_source_sha256=digest(inspect.getsourcefile(numberer)),
        hmmscan_sha256=digest(binary), hmm_sha256={p.name: digest(p) for p in assets},
        actual_defaults=defaults, scheme='imgt', chain_order=['heavy', 'light'],
        matching_domain='exactly one complete matching domain; ambiguity is an input error',
        layout=dict(segments=SEGMENTS, imgt_ranges=RANGES, widths=WIDTHS, length=298,
                    separator='X', right_padding='X', truncation=False))
    return numberer, identity


@contextmanager
def _numbering_scratch(directory):
    previous = tempfile.tempdir
    with tempfile.TemporaryDirectory(prefix='numbering_', dir=directory) as scratch:
        tempfile.tempdir = scratch
        try:
            yield
        finally:
            tempfile.tempdir = previous


def convert_pairs(records, numberer):
    """Actual chains -> ANARCI heavy then light -> strict native298, no fallback."""
    _validate_records(records)
    output = [dict(id=row['id'], ordinal=i, seq298=None, conversion_errors=[])
              for i, row in enumerate(records)]
    normalized = {}
    for chain in ('heavy', 'light'):
        legal = []
        for i, row in enumerate(records):
            sequence = row.get(chain)
            if not isinstance(sequence, str) or not sequence or not set(sequence) <= AA or len(sequence) < 60:
                output[i]['conversion_errors'].append(dict(stage='input', chain=chain,
                    code='missing_short_or_noncanonical_mutant_chain'))
            else:
                legal.append((str(i), sequence))
        if not legal:
            continue
        # Execution/resource errors are batch failures, not apparent low-scoring antibodies.
        numbering, details, _ = numberer(legal, scheme='imgt', output=False)
        if len(numbering) != len(legal) or len(details) != len(legal):
            raise MeasurementError('ANARCI returned an incomplete positional result list')
        for (key, _), domains, domain_details in zip(legal, numbering, details):
            if len(domains or []) != len(domain_details or []):
                raise MeasurementError('ANARCI domain/chain-identity cardinalities disagree')
            matches = []
            for domain, detail in zip(domains or [], domain_details or []):
                if str(detail.get('chain_type', '')).upper() not in ({'H'} if chain == 'heavy' else {'K', 'L'}):
                    continue
                segments = {name: ''.join(aa for position, aa in domain[0]
                    if low <= int(position[0]) <= high and aa != '-')
                    for name, (low, high) in zip(SEGMENTS, RANGES)}
                if all(segments.values()):
                    matches.append(segments)
            index = int(key)
            if len(matches) != 1:
                output[index]['conversion_errors'].append(dict(stage='numbering', chain=chain,
                    code='missing_or_multiple_complete_matching_domains'))
            else:
                normalized[(index, chain)] = matches[0]
    for i, row in enumerate(output):
        if row['conversion_errors']:
            continue
        try:
            row['seq298'] = build_layout298(normalized[(i, 'heavy')], normalized[(i, 'light')])
            row['positions_scored'] = native_positions(row['seq298'])
        except ValueError as exc:
            row['conversion_errors'].append(dict(stage='layout', code='invalid_native_slot', message=str(exc)))
    return output


def resource_error(error):
    message = str(error).lower()
    return isinstance(error, MemoryError) or 'outofmemory' in type(error).__name__.lower() or any(
        marker in message for marker in ('cuda out of memory', 'cuda error: out of memory',
                                        'cublas_status_alloc_failed', 'cannot allocate memory',
                                        "can't allocate memory"))


def _score_converted(records, converted, scorer, *, batch_size, metric_identity, source_identity, calibration):
    if (len(records) != len(converted) or any(row['id'] != conversion['id'] or conversion['ordinal'] != i
            for i, (row, conversion) in enumerate(zip(records, converted)))):
        raise MeasurementError('Conversion IDs/order differ from the original candidate records')
    valid = [row for row in converted if row['seq298'] is not None]
    sequences = [row['seq298'] for row in valid]
    results = scorer.score_many(sequences, batch_size=batch_size, strict=True,
                               return_per_position=True) if sequences else []
    if not isinstance(results, list) or len(results) != len(valid):
        raise MeasurementError('Native scorer returned a wrong result count')
    measured = {}
    for row, result in zip(valid, results):
        expected = native_positions(row['seq298'])
        if (not isinstance(result, dict) or result.get('seq_frcdr') != row['seq298']
                or result.get('positions_scored') != expected or result.get('n_positions') != len(expected)
                or result.get('sequence_length') != 298 or result.get('temperature') != 1.0
                or not _finite(result.get('average_log_likelihood'))
                or result.get('skipped_positions') != []):
            raise MeasurementError('Native score identity, coverage or numeric validation failed')
        per_position = result.get('per_position')
        if (not isinstance(per_position, list) or len(per_position) != len(expected)
                or any(not isinstance(item, dict) or item.get('position') != position
                       or item.get('aa') != row['seq298'][position] or not _finite(item.get('log_prob'))
                       for item, position in zip(per_position, expected))):
            raise MeasurementError('Native per-position evidence does not match the requested CDR sites')
        total = float(sum(float(item['log_prob']) for item in per_position))
        if result.get('log_likelihood') != total or result['average_log_likelihood'] != total/len(expected):
            raise MeasurementError('Native aggregate differs from its preserved per-position results')
        measured[row['ordinal']] = result
    output = []
    for i, (record, conversion) in enumerate(zip(records, converted)):
        result = measured.get(i)
        status = 'ok' if result is not None else 'conversion_error'
        score = float(result['average_log_likelihood']) if result is not None else None
        eligibility = None
        if calibration is not None:
            eligibility = 'missing' if score is None else 'below_reference_p5' if score < calibration['lower_cutoff'] else 'pass'
        output.append(dict(record, id=record['id'], ordinal=i, model=record['model'],
            group=record.get('group', record['model']), average_log_likelihood=score,
            status=status, error=conversion['conversion_errors'] or None,
            model_result=result, likelihood=dict(id=record['id'], score=score, status=status,
                heavy_sha256=hashlib.sha256(record['heavy'].encode()).hexdigest() if isinstance(record.get('heavy'), str) else None,
                light_sha256=hashlib.sha256(record['light'].encode()).hexdigest() if isinstance(record.get('light'), str) else None,
                native298_sha256=hashlib.sha256(conversion['seq298'].encode()).hexdigest() if conversion['seq298'] else None,
                metric_identity=metric_identity, source_identity=source_identity),
            p5_eligibility=eligibility, gc_selected=None))
    return output


def score_campaign_likelihood_pairs(records, output_dir, *, project_root, assets,
                                    device='cpu', batch_size=64, calibration=None):
    """Measure JM-Epi only; actual native loading is explicit, opt-in analysis.

    A failed inference attempt commits no scores.jsonl and propagates its error.
    No catch-all per-row fallback, no OOM->P5 rejection, no production hookup.
    Candidate scientific metadata is retained. Existing likelihood/average
    measurement fields are rejected even when null; reruns need fresh inputs.
    """
    records = deepcopy(list(records))
    _validate_records(records, campaign=True)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    output = new_output(project_root, output_dir)
    metadata = dict(schema='campaign-likelihood-analysis.v1', status='running',
        started_utc=datetime.now(timezone.utc).isoformat(), input_records=len(records),
        input_identity_sha256=identity_hash(records), scientific_clearance=False,
        production_connected=False, scores_committed=0, automatic_retry=False,
        error_is_p5_failure=False, cache_optimization=False)
    _dump(output/'metadata.json', metadata)
    stage = 'numbering'
    try:
        numberer, numbering_identity = _numbering_backend()
        with _numbering_scratch(output):
            converted = convert_pairs(records, numberer)
        _write_rows(output/'input_conversion.jsonl', converted)
        metadata['conversion_errors'] = sum(bool(r['conversion_errors']) for r in converted)
        stage = 'asset_loading'
        scorer, native_identity = load_native_scorer(project_root, assets, device=device)
        source_identity = dict(native=native_identity, numbering=numbering_identity,
            adapter_sha256=digest(__file__), loader_sha256=digest(Path(__file__).with_name('_likelihood_native.py')),
            score_contract=dict(batch_size=batch_size, strict=True, temperature=1.0,
                positions='native nonpadding CDR sites', normalization='AA20 log-softmax',
                reduction='original mean over native CDR sites', return_per_position=True,
                forward_groups='one antibody per native forward group; no cache optimization'))
        source_identity = json.loads(json.dumps(source_identity, allow_nan=False))
        metric_identity = 'sabdab298-native:' + identity_hash(source_identity)
        metadata.update(source_identity=source_identity, metric_identity=metric_identity)
        stage = 'calibration_validation'
        if calibration is not None:
            validate_campaign_calibration(calibration, metric_identity, source_identity)
        stage = 'scoring'
        rows = _score_converted(records, converted, scorer, batch_size=batch_size,
            metric_identity=metric_identity, source_identity=source_identity, calibration=calibration)
        stage = 'result_commit'
        # Validate full JSON before creating the final measurement artifact.
        json.dumps(rows, allow_nan=False)
        _write_rows(output/'scores.jsonl', rows)
        counts = Counter(input=len(rows), scored=sum(r['status'] == 'ok' for r in rows),
            conversion_errors=metadata['conversion_errors'])
        if calibration is not None:
            counts.update({key: sum(r['p5_eligibility'] == key for r in rows)
                           for key in ('pass', 'below_reference_p5', 'missing')})
        metadata.update(status='complete', scores_committed=len(rows),
            completed_utc=datetime.now(timezone.utc).isoformat(), counts=dict(counts),
            calibration=calibration, qc=score_qc([r['average_log_likelihood'] for r in rows]),
            output_sha256=digest(output/'scores.jsonl'),
            interpretation='Generator-related masked pseudo-loglikelihood, not measured affinity, specificity or an independent assessor')
        _dump(output/'metadata.json', metadata)
        return rows, metadata
    except BaseException as exc:
        metadata.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
            failed_stage=stage, error=dict(type=type(exc).__name__, message=str(exc),
                resource_error=resource_error(exc)), completed_utc=datetime.now(timezone.utc).isoformat())
        _dump(output/'metadata.json', metadata)
        raise


def score_abnativ_pairs(records, output_dir, models_dir, python_executable=sys.executable,
                        cpus=6, scratch_root=None, *, project_root, expected_checkpoint_sha256):
    """Official AbNatiV paired analysis for any model; no likelihood/P5 fields."""
    records = list(records)
    _validate_records(records)
    if type(cpus) is not int or cpus < 1:
        raise ValueError('cpus must be a positive integer')
    models = Path(models_dir)
    if not models.is_absolute():
        models = Path(project_root)/models
    checkpoint = verified_asset(project_root,
        dict(path=str(models/'vpaired2_model.ckpt'), sha256=expected_checkpoint_sha256), role='AbNatiV weights')
    models = checkpoint.parent
    output = new_output(project_root, output_dir)
    valid = [r for r in records if all(isinstance(r.get(c), str) and r[c] and set(r[c]) <= AA for c in ('heavy', 'light'))]
    with (output/'input.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['ID', 'vh_seq', 'vl_seq'])
        writer.writerows((r['id'], r['heavy'], r['light']) for r in valid)
    cache = output/'.cache'
    for name in ('matplotlib', 'tmp', 'numba'):
        (cache/name).mkdir(parents=True)
    scratch = Path(scratch_root or Path(project_root)/'.cache/tmp').resolve()
    if Path(project_root).resolve() not in scratch.parents or len(str(scratch).encode()) > 65:
        raise ValueError('Use a short temporary directory inside the explicit project')
    scratch.mkdir(parents=True, exist_ok=True)
    command = [str(python_executable), '-B', '-m', 'abnativ', 'paired_score', '-i', str(output/'input.csv'),
        '-cid', 'ID', '-cvh', 'vh_seq', '-cvl', 'vl_seq', '-odir', str(output/'raw'),
        '-oid', 'paired', '-mean', '-align', '-ncpu', str(cpus)]
    metadata = dict(status='running', metric=METRIC, scorer='AbNatiV-2 paired',
        checkpoint_sha256=expected_checkpoint_sha256, input_sha256=digest(output/'input.csv'),
        input_records=len(records), command=command, scientific_clearance=False,
        distinct_from_campaign_sabdab298_likelihood=True, selection_applied=False)
    _dump(output/'metadata.json', metadata)
    env = dict(os.environ)
    env.update(ABNATIV_MODELS_DIR=str(models), PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
        MPLCONFIGDIR=str(cache/'matplotlib'), NUMBA_CACHE_DIR=str(cache/'numba'),
        OMP_NUM_THREADS=str(cpus), OPENBLAS_NUM_THREADS=str(cpus))
    try:
        if not valid:
            raise ValueError('No valid mutated heavy/light pairs for AbNatiV')
        with tempfile.TemporaryDirectory(prefix='nat_', dir=scratch) as folder, (output/'scoring.log').open('x') as log:
            env['TMPDIR'] = folder
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=output)
        metadata['subprocess_returncode'] = process.returncode
        _dump(output/'metadata.json', metadata)
        if process.returncode:
            raise MeasurementError('AbNatiV process failed; no scores committed; inspect scoring.log')
        return finish_abnativ_pairs(records, output)
    except BaseException as exc:
        metadata.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                        error=dict(type=type(exc).__name__, message=str(exc)))
        _dump(output/'metadata.json', metadata)
        raise


def finish_abnativ_pairs(records, output_dir):
    _validate_records(records)
    output = Path(output_dir)
    metadata = json.loads((output/'metadata.json').read_text())
    if metadata.get('status') != 'running' or (output/'scores.jsonl').exists():
        raise ValueError('Postprocessing requires a running, not yet committed attempt')
    valid = {r['id']: r for r in records if all(isinstance(r.get(c), str) and r[c] and set(r[c]) <= AA for c in ('heavy', 'light'))}
    raw = output/'raw/paired_abnativ_seq_scores.csv'
    measured = {}
    with raw.open() as stream:
        for row in csv.DictReader(stream):
            key = row.get('ID') or row.get('seq_id')
            if key not in valid or key in measured:
                raise MeasurementError('Unexpected or duplicated AbNatiV output ID')
            if row.get('input_seq_vh') != valid[key]['heavy'] or row.get('input_seq_vl') != valid[key]['light']:
                raise MeasurementError('AbNatiV mutated-chain identity mismatch')
            measured[key] = row
    if metadata.get('subprocess_returncode') not in (None, 0):
        raise MeasurementError('Failed subprocess cannot be accepted as completed')
    if 'subprocess_returncode' not in metadata and set(measured) != set(valid):
        raise MeasurementError('Without a process receipt every expected valid ID must be present')
    rows = []
    for row in records:
        raw_score = measured.get(row['id'], {}).get(METRIC)
        score = float(raw_score) if _finite(raw_score) else None
        status = 'ok' if score is not None else 'invalid_input' if row['id'] not in valid else 'unmeasured_or_error'
        rows.append(dict(id=row['id'], group=row.get('group', row.get('model', 'all')),
                         abnativ_paired_score=score, status=status))
    _write_rows(output/'scores.jsonl', rows)
    metadata.update(status='complete', qc=score_qc([r['abnativ_paired_score'] for r in rows]),
        raw_score_sha256=digest(raw), output_sha256=digest(output/'scores.jsonl'),
        interpretation='Human-repertoire compatibility; not measured developability, binding or specificity')
    _dump(output/'metadata.json', metadata)
    return rows, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('campaign', 'paired'):
        command = commands.add_parser(name)
        command.add_argument('--project-root', type=Path, required=True)
        command.add_argument('--input', type=Path, required=True)
        command.add_argument('--output-dir', required=True, help='New project-relative directory')
    campaign = commands.choices['campaign']
    campaign.add_argument('--assets', type=Path, required=True)
    campaign.add_argument('--device', default='cpu')
    campaign.add_argument('--batch-size', type=int, default=64)
    campaign.add_argument('--calibration', type=Path)
    paired = commands.choices['paired']
    paired.add_argument('--models-dir', type=Path, required=True)
    paired.add_argument('--checkpoint-sha256', required=True)
    paired.add_argument('--python', default=sys.executable)
    paired.add_argument('--cpus', type=int, default=6)
    paired.add_argument('--scratch-root', type=Path)
    calibration = commands.add_parser('calibrate')
    calibration.add_argument('--reference', type=Path, required=True)
    calibration.add_argument('--column', required=True)
    calibration.add_argument('--metric-identity', required=True)
    calibration.add_argument('--source-identity', type=Path)
    calibration.add_argument('--lower-quantile', type=float)
    calibration.add_argument('--upper-quantile', type=float)
    calibration.add_argument('--independent-reference', action='store_true')
    calibration.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'campaign':
        rows, metadata = score_campaign_likelihood_pairs(read_pairs(args.input), args.output_dir,
            project_root=args.project_root, assets=json.loads(args.assets.read_text()),
            device=args.device, batch_size=args.batch_size,
            calibration=json.loads(args.calibration.read_text()) if args.calibration else None)
    elif args.command == 'paired':
        rows, metadata = score_abnativ_pairs(read_pairs(args.input), args.output_dir, args.models_dir,
            args.python, args.cpus, args.scratch_root, project_root=args.project_root,
            expected_checkpoint_sha256=args.checkpoint_sha256)
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        values = [r[args.column] for r in read_score_rows(args.reference)]
        metadata = calibrate(values, args.metric_identity, args.lower_quantile, args.upper_quantile,
            args.reference.name, source_identity=json.loads(args.source_identity.read_text()) if args.source_identity else None)
        metadata.update(reference_sha256=digest(args.reference), metric_column=args.column,
                        reference_not_candidates=args.independent_reference)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _dump(args.output, metadata)
    print(json.dumps(metadata, allow_nan=False))


if __name__ == '__main__':
    main()
