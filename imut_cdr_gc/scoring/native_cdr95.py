"""Strict actual-H/L native list-index95 successor; never reads generation cdr_seq."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import threading

from .common import AA, InputError, canonical_record, sequence_sha

PROTOCOL = 'deepcdr_esm2_hfirst_anarci_list_index95_v1'
CDR_RANGES = {'CDR1': (27, 38), 'CDR2': (56, 65), 'CDR3': (105, 117)}
WIDTHS = {'CDR1': 10, 'CDR2': 10, 'CDR3': 25}
_NUMBERING_TEMP_LOCK = threading.RLock()


def canonical_esm_input(record, *, project_root, numberer=None):
    row = canonical_record(record)
    pairs = [(role, row[role]) for role in ('heavy', 'light')]
    backend = {'kind': 'caller_supplied_numberer', 'scientific_equivalence_asserted': False}
    if numberer is None:
        import anarci as package
        from anarci import anarci
        numberer = anarci
        path = Path(package.__file__)
        backend = dict(kind='ANARCI', package_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       scheme='imgt', ncpu=1)
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise InputError('Explicit project_root must exist before numbering')
    temporary_root = (root / '.cache' / 'tmp').resolve()
    if root not in temporary_root.parents:
        raise InputError('Numbering temporary directory escapes project')
    temporary_root.mkdir(parents=True, exist_ok=True)
    # ANARCI uses Python tempfile internally. Confine only this serialized call,
    # restore the previous setting and remove the exact owned scratch directory.
    with _NUMBERING_TEMP_LOCK, tempfile.TemporaryDirectory(prefix='native_numbering_', dir=temporary_root) as scratch:
        old_temp = tempfile.tempdir
        try:
            tempfile.tempdir = scratch
            numbering, details, _ = numberer(pairs, scheme='imgt', output=False, ncpu=1, allow={'H', 'K', 'L'})
        finally:
            tempfile.tempdir = old_temp
    if len(numbering) != 2 or len(details) != 2:
        raise InputError('ANARCI did not return exactly the requested H/L chains')
    domains, windows, slots, truncated = {}, {}, [], []
    for index, (role, sequence) in enumerate(pairs):
        if not numbering[index] or len(numbering[index]) != 1 or not details[index] or len(details[index]) != 1:
            raise InputError('Missing or ambiguous ANARCI ' + role + ' domain')
        chain_type = str(details[index][0].get('chain_type', '')).upper()
        if chain_type not in (('H',) if role == 'heavy' else ('K', 'L')):
            raise InputError('ANARCI ' + role + ' role mismatch: ' + repr(chain_type))
        domain, start, end = numbering[index][0]
        if any(len(domain) <= stop for begin, stop in CDR_RANGES.values()):
            raise InputError('Truncated ANARCI numbered list for ' + role)
        if not isinstance(start, int) or isinstance(start, bool) or start < 0:
            raise InputError('Invalid ANARCI domain start')
        mapped, position = [], start
        for list_index, item in enumerate(domain):
            if item is None:
                continue
            label, aa = item
            if aa == '-':
                continue
            if aa not in AA or position >= len(sequence) or sequence[position] != aa:
                raise InputError('ANARCI domain does not map exactly to the actual mutant ' + role + ' sequence')
            mapped.append(dict(numbered_list_index_zero=list_index, chain_sequence_index_zero=position,
                               imgt_number=int(label[0]), imgt_insertion=str(label[1]).strip(), amino_acid=aa))
            position += 1
        domains[role] = dict(chain_type=chain_type, input_sha256=sequence_sha(sequence),
            numbered_list_length=len(domain), domain_start_zero=start, domain_end_reported=end,
            mapped_end_inclusive_zero=position - 1)
        for name, (begin, stop) in CDR_RANGES.items():
            selected = [s for s in mapped if begin <= s['numbered_list_index_zero'] <= stop]
            if not selected:
                raise InputError('Empty native ' + role + ' ' + name + ' window')
            windows[name.lower() + '_aa_' + role] = ''.join(s['amino_acid'] for s in selected)
            if slots:
                slots.append(dict(slot_zero=len(slots), kind='separator', amino_acid='X'))
            for offset in range(WIDTHS[name]):
                if offset < len(selected):
                    slots.append(dict(slot_zero=len(slots), kind='residue', role=role, cdr=name, **selected[offset]))
                else:
                    slots.append(dict(slot_zero=len(slots), kind='padding', role=role, cdr=name, amino_acid='X'))
            truncated.extend(dict(role=role, cdr=name, **s) for s in selected[WIDTHS[name]:])
    sequence = ''.join(s['amino_acid'] for s in slots)
    if len(sequence) != 95 or set(sequence) - set(AA + 'X'):
        raise InputError('Invalid native heavy-first list-index95 representation')
    return dict(id=row['id'], heavy_sha256=sequence_sha(row['heavy']), light_sha256=sequence_sha(row['light']),
        scorer_cdr_seq=sequence, scorer_cdr_sha256=sequence_sha(sequence), protocol=PROTOCOL,
        domains=domains, unpadded_cdrs=windows, slot_mapping=slots, width_truncated_residues=truncated,
        generation_cdr_seq_used=False, backend=backend,
        caveat='ANARCI list indices, not literal IMGT residue numbers or generator FR/CDR slices')
