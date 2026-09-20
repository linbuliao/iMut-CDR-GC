"""JM/JM-Epi joint proposals with explicit assets and truthful mutation accounting.

This module never imports a trainer, selects a GPU, downloads assets or performs
DeepCDR/likelihood screening. FR277 proposals require an independently verified
full-chain reconstruction before sequence/PDB delivery.
"""
from __future__ import annotations
from contextlib import nullcontext
import hashlib
import math

AA20 = 'ACDEFGHIKLMNPQRSTVWY'
MAX_REQUESTED_SITES = 10
FR_LENGTH = 277
CDR_SITES = frozenset(p for start, stop in
    ((26, 37), (55, 64), (102, 126), (165, 176), (194, 203), (241, 265))
    for p in range(start, stop + 1))
PROPOSAL_REPRESENTATION = 'fr_cdr_proposal_requires_full_chain_reconstruction'
MODEL_NAMES = {'jm': 'iMut-CDR-JM', 'jm-epi': 'iMut-CDR-JM-Epi'}


def _validate_record(record):
    if not isinstance(record, dict) or not isinstance(record.get('id'), str) or not record['id']:
        raise ValueError('A nonempty string id is required')
    for key in ('fr_cdr_seq', 'founder_fr_cdr_seq'):
        sequence = record.get(key)
        if not isinstance(sequence, str) or len(sequence) != FR_LENGTH or set(sequence) - set(AA20 + 'X'):
            raise ValueError('Supply explicit, lossless FR277 parent and founder sequences')
    sequence, founder = record['fr_cdr_seq'], record['founder_fr_cdr_seq']
    if any((a == 'X') != (b == 'X') for a, b in zip(sequence, founder)):
        raise ValueError('Founder/parent padding layout differs')
    if any(i not in CDR_SITES for i, (a, b) in enumerate(zip(sequence, founder)) if a != b):
        raise ValueError('Parent contains a framework mutation outside the explicit CDR layout')
    positions = record.get('proposal_positions')
    if not isinstance(positions, (list, tuple)) or not 1 <= len(positions) <= MAX_REQUESTED_SITES:
        raise ValueError('Each joint proposal must request 1 to 10 sites; no silent truncation')
    if any(type(p) is not int for p in positions) or len(set(positions)) != len(positions):
        raise ValueError('Requested sites must be unique integer indices')
    if any(p not in CDR_SITES or sequence[p] not in AA20 for p in positions):
        raise ValueError('Requested site is padding, a framework site or outside FR277')
    if 'mutable_positions' in record and not set(positions) <= set(record['mutable_positions']):
        raise ValueError('Requested site is outside caller-declared mutable positions')
    return list(positions)


def _validate_graph(graph, node_dim):
    import torch
    if not isinstance(graph, dict) or any(k not in graph for k in ('x', 'edge_index', 'edge_attr')):
        raise ValueError('Supply an explicit native antigen graph for each record')
    x, edges, attr = (graph[k] for k in ('x', 'edge_index', 'edge_attr'))
    if not all(torch.is_tensor(t) for t in (x, edges, attr)):
        raise ValueError('Graph fields must be tensors')
    if x.dtype != torch.float32 or x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != node_dim or not torch.isfinite(x).all():
        raise ValueError('Invalid, empty or nonfinite antigen node features')
    if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError('Graph edges must be a 2-by-E integer tensor')
    if edges.numel() and (edges.min() < 0 or edges.max() >= x.shape[0]):
        raise ValueError('Antigen graph edge index is outside the node array')
    if attr.dtype != torch.float32 or attr.shape != (edges.shape[1], 2) or not torch.isfinite(attr).all():
        raise ValueError('Native edge attributes must contain distance and sequence-edge flag')
    if attr.numel() and (attr[:, 0] < 0).any():
        raise ValueError('Graph distances must be nonnegative')


def _seed_for(record, seed):
    value = record.get('sampling_seed')
    if value is None:
        value = int.from_bytes(hashlib.sha256((str(seed) + ':' + record['id']).encode()).digest()[:8], 'big') % (2**63)
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError('sampling_seed must be an integer in [0, 2**63)')
    return value


def sample_aa20(logits, *, parent, aa_ids, temperature, top_k, penalty, generator):
    """Native V3 ordering: AA20 CPU-float logits / T, parental penalty, top-k.

    The parental amino acid is not forbidden. A requested site can stay exactly
    as it was, even with a soft nonzero penalty. Zero temperature is an explicit
    greedy option, not the recorded production sampling setting.
    """
    import torch
    values = logits[aa_ids].float().cpu()
    if not torch.isfinite(values).all():
        raise ValueError('Nonfinite model logits; no zero-score fallback')
    if temperature > 0:
        values = values / temperature
    values = values.clone()
    values[AA20.index(parent)] -= penalty
    if temperature == 0:
        return AA20[int(values.argmax())]
    k = min(top_k or len(AA20), len(AA20))
    selected, indices = values.topk(k)
    draw = int(torch.multinomial(selected.softmax(-1), 1, generator=generator))
    return AA20[int(indices[draw])]


class JointMutationGenerator:
    def __init__(self, model, *, temperature=1.0, top_k=0,
                 parental_residue_logit_penalty=0.0, seed=0, precision='float32', provenance=None,
                 model_kind='jm-epi'):
        import torch
        if model_kind not in MODEL_NAMES:
            raise ValueError('model_kind must be jm or jm-epi')
        if (model_kind == 'jm-epi') != hasattr(model, 'epi_config'):
            raise ValueError('The neural model architecture does not match model_kind')
        for key, expected in (('model_kind', model_kind), ('model', MODEL_NAMES[model_kind]),
                              ('antigen_conditioned', model_kind == 'jm-epi')):
            if provenance is not None and key in provenance and provenance[key] != expected:
                raise ValueError('Loaded model identity conflicts with model_kind: ' + key)
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError('temperature must be finite and nonnegative')
        if type(top_k) is not int or not 0 <= top_k <= 20:
            raise ValueError('top_k must be an integer from 0 to 20')
        if not math.isfinite(parental_residue_logit_penalty):
            raise ValueError('Nonfinite parental residue penalty')
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('seed must be an integer in [0, 2**63)')
        if precision not in ('float32', 'float16', 'bfloat16'):
            raise ValueError('Explicit precision must be float32, float16 or bfloat16')
        self.model = model.eval()
        self.model_kind = model_kind
        self.tokenizer = model.tokenizer
        self.device = next(model.parameters()).device
        if precision == 'float16' and self.device.type != 'cuda':
            raise ValueError('This interface only enables float16 autocast on an explicit CUDA device')
        self.precision, self.temperature, self.top_k = precision, float(temperature), top_k
        self.penalty, self.seed = float(parental_residue_logit_penalty), seed
        ids = [self.tokenizer.convert_tokens_to_ids(a) for a in AA20]
        if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != 20:
            raise ValueError('Tokenizer lacks twenty distinct canonical amino-acid tokens')
        self.aa_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        self.provenance = dict(provenance or {}, precision=precision,
            model_kind=model_kind, model=MODEL_NAMES[model_kind],
            antigen_conditioned=model_kind == 'jm-epi',
            sampling='AA20 float32 CPU; divide by temperature then subtract parental logit penalty; top-k; per-record CPU RNG',
            max_requested_sites=10, unchanged_residues_allowed=True,
            full_chain_reconstruction_verified=False, screening_applied=False)

    def propose(self, records, *, graphs=None):
        import torch
        records = list(records)
        if self.model_kind == 'jm':
            if graphs is not None:
                raise ValueError('Unconditioned JM does not accept antigen graphs')
        else:
            if graphs is None:
                raise ValueError('One antigen graph per proposal is required')
            graphs = list(graphs)
            if len(records) != len(graphs):
                raise ValueError('One antigen graph per proposal is required')
        if not records:
            return []
        masks = [_validate_record(r) for r in records]
        if len({r['id'] for r in records}) != len(records):
            raise ValueError('Duplicate proposal IDs')
        seeds = [_seed_for(r, self.seed) for r in records]
        if graphs is not None:
            for graph in graphs:
                _validate_graph(graph, self.model.epi_config.node_dim)
        sequences = [r['fr_cdr_seq'] for r in records]
        maximum = getattr(self.model, 'model_max_len', None)
        if maximum is not None and maximum < FR_LENGTH:
            raise ValueError('Encoder cannot hold FR277 without truncation')
        tokens = self.tokenizer(sequences, return_tensors='pt', padding=True,
                                truncation=False, add_special_tokens=False)
        ids, mask = tokens.input_ids.to(self.device), tokens.attention_mask.to(self.device)
        if ids.shape != (len(records), FR_LENGTH) or mask.shape != ids.shape or not torch.all(mask == 1):
            raise ValueError('Tokenizer did not preserve exactly one token per FR277 position')
        for i, sequence in enumerate(sequences):
            expected = torch.tensor([self.tokenizer.convert_tokens_to_ids(a) for a in sequence], device=self.device)
            if not torch.equal(ids[i], expected):
                raise ValueError('Token positions no longer correspond to the explicit FR277 sequence')
        context = (nullcontext() if self.precision == 'float32' else
            torch.autocast(self.device.type, dtype=getattr(torch, self.precision)))
        with torch.inference_mode(), context:
            if self.model_kind == 'jm':
                logits, _, _ = self.model(ids, mask, masks)
            else:
                logits, _, _ = self.model(ids, mask, masks, graphs)
        if logits.shape[:2] != ids.shape:
            raise ValueError('Model output position shape mismatch')
        outputs = []
        for i, (record, positions, seed) in enumerate(zip(records, masks, seeds)):
            parent, founder = record['fr_cdr_seq'], record['founder_fr_cdr_seq']
            chars = list(parent)
            rng = torch.Generator(device='cpu').manual_seed(seed)
            for p in positions:
                chars[p] = sample_aa20(logits[i, p], parent=parent[p], aa_ids=self.aa_ids,
                    temperature=self.temperature, top_k=self.top_k, penalty=self.penalty, generator=rng)
            sequence = ''.join(chars)
            changed = [p for p in positions if parent[p] != sequence[p]]
            cumulative = [p for p, (a, b) in enumerate(zip(founder, sequence)) if a != b]
            event = dict(iteration=len(record.get('iteration_trace', [])) + 1,
                parent_id=record.get('parent_id', record['id']), proposal_id=record['id'],
                sampling_seed=seed, requested_sites=positions, changed_sites=changed,
                requested_count=len(positions), changed_count=len(changed),
                cumulative_mutation_count=len(cumulative),
                mutations=[dict(position=p, from_aa=parent[p], to_aa=sequence[p]) for p in changed])
            trace = [*record.get('iteration_trace', []), event]
            # Do not carry parent heavy/light or old scoring fields forward as
            # if they represented the newly sampled FR sequence.
            output = {k:record[k] for k in ('design', 'version', 'antigen_id', 'founder_heavy',
                       'founder_light', 'mutable_positions') if k in record}
            output.update(id=record['id'], model=MODEL_NAMES[self.model_kind],
                antigen_conditioned=self.model_kind == 'jm-epi',
                representation=PROPOSAL_REPRESENTATION, parent_fr_cdr_seq=parent,
                founder_fr_cdr_seq=founder, fr_cdr_seq=sequence,
                requested_sites=positions, requested_count=len(positions),
                changed_sites=changed, changed_count=len(changed),
                actual_mutation_positions=cumulative, cumulative_mutation_count=len(cumulative),
                total_requested_count=sum(step['requested_count'] for step in trace),
                total_changed_count=sum(step['changed_count'] for step in trace), iteration_trace=trace,
                full_chain_reconstruction_required=True, screening_applied=False)
            outputs.append(output)
        return outputs

    def iterate(self, record, *, position_rounds, graph=None):
        """Unscreened proposal accounting helper, NOT a production V3 controller.

        V3 callers must instead gate each propose() result before it becomes a
        parent. Merely calling this helper does not implement DeepCDR selection.
        """
        position_rounds = list(position_rounds)
        if not position_rounds:
            raise ValueError('At least one explicit proposal round is required')
        result = dict(record)
        initial_id = result['id']
        for step, positions in enumerate(position_rounds, 1):
            previous_id = result['id']
            result = dict(result, id=f'{initial_id}.step{step}', parent_id=previous_id,
                          proposal_positions=list(positions))
            result.pop('sampling_seed', None)
            result, = self.propose([result], graphs=None if graph is None else [graph])
        return result


def load_local_generator(*, model_dir, checkpoint, expected_checkpoint_sha256, asset_sha256,
                         model_config=None, device='cpu', precision='float32', temperature=1.0,
                         top_k=0, parental_residue_logit_penalty=0.0, seed=0, model_kind='jm-epi'):
    from .models.loading import load_local_model
    model, identity = load_local_model(model_dir=model_dir, checkpoint=checkpoint,
        expected_checkpoint_sha256=expected_checkpoint_sha256, asset_sha256=asset_sha256,
        model_config=model_config, device=device, model_kind=model_kind)
    return JointMutationGenerator(model, temperature=temperature, top_k=top_k,
        parental_residue_logit_penalty=parental_residue_logit_penalty, seed=seed,
        precision=precision, provenance=identity, model_kind=model_kind)


def load_antigen_graph(path, expected_sha256, *, node_dim=30):
    from .models.antigen import load_antigen_graph as load
    return load(path, expected_sha256, node_dim=node_dim)
