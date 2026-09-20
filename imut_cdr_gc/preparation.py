"""Lossless founder-chain reconstruction from explicit FR277 CDR mappings.

The model representation can truncate framework segments. Removing its X
characters is therefore never used to reconstruct the full antibody chains.
Only an unambiguous founder CDR-to-chain mapping permits a substitution.
"""
from __future__ import annotations

from copy import deepcopy

from .records import AA, sequence

WINDOWS = (("light", 26, 37), ("light", 55, 64), ("light", 102, 126),
           ("heavy", 165, 176), ("heavy", 194, 203), ("heavy", 241, 265))


def fr_sequence(value):
    if not isinstance(value, str) or len(value) != 277 or set(value) - (AA | {"X"}):
        raise ValueError("Expected the native 277-character standard-AA/X representation")
    return value


def prepare_founder(founder):
    """Validate an optional explicit zero-based position map or derive uniquely.

    Supply ``cdr_position_map`` for repeated substrings; every position, residue,
    window/chain and within-window ordering is independently checked.
    """
    result = deepcopy(founder)
    for field in ("id", "design", "antigen_id"):
        if not isinstance(result.get(field), str) or not result[field]:
            raise ValueError(f"Explicit {field} is required")
    heavy = sequence(result.get("heavy"), "founder heavy")
    light = sequence(result.get("light"), "founder light")
    fr = fr_sequence(result.get("fr_cdr_seq"))
    supplied = result.get("cdr_position_map")
    if supplied is not None and not isinstance(supplied, dict):
        raise ValueError("cdr_position_map must be a dictionary")
    mapping, used = {}, set()
    for chain, low, high in WINDOWS:
        source = heavy if chain == "heavy" else light
        positions = [p for p in range(low, high + 1) if fr[p] in AA]
        motif = "".join(fr[p] for p in positions)
        if not motif:
            raise ValueError("Every CDR window must contain real residues")
        if supplied is None:
            matches = [p for p in range(len(source) - len(motif) + 1)
                       if source[p:p + len(motif)] == motif]
            if len(matches) != 1:
                raise ValueError("Absent/ambiguous founder CDR motif; provide a verified position map")
            entries = [{"chain": chain, "index": matches[0] + offset}
                       for offset in range(len(positions))]
        else:
            entries = [supplied.get(str(p)) for p in positions]
        last = None
        for p, entry in zip(positions, entries):
            if not isinstance(entry, dict) or entry.get("chain") != chain:
                raise ValueError("CDR position has a missing or wrong chain mapping")
            index = entry.get("index")
            if type(index) is not int or not 0 <= index < len(source) or source[index] != fr[p]:
                raise ValueError("CDR mapping does not match the actual founder residue")
            if last is not None and index != last + 1:
                raise ValueError("Mapped CDR residues must be contiguous and ordered")
            if (chain, index) in used:
                raise ValueError("Two model positions map to the same chain residue")
            used.add((chain, index)); last = index
            mapping[str(p)] = {"chain": chain, "index": index}
    if supplied is not None and set(supplied) != set(mapping):
        raise ValueError("Unexpected or missing CDR position-map entries")
    result.update(cdr_position_map=mapping, representation="native_fr277_with_full_chain_mapping",
                  position_index_base=0, prepared=True)
    return result


def reconstruct(founder, mutant_fr):
    """Apply only validated CDR substitutions; preserve every framework residue."""
    founder = prepare_founder(founder)
    mutant_fr = fr_sequence(mutant_fr)
    chains = {name: list(founder[name]) for name in ("heavy", "light")}
    mutations = []
    for p, (old, new) in enumerate(zip(founder["fr_cdr_seq"], mutant_fr)):
        if old == new:
            continue
        entry = founder["cdr_position_map"].get(str(p))
        if entry is None or old not in AA or new not in AA:
            raise ValueError("Mutation changes framework, padding or an unmapped position")
        chain, index = entry["chain"], entry["index"]
        chains[chain][index] = new
        mutations.append({"fr_position": p, "chain": chain, "chain_index": index,
                          "from": old, "to": new})
    return {"heavy": "".join(chains["heavy"]), "light": "".join(chains["light"]),
            "mutations": mutations, "mutation_count": len(mutations),
            "position_index_base": 0}
