# Selecting an eligible library

`imut-cdr-gc select` first checks mutation burden, prior-library exclusions,
the reference P5 likelihood lower bound and the declared native DeepCDR route.
It then deduplicates exact H/L sequence pairs and selects from the remaining
pool. Only JM-Epi candidates enter this stage. V3 inputs must retain a passing
DeepCDR measurement and their sequence lineage at each joint proposal.

## Random sampling without replacement

Add these fields to a policy containing your existing eligibility calibration:

```json
{
  "selection_method": "random_without_replacement",
  "random_seed": "imut-cdr-gc-random2000-v1",
  "target": 2000
}
```

```bash
imut-cdr-gc select --input outputs/scored/scores.jsonl \
  --policy inputs/selection_policy.json --output-dir outputs/selected
```

Supply the **complete eligible sampling frame**, not a previously ranked
Top-10,000 subset. Eligible unique sequence pairs are ordered by
`SHA256(canonical JSON [seed, design/version, heavy, light])`; the first
`target` pairs form a reproducible pseudorandom sample. Canonical JSON uses
ASCII escaping, no spaces and UTF-8 encoding. A sequence-pair SHA256 breaks
hash-priority ties. The order is independent of passing score magnitude,
record ID and PDB availability. Duplicate provenance is selected by source
path and zero-based source row when supplied, then by record ID; it never
prefers higher scores. Without source pointers, record ID breaks that tie.

Use the same H/L sequence representation when replaying a sample. Adding
constant domains changes the hash input. The study's frozen lottery used
variable-domain H/L sequences. Its upstream pool assembly, provenance and
historical V3 exclusions remain necessary to reproduce the study membership;
the portable stage does not reconstruct those unavailable inputs.

Run each design/version separately. For cross-version deduplication, process
V1, then V2, then V3, passing earlier selected H/L pairs with `--excluded`.
V3's exclusion file should contain the union of selected V1 and V2 records
for that design. Do not exclude sequences selected for another design.

The summary records the seed, cohort, input and policy hashes, eligibility
losses, duplicate count, selected count and eligible-but-unsampled count.
`sampling_order` is **not a score rank**. `actual_boundary` and
`score_ranking_cutoff` are null; observed score ranges describe the sample,
not an extra cutoff. Too few eligible sequences produces
`insufficient_eligible`; a pipeline then stops before export.

## Score-ranked selection

`"selection_method": "score_top_k"` selects by active DeepCDR scores, followed
by likelihood and ID. This remains the default for existing configurations.
It reports an observed ranking boundary separately from eligibility cutoffs.
An OR route ranks by the larger active score, then the smaller; it does not
apply two independently chosen model thresholds.

Both selection modes preserve the existing terminal measurement contract:
all active scores must be measured. Missing scores are recorded as missing,
not zero. V3 intermediate OR gates can instead short-circuit on a measured
passing score. Neither sampling mode changes that lineage evidence or
verifies the final complex structures.
