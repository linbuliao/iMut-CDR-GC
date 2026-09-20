# Software validation boundaries

This is an alpha software release, not a completed scientific data release.
The checks below use CPU-only fixtures and explicit mocks unless stated
otherwise. They are software checks, not evidence of biological superiority.

## Completed installation checks

The portable source built an `imut_cdr_gc-0.2.0a2` wheel offline. It installed in
a separate virtual environment, and the installed package, scoring, folding,
rebuilding and naturalness interfaces imported successfully. Both command-line
help entry points worked. The basic imports did not import Torch. The wheel
contained neither internal execution records nor learned weights. Every
packaged module and feature JSON matched the corresponding source bytes.
This installed-wheel check used Python 3.13.5, separately from the source-level
Python 3.10 check below. The package still declares Python >=3.11.

## Test status

The full collected suite exited successfully: **247 passed, four skipped, no
failures or errors** (251 collected). Source identities were unchanged during
the run. The four skipped checks require optional inputs or dependencies:

- One naturalness oracle requires an original research-source layout that is
  not distributed in this repository.
- Three graph/model checks require `torch_geometric`, which was absent from
  this test environment. These three checks are not claimed as validated here.

The suite covers antigen graphs, folding and reconstruction contracts,
generation CLI/model/proposal contracts, JM versus JM-Epi model identity,
scoring and stage-input identities, naturalness, selection, and pipeline
configuration and execution. Tiny model fixtures test strict state loading,
forward passes and rejection of incompatible inputs. They do not use learned
JM weights. The full ESM650M architecture is checked on a meta device only.

The completed suite used Python 3.13.5, NumPy 2.1.3, Torch 2.12.0,
Transformers 4.46.3 and pytest 8.3.4, on CPU. These identify the software-test
environment, not a validated replacement for each model's recorded inference
environment.

The documented no-model quickstart also completed: eight synthetic records
produced four eligible records, three unique eligible sequences and two
selected records. An independent checker verified counts, content hashes,
score labels and cutoffs. Its scores are invented test values, not results
from DeepCDR or a naturalness model.

## Reference-checkpoint CPU check

The exact JM-Epi checkpoint
`a19bd33544bbfcb8a18dfb88bb007439cd57e702c4d89963e143df79991c144e`
passed a separate source-level check under Python 3.10.12, Torch 2.6.0+cu124
and Transformers 4.46.3, using CPU float32 with no AMP or GPU. All 699 saved
keys loaded strictly after the exact saved-layout compatibility check. A fresh
independent encoder and original model definitions produced bit-identical
sequence logits, anchor and positive embeddings; maximum absolute differences
were all zero. Source and asset identities were unchanged.

This used one fixed 277-position founder, one ten-site query and a synthetic
four-node antigen graph. It generated no sequences and did not validate real
antigen preprocessing, sampling, mixed precision or an end-to-end library.
It is not a Python 3.10 wheel-installation test. Per-call model identity does
not automatically claim a new independent parity check for every later input.

## What these checks do not establish

Passing synthetic tests does not by itself establish real-checkpoint compatibility,
physical validity of a reconstructed complex, benchmark superiority, predictive
accuracy, or a completed 90,000-member delivery. The separate real-weight
check has the single-input scope described above. There is no partial state
loading or replacement of missing learned parameters. Native
asset validation limits for folding and rebuilding remain in their guides.
Historical JM learned-checkpoint inference, model training, and a full
production V3 iteration schedule have not been reproduced by this release.

See [scientific status](docs/status.md) for ongoing comparisons and paused MD,
and [license review](LICENSE_REVIEW.md) before redistribution.
