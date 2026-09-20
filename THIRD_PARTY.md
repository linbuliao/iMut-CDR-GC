# Third-party code, assets and output terms

Dependencies, model weights and outputs may have separate terms. They are not
automatically covered by the eventual iMut-CDR-GC project license.

| Component | Source and included material | Review boundary |
| --- | --- | --- |
| iMut-CDR/iMut-CDR-Epi | [Author repository](https://github.com/linbuliao/iMut-CDR); project-derived model definitions and adapters | Preserve provenance; do not invent unpublished paper metadata |
| DeepCDR-ESM2 | [Author repository](https://github.com/haiping1010/DeepCDR_esm2); derived model/graph code and small residue-feature table | Redistribution terms remain to be confirmed; no weights bundled |
| DeepCDR-3D | Project-derived dual-graph implementation and small residue-feature table | Project/underlying material rights remain distinct; no weights bundled |
| ImmuneBuilder / ANARCI | Caller-supplied native sources, checkpoints, HMMs and executable | Not bundled; retain upstream terms and explicit asset identities |
| AbNatiV2 / ESM model assets | Caller-supplied local scorer/model resources | Not bundled; no implied permission or automatic download |
| AntiBERTa2 | [Author model repository](https://huggingface.co/alchemab/antiberta2) | Not bundled or required for this package; author's modified terms include research/non-commercial and generated-output restrictions. Consult the actual model license before use or redistribution |
| Other external comparison methods | Separately run research baselines | No pretrained baseline code, weights or generated results are included here |

The AntiBERTa2 provenance review inspected model revision
`3123350a3ee741c02a5dcf5efc771f3fb588ae77`. Its `LICENSE.md` SHA256 is
`04e114bb91592908e66a6645b211f449a07ea3e47c2f6e0467f934df9c0be75a`.
This identifier is not a replacement for the license text and does not turn
the author's modified terms into an unrestricted Apache grant.

Do not publish private model files or generated sequence datasets through this
repository merely because an internal research run was authorized. For future
data releases, review the originating model/data terms separately.
