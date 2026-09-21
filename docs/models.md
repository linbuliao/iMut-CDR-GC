# Trained models

Download the evaluated models from
[linbuliao/iMut-CDR-GC on Hugging Face](https://huggingface.co/linbuliao/iMut-CDR-GC).
All three models are public; downloading does not require an account.

| Model folder | Weight size | Use |
| --- | --- | --- |
| `iMut-CDR-JM/` | 2.85 GB | Joint CDR residue proposals |
| `iMut-CDR-JM-Epi/` | 3.00 GB | Antigen-conditioned joint proposals |
| `DeepCDR-3D/` | 24.3 MB | Structure-based interaction scoring |

The immutable release revision is
`a5b8d7fa591778d4f5132b1ca6c35c850b6b6deb`.
Each folder contains a model card and `release_manifest.json` with file sizes,
SHA256 hashes and the compatible source revision. JM and JM-Epi include their
fine-tuned ESM2 encoder; no separate ESM2 weights are needed.

## Download

Run from your working project's root:

```bash
python -m pip install huggingface_hub
HF_HOME=.cache/huggingface hf download linbuliao/iMut-CDR-GC \
  --revision a5b8d7fa591778d4f5132b1ca6c35c850b6b6deb \
  --include 'iMut-CDR-JM-Epi/*' \
  --local-dir assets/imut_cdr_gc
```

Replace the include pattern with `iMut-CDR-JM/*` or `DeepCDR-3D/*` for those
components, or omit `--include` to download the full release (about 5.87 GB).
Keep the downloaded manifests with the weights.

## Configure inference

Copy the appropriate [configuration template](../examples/configs/README.md)
to your project and fill its asset entries from the downloaded manifest:

- JM/JM-Epi: set `generator.model_dir` to the model folder's
  `esm2_config_and_tokenizer/`; set `generator.checkpoint.path` to its `model.pt`.
  Use the manifest's checkpoint hash and the four tokenizer/configuration
  hashes in `generator.asset_sha256`.
- DeepCDR-3D: set `assets.3d.weights` and `assets.3d.features` to its `model.pt`
  and `amino_acid_vectors_30dim.json`, with their manifest hashes.
- Provide your own sequence inputs and antigen graph or pocket as described
  in the [generation](generation.md), [graph](antigen_graph.md) and
  [scoring](scoring.md) guides. All paths remain relative to the project root.

For example, a JM-Epi checkpoint entry is:

```json
{
  "path": "assets/imut_cdr_gc/iMut-CDR-JM-Epi/model.pt",
  "sha256": "a19bd33544bbfcb8a18dfb88bb007439cd57e702c4d89963e143df79991c144e"
}
```

The model cards also contain direct Python loading examples. These are custom
PyTorch networks, not Transformers `AutoModel` repositories. The inference
interfaces verify local asset hashes before loading.

DeepCDR-ESM2 is maintained in its
[original repository](https://github.com/haiping1010/DeepCDR_esm2).
The separate reference-likelihood model used for P5 calibration is not part of
this release; its asset requirements are documented in [naturalness](naturalness.md).
