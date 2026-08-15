# image-analysis-classification

Concrete image classification contracts and presets for `moritzbrantner-video-analysis`.

Model download and bundle handling belong to `moritzbrantner-model-runtime`; this crate owns
classification request/response types, classifier backend traits, and
classification catalog metadata.

## Runtime Surface

- Primary workflow: `image.classification.classify` is the server-only local
  model path for the default `Xenova/vit-base-patch16-224` ONNX preset. It
  declares filesystem reads, filesystem writes, and network access because
  first run may materialize the bundle under `.model-runtime`.
- Workflow operations: `image.classification.imported` validates caller-supplied
  labels and scores, `image.classification.topLabels` ranks normalized labels,
  and `image.classification.thresholdLabels` splits labels by minimum score.
- Debug operations: `image.classification.models`,
  `image.classification.schema`, and `describe` inspect catalogs, schemas, and
  package metadata.
- WASM keeps imported/ranking/threshold workflows. Native ONNX execution lives
  in this crate with low-level session execution delegated to `moritzbrantner-runtime-onnx`.
