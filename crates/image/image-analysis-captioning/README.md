# image-analysis-captioning

Concrete image captioning contracts and presets for `moritzbrantner-video-analysis`.

Model download and bundle handling belong to `moritzbrantner-model-runtime`; this crate owns
caption outputs, captioner backend traits, and captioning catalog metadata.

## Runtime Surface

- Primary workflow: `image.captioning.caption` is the server-only local model
  path for the default `Xenova/vit-gpt2-image-captioning` ONNX preset. It
  declares filesystem reads, filesystem writes, and network access because
  first run may materialize the bundle under `.model-runtime`.
- Workflow operations: `image.captioning.imported` validates caller-supplied
  captions and scores, `image.captioning.rankCaptions` ranks normalized
  captions, and `image.captioning.captionReport` summarizes caption-set
  quality fields.
- Debug operations: `image.captioning.models`, `image.captioning.schema`, and
  `describe` inspect catalogs, schemas, and package metadata.
- `Salesforce/blip-image-captioning-base` remains catalog metadata and is
  reference-only until a BLIP-specific generator is implemented. WASM keeps
  imported/ranking/report workflows.
