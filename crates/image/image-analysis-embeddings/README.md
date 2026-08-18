# image-analysis-embeddings

Concrete image and face embedding contracts and presets for `moritzbrantner-video-analysis`.

Model download and bundle handling belong to `moritzbrantner-model-runtime`; this crate owns
embedding values, embedder backend traits, and embedding catalog metadata.

## Runtime Surface

- Workflow operations: `image.embeddings.validate` validates imported image or
  face vectors and optional face regions.
- Debug operations: `image.embeddings.models`, `image.embeddings.schema`, and
  `describe` inspect catalogs, schemas, and package metadata.
- The surface does not download models or compute learned embeddings.
