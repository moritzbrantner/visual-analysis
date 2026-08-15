# Release checklist

This bootstrap does not authorize publication.

Before any future release:

1. Open an exact destination-local release issue and manifest.
2. Resolve `moenarch-audio-contracts =0.1.0` and `scenedetect-core =0.1.0`
   from the registry without path or Git fallbacks.
3. Generate a clean lockfile and run the verification policy authorized by that
   release issue.
4. Prove registry consumers separately from workspace source.
5. Keep npm/WASM publication separately authorized.

Source removal from rust-packages requires a later, independent issue.
