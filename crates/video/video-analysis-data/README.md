# video-analysis-data

Normalized stream records and online aggregation for `moritzbrantner-video-analysis`.

## Feature flags

- No optional feature flags today.

## Example

```rust,ignore
use video_analysis_data::{DataBucketOptions, StreamAggregator};

let mut aggregator = StreamAggregator::new(DataBucketOptions::default())?;
let _ = aggregator.finish();
```

## Related crates

- `video-analysis-core`
- `video-analysis-dataset`
- `dense-data`

## Package surface

Workflow operations:

- `video.data.recordSummary`
- `video.data.eventTimeline`

Debug operations:

- `describe`
- `video.data.joinPlan`

Runtime limits:

Operations are deterministic, local-first, and side-effect free. They return inline JSON reports and do not download models, write files, or run native tools.

Invalid input returns a clear error through `run_surface_operation`; successful
responses include `operation`, `title`, `message`, `summary`, and `result` while
keeping existing top-level domain fields for compatibility.
