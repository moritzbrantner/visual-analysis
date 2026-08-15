# video-analysis-tracking-server

Thin HTTP API adapter for `moritzbrantner-video-analysis-tracking`.

Run:

```bash
cargo run -p video-analysis-tracking-server -- --addr 127.0.0.1:3000
```

Endpoints:

- `GET /health`
- `GET /api/package`
- `GET /api/schema`
- `GET /api/operations`
- `POST /api/run`
- `POST /api/<operation-id>`
