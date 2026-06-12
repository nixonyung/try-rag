# try-rag

## Getting Started

```fish
uv sync
uv run main.py # add --force to ignore existing `data/processed`
```

### Using `jq`

If you have [`jq`](https://github.com/jqlang/jq), you can run this to view NDJSON data files:

`jq --compact-output '{id, dense: .vector.dense}' data/processed/v1/ko-20251231/04-doc_get_points.ndjson`
