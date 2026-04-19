# PMSR MCP Search Server

PMSR exposes read-only retrieval tools through the Model Context Protocol (MCP), so Codex and other MCP clients can query the same textual and multimodal knowledge bases used by the PMSR agent.

The tool descriptions follow the PMSR paper: PMSR performs joint search over heterogeneous textual and multimodal KBs; textual retrieval uses dense text-text semantic similarity; multimodal retrieval uses image-text pairs and combines image similarity with query-conditioned text similarity.

## Tools

### `text_search(query, top_k=5)`

Retrieve text passages from the textual Wikipedia KB using dense text-text semantic similarity.

Required `.env`:

```bash
TEXT_KB=/path/to/text.index
TEXT_METADATA=/path/to/text_metadata.jsonl
TEXT_EMBED_API_BASE=http://<host>:<port>
TEXT_MODEL=Qwen/Qwen3-Embedding-0.6B
```

### `image_search(image, query, top_k=5)`

Retrieve image-text pairs from the multimodal Wikipedia KB for an input image and query. `image` can be a local path, an HTTP(S) URL, or a data URL.

If `MLLM_KB`, `MLLM_METADATA`, and `MLLM_EMBED_API_BASE` are set, PMSR uses MLLM fusion and encodes the image and query jointly. Otherwise it uses concat fusion with separate image and text embedding APIs.

Required `.env`:

```bash
PMSR_KB=/path/to/pmsr.index
PMSR_METADATA=/path/to/pmsr_metadata.csv
IMAGE_EMBED_API_BASE=http://<host>:<port>
QWEN_TEXT_EMBED_API_BASE=http://<host>:<port>

# Optional MLLM fusion override
MLLM_KB=/path/to/mllm.index
MLLM_METADATA=/path/to/mllm_metadata.csv
MLLM_EMBED_API_BASE=http://<host>:<port>
MLLM_EMBED_MODEL=Qwen/Qwen3-VL-Embedding-2B
```

### `pmsr_multimodal_search(image, record_level_query, trajectory_level_query, top_k=5)`

Run PMSR dual-scope retrieval for external evidence gathering. A calling sub-agent should generate:

- `record_level_query`: a local query from the latest compact reasoning record.
- `trajectory_level_query`: a global query from the full reasoning trajectory.

The MCP tool searches both query scopes over text and multimodal KBs, then merges duplicates:

- `text_results`: text passages from the textual KB.
- `image_results`: image-text pairs from the multimodal KB.

`top_k` is capped at 20 for each tool call.

## Run Locally

```bash
python -m mcp_server.search_server
```

The default transport is `stdio`, which is the recommended local Codex setup because it avoids opening a network port and keeps FAISS indexes loaded in a long-lived process.

For debugging with MCP Inspector:

```bash
npx -y @modelcontextprotocol/inspector python -m mcp_server.search_server
```

## Codex MCP Configuration

Use this shape in the Codex MCP configuration:

```json
{
  "mcpServers": {
    "pmsr-search": {
      "command": "python",
      "args": ["-m", "mcp_server.search_server"],
      "cwd": "/drl_nas2/ckddls1321/dev/PMSR",
      "env": {
        "PYTHONPATH": "/drl_nas2/ckddls1321/dev/PMSR"
      }
    }
  }
}
```

The server reads `.env` from the working directory by default. Use `--env-file` for a different file:

```bash
python -m mcp_server.search_server --env-file /path/to/.env
```

Streamable HTTP is available for shared deployments:

```bash
python -m mcp_server.search_server \
  --transport streamable-http \
  --host 0.0.0.0 \
  --port 8765
```
