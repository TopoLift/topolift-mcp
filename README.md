# topolift-mcp

MCP server exposing **TopoLift's atom-grounded negotiation reasoning** to any MCP-capable agent — Claude Code, Cursor, OpenAI Agents SDK, etc.

## What you get

Two tools:

- **`topolift_dialect`** — fetches TopoLift's published vocabulary (regimes, canonical strategies, signal keys, citation grammar). Call this once at session start; no auth required.
- **`topolift_negotiate`** — sends a negotiation context to TopoLift's reasoning engine and returns a **bilingual response**:
  - A typed `topology` slot (`regime`, `load_bearing_strategies[]`, `bridge_pivots[]`, `topology_signals{}`) drawn from a closed vocabulary
  - Prose fields with inline `[Cluster_X#strategy1,strategy2]` citation tokens anchoring every claim to the cluster + load-bearing atoms that drove it

Atoms stay on the TopoLift server. What travels is the *grammar* of the dialect — the vocabulary your agent uses to read structural reasoning.

## Install

```bash
pip install topolift-mcp
```

Set your API key (get one at <https://topolift.ai>):

```bash
export TOPOLIFT_API_KEY=tl-...
```

### Claude Code

```bash
claude mcp add topolift-negotiation -- topolift-mcp
```

…or, with the API key inline:

```bash
claude mcp add topolift-negotiation -e TOPOLIFT_API_KEY=tl-... -- topolift-mcp
```

### Cursor / Continue / other MCP clients

Add to your MCP config (`~/.cursor/mcp.json` or equivalent):

```json
{
  "mcpServers": {
    "topolift-negotiation": {
      "command": "topolift-mcp",
      "env": {
        "TOPOLIFT_API_KEY": "tl-..."
      }
    }
  }
}
```

### Running directly (no install)

```bash
TOPOLIFT_API_KEY=tl-... python -m topolift_mcp.server
```

## Configuration

Environment variables:

| Var | Default | Purpose |
|---|---|---|
| `TOPOLIFT_API_KEY` | *(required for negotiate)* | Bearer key; get one at <https://topolift.ai> |
| `TOPOLIFT_API_URL` | `https://api.topolift.ai` | API base URL |
| `TOPOLIFT_TIMEOUT` | `600` | Negotiate-call timeout in seconds |
| `TOPOLIFT_LOG_LEVEL` | `INFO` | Python logging level |

The dialect tool works without `TOPOLIFT_API_KEY` — only `topolift_negotiate` requires it.

## Pricing

- **Bearer key** (this MCP server's path): one-time / monthly plans starting at $50. See <https://topolift.ai>.
- **x402 micropayment**: agents can pay $0.10 USDC per call directly against `https://api.topolift.ai/v1/negotiate` with no API key — see the API's 402 challenge for details.

## How it works

1. Your agent calls `topolift_dialect` once and learns the vocabulary.
2. Your agent calls `topolift_negotiate` with a scenario.
3. TopoLift retrieves the most relevant atom clusters from its server-side topology, runs a Mistral-Small-4-119B reasoning pass with a closed-vocabulary prompt, and returns the bilingual response.
4. Your agent parses `topology` for machine-readable structure and reads the prose fields with citation-traceable evidence.

The atoms — the structural primitives — never leave TopoLift's servers. The dialect — the names and the grammar — is published openly so any agent can read structural reasoning fluently.

## License

MIT
