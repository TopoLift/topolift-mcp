"""
TopoLift MCP Server — stdio transport.

Exposes TopoLift's atom-grounded negotiation reasoning to any MCP-capable agent
(Claude Code, Cursor, OpenAI Agents SDK, etc) as two tools:

  - `topolift_dialect`   — fetch the canonical vocabulary (regimes, strategies,
                           signal keys, citation grammar) so the agent learns
                           the dialect it is about to read.
  - `topolift_negotiate` — get atom-grounded structural reasoning about a
                           specific negotiation context. Returns the bilingual
                           response: typed `topology` slot + prose with inline
                           [Cluster_X#strat1,strat2] citations.

Atoms stay on the TopoLift server — this MCP layer is a thin client that
forwards a request, returns the response, and lets the calling agent reason
against the published dialect.

Configuration (env vars):
  TOPOLIFT_API_URL    — base URL for the API. Default: https://api.topolift.ai
  TOPOLIFT_API_KEY    — Bearer key for /v1/negotiate. Required for negotiation
                        calls (humans pay via Stripe; autonomous agents pay via
                        x402 directly to the API and don't need this layer).
  TOPOLIFT_TIMEOUT    — seconds to wait for /v1/negotiate. Default 600.
"""

from __future__ import annotations

import contextvars
import logging
import os
import sys
from typing import Any, Optional

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

logger = logging.getLogger("topolift.mcp")

API_URL = os.environ.get("TOPOLIFT_API_URL", "https://api.topolift.ai").rstrip("/")
API_KEY = os.environ.get("TOPOLIFT_API_KEY", "").strip()
NEGOTIATE_TIMEOUT = float(os.environ.get("TOPOLIFT_TIMEOUT", "600"))

# Per-request auth — only used in Streamable HTTP mode where each MCP request
# carries its own Authorization header. The ASGI middleware below captures it
# into this ContextVar so the tools can pick it up. In stdio mode this stays
# empty and tools fall back to TOPOLIFT_API_KEY env var.
_request_auth: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "topolift_request_auth", default=None
)


def _auth_header() -> Optional[str]:
    """Return the Authorization header value for the outbound API call.

    Order: per-request context (HTTP mode) → TOPOLIFT_API_KEY env var (stdio mode).
    """
    val = _request_auth.get()
    if val:
        return val
    if API_KEY:
        return f"Bearer {API_KEY}"
    return None

# DNS rebinding protection: FastMCP auto-enables a localhost-only allow-list
# when host=127.0.0.1, which breaks deployments behind a TLS-terminating
# reverse proxy on a public hostname. We allow the hosted public domain plus
# the loopback addresses (for direct localhost testing). Additional public
# hosts can be passed in via TOPOLIFT_MCP_ALLOWED_HOSTS (comma-separated).
_extra_hosts = [
    h.strip() for h in os.environ.get(
        "TOPOLIFT_MCP_ALLOWED_HOSTS", "mcp.topolift.ai"
    ).split(",") if h.strip()
]
_transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=[
        "127.0.0.1:*", "localhost:*", "[::1]:*",
        *_extra_hosts,
    ],
    allowed_origins=[
        "http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*",
        *(f"https://{h}" for h in _extra_hosts),
    ],
)

mcp = FastMCP(
    "topolift-negotiation",
    instructions=(
        "TopoLift exposes atom-grounded negotiation reasoning. Before your first "
        "negotiate call, fetch the dialect with `topolift_dialect` so you can read "
        "the typed `topology` slot and inline [Cluster_X#strat] citations in the "
        "negotiate response with full structural fluency. Atoms stay server-side; "
        "what you receive is the structural reasoning that the topology produced."
    ),
    transport_security=_transport_security,
)


# ────────────────────────────────────────────────────────────────────────────
# Tool: topolift_dialect
# ────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def topolift_dialect() -> dict[str, Any]:
    """Fetch TopoLift's negotiation dialect — the closed vocabulary of regimes,
    canonical strategies, topology signal keys, metric roles, and the inline
    citation format used in negotiate responses.

    Call this once at the start of a session to learn what values to expect in
    the typed `topology` field of negotiate responses, and how to parse the
    `[Cluster_X#strategies]` citation tokens that anchor each prose claim.

    No authentication required.

    Returns:
        A dict with keys: version, description, regimes (list of cluster→regime
        mappings), canonical_regimes, canonical_strategies, topology_signal_keys,
        metric_roles, citation, response_fields.
    """
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(f"{API_URL}/v1/dialect")
        resp.raise_for_status()
        return resp.json()


# ────────────────────────────────────────────────────────────────────────────
# Tool: topolift_negotiate
# ────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def topolift_negotiate(
    scenario: str,
    goals: str,
    reservation_price: Optional[float] = None,
    aspiration_price: Optional[float] = None,
    constraints: Optional[str] = None,
    relationship_priority: str = "short_term",
    phase: str = "unknown",
    urgency: str = "medium",
    counterparty_known_preferences: Optional[str] = None,
    counterparty_known_constraints: Optional[str] = None,
    counterparty_behavior_signals: Optional[str] = None,
    counterparty_is_ai_agent: Optional[bool] = None,
    current_offer_on_table: Optional[str] = None,
    conversation_history: Optional[str] = None,
    question: Optional[str] = None,
) -> dict[str, Any]:
    """Get atom-grounded structural reasoning for a negotiation scenario.

    The response is BILINGUAL:
      - `topology` (object) is the typed dialect view: regime,
        load_bearing_strategies[], bridge_pivots[], topology_signals{}.
        Every value is drawn from the closed vocabulary — fetch the dialect
        with `topolift_dialect` to learn the canonical names.
      - All prose fields (situation_read, primary_recommendation.rationale,
        fallback_recommendation.rationale, counterparty_read) end every claim
        with an inline citation token of the form
        `[Cluster_X#strategy1,strategy2]` that anchors the claim to the cluster
        + load-bearing atoms that drove it.

    Args:
        scenario: What is being negotiated, what has happened so far.
        goals: What the principal wants to achieve.
        reservation_price: Absolute limit — never exceed.
        aspiration_price: Ideal outcome to push toward.
        constraints: Time pressure, budget limits, dependencies, etc.
        relationship_priority: one_shot | short_term | long_term |
            strategic_partnership.
        phase: opening | mid_game | closing | post_deal | unknown.
        urgency: low | medium | high | critical.
        counterparty_known_preferences: Any revealed preferences, priorities,
            or stated positions.
        counterparty_known_constraints: Any known time pressure, budget limits,
            or dependencies.
        counterparty_behavior_signals: Observed behavior — aggressive,
            cooperative, evasive, etc.
        counterparty_is_ai_agent: Is the counterparty an AI agent?
        current_offer_on_table: The most recent offer or proposal.
        conversation_history: Recent conversation turns (last 5–10 exchanges).
        question: Specific question to answer (e.g. "Should I accept this?").

    Returns:
        The full NegotiateResponse: situation_read, phase_assessment,
        zopa_assessment, primary_recommendation, fallback_recommendation,
        risk_assessment, counterparty_read, answer, topology,
        topological_reasoning, cited_clusters, reasoning_depth.

    Requires the TOPOLIFT_API_KEY env var to be set with a valid TopoLift
    Bearer key. Get one at https://topolift.ai or pay per call via x402
    directly against the API.
    """
    auth = _auth_header()
    if not auth:
        raise RuntimeError(
            "No authentication. In stdio mode, set TOPOLIFT_API_KEY env var. "
            "In HTTP mode (mcp.topolift.ai), pass an Authorization: Bearer "
            "tl-... header on the MCP request. Get a key at https://topolift.ai "
            "or call /v1/negotiate directly with an x402 payment."
        )

    body: dict[str, Any] = {
        "scenario": scenario,
        "phase": phase,
        "urgency": urgency,
        "principal": {
            "goals": goals,
            "reservation_price": reservation_price,
            "aspiration_price": aspiration_price,
            "relationship_priority": relationship_priority,
            "constraints": constraints,
        },
    }
    if any([
        counterparty_known_preferences,
        counterparty_known_constraints,
        counterparty_behavior_signals,
        counterparty_is_ai_agent is not None,
    ]):
        body["counterparty"] = {
            "known_preferences": counterparty_known_preferences,
            "known_constraints": counterparty_known_constraints,
            "behavior_signals": counterparty_behavior_signals,
            "is_ai_agent": counterparty_is_ai_agent,
        }
    if current_offer_on_table:
        body["current_offer_on_table"] = current_offer_on_table
    if conversation_history:
        body["conversation_history"] = conversation_history
    if question:
        body["question"] = question

    with httpx.Client(timeout=NEGOTIATE_TIMEOUT) as client:
        resp = client.post(
            f"{API_URL}/v1/negotiate",
            headers={
                "Authorization": auth,
                "Content-Type": "application/json",
            },
            json=body,
        )
        if resp.status_code == 401:
            raise RuntimeError(
                "TopoLift rejected the API key (HTTP 401). Check TOPOLIFT_API_KEY."
            )
        if resp.status_code == 402:
            raise RuntimeError(
                "TopoLift requires payment (HTTP 402). The Bearer key was missing "
                "or the server is configured for x402 — get an API key at "
                "https://topolift.ai."
            )
        resp.raise_for_status()
        return resp.json()


def main() -> None:
    """Entry point for the topolift-mcp console script (stdio transport)."""
    logging.basicConfig(level=os.environ.get("TOPOLIFT_LOG_LEVEL", "INFO"))
    if not API_KEY:
        # Don't refuse to start — `topolift_dialect` works without auth, and the
        # user may set the env var via their MCP client config. But emit a hint
        # so it's obvious why a negotiate call later returns an error.
        print(
            "topolift-mcp: warning — TOPOLIFT_API_KEY is not set. "
            "Dialect calls will work; negotiate calls will fail until set.",
            file=sys.stderr,
        )
    mcp.run()


# ────────────────────────────────────────────────────────────────────────────
# Streamable HTTP variant — for hosted deployment at mcp.topolift.ai
#
# Each incoming MCP request can carry its own `Authorization: Bearer tl-...`
# header. An ASGI middleware copies that header into the per-request context
# so the topolift_negotiate tool forwards it to the API. This is the multi-
# tenant model: the host doesn't know API keys; clients bring their own.
# ────────────────────────────────────────────────────────────────────────────


class _AuthCaptureMiddleware:
    """Pure-ASGI middleware. Reads Authorization from each HTTP request and
    sets the per-request ContextVar so the tool sees it. Resets on every
    request so values don't leak between concurrent agents."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        token = None
        for raw_name, raw_value in scope.get("headers", []):
            if raw_name.lower() == b"authorization":
                token = _request_auth.set(raw_value.decode("latin-1"))
                break
        try:
            await self.app(scope, receive, send)
        finally:
            if token is not None:
                _request_auth.reset(token)


def main_http() -> None:
    """Entry point for the topolift-mcp-http console script (Streamable HTTP).

    Listens on TOPOLIFT_MCP_HTTP_HOST:TOPOLIFT_MCP_HTTP_PORT (default
    127.0.0.1:8401). Designed to run behind a reverse proxy (Caddy) at
    mcp.topolift.ai. Each MCP request must carry its own
    `Authorization: Bearer tl-...` header for negotiate calls; dialect calls
    are unauthenticated.

    Run directly:  topolift-mcp-http
    Run with:      TOPOLIFT_MCP_HTTP_PORT=8401 topolift-mcp-http
    """
    import uvicorn

    logging.basicConfig(level=os.environ.get("TOPOLIFT_LOG_LEVEL", "INFO"))

    host = os.environ.get("TOPOLIFT_MCP_HTTP_HOST", "127.0.0.1")
    port = int(os.environ.get("TOPOLIFT_MCP_HTTP_PORT", "8401"))

    app = mcp.streamable_http_app()
    app = _AuthCaptureMiddleware(app)

    logger.info(
        "Serving topolift-mcp Streamable HTTP on %s:%d (forwarding to %s)",
        host, port, API_URL,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
