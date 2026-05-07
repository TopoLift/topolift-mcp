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

import logging
import os
import sys
from typing import Any, Optional

import httpx
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("topolift.mcp")

API_URL = os.environ.get("TOPOLIFT_API_URL", "https://api.topolift.ai").rstrip("/")
API_KEY = os.environ.get("TOPOLIFT_API_KEY", "").strip()
NEGOTIATE_TIMEOUT = float(os.environ.get("TOPOLIFT_TIMEOUT", "600"))

mcp = FastMCP(
    "topolift-negotiation",
    instructions=(
        "TopoLift exposes atom-grounded negotiation reasoning. Before your first "
        "negotiate call, fetch the dialect with `topolift_dialect` so you can read "
        "the typed `topology` slot and inline [Cluster_X#strat] citations in the "
        "negotiate response with full structural fluency. Atoms stay server-side; "
        "what you receive is the structural reasoning that the topology produced."
    ),
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
    if not API_KEY:
        raise RuntimeError(
            "TOPOLIFT_API_KEY is not set. Set the env var to your TopoLift API "
            "key (get one at https://topolift.ai), or call /v1/negotiate "
            "directly with an x402 payment if you prefer pay-per-call."
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
                "Authorization": f"Bearer {API_KEY}",
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
    """Entry point for the topolift-mcp console script."""
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


if __name__ == "__main__":
    main()
