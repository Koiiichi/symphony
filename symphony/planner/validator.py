"""Pre-execution graph validator.

Checks a TaskGraph for common structural problems that would cause silent
failures at execution time.  Returns a list of human-readable error strings;
an empty list means the graph is valid.

Rejected conditions:
  - api_check node missing url, method, or expected_status in config
  - navigate action with a relative URL (no scheme) and no base_url provided
  - required verification claim with no executable assertion linked to it
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from symphony.planner.schema import TaskGraph


def validate_graph(graph: "TaskGraph", *, base_url: str = "") -> list[str]:
    """Return a list of validation errors, or [] if the graph is valid."""
    errors: list[str] = []

    # Collect claim IDs covered by at least one assertion
    covered_claims: set[str] = set()
    for node in graph.nodes:
        for action in list(node.assertions) + list(node.actions):
            if action.action.value in ("assert_text", "assert_http_status", "assert_banner"):
                cid = action.params.get("claim_id")
                if cid:
                    covered_claims.add(str(cid))
        # api_check nodes can cover a claim via config
        if node.type.value == "api_check":
            cid = node.config.get("claim_id")
            if cid:
                covered_claims.add(str(cid))

    for node in graph.nodes:
        ntype = node.type.value

        # --- api_check: require url, method, expected_status ---
        if ntype == "api_check":
            for field in ("url", "method", "expected_status"):
                if not node.config.get(field):
                    errors.append(
                        f"Node '{node.id}' (api_check) missing config.{field}. "
                        f"api_check config must look like: "
                        f'{{\"url\": \"http://localhost:3000/api/endpoint\", '
                        f'\"method\": \"POST\", \"expected_status\": 401}} — '
                        f"expected_status is ALWAYS required (use 401 for auth failures, "
                        f"201 for created, 200 for success, etc.)"
                    )
            # Relative URL is only an error if we have no base_url to resolve against.
            url = node.config.get("url", "")
            if url and not url.startswith(("http://", "https://")) and not base_url:
                errors.append(
                    f"Node '{node.id}' (api_check) has relative URL '{url}' "
                    "but no base_url is known (service_start not yet run or missing port). "
                    f"Use an absolute URL like 'http://localhost:3000{url}'"
                )

        # --- web_flow_test: check for relative navigate without base_url ---
        if ntype == "web_flow_test":
            for action in node.actions:
                if action.action.value == "navigate" and action.value:
                    url = action.value.strip()
                    if not url.startswith(("http://", "https://", "file://")):
                        if not base_url:
                            errors.append(
                                f"Node '{node.id}' has relative navigate '{url}' "
                                "but no base_url is known (service_start not yet run "
                                "or missing port)"
                            )
                        # If base_url is present, the executor will resolve it —
                        # not an error here.

    # --- verification_contract: required claims must be covered ---
    for claim in graph.verification_contract:
        if claim.required and claim.id not in covered_claims:
            errors.append(
                f"Required claim '{claim.id}' ({claim.description!r}) "
                "has no executable assertion linked via params.claim_id"
            )

    return errors
