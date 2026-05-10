"""OpenTermsCallbackHandler: logs openterms.json permission checks
whenever a LangChain agent invokes a tool against a URL.

This is a passive observer only — it logs and records checks but does NOT
block tool execution. Use OpenTermsGuard for enforcement.

Usage:
    from langchain_openterms import OpenTermsCallbackHandler

    handler = OpenTermsCallbackHandler(
        default_action="read_content",
        on_check=lambda r: print(f"Checked {r['domain']}: allowed={r['allowed']}"),
    )

    agent.invoke({"input": "..."}, config={"callbacks": [handler]})

    # Review logged checks after the run:
    for check in handler.checks:
        if check["allowed"] is not True:
            print(f"BLOCKED domain would be: {check['domain']} ({check['reason']})")
"""

import logging
from typing import Any, Callable, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from langchain_openterms.client import OpenTermsClient
from langchain_openterms.guard import _extract_domain

logger = logging.getLogger("langchain_openterms")


class OpenTermsCallbackHandler(BaseCallbackHandler):
    """Callback handler that checks openterms.json when tools are invoked.

    Passive observer: logs permission checks but does not block tool execution.
    Use OpenTermsGuard for enforcement.

    A check result with ``allowed`` not equal to True (i.e., None, False) means
    the action would be blocked under fail-closed semantics. This handler
    logs that as a warning but does not prevent execution.

    Args:
        default_action: The permission key to check (default: "read_content").
            Must be one of the 7 canonical keys: read_content, scrape_data,
            api_access, create_account, make_purchases, post_content, allow_training.
        client: Optional OpenTermsClient instance.
        on_check: Optional callback invoked with the check result dict.
    """

    def __init__(
        self,
        default_action: str = "read_content",
        client: Optional[OpenTermsClient] = None,
        on_check: Optional[Callable[[dict], None]] = None,
    ):
        self.default_action = default_action
        self.client = client or OpenTermsClient()
        self.on_check = on_check
        self.checks: list[dict] = []

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        domain = _extract_domain(input_str)
        if inputs and not domain:
            domain = _extract_domain(inputs)

        if domain is None:
            return

        result = self.client.check(domain, self.default_action)

        entry = {
            "tool": serialized.get("name", "unknown"),
            "domain": domain,
            "action": self.default_action,
            "allowed": result["allowed"],
            "reason": result["reason"],
        }

        self.checks.append(entry)

        if result["allowed"] is not True:
            logger.warning(
                "OpenTerms: %s — '%s' on %s is NOT explicitly allowed (allowed=%r). "
                "Reason: %s. (This handler is passive — use OpenTermsGuard to enforce.)",
                serialized.get("name", "tool"),
                self.default_action,
                domain,
                result["allowed"],
                result["reason"],
            )
        else:
            logger.info(
                "OpenTerms: %s — '%s' on %s is explicitly allowed.",
                serialized.get("name", "tool"),
                self.default_action,
                domain,
            )

        if self.on_check:
            self.on_check(entry)
