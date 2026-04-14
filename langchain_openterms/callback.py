"""OpenTermsCallbackHandler: logs openterms.json permission checks
whenever a LangChain agent invokes a tool against a URL.

Usage:
    from langchain_openterms import OpenTermsCallbackHandler

    handler = OpenTermsCallbackHandler(
        default_action="read_content",
        on_check=lambda r: print(f"Checked {r['domain']}: {r['allowed']}"),
    )

    agent.invoke({"input": "..."}, config={"callbacks": [handler]})
"""

import json
import logging
from typing import Any, Callable, Optional, Sequence, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from langchain_openterms.client import OpenTermsClient
from langchain_openterms.guard import _extract_domain

logger = logging.getLogger("langchain_openterms")


class OpenTermsCallbackHandler(BaseCallbackHandler):
    """Callback handler that checks openterms.json when tools are invoked.

    This is a passive observer: it logs permission checks but does not
    block tool execution. Use OpenTermsGuard for enforcement.

    Args:
        default_action: The permission key to check (default: "read_content").
        client: Optional OpenTermsClient instance.
        on_check: Optional callback invoked with the check result dict.
        log_receipts: If True, generate and log ORS receipts (default: True).
    """

    def __init__(
        self,
        default_action: str = "read_content",
        client: Optional[OpenTermsClient] = None,
        on_check: Optional[Callable[[dict], None]] = None,
        log_receipts: bool = True,
    ):
        self.default_action = default_action
        self.client = client or OpenTermsClient()
        self.on_check = on_check
        self.log_receipts = log_receipts
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

        if self.log_receipts:
            entry["receipt"] = self.client.receipt(domain, self.default_action, result)

        self.checks.append(entry)

        if result["allowed"] is False:
            logger.warning(
                "OpenTerms: %s denied '%s' on %s",
                serialized.get("name", "tool"),
                self.default_action,
                domain,
            )
        else:
            logger.info(
                "OpenTerms: %s checked '%s' on %s -> %s",
                serialized.get("name", "tool"),
                self.default_action,
                domain,
                result["allowed"],
            )

        if self.on_check:
            self.on_check(entry)
