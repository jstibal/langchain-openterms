"""OpenTermsChecker: a standalone LangChain tool that agents can call
to check what they're allowed to do on a domain before acting.

Returns the permission check result as JSON. Agents should gate execution
on the ``allowed`` field being explicitly ``true``.

Usage:
    from langchain_openterms import OpenTermsChecker

    checker = OpenTermsChecker()

    # Agent calls this tool with a domain and action:
    result = checker.invoke("example.com read_content")
    import json
    parsed = json.loads(result)
    if parsed["check"]["allowed"] is True:
        # explicitly allowed — proceed
        ...
    else:
        # blocked: denied, not_specified, missing file, low-confidence, etc.
        print("Cannot proceed:", parsed["check"]["reason"])
"""

import json
from typing import Any, Optional

from langchain_core.tools import BaseTool

from langchain_openterms.client import OpenTermsClient


class OpenTermsChecker(BaseTool):
    """Tool that checks a domain's openterms.json permissions.

    Input format: "<domain> <action>"
    Examples:
        "github.com read_content"
        "stripe.com api_access"
        "example.com scrape_data"

    Returns a JSON string with the permission check result including
    domain, action, allowed (True/False/None), and reason.

    The ``allowed`` field is True only for an explicitly allowed permission.
    All other states (denied, not_specified, missing file, low-confidence,
    conditional) return allowed=False or allowed=None and must be treated
    as blocked.

    Canonical permission keys:
        read_content, scrape_data, api_access, create_account,
        make_purchases, post_content, allow_training
    """

    name: str = "openterms_check"
    description: str = (
        "Check what an AI agent is permitted to do on a website. "
        "Input: '<domain> <action>' where action is one of the 7 canonical keys: "
        "read_content, scrape_data, api_access, create_account, "
        "make_purchases, post_content, allow_training. "
        "Returns allowed=true only for explicitly permitted actions. "
        "All other results (denied, not_specified, missing file, "
        "low-confidence, conditional) must be treated as blocked."
    )
    client: OpenTermsClient = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **kwargs: Any):
        if "client" not in kwargs or kwargs["client"] is None:
            kwargs["client"] = OpenTermsClient()
        super().__init__(**kwargs)

    @staticmethod
    def _split_input(tool_input: str) -> tuple[str, str]:
        parts = tool_input.strip().split()
        if len(parts) < 2:
            raise ValueError(
                f"Expected '<domain> <action>', got: '{tool_input}'. "
                f"Example: 'github.com read_content'"
            )
        return parts[0], parts[1]

    def _run(self, tool_input: str, **kwargs: Any) -> str:
        domain, action = self._split_input(tool_input)
        result = self.client.check(domain, action)
        receipt = self.client.receipt(domain, action, result)
        return json.dumps({"check": result, "receipt": receipt}, indent=2)

    async def _arun(self, tool_input: str, **kwargs: Any) -> str:
        return self._run(tool_input, **kwargs)
