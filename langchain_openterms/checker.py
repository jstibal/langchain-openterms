"""OpenTermsChecker: a standalone LangChain tool that agents can call
to check what they're allowed to do on a domain before acting.

Usage:
    from langchain_openterms import OpenTermsChecker

    checker = OpenTermsChecker()

    # Agent calls this tool with a domain and action:
    result = checker.invoke("example.com read_content")
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
    domain, action, whether it's allowed, and the reason.
    """

    name: str = "openterms_check"
    description: str = (
        "Check what an AI agent is permitted to do on a website. "
        "Input: '<domain> <action>' where action is one of: "
        "read_content, scrape_data, api_access, create_account, "
        "make_purchases, post_content, execute_code. "
        "Returns whether the action is allowed, denied, or unspecified."
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
