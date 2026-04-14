"""OpenTermsGuard: wraps a LangChain tool so it checks openterms.json
before executing against a domain.

Usage:
    from langchain_community.tools import WebSearchTool
    from langchain_openterms import OpenTermsGuard

    search = WebSearchTool()
    guarded_search = OpenTermsGuard(
        tool=search,
        action="read_content",
    )

    # Agent uses guarded_search instead of search.
    # If the target domain denies "read_content", the tool returns
    # a denial message instead of executing.
"""

from typing import Any, Callable, Optional
from urllib.parse import urlparse

from langchain_core.tools import BaseTool

from langchain_openterms.client import OpenTermsClient


def _extract_domain(tool_input: Any) -> Optional[str]:
    """Best-effort domain extraction from tool input."""
    text = ""
    if isinstance(tool_input, str):
        text = tool_input
    elif isinstance(tool_input, dict):
        for key in ("url", "query", "input", "site"):
            if key in tool_input:
                text = str(tool_input[key])
                break
        if not text:
            text = str(tool_input)

    parsed = urlparse(text)
    if parsed.netloc:
        return parsed.netloc

    # Try treating the whole string as a domain
    for word in text.split():
        if "." in word:
            cleaned = word.strip("\"'<>()[]")
            parsed = urlparse(f"https://{cleaned}")
            if parsed.netloc:
                return parsed.netloc

    return None


class OpenTermsGuard(BaseTool):
    """Wraps another tool with an openterms.json permission check.

    Before the wrapped tool executes, OpenTermsGuard fetches the target
    domain's openterms.json and checks whether the specified action is
    allowed. If denied, the tool returns a message explaining the denial
    instead of executing. If no openterms.json is found or the action is
    unspecified, the tool executes normally (permissive default).

    Args:
        tool: The LangChain tool to wrap.
        action: The openterms.json permission key to check (e.g., "read_content",
            "scrape_data", "api_access").
        client: Optional OpenTermsClient instance. Created automatically if omitted.
        on_denied: Optional callback invoked with (domain, action, check_result)
            when a permission is denied. Useful for logging.
        strict: If True, block execution when openterms.json is absent (default False).
        domain_extractor: Optional function to extract a domain from tool input.
            Defaults to a built-in heuristic.
    """

    tool: BaseTool
    action: str
    client: OpenTermsClient = None
    on_denied: Optional[Callable] = None
    strict: bool = False
    domain_extractor: Optional[Callable] = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **kwargs: Any):
        if "client" not in kwargs or kwargs["client"] is None:
            kwargs["client"] = OpenTermsClient()
        tool = kwargs["tool"]
        kwargs.setdefault("name", f"guarded_{tool.name}")
        kwargs.setdefault(
            "description",
            f"{tool.description} (permission-checked via openterms.json)",
        )
        super().__init__(**kwargs)

    def _run(self, tool_input: Any, **kwargs: Any) -> Any:
        extractor = self.domain_extractor or _extract_domain
        domain = extractor(tool_input)

        if domain is None:
            # Cannot determine domain; pass through
            return self.tool._run(tool_input, **kwargs)

        result = self.client.check(domain, self.action)

        if result["allowed"] is False:
            if self.on_denied:
                self.on_denied(domain, self.action, result)
            return (
                f"Action blocked by openterms.json: {result['reason']} "
                f"Domain '{domain}' does not permit '{self.action}'."
            )

        if result["allowed"] is None and self.strict:
            return (
                f"No openterms.json found for '{domain}' and strict mode is enabled. "
                f"Cannot verify permission for '{self.action}'."
            )

        return self.tool._run(tool_input, **kwargs)

    async def _arun(self, tool_input: Any, **kwargs: Any) -> Any:
        # Synchronous check, async tool execution
        extractor = self.domain_extractor or _extract_domain
        domain = extractor(tool_input)

        if domain is None:
            return await self.tool._arun(tool_input, **kwargs)

        result = self.client.check(domain, self.action)

        if result["allowed"] is False:
            if self.on_denied:
                self.on_denied(domain, self.action, result)
            return (
                f"Action blocked by openterms.json: {result['reason']} "
                f"Domain '{domain}' does not permit '{self.action}'."
            )

        if result["allowed"] is None and self.strict:
            return (
                f"No openterms.json found for '{domain}' and strict mode is enabled. "
                f"Cannot verify permission for '{self.action}'."
            )

        return await self.tool._arun(tool_input, **kwargs)
