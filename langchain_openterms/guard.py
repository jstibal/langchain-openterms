"""OpenTermsGuard: wraps a LangChain tool so it checks openterms.json
before executing against a domain.

Fail-closed by default. Execution is blocked unless the permission check
returns an explicitly allowed result. To opt into permissive behavior for
missing or unspecified permissions, pass fail_closed=False.

Usage:
    from langchain_community.tools import WebSearchTool
    from langchain_openterms import OpenTermsGuard

    search = WebSearchTool()

    # Default: fail-closed — blocks on missing/unspecified/denied
    guarded_search = OpenTermsGuard(
        tool=search,
        action="read_content",
    )

    result = guarded_search.invoke("https://example.com/page")
    if "blocked" in result.lower():
        print("Permission denied — not executing")

    # Permissive opt-in (not recommended for production):
    permissive_search = OpenTermsGuard(
        tool=search,
        action="read_content",
        fail_closed=False,  # pass through when no openterms.json found
    )
"""

from typing import Any, Callable, Optional
from urllib.parse import urlparse

from langchain_core.tools import BaseTool

from langchain_openterms.client import OpenTermsClient


def _is_explicitly_allowed(result: dict) -> bool:
    """Return True only when the check result is an explicit allow.

    Every other state — None, not_specified, no_openterms_json, low-confidence,
    conditional, denied — is treated as blocked under fail-closed semantics.
    """
    return result.get("allowed") is True


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

    Fail-closed by default: execution is blocked unless the permission check
    returns an explicitly allowed result. The following states all block
    by default:

        - null / unknown (domain unreachable, SDK error)
        - no_openterms_json (no file found)
        - not_specified (key absent from openterms.json)
        - low-confidence (validator confidence too low)
        - conditional (permission is conditional on unverified criteria)
        - denied (explicit deny)

    Only an explicit ``allowed: true`` permits execution.

    Args:
        tool: The LangChain tool to wrap.
        action: The openterms.json permission key to check. Must be one of
            the 7 canonical keys: read_content, scrape_data, api_access,
            create_account, make_purchases, post_content, allow_training.
        client: Optional OpenTermsClient instance. Created automatically if omitted.
        on_denied: Optional callback invoked with (domain, action, check_result)
            when execution is blocked. Useful for logging.
        fail_closed: If True (default), block execution for any non-explicitly-
            allowed result including missing files and unspecified permissions.
            Set to False only for explicit permissive opt-in.
        domain_extractor: Optional function to extract a domain from tool input.
            Defaults to a built-in heuristic.
    """

    tool: BaseTool
    action: str
    client: OpenTermsClient = None
    on_denied: Optional[Callable] = None
    fail_closed: bool = True
    domain_extractor: Optional[Callable] = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **kwargs: Any):
        # Legacy compat: map deprecated strict= to fail_closed=
        if "strict" in kwargs and "fail_closed" not in kwargs:
            kwargs["fail_closed"] = kwargs.pop("strict")
        else:
            kwargs.pop("strict", None)
        if "client" not in kwargs or kwargs["client"] is None:
            kwargs["client"] = OpenTermsClient()
        tool = kwargs["tool"]
        kwargs.setdefault("name", f"guarded_{tool.name}")
        kwargs.setdefault(
            "description",
            f"{tool.description} (permission-checked via openterms.json)",
        )
        super().__init__(**kwargs)

    def _check_and_block(self, tool_input: Any) -> Optional[str]:
        """Run permission check. Returns a block message, or None if allowed."""
        extractor = self.domain_extractor or _extract_domain
        domain = extractor(tool_input)

        if domain is None:
            if self.fail_closed:
                return (
                    "OpenTermsGuard: cannot determine target domain from input. "
                    "Execution blocked (fail_closed=True). "
                    "Pass fail_closed=False to allow execution when domain is unknown."
                )
            return None  # permissive: pass through

        result = self.client.check(domain, self.action)

        if _is_explicitly_allowed(result):
            return None  # explicitly allowed — proceed

        # All other states block under fail-closed
        if self.fail_closed:
            if self.on_denied:
                self.on_denied(domain, self.action, result)
            reason = result.get("reason", "Permission not explicitly granted.")
            return (
                f"Action blocked by openterms.json: {reason} "
                f"Domain '{domain}' does not explicitly permit '{self.action}'. "
                f"(fail_closed=True blocks null, not_specified, low-confidence, "
                f"conditional, and denied results.)"
            )

        # fail_closed=False: only block on explicit deny
        if result["allowed"] is False:
            if self.on_denied:
                self.on_denied(domain, self.action, result)
            return (
                f"Action blocked by openterms.json: {result['reason']} "
                f"Domain '{domain}' does not permit '{self.action}'."
            )

        return None  # permissive pass-through for None/not_specified

    def _run(self, tool_input: Any, **kwargs: Any) -> Any:
        block_msg = self._check_and_block(tool_input)
        if block_msg is not None:
            return block_msg
        return self.tool._run(tool_input, **kwargs)

    async def _arun(self, tool_input: Any, **kwargs: Any) -> Any:
        block_msg = self._check_and_block(tool_input)
        if block_msg is not None:
            return block_msg
        return await self.tool._arun(tool_input, **kwargs)
