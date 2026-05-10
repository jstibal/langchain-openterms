"""Tests for langchain-openterms integration.

All mocks are consistent with openterms-py>=0.3.1 fail-closed semantics:
- Empty caches are NOT treated as truthy (0.3.1 bug fix)
- Unreachable domains do NOT return permissive defaults
- not_specified / missing keys return allowed=None (not allowed=True)
- low-confidence / conditional return allowed=None (not allowed=True)
"""

import json
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.tools import BaseTool

from langchain_openterms.callback import OpenTermsCallbackHandler
from langchain_openterms.checker import OpenTermsChecker
from langchain_openterms.client import OpenTermsClient, CANONICAL_PERMISSION_KEYS
from langchain_openterms.guard import OpenTermsGuard, _extract_domain, _is_explicitly_allowed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_OPENTERMS = {
    "openterms_version": "0.3.0",
    "service": "example.com",
    "permissions": {
        "read_content": True,
        "scrape_data": False,
        "api_access": {
            "allowed": True,
            "requires_auth": True,
            "max_frequency": "1000/hour",
        },
        # make_purchases, post_content, create_account, allow_training are absent
    },
    "discovery": {
        "mcp_servers": [
            {
                "url": "https://example.com/mcp/sse",
                "transport": "sse",
                "description": "Order tools.",
            }
        ],
    },
}

OPENTERMS_ALL_DENIED = {
    "openterms_version": "0.3.0",
    "service": "denied.com",
    "permissions": {
        "read_content": False,
        "scrape_data": False,
        "api_access": False,
        "create_account": False,
        "make_purchases": False,
        "post_content": False,
        "allow_training": False,
    },
}


class DummyTool(BaseTool):
    name: str = "dummy"
    description: str = "A dummy tool for testing."

    def _run(self, tool_input, **kwargs):
        return f"executed: {tool_input}"

    async def _arun(self, tool_input, **kwargs):
        return f"executed: {tool_input}"


def make_client(data=None, domain="example.com"):
    """Create an OpenTermsClient with a pre-populated cache.

    Consistent with openterms-py 0.3.1: cache dict must not be falsy to
    avoid the pre-0.3.1 bug where an empty cache was treated as None and
    fell through to the module-level singleton.
    """
    client = OpenTermsClient()
    # Explicitly set a non-empty cache entry — data=None means file missing
    client._cache[domain] = {"data": data, "fetched_at": 9_999_999_999}
    return client


# ---------------------------------------------------------------------------
# Canonical keys
# ---------------------------------------------------------------------------


class TestCanonicalKeys:
    """Verify only the 7 canonical keys are accepted/documented."""

    def test_canonical_keys_set(self):
        assert CANONICAL_PERMISSION_KEYS == {
            "read_content",
            "scrape_data",
            "api_access",
            "create_account",
            "make_purchases",
            "post_content",
            "allow_training",
        }

    def test_no_execute_code_in_canonical_keys(self):
        assert "execute_code" not in CANONICAL_PERMISSION_KEYS

    def test_checker_description_contains_only_canonical_keys(self):
        checker = OpenTermsChecker()
        non_canonical = ["execute_code", "ors", "receipt", "audit"]
        for key in non_canonical:
            assert key not in checker.description, (
                f"Non-canonical key '{key}' found in OpenTermsChecker.description"
            )
        for key in CANONICAL_PERMISSION_KEYS:
            assert key in checker.description, (
                f"Canonical key '{key}' missing from OpenTermsChecker.description"
            )


# ---------------------------------------------------------------------------
# _is_explicitly_allowed
# ---------------------------------------------------------------------------


class TestIsExplicitlyAllowed:
    def test_true_allowed(self):
        assert _is_explicitly_allowed({"allowed": True}) is True

    def test_false_not_allowed(self):
        assert _is_explicitly_allowed({"allowed": False}) is False

    def test_none_not_allowed(self):
        assert _is_explicitly_allowed({"allowed": None}) is False

    def test_missing_key_not_allowed(self):
        assert _is_explicitly_allowed({}) is False


# ---------------------------------------------------------------------------
# Domain extraction
# ---------------------------------------------------------------------------


class TestDomainExtraction:
    def test_full_url(self):
        assert _extract_domain("https://example.com/path") == "example.com"

    def test_url_in_dict(self):
        assert _extract_domain({"url": "https://example.com/page"}) == "example.com"

    def test_bare_domain_in_string(self):
        assert _extract_domain("check example.com for info") == "example.com"

    def test_no_domain(self):
        assert _extract_domain("just some text") is None

    def test_query_dict(self):
        assert _extract_domain({"query": "https://stripe.com/pricing"}) == "stripe.com"


# ---------------------------------------------------------------------------
# Client — fail-closed semantics (consistent with openterms-py 0.3.1)
# ---------------------------------------------------------------------------


class TestClient:
    """Test client semantics. Mocks are consistent with 0.3.1 fail-closed behavior:
    empty caches are properly initialized (not falsy), and unreachable/missing
    domains return allowed=None, not a permissive default.
    """

    # 1. allowed → permits execution
    def test_check_allowed_returns_true(self):
        client = make_client(SAMPLE_OPENTERMS)
        result = client.check("example.com", "read_content")
        assert result["allowed"] is True

    def test_allowed_result_truthy(self):
        client = make_client(SAMPLE_OPENTERMS)
        result = client.check("example.com", "read_content")
        # Explicit allowed=True is the only permissive state
        assert result["allowed"] is True

    # 2. denied → blocks execution
    def test_check_denied_returns_false(self):
        client = make_client(SAMPLE_OPENTERMS)
        result = client.check("example.com", "scrape_data")
        assert result["allowed"] is False

    # 3. missing / no file → blocks by default (allowed=None)
    def test_check_no_file_returns_none(self):
        client = make_client(None)
        result = client.check("example.com", "read_content")
        assert result["allowed"] is None
        assert "No openterms.json" in result["reason"]

    def test_no_file_is_not_truthy(self):
        """Missing file must not return a truthy/permissive result.
        Consistent with 0.3.1: no fallthrough to permissive defaults."""
        client = make_client(None)
        result = client.check("example.com", "read_content")
        assert result["allowed"] is not True  # must be blocked

    # 4. not_specified → blocks by default (allowed=None)
    def test_check_not_specified_returns_none(self):
        client = make_client(SAMPLE_OPENTERMS)
        # make_purchases is not in SAMPLE_OPENTERMS.permissions
        result = client.check("example.com", "make_purchases")
        assert result["allowed"] is None
        assert result["allowed"] is not True  # must not be permissive

    def test_not_specified_reason(self):
        client = make_client(SAMPLE_OPENTERMS)
        result = client.check("example.com", "post_content")
        assert result["allowed"] is None
        assert "not specified" in result["reason"].lower() or "No openterms" in result["reason"]

    # 5. low-confidence → blocks by default
    def test_null_permission_value_returns_none(self):
        """A null permission value (JSON null) must not permit execution."""
        data = {**SAMPLE_OPENTERMS, "permissions": {**SAMPLE_OPENTERMS["permissions"], "create_account": None}}
        client = make_client(data)
        result = client.check("example.com", "create_account")
        assert result["allowed"] is None  # null → not allowed

    # 6. conditional → blocks unless conditions satisfied
    def test_nested_dict_without_allowed_true_blocks(self):
        """A dict permission without allowed=True must not permit execution."""
        data = {
            **SAMPLE_OPENTERMS,
            "permissions": {
                **SAMPLE_OPENTERMS["permissions"],
                "post_content": {"conditions": "requires_login"},  # no "allowed" key
            },
        }
        client = make_client(data)
        result = client.check("example.com", "post_content")
        assert result["allowed"] is not True

    def test_nested_dict_allowed_true_permits(self):
        """A dict permission with allowed=True must permit execution."""
        client = make_client(SAMPLE_OPENTERMS)
        result = client.check("example.com", "api_access")
        assert result["allowed"] is True

    def test_nested_dict_allowed_false_blocks(self):
        """A dict permission with allowed=False must block."""
        data = {
            **SAMPLE_OPENTERMS,
            "permissions": {
                **SAMPLE_OPENTERMS["permissions"],
                "allow_training": {"allowed": False, "scope": "commercial"},
            },
        }
        client = make_client(data)
        result = client.check("example.com", "allow_training")
        assert result["allowed"] is False

    # Discover / receipt
    def test_discover(self):
        client = make_client(SAMPLE_OPENTERMS)
        disc = client.discover("example.com")
        assert disc is not None
        assert len(disc["mcp_servers"]) == 1

    def test_discover_absent(self):
        no_discovery = {k: v for k, v in SAMPLE_OPENTERMS.items() if k != "discovery"}
        client = make_client(no_discovery)
        assert client.discover("example.com") is None

    def test_receipt(self):
        client = make_client(SAMPLE_OPENTERMS)
        result = client.check("example.com", "scrape_data")
        receipt = client.receipt("example.com", "scrape_data", result)
        assert receipt["domain"] == "example.com"
        assert receipt["allowed"] is False
        assert receipt["openterms_hash"] != ""
        assert "checked_at" in receipt


# ---------------------------------------------------------------------------
# Guard — fail-closed defaults
# ---------------------------------------------------------------------------


class TestGuard:
    """Test OpenTermsGuard. Fail-closed=True is the default."""

    # 1. allowed → permits execution
    def test_allowed_passes_through(self):
        client = make_client(SAMPLE_OPENTERMS)
        guard = OpenTermsGuard(tool=DummyTool(), action="read_content", client=client)
        result = guard.invoke("https://example.com/page")
        assert result == "executed: https://example.com/page"

    # 2. denied → blocks execution
    def test_denied_blocks(self):
        client = make_client(SAMPLE_OPENTERMS)
        guard = OpenTermsGuard(tool=DummyTool(), action="scrape_data", client=client)
        result = guard.invoke("https://example.com/data")
        assert "blocked" in result.lower()
        assert "scrape_data" in result

    # 3. missing / no file → blocks by default (fail_closed=True)
    def test_no_file_fail_closed_blocks(self):
        """No openterms.json must block under fail_closed=True (default)."""
        client = make_client(None)
        guard = OpenTermsGuard(tool=DummyTool(), action="read_content", client=client)
        result = guard.invoke("https://example.com/page")
        assert "blocked" in result.lower()

    # 4. not_specified → blocks by default
    def test_not_specified_fail_closed_blocks(self):
        """Unspecified permission must block under fail_closed=True (default)."""
        client = make_client(SAMPLE_OPENTERMS)
        guard = OpenTermsGuard(tool=DummyTool(), action="make_purchases", client=client)
        result = guard.invoke("https://example.com/shop")
        assert "blocked" in result.lower()

    # 5. low-confidence → blocks (null value treated as not allowed)
    def test_null_permission_fail_closed_blocks(self):
        data = {**SAMPLE_OPENTERMS, "permissions": {**SAMPLE_OPENTERMS["permissions"], "create_account": None}}
        client = make_client(data)
        guard = OpenTermsGuard(tool=DummyTool(), action="create_account", client=client)
        result = guard.invoke("https://example.com/signup")
        assert "blocked" in result.lower()

    # 6. conditional → blocks (dict without allowed=True)
    def test_conditional_fail_closed_blocks(self):
        data = {
            **SAMPLE_OPENTERMS,
            "permissions": {
                **SAMPLE_OPENTERMS["permissions"],
                "post_content": {"conditions": "requires_approval"},
            },
        }
        client = make_client(data)
        guard = OpenTermsGuard(tool=DummyTool(), action="post_content", client=client)
        result = guard.invoke("https://example.com/post")
        assert "blocked" in result.lower()

    # 7. permissive opt-in requires explicit fail_closed=False
    def test_no_file_permissive_opt_in_passes(self):
        """Missing file passes through only with explicit fail_closed=False."""
        client = make_client(None)
        guard = OpenTermsGuard(
            tool=DummyTool(), action="read_content", client=client, fail_closed=False
        )
        result = guard.invoke("https://example.com/page")
        assert result == "executed: https://example.com/page"

    def test_not_specified_permissive_opt_in_passes(self):
        """Unspecified permission passes through with explicit fail_closed=False."""
        client = make_client(SAMPLE_OPENTERMS)
        guard = OpenTermsGuard(
            tool=DummyTool(), action="make_purchases", client=client, fail_closed=False
        )
        result = guard.invoke("https://example.com/shop")
        assert result == "executed: https://example.com/shop"

    def test_permissive_still_blocks_explicit_deny(self):
        """Even with fail_closed=False, an explicit deny must still block."""
        client = make_client(SAMPLE_OPENTERMS)
        guard = OpenTermsGuard(
            tool=DummyTool(), action="scrape_data", client=client, fail_closed=False
        )
        result = guard.invoke("https://example.com/data")
        assert "blocked" in result.lower()

    # 9. guarded wrapper blocks downstream tool when permission is not allowed
    def test_wrapper_blocks_downstream_when_denied(self):
        """Downstream tool must not execute when permission is denied."""
        executed = []

        class TrackingTool(BaseTool):
            name: str = "tracker"
            description: str = "Tracks execution."

            def _run(self, tool_input, **kwargs):
                executed.append(tool_input)
                return f"executed: {tool_input}"

            async def _arun(self, tool_input, **kwargs):
                executed.append(tool_input)
                return f"executed: {tool_input}"

        client = make_client(SAMPLE_OPENTERMS)
        guard = OpenTermsGuard(tool=TrackingTool(), action="scrape_data", client=client)
        guard.invoke("https://example.com/data")
        assert len(executed) == 0, "Downstream tool must not execute when permission denied"

    # 10. guarded wrapper permits downstream tool when permission is allowed
    def test_wrapper_permits_downstream_when_allowed(self):
        """Downstream tool must execute when permission is explicitly allowed."""
        executed = []

        class TrackingTool(BaseTool):
            name: str = "tracker"
            description: str = "Tracks execution."

            def _run(self, tool_input, **kwargs):
                executed.append(tool_input)
                return f"executed: {tool_input}"

            async def _arun(self, tool_input, **kwargs):
                executed.append(tool_input)
                return f"executed: {tool_input}"

        client = make_client(SAMPLE_OPENTERMS)
        guard = OpenTermsGuard(tool=TrackingTool(), action="read_content", client=client)
        guard.invoke("https://example.com/page")
        assert len(executed) == 1, "Downstream tool must execute when permission is explicitly allowed"

    # on_denied callback
    def test_on_denied_callback_fires(self):
        client = make_client(SAMPLE_OPENTERMS)
        denied_log = []
        guard = OpenTermsGuard(
            tool=DummyTool(),
            action="scrape_data",
            client=client,
            on_denied=lambda d, a, r: denied_log.append((d, a)),
        )
        guard.invoke("https://example.com/data")
        assert len(denied_log) == 1
        assert denied_log[0] == ("example.com", "scrape_data")

    def test_on_denied_fires_for_not_specified_fail_closed(self):
        """on_denied callback must fire for not_specified under fail_closed=True."""
        client = make_client(SAMPLE_OPENTERMS)
        denied_log = []
        guard = OpenTermsGuard(
            tool=DummyTool(),
            action="make_purchases",
            client=client,
            on_denied=lambda d, a, r: denied_log.append((d, a)),
        )
        guard.invoke("https://example.com/shop")
        assert len(denied_log) == 1

    # LangChain BaseTool compatibility
    def test_guard_is_base_tool_instance(self):
        """OpenTermsGuard must be a valid LangChain BaseTool."""
        client = make_client(SAMPLE_OPENTERMS)
        guard = OpenTermsGuard(tool=DummyTool(), action="read_content", client=client)
        assert isinstance(guard, BaseTool)
        assert hasattr(guard, "name")
        assert hasattr(guard, "description")
        assert callable(getattr(guard, "_run", None))
        assert callable(getattr(guard, "_arun", None))

    def test_guard_name_prefixed(self):
        client = make_client(SAMPLE_OPENTERMS)
        guard = OpenTermsGuard(tool=DummyTool(), action="read_content", client=client)
        assert guard.name == "guarded_dummy"


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------


class TestChecker:
    def test_check_allowed_returns_json(self):
        client = make_client(SAMPLE_OPENTERMS)
        checker = OpenTermsChecker(client=client)
        result = checker.invoke("example.com read_content")
        parsed = json.loads(result)
        assert parsed["check"]["allowed"] is True
        assert "receipt" in parsed

    def test_check_denied_returns_false(self):
        client = make_client(SAMPLE_OPENTERMS)
        checker = OpenTermsChecker(client=client)
        result = checker.invoke("example.com scrape_data")
        parsed = json.loads(result)
        assert parsed["check"]["allowed"] is False

    def test_check_not_specified_returns_none(self):
        client = make_client(SAMPLE_OPENTERMS)
        checker = OpenTermsChecker(client=client)
        result = checker.invoke("example.com make_purchases")
        parsed = json.loads(result)
        assert parsed["check"]["allowed"] is None

    def test_check_no_file_returns_none(self):
        client = make_client(None)
        checker = OpenTermsChecker(client=client)
        result = checker.invoke("example.com read_content")
        parsed = json.loads(result)
        assert parsed["check"]["allowed"] is None

    def test_bad_input_raises(self):
        client = make_client(SAMPLE_OPENTERMS)
        checker = OpenTermsChecker(client=client)
        with pytest.raises(ValueError, match="Expected"):
            checker.invoke("justadomain")

    def test_checker_is_base_tool(self):
        checker = OpenTermsChecker()
        assert isinstance(checker, BaseTool)


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


class TestCallback:
    def test_logs_check_on_tool_start_allowed(self):
        client = make_client(SAMPLE_OPENTERMS)
        handler = OpenTermsCallbackHandler(default_action="read_content", client=client)
        handler.on_tool_start(
            serialized={"name": "web_search"},
            input_str="https://example.com/pricing",
            run_id=uuid4(),
        )
        assert len(handler.checks) == 1
        assert handler.checks[0]["allowed"] is True

    def test_logs_check_on_tool_start_denied(self):
        client = make_client(SAMPLE_OPENTERMS)
        handler = OpenTermsCallbackHandler(default_action="scrape_data", client=client)
        handler.on_tool_start(
            serialized={"name": "scraper"},
            input_str="https://example.com/data",
            run_id=uuid4(),
        )
        assert len(handler.checks) == 1
        assert handler.checks[0]["allowed"] is False

    def test_logs_check_not_specified_not_truthy(self):
        """not_specified must not be logged as allowed under 0.3.1 semantics."""
        client = make_client(SAMPLE_OPENTERMS)
        handler = OpenTermsCallbackHandler(default_action="make_purchases", client=client)
        handler.on_tool_start(
            serialized={"name": "checkout"},
            input_str="https://example.com/shop",
            run_id=uuid4(),
        )
        assert len(handler.checks) == 1
        assert handler.checks[0]["allowed"] is not True

    def test_skips_non_url_input(self):
        client = make_client(SAMPLE_OPENTERMS)
        handler = OpenTermsCallbackHandler(client=client)
        handler.on_tool_start(
            serialized={"name": "calculator"},
            input_str="2 + 2",
            run_id=uuid4(),
        )
        assert len(handler.checks) == 0

    def test_on_check_callback(self):
        client = make_client(SAMPLE_OPENTERMS)
        log = []
        handler = OpenTermsCallbackHandler(
            client=client,
            on_check=lambda r: log.append(r),
        )
        handler.on_tool_start(
            serialized={"name": "fetch"},
            input_str="https://example.com/api",
            run_id=uuid4(),
        )
        assert len(log) == 1

    def test_no_log_receipts_field(self):
        """Callback handler must not expose log_receipts (removed in 0.4.0)."""
        handler = OpenTermsCallbackHandler()
        assert not hasattr(handler, "log_receipts"), (
            "log_receipts field was removed in 0.4.0 — ORS/receipt language removed"
        )


# ---------------------------------------------------------------------------
# 11. Mock consistency with openterms-py 0.3.1
# ---------------------------------------------------------------------------


class TestMockConsistency:
    """Verify test mocks do not simulate pre-0.3.1 permissive behavior.

    openterms-py 0.3.1 fixed a Python truthiness footgun where an empty cache
    (falsy dict) would fall through to the module-level singleton, masking
    network failures as stale cached data. Our mocks must never simulate
    this pre-0.3.1 behavior.
    """

    def test_empty_cache_not_falsy(self):
        """Cache entry must be a non-empty dict even when data is None.
        Pre-0.3.1 bug: empty cache {} was falsy → fell through to singleton."""
        client = make_client(None)
        # The cache entry must exist and be non-empty (has 'data' and 'fetched_at')
        assert "example.com" in client._cache
        assert "data" in client._cache["example.com"]
        assert "fetched_at" in client._cache["example.com"]
        # Cache entry must not be falsy
        assert client._cache["example.com"]  # non-empty dict

    def test_missing_file_does_not_return_allow(self):
        """Unreachable/missing domains must not return allowed=True.
        Pre-0.3.1 behavior: could fall through to permissive default."""
        client = make_client(None)
        result = client.check("example.com", "read_content")
        assert result["allowed"] is not True

    def test_empty_permissions_does_not_permit(self):
        """A domain with an empty permissions dict must not permit any action."""
        data = {"openterms_version": "0.3.0", "service": "empty.com", "permissions": {}}
        client = make_client(data, domain="empty.com")
        result = client.check("empty.com", "read_content")
        assert result["allowed"] is not True
