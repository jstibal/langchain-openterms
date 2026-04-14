"""Tests for langchain-openterms integration."""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import BaseTool

from langchain_openterms.callback import OpenTermsCallbackHandler
from langchain_openterms.checker import OpenTermsChecker
from langchain_openterms.client import OpenTermsClient
from langchain_openterms.guard import OpenTermsGuard, _extract_domain


# --- Fixtures ---

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


class DummyTool(BaseTool):
    name: str = "dummy"
    description: str = "A dummy tool for testing."

    def _run(self, tool_input, **kwargs):
        return f"executed: {tool_input}"

    async def _arun(self, tool_input, **kwargs):
        return f"executed: {tool_input}"


def mock_client(data=None):
    """Create an OpenTermsClient with a pre-populated cache."""
    client = OpenTermsClient()
    if data is not None:
        client._cache["example.com"] = {"data": data, "fetched_at": 9999999999}
    return client


# --- Domain extraction ---


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


# --- Client ---


class TestClient:
    def test_check_allowed(self):
        client = mock_client(SAMPLE_OPENTERMS)
        result = client.check("example.com", "read_content")
        assert result["allowed"] is True

    def test_check_denied(self):
        client = mock_client(SAMPLE_OPENTERMS)
        result = client.check("example.com", "scrape_data")
        assert result["allowed"] is False

    def test_check_nested(self):
        client = mock_client(SAMPLE_OPENTERMS)
        result = client.check("example.com", "api_access")
        assert result["allowed"] is True

    def test_check_unspecified(self):
        client = mock_client(SAMPLE_OPENTERMS)
        result = client.check("example.com", "make_purchases")
        assert result["allowed"] is None

    def test_check_no_file(self):
        client = mock_client(None)
        result = client.check("example.com", "read_content")
        assert result["allowed"] is None
        assert "No openterms.json" in result["reason"]

    def test_discover(self):
        client = mock_client(SAMPLE_OPENTERMS)
        disc = client.discover("example.com")
        assert disc is not None
        assert len(disc["mcp_servers"]) == 1

    def test_discover_absent(self):
        no_discovery = {**SAMPLE_OPENTERMS}
        del no_discovery["discovery"]
        client = mock_client(no_discovery)
        assert client.discover("example.com") is None

    def test_receipt(self):
        client = mock_client(SAMPLE_OPENTERMS)
        result = client.check("example.com", "scrape_data")
        receipt = client.receipt("example.com", "scrape_data", result)
        assert receipt["domain"] == "example.com"
        assert receipt["allowed"] is False
        assert receipt["openterms_hash"] != ""
        assert "checked_at" in receipt


# --- Guard ---


class TestGuard:
    def test_allowed_passes_through(self):
        client = mock_client(SAMPLE_OPENTERMS)
        dummy = DummyTool()
        guard = OpenTermsGuard(tool=dummy, action="read_content", client=client)
        result = guard.invoke("https://example.com/page")
        assert result == "executed: https://example.com/page"

    def test_denied_blocks(self):
        client = mock_client(SAMPLE_OPENTERMS)
        dummy = DummyTool()
        guard = OpenTermsGuard(tool=dummy, action="scrape_data", client=client)
        result = guard.invoke("https://example.com/data")
        assert "blocked" in result.lower()
        assert "scrape_data" in result

    def test_no_file_permissive_default(self):
        client = mock_client(None)
        dummy = DummyTool()
        guard = OpenTermsGuard(tool=dummy, action="read_content", client=client)
        result = guard.invoke("https://example.com/page")
        assert result == "executed: https://example.com/page"

    def test_no_file_strict_blocks(self):
        client = mock_client(None)
        dummy = DummyTool()
        guard = OpenTermsGuard(
            tool=dummy, action="read_content", client=client, strict=True
        )
        result = guard.invoke("https://example.com/page")
        assert "strict mode" in result.lower()

    def test_no_domain_passes_through(self):
        client = mock_client(SAMPLE_OPENTERMS)
        dummy = DummyTool()
        guard = OpenTermsGuard(tool=dummy, action="scrape_data", client=client)
        result = guard.invoke("some query without a domain")
        assert result == "executed: some query without a domain"

    def test_on_denied_callback(self):
        client = mock_client(SAMPLE_OPENTERMS)
        dummy = DummyTool()
        denied_log = []
        guard = OpenTermsGuard(
            tool=dummy,
            action="scrape_data",
            client=client,
            on_denied=lambda d, a, r: denied_log.append((d, a)),
        )
        guard.invoke("https://example.com/data")
        assert len(denied_log) == 1
        assert denied_log[0] == ("example.com", "scrape_data")


# --- Checker ---


class TestChecker:
    def test_check_returns_json(self):
        client = mock_client(SAMPLE_OPENTERMS)
        checker = OpenTermsChecker(client=client)
        result = checker.invoke("example.com read_content")
        parsed = json.loads(result)
        assert parsed["check"]["allowed"] is True
        assert "receipt" in parsed

    def test_check_denied(self):
        client = mock_client(SAMPLE_OPENTERMS)
        checker = OpenTermsChecker(client=client)
        result = checker.invoke("example.com scrape_data")
        parsed = json.loads(result)
        assert parsed["check"]["allowed"] is False

    def test_bad_input(self):
        client = mock_client(SAMPLE_OPENTERMS)
        checker = OpenTermsChecker(client=client)
        with pytest.raises(ValueError, match="Expected"):
            checker.invoke("justadomain")


# --- Callback ---


class TestCallback:
    def test_logs_check_on_tool_start(self):
        client = mock_client(SAMPLE_OPENTERMS)
        handler = OpenTermsCallbackHandler(
            default_action="read_content", client=client
        )
        from uuid import uuid4

        handler.on_tool_start(
            serialized={"name": "web_search"},
            input_str="https://example.com/pricing",
            run_id=uuid4(),
        )
        assert len(handler.checks) == 1
        assert handler.checks[0]["allowed"] is True
        assert "receipt" in handler.checks[0]

    def test_skips_non_url_input(self):
        client = mock_client(SAMPLE_OPENTERMS)
        handler = OpenTermsCallbackHandler(client=client)
        from uuid import uuid4

        handler.on_tool_start(
            serialized={"name": "calculator"},
            input_str="2 + 2",
            run_id=uuid4(),
        )
        assert len(handler.checks) == 0

    def test_on_check_callback(self):
        client = mock_client(SAMPLE_OPENTERMS)
        log = []
        handler = OpenTermsCallbackHandler(
            client=client,
            on_check=lambda r: log.append(r),
        )
        from uuid import uuid4

        handler.on_tool_start(
            serialized={"name": "fetch"},
            input_str="https://example.com/api",
            run_id=uuid4(),
        )
        assert len(log) == 1
