"""Client adapter for OpenTerms protocol.

Uses the openterms-py SDK (>=0.3.1) if installed, adapting its dataclass-based
API to the dict-based interface the framework integrations expect.
Falls back to a minimal built-in implementation if the SDK is not installed.

Fail-closed semantics: ``allowed`` is True only for an explicit allow decision.
All other states (None, not_specified, no_openterms_json, low-confidence,
conditional, denied) map to a non-True ``allowed`` value. The callers
(guard, checker) treat any non-True allowed as blocked under fail_closed=True.

SDK (openterms-py>=0.3.1) returns:
    check()    -> result.decision ("allow"/"deny"/"not_specified"), result.raw_value
    discover() -> DiscoveryResult dataclass (.mcp_servers, .api_specs)
    receipt()  -> receipt.decision, receipt.timestamp
    fetch()    -> dict | None

Framework integrations expect:
    check()    -> {"allowed": bool|None, "reason": str, ...}
    discover() -> {"mcp_servers": [...], "api_specs": [...]} | None
    receipt()  -> {"allowed": bool|None, "checked_at": str, ...}
    fetch()    -> dict | None
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Optional


# Canonical permission keys per OpenTerms Schema v0.3.0
CANONICAL_PERMISSION_KEYS = frozenset([
    "read_content",
    "scrape_data",
    "api_access",
    "create_account",
    "make_purchases",
    "post_content",
    "allow_training",
])


try:
    from openterms.client import OpenTermsClient as _SDKClient
    _HAS_SDK = True
except ImportError:
    try:
        from openterms import OpenTermsClient as _SDKClient
        _HAS_SDK = True
    except ImportError:
        _HAS_SDK = False


def _decision_to_allowed(decision) -> Optional[bool]:
    """Convert SDK decision string to allowed bool.

    Returns True only for explicit "allow". All other values
    (deny, not_specified, no_openterms_json, low-confidence,
    conditional, or unknown types) return False or None.
    """
    if isinstance(decision, bool):
        return decision if decision else False
    if isinstance(decision, str):
        if decision == "allow":
            return True
        elif decision in ("deny", "denied"):
            return False
        # not_specified, no_openterms_json, low-confidence, conditional, unknown
        return None
    return None


class OpenTermsClient:
    """Wraps the SDK or uses a built-in fallback. All methods return plain dicts.

    Requires openterms-py>=0.3.1 for fail-closed SDK behavior. If the SDK is
    not installed, the built-in fallback applies fail-closed semantics directly.
    """

    WELL_KNOWN_PATH = "/.well-known/openterms.json"
    FALLBACK_PATH = "/openterms.json"

    def __init__(self, cache_ttl: int = 3600, timeout: int = 5, registry_url: Optional[str] = None):
        self._cache_ttl = cache_ttl
        self._timeout = timeout

        if _HAS_SDK:
            kwargs = {}
            if cache_ttl:
                kwargs["default_ttl"] = cache_ttl
            if timeout:
                kwargs["timeout"] = timeout
            if registry_url:
                kwargs["registry_url"] = registry_url
            try:
                self._sdk = _SDKClient(**kwargs)
            except TypeError:
                try:
                    self._sdk = _SDKClient(default_ttl=cache_ttl)
                except TypeError:
                    self._sdk = _SDKClient()
        else:
            self._sdk = None
        self._cache: dict[str, dict[str, Any]] = {}

    @property
    def using_sdk(self) -> bool:
        return self._sdk is not None

    def fetch(self, domain: str) -> Optional[dict]:
        if self._sdk and not self._cache:
            return self._sdk.fetch(domain)
        return self._fetch_fallback(domain)

    def check(self, domain: str, action: str) -> dict:
        """Check whether ``action`` is permitted on ``domain``.

        Returns a dict with:
          - allowed: True (explicit allow), False (explicit deny), or None
                     (missing file, unspecified, low-confidence, conditional)
          - reason: human-readable explanation
          - domain, action

        Callers must treat any non-True ``allowed`` as blocked under
        fail-closed semantics.
        """
        if self._sdk and not self._cache:
            result = self._sdk.check(domain, action)
            if isinstance(result, dict):
                decision = result.get("decision")
                raw = result.get("raw_value")
            else:
                decision = getattr(result, "decision", None)
                raw = getattr(result, "raw_value", None)

            allowed = _decision_to_allowed(decision)

            if allowed is True:
                if isinstance(raw, dict):
                    reason = f"Permission '{action}' explicitly allowed: {json.dumps(raw)}"
                else:
                    reason = f"Permission '{action}' is explicitly allowed by openterms.json."
            elif allowed is False:
                reason = f"Permission '{action}' is denied by openterms.json."
            else:
                # None: not_specified, no_openterms_json, low-confidence, conditional, unknown
                if decision in (None, "no_openterms_json"):
                    reason = f"No openterms.json found for this domain. Permission '{action}' cannot be verified."
                elif decision == "not_specified":
                    reason = f"Permission '{action}' is not specified in openterms.json."
                elif decision == "low-confidence":
                    reason = f"Permission '{action}' check returned low-confidence result."
                elif decision == "conditional":
                    reason = f"Permission '{action}' is conditional — conditions cannot be automatically verified."
                else:
                    reason = f"Permission '{action}' status unknown (decision: {decision!r})."

            return {"domain": domain, "action": action, "allowed": allowed, "reason": reason}
        return self._check_fallback(domain, action)

    def discover(self, domain: str) -> Optional[dict]:
        if self._sdk and not self._cache:
            result = self._sdk.discover(domain)
            if result is None:
                return None
            mcp = getattr(result, "mcp_servers", []) or []
            apis = getattr(result, "api_specs", []) or []
            return {
                "mcp_servers": [vars(s) if hasattr(s, "__dict__") else s for s in mcp],
                "api_specs": [vars(s) if hasattr(s, "__dict__") else s for s in apis],
            }
        return self._discover_fallback(domain)

    def receipt(self, domain: str, action: str, decision: dict) -> dict:
        return self._receipt_fallback(domain, action, decision)

    # --- Fallback implementations (no SDK) ---

    def _fetch_fallback(self, domain: str) -> Optional[dict]:
        import requests
        now = time.time()
        cached = self._cache.get(domain)
        if cached and (now - cached["fetched_at"]) < self._cache_ttl:
            return cached["data"]
        for path in [self.WELL_KNOWN_PATH, self.FALLBACK_PATH]:
            try:
                resp = requests.get(f"https://{domain}{path}", timeout=self._timeout, allow_redirects=True)
                if resp.status_code == 200:
                    data = resp.json()
                    self._cache[domain] = {"data": data, "fetched_at": now}
                    return data
            except Exception:
                continue
        self._cache[domain] = {"data": None, "fetched_at": now}
        return None

    def _check_fallback(self, domain: str, action: str) -> dict:
        data = self.fetch(domain)
        if data is None:
            return {
                "domain": domain,
                "action": action,
                "allowed": None,
                "reason": f"No openterms.json found for this domain. Permission '{action}' cannot be verified.",
            }
        permissions = data.get("permissions", {})
        if action in permissions:
            value = permissions[action]
            if isinstance(value, bool):
                reason = f"Permission '{action}' is {'explicitly allowed' if value else 'denied'} by openterms.json."
                return {"domain": domain, "action": action, "allowed": value, "reason": reason}
            if isinstance(value, dict):
                inner_allowed = value.get("allowed")
                if inner_allowed is True:
                    reason = f"Permission '{action}' explicitly allowed: {json.dumps(value)}"
                elif inner_allowed is False:
                    reason = f"Permission '{action}' denied: {json.dumps(value)}"
                else:
                    reason = f"Permission '{action}' is conditional or unspecified: {json.dumps(value)}"
                return {"domain": domain, "action": action, "allowed": inner_allowed, "reason": reason}
            if value is None:
                return {
                    "domain": domain,
                    "action": action,
                    "allowed": None,
                    "reason": f"Permission '{action}' is explicitly null in openterms.json.",
                }
        return {
            "domain": domain,
            "action": action,
            "allowed": None,
            "reason": f"Permission '{action}' is not specified in openterms.json.",
        }

    def _discover_fallback(self, domain: str) -> Optional[dict]:
        data = self.fetch(domain)
        if data is None:
            return None
        return data.get("discovery")

    def _receipt_fallback(self, domain: str, action: str, decision: dict) -> dict:
        data = self.fetch(domain)
        content_hash = ""
        if data is not None:
            content_hash = hashlib.sha256(
                json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        return {
            "domain": domain,
            "action": action,
            "allowed": decision.get("allowed"),
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "openterms_hash": "unknown",
        }
