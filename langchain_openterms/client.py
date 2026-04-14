"""Client adapter for OpenTerms protocol.

Uses the openterms-py SDK if installed, adapting its dataclass-based API
to the dict-based interface the framework integrations expect.
Falls back to a minimal built-in implementation if the SDK is not installed.

SDK returns:
    check()    -> result.decision ("allow"/"deny"/"not_specified"), result.raw_value
    discover() -> DiscoveryResult dataclass (.mcp_servers, .api_specs)
    receipt()  -> receipt.decision, receipt.timestamp
    fetch()    -> dict | None (matches)

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
    if isinstance(decision, bool):
        return decision
    if isinstance(decision, str):
        if decision == "allow":
            return True
        elif decision == "deny":
            return False
    return None


class OpenTermsClient:
    """Wraps the SDK or uses a built-in fallback. All methods return plain dicts."""

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
                # If constructor doesn't accept all kwargs, try minimal
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
        if self._sdk:
            return self._sdk.fetch(domain)
        return self._fetch_fallback(domain)

    def check(self, domain: str, action: str) -> dict:
        if self._sdk:
            result = self._sdk.check(domain, action)
            decision = getattr(result, "decision", None)
            allowed = _decision_to_allowed(decision)
            raw = getattr(result, "raw_value", None)
            if allowed is None and raw is None:
                reason = f"Permission '{action}' not specified in openterms.json."
            elif allowed is None:
                reason = "No openterms.json found for this domain."
            elif isinstance(raw, dict):
                reason = f"Permission '{action}': {json.dumps(raw)}"
            else:
                reason = f"Permission '{action}' is {'allowed' if allowed else 'denied'} by openterms.json."
            return {"domain": domain, "action": action, "allowed": allowed, "reason": reason}
        return self._check_fallback(domain, action)

    def discover(self, domain: str) -> Optional[dict]:
        if self._sdk:
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
        if self._sdk:
            try:
                # SDK expects decision as string, not dict
                allowed = decision.get("allowed")
                if allowed is True:
                    sdk_decision = "allow"
                elif allowed is False:
                    sdk_decision = "deny"
                else:
                    sdk_decision = "not_specified"
                result = self._sdk.receipt(domain, action, sdk_decision)
                return {
                    "domain": domain,
                    "action": action,
                    "allowed": _decision_to_allowed(getattr(result, "decision", sdk_decision)),
                    "checked_at": getattr(result, "timestamp", datetime.now(timezone.utc).isoformat()),
                    "openterms_hash": getattr(result, "openterms_hash", ""),
                }
            except Exception:
                pass
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
            return {"domain": domain, "action": action, "allowed": None, "reason": "No openterms.json found for this domain."}
        permissions = data.get("permissions", {})
        if action in permissions:
            value = permissions[action]
            if isinstance(value, bool):
                return {"domain": domain, "action": action, "allowed": value, "reason": f"Permission '{action}' is {'allowed' if value else 'denied'} by openterms.json."}
            if isinstance(value, dict):
                return {"domain": domain, "action": action, "allowed": value.get("allowed"), "reason": f"Permission '{action}': {json.dumps(value)}"}
        return {"domain": domain, "action": action, "allowed": None, "reason": f"Permission '{action}' not specified in openterms.json."}

    def _discover_fallback(self, domain: str) -> Optional[dict]:
        data = self.fetch(domain)
        if data is None:
            return None
        return data.get("discovery")

    def _receipt_fallback(self, domain: str, action: str, decision: dict) -> dict:
        data = self.fetch(domain)
        content_hash = ""
        if data is not None:
            content_hash = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {
            "domain": domain, "action": action, "allowed": decision.get("allowed"),
            "checked_at": datetime.now(timezone.utc).isoformat(), "openterms_hash": content_hash,
        }
