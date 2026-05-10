# langchain-openterms

Permission-aware LangChain agents that check a domain's `openterms.json`
before taking action.

**Fail-closed by default.** If the site has no `openterms.json`, or the
permission is not explicitly granted, execution is blocked. You must
opt in to permissive behavior explicitly with `fail_closed=False`.

## Installation

```bash
pip install langchain-openterms

# With openterms-py SDK (recommended, requires >=0.3.1):
pip install "langchain-openterms[sdk]"
```

## Canonical Permission Keys

Only these 7 keys are recognized:

| Key | Meaning |
|-----|---------|
| `read_content` | Read / display page content |
| `scrape_data` | Automated scraping / crawling |
| `api_access` | Access the domain's API |
| `create_account` | Create a user account |
| `make_purchases` | Complete a purchase |
| `post_content` | Post or publish content |
| `allow_training` | Use content for model training |

## Fail-Closed Defaults

The following states **block execution by default**:

- `null` / `None` — domain unreachable or SDK error
- `no_openterms_json` — no `openterms.json` file found
- `not_specified` — key absent from `openterms.json`
- `low-confidence` — validator confidence too low
- `conditional` — permission has unverifiable conditions
- `denied` — explicit deny

Only an explicit `allowed: true` in `openterms.json` permits execution.

## Quick Start: OpenTermsGuard

Wrap any LangChain tool. Execution is blocked unless permission is explicitly
granted.

```python
from langchain_community.tools import BraveSearch
from langchain_openterms import OpenTermsGuard

search = BraveSearch.from_api_key(api_key="...", search_kwargs={"count": 3})

# Fail-closed by default — blocks if no openterms.json or not explicitly allowed
guarded_search = OpenTermsGuard(
    tool=search,
    action="read_content",
)

result = guarded_search.invoke("https://example.com/pricing")

if "blocked" in result.lower():
    # denied, not_specified, missing file, low-confidence — all blocked
    print("Cannot proceed:", result)
else:
    print("Allowed:", result)
```

**The guard never silently proceeds on ambiguous results.** If the check
returns anything other than an explicit allow, the tool returns a block
message.

### Permissive Opt-In (explicit, not recommended for production)

```python
# Only use fail_closed=False when you have a deliberate reason
permissive_guard = OpenTermsGuard(
    tool=search,
    action="read_content",
    fail_closed=False,  # pass through when no openterms.json found
)
```

With `fail_closed=False`, only an explicit `denied` blocks. Missing files
and unspecified permissions pass through.

### Denial Callback

```python
denied_domains = []

guard = OpenTermsGuard(
    tool=search,
    action="scrape_data",
    on_denied=lambda domain, action, result: denied_domains.append(domain),
)
```

## Agent Tool: OpenTermsChecker

Agents can call this tool directly to check permissions before deciding
whether to proceed.

```python
import json
from langchain_openterms import OpenTermsChecker

checker = OpenTermsChecker()

# Agent calls: "<domain> <action>"
result_json = checker.invoke("example.com scrape_data")
parsed = json.loads(result_json)

# Gate strictly on allowed=True
if parsed["check"]["allowed"] is True:
    # Only here is execution safe
    pass
else:
    # blocked: denied, not_specified, missing file, low-confidence, conditional
    print("Blocked:", parsed["check"]["reason"])
```

## Passive Observer: OpenTermsCallbackHandler

Logs permission checks without blocking. Use this for monitoring only.
**Does not enforce permissions** — use `OpenTermsGuard` for that.

```python
from langchain_openterms import OpenTermsCallbackHandler

handler = OpenTermsCallbackHandler(
    default_action="read_content",
    on_check=lambda r: print(f"{r['domain']}: allowed={r['allowed']}"),
)

agent.invoke({"input": "..."}, config={"callbacks": [handler]})

# Review after the run — these are domains where action was NOT explicitly allowed
for check in handler.checks:
    if check["allowed"] is not True:
        print(f"Would be blocked: {check['domain']} — {check['reason']}")
```

## bool() Truthiness

Check results use strict truthiness: `bool(result)` is `True` only for
explicitly allowed results. Do not rely on truthiness for dict results —
always check `result["allowed"] is True` explicitly.

## Using with openterms-py SDK

Install with SDK support for the most accurate results:

```bash
pip install "langchain-openterms[sdk]"
# Requires openterms-py>=0.3.1
```

The SDK applies fail-closed semantics at the check level. Empty caches,
unreachable domains, and missing files all return non-allow decisions —
never a permissive default.

## openterms-py Dependency

- **With SDK** (`[sdk]` extra or `openterms-py>=0.3.1` installed): Uses the SDK
  for all permission checks. Requires openterms-py>=0.3.1 for correct
  fail-closed behavior.
- **Without SDK**: Falls back to a built-in HTTP client with equivalent
  fail-closed semantics. Missing files and unspecified keys return
  `allowed=None`, which blocks under `fail_closed=True` (the default).

## Version History

- **0.4.0** — Fail-closed by default (`fail_closed=True`). Blocks null,
  not_specified, no_openterms_json, low-confidence, conditional, denied.
  `fail_closed=False` opt-in for permissive behavior. Canonical keys only.
  openterms-py>=0.3.1 required for SDK mode.
- **0.3.1** — Initial public release.
